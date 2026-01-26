"""
ForceVLA model implementation in JAX/Flax for TactilePi05.
Ported from ForceVLA's pi0_force.py.

Key architecture:
1. Pi0.5 base model with flow matching for action generation
2. LIMoE-based tactile fusion
3. Tactile history (16 timesteps x 30 dims) used for fusion
4. No tactile prediction loss - only action prediction

The model follows Pi0.5's flow matching approach for action generation,
with LIMoE-based tactile fusion added to the prefix output.
"""

import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.limoe_simple as _limoe
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0ForceVLAConfig(_model.BaseModelConfig):
    """Configuration for ForceVLA model."""
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Model specific defaults
    action_dim: int = 32
    action_horizon: int = 16
    max_token_len: int = 200  # Pi0.5 default

    # Tactile configuration
    tactile_input_dim: int = 30  # 15 left + 15 right
    tactile_history_len: int = 16  # Number of history timesteps

    # Pi0.5 mode (uses adaRMSNorm for timestep injection)
    pi05: bool = True

    # For Pi0.5: state input is part of discrete language tokens
    # This is used by ModelTransformFactory for tokenization
    discrete_state_input: bool = True

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05 if self.pi05 else _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0ForceVLA":
        return Pi0ForceVLA(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                # Tactile history: (batch, history_len, 30)
                tactile=jax.ShapeDtypeStruct([batch_size, self.tactile_history_len, self.tactile_input_dim], jnp.float32),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        import openpi.shared.nnx_utils as nnx_utils
        
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter)
            if "lora" not in self.action_expert_variant:
                filters.append(nnx.Not(action_expert_params_filter))
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter)
            has_lora = True

        if has_lora:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0ForceVLA(_model.BaseModel):
    """ForceVLA model: Pi0.5 with LIMoE-based tactile fusion.
    
    Architecture:
    1. Prefix (VLM): Images + Text → SigLIP + Gemma Embed
    2. Tactile Fusion: LIMoE(concat[prefix_out, tactile_tokens])
    3. Suffix (Action Expert): Noisy Actions + Time → Action Expert with adaRMS
    4. Action Output: limoe_out[:, -action_horizon:] + suffix_out[:, -action_horizon:]
    """

    def __init__(self, config: Pi0ForceVLAConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.tactile_input_dim = config.tactile_input_dim
        self.tactile_history_len = config.tactile_history_len

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # PaliGemma LLM (2-stream: VLM + Action Expert)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])

        # SigLIP image encoder
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Tactile projection: (history_len, 30) → (history_len, paligemma_width)
        self.tactile_in_proj = nnx.Linear(config.tactile_input_dim, paligemma_config.width, rngs=rngs)

        # LIMoE for tactile fusion
        self.limoe = nnx_bridge.ToNNX(
            _limoe.LIMoEBlock(
                mlp_dim=paligemma_config.width,
                num_experts=4,
                num_top_k=1,
                num_heads=paligemma_config.num_heads,
                out_dim=action_expert_config.width,
            )
        )
        # Initialize LIMoE with dummy input
        self.limoe.lazy_init(jnp.zeros((32, 200, paligemma_config.width)), True, rngs=rngs)

        # Action projections
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Time embedding MLP (for Pi0.5 with adaRMS)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed images and language tokens for prefix (VLM stream)."""
        input_mask = []
        ar_mask = []
        tokens = []

        # Embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            ar_mask += [False] * image_tokens.shape[1]

        # Embed language tokens
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """Embed noisy actions and timestep for suffix (Action stream)."""
        input_mask = []
        ar_mask = []
        tokens = []

        if not self.pi05:
            # Add state token for Pi0 (not Pi0.5)
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        # Action embedding
        action_tokens = self.action_in_proj(noisy_actions)

        # Time embedding
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        if self.pi05:
            # Pi0.5: Use adaRMS for timestep injection
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # Pi0: Mix timestep + action using MLP
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def embed_tactile_history(self, tactile_history: at.Array) -> at.Array:
        """Embed tactile history to multiple tokens.
        
        Args:
            tactile_history: (batch, history_len, 30) tactile sensor history
            
        Returns:
            tactile_tokens: (batch, history_len, paligemma_width)
        """
        # Project each timestep: (B, T, 30) → (B, T, paligemma_width)
        tactile_tokens = self.tactile_in_proj(tactile_history)
        return tactile_tokens

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """Compute flow matching loss for action prediction."""
        preprocess_rng, noise_rng, time_rng, dropout_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embed prefix (images + text)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # Embed suffix (noisy actions + time)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

        # Embed tactile history: (B, history_len, 30) → (B, history_len, paligemma_width)
        # Get tactile from observation (should be set by data loader)
        tactile_history = getattr(observation, 'tactile', None)
        if tactile_history is None:
            # Fallback: zero tactile history
            batch_size = actions.shape[0]
            tactile_history = jnp.zeros((batch_size, self.tactile_history_len, self.tactile_input_dim))
        
        tactile_tokens = self.embed_tactile_history(tactile_history)

        # Construct attention masks
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # Forward pass through PaliGemma + Action Expert
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], 
            mask=attn_mask, 
            positions=positions,
            adarms_cond=[None, adarms_cond]
        )

        # LIMoE fusion: concat prefix_out with tactile_tokens
        limoe_input = jnp.concatenate([prefix_out, tactile_tokens], axis=1)
        # Pass rngs for dropout and jitter (MoE router) when training (deterministic=False)
        if train:
            dropout_rng, jitter_rng = jax.random.split(dropout_rng, 2)
            limoe_rngs = nnx.Rngs(dropout=dropout_rng, jitter=jitter_rng)
        else:
            limoe_rngs = None
        limoe_out = self.limoe(limoe_input, not train, rngs=limoe_rngs)  # deterministic = not train

        # Extract action output from LIMoE
        # The last `tactile_history_len` tokens of limoe_out correspond to tactile positions
        tactile_hist_len = tactile_tokens.shape[1]
        limoe_action_out = limoe_out[0][:, -tactile_hist_len:]

        # Suffix output's last action_horizon tokens
        suffix_action_out = suffix_out[:, -self.action_horizon:]

        # Handle case where tactile_hist_len != action_horizon
        if tactile_hist_len != self.action_horizon:
            limoe_action_out = limoe_action_out[:, -self.action_horizon:]

        # Combine LIMoE output with suffix output
        combined_out = limoe_action_out + suffix_action_out

        # Project to action dimension
        v_t = self.action_out_proj(combined_out)

        # Flow matching loss
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Sample actions using flow matching ODE integration."""
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Embed prefix and compute KV cache
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out_fix, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], 
            mask=prefix_attn_mask, 
            positions=positions
        )

        # Embed tactile history
        tactile_history = getattr(observation, 'tactile', None)
        if tactile_history is None:
            tactile_history = jnp.zeros((batch_size, self.tactile_history_len, self.tactile_input_dim))
        tactile_tokens = self.embed_tactile_history(tactile_history)

        # Pre-compute LIMoE output (fixed during denoising)
        limoe_input = jnp.concatenate([prefix_out_fix, tactile_tokens], axis=1)
        limoe_out_fixed = self.limoe(limoe_input, True)  # deterministic=True
        tactile_hist_len = tactile_tokens.shape[1]

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )

            # Attention mask for suffix attending to prefix (via cache) and itself
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            # Position IDs
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            # Forward with cache
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None

            # Combine with LIMoE output
            limoe_action_out = limoe_out_fixed[0][:, -tactile_hist_len:]
            suffix_action_out = suffix_out[:, -self.action_horizon:]

            if tactile_hist_len != self.action_horizon:
                limoe_action_out = limoe_action_out[:, -self.action_horizon:]

            combined_out = limoe_action_out + suffix_action_out
            v_t = self.action_out_proj(combined_out)

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
