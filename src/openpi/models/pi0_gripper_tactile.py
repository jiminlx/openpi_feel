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
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
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


class Pi0JaxWithGripperTactile(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.tactile_dim = config.tactile_input_dim

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        tactile_expert_config = _gemma.get_config(config.tactile_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config, tactile_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True, True] if config.pi05 else [False, False, False])
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
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

            self.tactile_mlp_in = nnx.Linear(tactile_expert_config.width, tactile_expert_config.width, rngs=rngs)
            self.tactile_mlp_out = nnx.Linear(tactile_expert_config.width, tactile_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.tactile_in_proj = nnx.Linear(self.tactile_dim, tactile_expert_config.width, rngs=rngs)
        self.tactile_out_proj = nnx.Linear(tactile_expert_config.width, self.tactile_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
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
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def embed_tactile(
        self, 
        tactile_history: at.Float[at.Array, "b h d"], 
        noisy_tactile_future: at.Float[at.Array, "b f d"], 
        timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """Embeds Tactile Stream: History (Static) + Future (Noisy & Time-Conditioned)"""
        input_mask = []
        ar_mask = []
        tokens = []

        # --- Stream 2-A: Tactile History ---
        # (Batch, Hist_Len, Dim) -> (Batch, Hist_Len, Emb)
        hist_tokens = self.tactile_in_proj(tactile_history)
        tokens.append(hist_tokens)
        input_mask.append(jnp.ones(hist_tokens.shape[:2], dtype=jnp.bool_))
        # History shares context with Prefix (PyTorch: [0] * seq_len)
        ar_mask += [False] * hist_tokens.shape[1]

        # --- Stream 2-B: Tactile Future ---
        # (Batch, Future_Len, Dim) -> (Batch, Future_Len, Emb)
        future_tokens = self.tactile_in_proj(noisy_tactile_future)
        
        # Time Embedding
        time_emb = posemb_sincos(timestep, self.tactile_in_proj.out_features, min_period=4e-3, max_period=4.0)
        
        if self.pi05:
            # Time MLP for adaRMS
            time_emb = self.tactile_time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.tactile_time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            adarms_cond = time_emb
        else:
            # Fallback if not pi05 (simplified for brevity, matching action logic)
            adarms_cond = None 
            # Note: Add non-adaRMS MLP injection here if needed
            
        tokens.append(future_tokens)
        input_mask.append(jnp.ones(future_tokens.shape[:2], dtype=jnp.bool_))
        
        # Attention Mask: [True] (New Block) + [False] (Internal Attention)
        # PyTorch: [1] + [0] * (seq_len - 1)
        ar_mask += [True] + ([False] * (future_tokens.shape[1] - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        
        return tokens, input_mask, ar_mask, adarms_cond

    @at.typecheck
    def embed_action(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        # Rename of embed_suffix to correspond to Action stream specifically
        input_mask = []
        ar_mask = []
        tokens = []
        
        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
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

    @override
    def compute_loss(
        self, 
        rng: at.KeyArrayLike, 
        observation: _model.Observation, 
        actions: _model.Actions, 
        *, 
        train: bool = False,
        **kwargs
    ) -> at.Float[at.Array, "*b ah"]:

        preprocess_rng, noise_rng_act, noise_rng_tac, time_rng = jax.random.split(rng, 4)


        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        tactile_future = kwargs.get("tactile_future", None)
        # 1. Noise Generation (Action & Tactile)
        noise_act = jax.random.normal(noise_rng_act, actions.shape)
        noise_tac = jax.random.normal(noise_rng_tac, tactile_future.shape)
        
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        
        # Diffused Inputs
        x_t_act = time_expanded * noise_act + (1 - time_expanded) * actions
        x_t_tac = time_expanded * noise_tac + (1 - time_expanded) * tactile_future
        
        # Targets
        u_t_act = noise_act - actions
        u_t_tac = noise_tac - tactile_future

        # 2. Embedding Streams
        # Stream 1: Prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        
        # Stream 2: Tactile (History + Future)
        # Assuming observation has tactile_history
        tactile_tokens, tactile_mask, tactile_ar_mask, adarms_tac = self.embed_tactile(
            observation.tactile_history, x_t_tac, time
        )
        
        # Stream 3: Action
        action_tokens, action_mask, action_ar_mask, adarms_act = self.embed_action(
            observation, x_t_act, time
        )

        # 3. Combine & Mask
        # Order: [Prefix, Tactile, Action]
        input_mask = jnp.concatenate([prefix_mask, tactile_mask, action_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, tactile_ar_mask, action_ar_mask], axis=0)
        
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # 4. Forward Pass
        # Pass list of 3 inputs for the 3 configs
        (prefix_out, tactile_out, action_out), _ = self.PaliGemma.llm(
            [prefix_tokens, tactile_tokens, action_tokens], 
            mask=attn_mask, 
            positions=positions, 
            adarms_cond=[None, adarms_tac, adarms_act]
        )

        # 5. Projection & Loss
        # Tactile Output: Only last 'future_len' tokens
        future_len = tactile_future.shape[1]
        v_t_tac = self.tactile_out_proj(tactile_out[:, -future_len:])
        
        # Action Output
        v_t_act = self.action_out_proj(action_out[:, -self.action_horizon:])

        loss_tac = jnp.mean(jnp.square(v_t_tac - u_t_tac), axis=-1).mean(axis=-1)
        loss_act = jnp.mean(jnp.square(v_t_act - u_t_act), axis=-1).mean(axis=-1)
        
        # Weighting (assuming 0.1 from PyTorch code)
        w_tactile = 0.1
        
        return loss_tac * w_tactile + loss_act

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise_action: at.Float[at.Array, "b ah ad"] | None = None,
        noise_tactile: at.Float[at.Array, "b fh td"] | None = None,
    ) -> tuple[_model.Actions, at.Float[at.Array, "b fh td"]]:
        
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        
        # Initialize Noise
        if noise_action is None:
            noise_action = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        if noise_tactile is None:
            # Assuming tactile horizon same as action horizon or defined in config
            noise_tactile = jax.random.normal(rng, (batch_size, self.action_horizon, self.tactile_dim))

        # 1. Fill KV Cache with Prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        
        # Pass [Prefix, None, None] to prime the cache
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None], 
            mask=prefix_attn_mask, 
            positions=positions
        )

        # Denoising Loop
        def step(carry):
            x_t_act, x_t_tac, time = carry
            
            # Embed Dynamic parts
            # Note: Tactile History is re-embedded here with Future. 
            # Optimization: Could cache history embedding, but it's cheap compared to LLM.
            
            # Tactile Stream Embedding (History + Future)
            tactile_tokens, tactile_mask, tactile_ar_mask, adarms_tac = self.embed_tactile(
                observation.tactile_history, x_t_tac, jnp.broadcast_to(time, batch_size)
            )
            
            # Action Stream Embedding
            action_tokens, action_mask, action_ar_mask, adarms_act = self.embed_action(
                observation, x_t_act, jnp.broadcast_to(time, batch_size)
            )

            # Construct Masks (Generation Phase)
            # Generation block = [Tactile, Action]
            gen_mask = jnp.concatenate([tactile_mask, action_mask], axis=1)
            gen_ar_mask = jnp.concatenate([tactile_ar_mask, action_ar_mask], axis=0)
            
            # Mask logic:
            # 1. Gen attends to Prefix (via KV Cache) -> needs prefix_len
            # 2. Gen attends to itself (via gen_ar_mask)
            
            gen_attn_mask = make_attn_mask(gen_mask, gen_ar_mask)
            
            # Prefix cross-attention mask: repeat prefix mask for gen rows
            prefix_attn_mask_for_gen = einops.repeat(
                prefix_mask, "b p -> b s p", s=gen_mask.shape[1]
            )
            
            full_attn_mask = jnp.concatenate([prefix_attn_mask_for_gen, gen_attn_mask], axis=-1)
            
            # Positions
            prefix_len = jnp.sum(prefix_mask, axis=-1)[:, None]
            positions = prefix_len + jnp.cumsum(gen_mask, axis=-1) - 1

            # Forward (No prefix input, just generation streams)
            (prefix_out, tactile_out, action_out), _ = self.PaliGemma.llm(
                [None, tactile_tokens, action_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_tac, adarms_act],
            )

            # Project Outputs
            v_t_tac = self.tactile_out_proj(tactile_out[:, -noise_tactile.shape[1]:])
            v_t_act = self.action_out_proj(action_out[:, -self.action_horizon:])

            # Euler Step
            next_x_act = x_t_act + dt * v_t_act
            next_x_tac = x_t_tac + dt * v_t_tac
            
            return next_x_act, next_x_tac, time + dt

        def cond(carry):
            _, _, time = carry
            return time >= -dt / 2

        x_0_act, x_0_tac, _ = jax.lax.while_loop(cond, step, (noise_action, noise_tactile, 1.0))
        
        return x_0_act, x_0_tac
