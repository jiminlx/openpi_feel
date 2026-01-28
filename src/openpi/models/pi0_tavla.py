import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
import openpi.models.pi0_config as pi0_config
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

logger = logging.getLogger("openpi")

# --- Helper Functions (유지) ---
def make_attn_mask(input_mask, mask_ar):
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)

@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
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


# --- Configuration ---

# --- Model Implementation (Method 1: Joint Torque) ---
class Pi0JaxTavla(_model.BaseModel):
    """
    EXPERT_HIS_C_L_FUT Implementation:
    - EXPERT: Torque is processed by the Action Expert (Suffix).
    - HIS: Uses Torque History.
    - C_L: Concatenates the Linearly projected history with the action tokens.
    - FUT: Trains on Future Torque (handled in compute_loss).
    """
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.torque_dim = 7
        self.torque_loss_weight = config.loss_torque_weight
        self.history_len = config.history_len
        
        # 1. Initialize Backbones
        paligemma_config = _gemma.get_config(config.paligemma_variant)      # Width ~ 2048
        action_expert_config = _gemma.get_config(config.action_expert_variant) # Width ~ 1024

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        
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

        # 2. Projection Layers for Torque History
        # We project Flattened History -> Expert Width (1024)
        # Fix: Ensure input dim accounts for history length (D * H)
        self.torque_proj_in = nnx.Linear(
            self.torque_dim * config.history_len, 
            2 * action_expert_config.width, 
            rngs=rngs
        )
        # OUTPUT goes to Expert, so use action_expert_config.width (1024)
        self.torque_proj_out = nnx.Linear(
            2 * action_expert_config.width, 
            action_expert_config.width, 
            rngs=rngs
        )

        # 3. Projection Layers for Diffusion (Suffix)
        # Input/Output dimensions are (action_dim + torque_dim)
        joint_dim = config.action_dim + self.torque_dim
        
        self.joint_in_proj = nnx.Linear(joint_dim, action_expert_config.width, rngs=rngs)
        
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        
        self.joint_out_proj = nnx.Linear(action_expert_config.width, joint_dim, rngs=rngs)

    def _project_torque_history(self, torque_history: at.Float[at.Array, "b h d"]) -> at.Float[at.Array, "b 1 emb"]:
        """Projects torque history to a single context token for the Expert."""
        batch_size = torque_history.shape[0]
        # Flatten history: [B, H, D] -> [B, H*D]
        torque_flat = torque_history.reshape(batch_size, -1)
        
        torque_hidden = self.torque_proj_in(torque_flat) 
        torque_hidden = nnx.swish(torque_hidden)
        # Add sequence dimension [B, 1, Emb]
        torque_token = self.torque_proj_out(torque_hidden)[:, None, :]
        return torque_token

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """
        Encodes Images and Text for the VLM.
        NOTE: Torque is REMOVED from here to avoid dimension mismatch.
        """
        input_mask = []
        ar_mask = []
        tokens = []

        # 1. Embed Images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

        # 2. Embed Language
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        # Concatenate VLM tokens (Width 2048)
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: at.Float[at.Array, "b h d"], timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """
        Encodes inputs for the Action Expert.
        Includes: Torque History (Context) + Noisy Joint State (Generation).
        """
        input_mask = []
        ar_mask = []
        tokens = []

        # --- 1. Embed Torque History (Context for Expert) ---
        # This is where "EXPERT_HIS" happens.
        torque_token = self._project_torque_history(obs.torque_history)
        tokens.append(torque_token)
        input_mask.append(jnp.ones(torque_token.shape[:2], dtype=jnp.bool_))
        # False means this token is NOT generated / is visible context
        ar_mask += [False] 
        #import pdb; pdb.set_trace()
        # --- 2. Embed Joint Action+Torque (Noisy Input) ---
        action_tokens = self.joint_in_proj(noisy_actions)
        
        # Time Embedding
        time_emb = posemb_sincos(timestep, self.joint_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        
        # Mix Action + Time
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_expert_tokens = self.action_time_mlp_out(action_time_tokens)
        
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        
        # Causal Masking: First action token is "start of generation" (True), others follow
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        # Concatenate Expert tokens (Width 1024)
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        
        adarms_cond = None
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False, **kwargs
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # --- KEY STEP 1: PREPARE JOINT TARGET ---
        # Get future torque from observation (assuming observation contains full sequence in .torque
        torque_future = kwargs.get("torque_future", None)
        
        # Update observation to remove future (so we don't cheat in prefix)
        obs_prefix = observation.replace(torque_history=observation.torque_history)
        
        # Concatenate: [Action, Torque]
        joint_targets = jnp.concatenate([actions, torque_future], axis=-1)
        
        # --- KEY STEP 2: DIFFUSION PROCESS ---
        batch_shape = joint_targets.shape[:-2]
        noise = jax.random.normal(noise_rng, joint_targets.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        
        # Forward Process (Add Noise)
        x_t = time_expanded * noise + (1 - time_expanded) * joint_targets
        u_t = noise - joint_targets 

        # --- KEY STEP 3: MODEL FORWARD ---
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs_prefix)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(obs_prefix, x_t, time)
        
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )

        # --- KEY STEP 4: PREDICT & SEPARATE ---
        # Project back to joint dimension
        v_t_joint = self.joint_out_proj(suffix_out[:, -self.action_horizon:])
        
        # Split back into Action and Torque
        v_t_action = v_t_joint[..., :self.action_dim]
        v_t_torque = v_t_joint[..., self.action_dim:]
        
        u_t_action = u_t[..., :self.action_dim]
        u_t_torque = u_t[..., self.action_dim:]

        # --- KEY STEP 5: CALCULATE LOSS ---
        action_loss = jnp.mean(jnp.square(v_t_action - u_t_action), axis=-1)
        torque_loss = jnp.mean(jnp.square(v_t_torque - u_t_torque), axis=-1)
        self.torque_loss_weight = 1.0
        return action_loss + self.torque_loss_weight * torque_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        # Initialize Noise for JOINT vector (Action + Torque)
        joint_dim = self.action_dim + self.torque_dim
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, joint_dim))

        # Fill KV Cache
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            # Embed x_t which contains BOTH action and torque
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache, adarms_cond=[None, adarms_cond]
            )
            
            # Prediction is also JOINT
            v_t = self.joint_out_proj(suffix_out[:, -self.action_horizon:])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0_joint, _ = jax.lax.while_loop(cond, step, (noise, 1.0))

        # Return only the action part (usually what we need for control), 
        # or return both if needed.
        x_0_action = x_0_joint[..., :self.action_dim]
        
        return x_0_action