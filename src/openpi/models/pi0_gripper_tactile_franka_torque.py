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
    (Docstring omitted for brevity, same as original)
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


class Pi0JaxWithGripperTactileTorque(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.tactile_dim = config.tactile_input_dim
        self.torque_dim = config.torque_input_dim # Config에서 Torque Dim 가져오기

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        tactile_expert_config = _gemma.get_config(config.tactile_expert_variant)
        torque_expert_config = _gemma.get_config(config.torque_expert_variant)
        
        # NOTE: configs 순서를 데이터 입력 순서 [Prefix, Tactile, Torque, Action]에 맞게 정렬합니다.
        # 이렇게 해야 expert weight가 올바른 입력 스트림에 매핑됩니다.
        llm_configs = [paligemma_config, tactile_expert_config, torque_expert_config, action_expert_config]
        use_adarms = [False, True, True, True] if config.pi05 else [False, False, False, False]

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=llm_configs,
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=use_adarms)
        
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
        
        # --- Action Projections ---
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # --- Tactile Projections ---
        self.tactile_in_proj = nnx.Linear(self.tactile_dim, tactile_expert_config.width, rngs=rngs)
        self.tactile_out_proj = nnx.Linear(tactile_expert_config.width, self.tactile_dim, rngs=rngs)

        # --- Torque Projections ---
        self.torque_in_proj = nnx.Linear(self.torque_dim, torque_expert_config.width, rngs=rngs)
        self.torque_out_proj = nnx.Linear(torque_expert_config.width, self.torque_dim, rngs=rngs)

        if config.pi05:
            # Action Time MLP
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

            # Tactile Time MLP
            self.tactile_mlp_in = nnx.Linear(tactile_expert_config.width, tactile_expert_config.width, rngs=rngs)
            self.tactile_mlp_out = nnx.Linear(tactile_expert_config.width, tactile_expert_config.width, rngs=rngs)

            # Torque Time MLP (Added)
            self.torque_mlp_in = nnx.Linear(torque_expert_config.width, torque_expert_config.width, rngs=rngs)
            self.torque_mlp_out = nnx.Linear(torque_expert_config.width, torque_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            # Note: Non-pi05 fallback for auxiliary streams omitted for brevity/consistency with original code

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
        """Embeds Tactile Stream"""
        input_mask = []
        ar_mask = []
        tokens = []

        # History
        hist_tokens = self.tactile_in_proj(tactile_history)
        tokens.append(hist_tokens)
        input_mask.append(jnp.ones(hist_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [False] * hist_tokens.shape[1]

        # Future
        future_tokens = self.tactile_in_proj(noisy_tactile_future)
        
        time_emb = posemb_sincos(timestep, self.tactile_in_proj.out_features, min_period=4e-3, max_period=4.0)
        
        if self.pi05:
            time_emb = self.tactile_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.tactile_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            adarms_cond = time_emb
        else:
            adarms_cond = None 
            
        tokens.append(future_tokens)
        input_mask.append(jnp.ones(future_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (future_tokens.shape[1] - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        
        return tokens, input_mask, ar_mask, adarms_cond

    def embed_torque(
        self, 
        torque_history: at.Float[at.Array, "b h d"], 
        noisy_torque_future: at.Float[at.Array, "b f d"], 
        timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """Embeds Torque Stream: Identical logic to Tactile"""
        input_mask = []
        ar_mask = []
        tokens = []

        # History
        hist_tokens = self.torque_in_proj(torque_history)
        tokens.append(hist_tokens)
        input_mask.append(jnp.ones(hist_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [False] * hist_tokens.shape[1]

        # Future
        future_tokens = self.torque_in_proj(noisy_torque_future)
        
        time_emb = posemb_sincos(timestep, self.torque_in_proj.out_features, min_period=4e-3, max_period=4.0)
        
        if self.pi05:
            time_emb = self.torque_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.torque_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            adarms_cond = time_emb
        else:
            adarms_cond = None 
            
        tokens.append(future_tokens)
        input_mask.append(jnp.ones(future_tokens.shape[:2], dtype=jnp.bool_))
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

        preprocess_rng, noise_rng_act, noise_rng_tac, noise_rng_tor, time_rng = jax.random.split(rng, 5)

        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        tactile_future = kwargs.get("tactile_future", None)
        torque_future = kwargs.get("torque_future", None) # Kwargs에서 Torque Future 가져오기

        # 1. Noise Generation (Action, Tactile, Torque)
        noise_act = jax.random.normal(noise_rng_act, actions.shape)
        noise_tac = jax.random.normal(noise_rng_tac, tactile_future.shape)
        noise_tor = jax.random.normal(noise_rng_tor, torque_future.shape)
        
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        
        # Diffused Inputs
        x_t_act = time_expanded * noise_act + (1 - time_expanded) * actions
        x_t_tac = time_expanded * noise_tac + (1 - time_expanded) * tactile_future
        x_t_tor = time_expanded * noise_tor + (1 - time_expanded) * torque_future
        
        # Targets
        u_t_act = noise_act - actions
        u_t_tac = noise_tac - tactile_future
        u_t_tor = noise_tor - torque_future

        # 2. Embedding Streams
        # Stream 1: Prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        
        # Stream 2: Tactile
        tactile_tokens, tactile_mask, tactile_ar_mask, adarms_tac = self.embed_tactile(
            observation.tactile_history, x_t_tac, time
        )
        
        # Stream 3: Torque
        torque_tokens, torque_mask, torque_ar_mask, adarms_tor = self.embed_torque(
            observation.torque_history, x_t_tor, time
        )
        
        # Stream 4: Action
        action_tokens, action_mask, action_ar_mask, adarms_act = self.embed_action(
            observation, x_t_act, time
        )

        # 3. Combine & Mask
        # Order: [Prefix, Tactile, Torque, Action]
        input_mask = jnp.concatenate([prefix_mask, tactile_mask, torque_mask, action_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, tactile_ar_mask, torque_ar_mask, action_ar_mask], axis=0)
        
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # 4. Forward Pass
        (prefix_out, tactile_out, torque_out, action_out), _ = self.PaliGemma.llm(
            [prefix_tokens, tactile_tokens, torque_tokens, action_tokens], 
            mask=attn_mask, 
            positions=positions, 
            adarms_cond=[None, adarms_tac, adarms_tor, adarms_act]
        )

        # 5. Projection & Loss
        future_len_tac = tactile_future.shape[1]
        v_t_tac = self.tactile_out_proj(tactile_out[:, -future_len_tac:])
        
        future_len_tor = torque_future.shape[1]
        v_t_tor = self.torque_out_proj(torque_out[:, -future_len_tor:])
        
        v_t_act = self.action_out_proj(action_out[:, -self.action_horizon:])

        loss_tac = jnp.mean(jnp.square(v_t_tac - u_t_tac), axis=-1).mean(axis=-1)
        loss_tor = jnp.mean(jnp.square(v_t_tor - u_t_tor), axis=-1).mean(axis=-1)
        loss_act = jnp.mean(jnp.square(v_t_act - u_t_act), axis=-1).mean(axis=-1)
        
        # Weighting
        w_tactile = 0.1
        w_torque = 0.1
        
        return loss_tac * w_tactile + loss_tor * w_torque + loss_act

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise_action: at.Float[at.Array, "b ah ad"] | None = None,
        noise_tactile: at.Float[at.Array, "b fh td"] | None = None,
        noise_torque: at.Float[at.Array, "b fh trd"] | None = None,
    ) -> tuple[_model.Actions, at.Float[at.Array, "b fh td"], at.Float[at.Array, "b fh trd"]]:
        
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        
        # Initialize Noise
        if noise_action is None:
            noise_action = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        if noise_tactile is None:
            noise_tactile = jax.random.normal(rng, (batch_size, self.action_horizon, self.tactile_dim))
        if noise_torque is None:
            noise_torque = jax.random.normal(rng, (batch_size, self.action_horizon, self.torque_dim))

        # 1. Fill KV Cache with Prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        
        # Pass [Prefix, None, None, None]
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None, None], 
            mask=prefix_attn_mask, 
            positions=positions
        )

        # Denoising Loop
        def step(carry):
            x_t_act, x_t_tac, x_t_tor, time = carry
            
            # Embed Dynamic parts (Tactile, Torque, Action)
            tactile_tokens, tactile_mask, tactile_ar_mask, adarms_tac = self.embed_tactile(
                observation.tactile_history, x_t_tac, jnp.broadcast_to(time, batch_size)
            )
            
            torque_tokens, torque_mask, torque_ar_mask, adarms_tor = self.embed_torque(
                observation.torque_history, x_t_tor, jnp.broadcast_to(time, batch_size)
            )
            
            action_tokens, action_mask, action_ar_mask, adarms_act = self.embed_action(
                observation, x_t_act, jnp.broadcast_to(time, batch_size)
            )

            # Construct Masks (Generation Phase)
            # Generation block = [Tactile, Torque, Action]
            gen_mask = jnp.concatenate([tactile_mask, torque_mask, action_mask], axis=1)
            gen_ar_mask = jnp.concatenate([tactile_ar_mask, torque_ar_mask, action_ar_mask], axis=0)
            
            gen_attn_mask = make_attn_mask(gen_mask, gen_ar_mask)
            
            # Prefix cross-attention mask
            prefix_attn_mask_for_gen = einops.repeat(
                prefix_mask, "b p -> b s p", s=gen_mask.shape[1]
            )
            
            full_attn_mask = jnp.concatenate([prefix_attn_mask_for_gen, gen_attn_mask], axis=-1)
            
            # Positions
            prefix_len = jnp.sum(prefix_mask, axis=-1)[:, None]
            positions = prefix_len + jnp.cumsum(gen_mask, axis=-1) - 1


            @nnx.remat
            def llm_forward_pass(module, inputs, mask, positions, adarms_cond):
                return module(inputs, mask=mask, positions=positions, adarms_cond=adarms_cond)
            
            # Forward
            (prefix_out, tactile_out, torque_out, action_out), _ = llm_forward_pass(
                self.PaliGemma.llm,
                [None, tactile_tokens, torque_tokens, action_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_tac, adarms_tor, adarms_act],
            )

            # Project Outputs
            v_t_tac = self.tactile_out_proj(tactile_out[:, -noise_tactile.shape[1]:])
            v_t_tor = self.torque_out_proj(torque_out[:, -noise_torque.shape[1]:])
            v_t_act = self.action_out_proj(action_out[:, -self.action_horizon:])

            # Euler Step
            next_x_act = x_t_act + dt * v_t_act
            next_x_tac = x_t_tac + dt * v_t_tac
            next_x_tor = x_t_tor + dt * v_t_tor
            
            return next_x_act, next_x_tac, next_x_tor, time + dt

        def cond(carry):
            _, _, _, time = carry
            return time >= -dt / 2

        x_0_act, x_0_tac, x_0_tor, _ = jax.lax.while_loop(
            cond, step, (noise_action, noise_tactile, noise_torque, 1.0)
        )
        
        return x_0_act, x_0_tac, x_0_tor