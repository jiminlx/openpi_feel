"""
ForceVLA model implementation in PyTorch.
Ported from ForceVLA's pi0_force.py (JAX/Flax implementation).

Key architecture differences from TactilePi05:
1. Uses LIMoE for tactile fusion (not decoupled stream)
2. Tactile is used as context only (no future prediction)
3. Tactile history (16 timesteps x 30 dims) is used for LIMoE fusion

The model follows Pi0.5's flow matching approach for action generation,
with LIMoE-based tactile fusion added to the prefix output.

LIMoE Fusion Details:
- Input: concat[prefix_out (N tokens), tactile_tokens (16 tokens)]
- Output: LIMoE processes all tokens with self-attention + MoE
- Action prediction uses: limoe_out[:, -action_horizon:] + suffix_out[:, -action_horizon:]
  (Since tactile_history has 16 tokens = action_horizon, the tactile positions' 
   LIMoE outputs are used for action prediction, which aligns with the original 
   ForceVLA design where tactile information should influence action generation)
"""

import logging
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch_forcevla import PaliGemmaWithExpertForceVLA
from openpi.models_pytorch.limoe_pytorch import LIMoEBlock
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Create 2D attention masks from padding and AR masks."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0PytorchForceVLA(nn.Module):
    """ForceVLA model: Pi0.5 with LIMoE-based tactile fusion.
    
    Architecture:
    1. Prefix (VLM): Images + Text → SigLIP + Gemma Embed
    2. Suffix (Action): State + Noisy Actions → Action Expert
    3. Tactile Fusion: LIMoE(concat[prefix_out, tactile_tokens])
    4. Action Output: limoe_out[:, -action_horizon:] + suffix_out[:, -action_horizon:] → action_out_proj
    
    Key difference from TactilePi05:
    - Tactile history (16×30) is used for LIMoE fusion (not decoupled stream)
    - LIMoE fusion instead of joint attention
    - No tactile prediction loss (action-only)
    
    Tactile History Usage:
    - Input: (B, tactile_history_len, 30) where tactile_history_len = 16
    - Projected to: (B, 16, paligemma_width)
    - LIMoE input: concat[prefix_out, tactile_tokens] → (B, prefix_len + 16, width)
    - LIMoE output's last `action_horizon` tokens are used for action prediction
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # PaliGemma with Action Expert (2-stream)
        self.paligemma_with_expert = PaliGemmaWithExpertForceVLA(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        # Tactile projection (30 dims → paligemma width)
        self.tactile_dim = config.tactile_input_dim
        self.tactile_in_proj = nn.Linear(self.tactile_dim, paligemma_config.width)

        # LIMoE for tactile fusion
        self.limoe = LIMoEBlock(
            mlp_dim=paligemma_config.width,
            num_experts=4,
            num_top_k=1,
            num_heads=paligemma_config.num_heads,
            out_dim=action_expert_config.width,
            dropout_rate=0.1,
        )

        # Action projections
        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        # State projection (for pi0, not pi05)
        if not self.pi05:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")

        # Gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Check transformers_replace installation
        msg = "transformers_replace is not installed correctly."
        try:
            from transformers.models.siglip import check
            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for ForceVLA model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for ForceVLA model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images and language tokens for prefix (VLM stream)."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state and noisy actions for suffix (Action stream)."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)
            att_masks += [1]

        # Time embedding
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Action embedding
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def embed_tactile_history(self, tactile_history: torch.Tensor) -> torch.Tensor:
        """Embed tactile history to multiple tokens.
        
        Args:
            tactile_history: (batch, history_len, 30) tactile sensor history
                             where history_len is typically 16
            
        Returns:
            tactile_tokens: (batch, history_len, paligemma_width)
        """
        # Project each timestep: (B, T, 30) → (B, T, paligemma_width)
        tactile_tokens = self.tactile_in_proj(tactile_history)
        return tactile_tokens

    def forward(
        self, 
        observation, 
        actions, 
        tactile_history=None,  # (B, hist_len, 30) - full history used for LIMoE fusion
        tactile_future=None,   # Not used in ForceVLA (no tactile prediction)
        noise=None, 
        time=None
    ) -> dict:
        """Training forward pass with flow matching loss.
        
        ForceVLA uses the full tactile history (16×30) for LIMoE fusion.
        No tactile prediction loss is computed - only action prediction.
        
        Args:
            observation: Observation dict with images, state, prompt
            actions: (B, action_horizon, action_dim) target actions
            tactile_history: (B, hist_len, 30) tactile sensor history
            tactile_future: Not used (ignored)
            noise: Optional noise for flow matching
            time: Optional timestep for flow matching
            
        Returns:
            dict with 'loss' and 'loss_action' keys
        """
        bsize = actions.shape[0]
        
        # Handle tactile history
        if tactile_history is not None:
            # Use full tactile history: (B, hist_len, 30)
            tactile_hist = tactile_history
        else:
            # Fallback: zero tactile history with default length matching action_horizon
            tactile_hist = torch.zeros(
                bsize, self.config.action_horizon, self.tactile_dim, 
                device=actions.device, dtype=actions.dtype
            )

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(bsize, actions.device)

        time_expanded = time[:, None, None]

        # Noisy actions (flow matching)
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions  # Target velocity

        # Preprocess observation
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        # Embed prefix (images + text)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Embed suffix (state + noisy actions)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        # Embed tactile history: (B, hist_len, 30) → (B, hist_len, paligemma_width)
        tactile_tokens = self.embed_tactile_history(tactile_hist)

        # Construct attention masks
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Cast to bfloat16 if needed
        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            tactile_tokens = tactile_tokens.to(dtype=torch.bfloat16)

        # Forward pass through PaliGemma + Action Expert
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            outputs, _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return outputs[0], outputs[1]

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        # LIMoE fusion: concat prefix_out with tactile_tokens
        # Input shape: (B, prefix_len + tactile_hist_len, paligemma_width)
        limoe_input = torch.cat([prefix_out, tactile_tokens], dim=1)
        limoe_out = self.limoe(limoe_input, deterministic=not self.training)

        # Extract action output from LIMoE
        # The last `action_horizon` tokens of limoe_out correspond to tactile positions
        # (since tactile_hist_len == action_horizon == 16)
        # This aligns with original ForceVLA where tactile info influences action prediction
        tactile_hist_len = tactile_tokens.shape[1]
        limoe_action_out = limoe_out[:, -tactile_hist_len:]  # (B, tactile_hist_len, action_expert_width)
        
        # suffix_out's last action_horizon tokens
        suffix_action_out = suffix_out[:, -self.config.action_horizon:]

        # Combine LIMoE output with suffix output
        # Note: tactile_hist_len should equal action_horizon for proper alignment
        # If they differ, we take the last action_horizon tokens from limoe_action_out
        if tactile_hist_len != self.config.action_horizon:
            limoe_action_out = limoe_action_out[:, -self.config.action_horizon:]
        
        combined_out = limoe_action_out + suffix_action_out
        combined_out = combined_out.to(dtype=torch.float32)

        v_t = self.action_out_proj(combined_out)

        # Flow matching loss (action only - no tactile prediction)
        loss = F.mse_loss(v_t, u_t, reduction="none").mean()

        return {"loss": loss, "loss_action": loss}

    def sample_actions(
        self, 
        device, 
        observation, 
        tactile_history=None,
        noise=None, 
        num_steps=10
    ) -> Tensor:
        """Inference: sample actions using flow matching ODE integration.
        
        Args:
            device: Target device
            observation: Observation dict
            tactile_history: (B, hist_len, 30) - full history used for LIMoE fusion
            noise: Optional initial noise
            num_steps: Number of denoising steps
            
        Returns:
            actions: (B, action_horizon, action_dim)
        """
        bsize = observation.state.shape[0]

        # Handle tactile history
        if tactile_history is not None:
            tactile_hist = tactile_history.to(device)  # (B, hist_len, 30)
        else:
            # Fallback: zero tactile history
            tactile_hist = torch.zeros(
                bsize, self.config.action_horizon, self.tactile_dim, 
                device=device, dtype=torch.float32
            )

        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Preprocess observation
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        # Embed prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute prefix KV cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Embed tactile history: (B, hist_len, 30) → (B, hist_len, paligemma_width)
        tactile_tokens = self.embed_tactile_history(tactile_hist)
        
        # We need prefix_out for LIMoE - run prefix forward again to get output
        prefix_out, _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        prefix_out = prefix_out[0]  # Get VLM output

        # Pre-compute LIMoE output (fixed during denoising)
        if prefix_out.dtype == torch.bfloat16:
            tactile_tokens = tactile_tokens.to(dtype=torch.bfloat16)
        limoe_input = torch.cat([prefix_out, tactile_tokens], dim=1)
        limoe_out_fixed = self.limoe(limoe_input, deterministic=True)
        
        # Store tactile history length for extraction
        tactile_hist_len = tactile_tokens.shape[1]

        # Denoising loop
        dt = -1.0 / num_steps
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            v_t = self._denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                limoe_out_fixed,
                tactile_hist_len,
            )

            x_t = x_t + dt * v_t
            time = time + dt

        return x_t

    def _denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        limoe_out_fixed,
        tactile_hist_len,
    ):
        """Single denoising step.
        
        Args:
            state: Robot state
            prefix_pad_masks: Padding masks for prefix
            past_key_values: KV cache from prefix forward
            x_t: Current noisy actions
            timestep: Current diffusion timestep
            limoe_out_fixed: Pre-computed LIMoE output (fixed during denoising)
            tactile_hist_len: Length of tactile history for proper extraction
        """
        # Embed suffix
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        bsize = prefix_pad_masks.shape[0]
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]

        # Attention mask: suffix attends to prefix (via cache) and itself
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        # Position IDs
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Cast if needed
        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        # Forward with cache
        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs[1]

        # Combine with LIMoE output
        # Extract last tactile_hist_len tokens from LIMoE output
        limoe_action_out = limoe_out_fixed[:, -tactile_hist_len:]
        suffix_action_out = suffix_out[:, -self.config.action_horizon:]
        
        # Handle case where tactile_hist_len != action_horizon
        if tactile_hist_len != self.config.action_horizon:
            limoe_action_out = limoe_action_out[:, -self.config.action_horizon:]

        combined_out = limoe_action_out + suffix_action_out
        combined_out = combined_out.to(dtype=torch.float32)

        v_t = self.action_out_proj(combined_out)

        return v_t
