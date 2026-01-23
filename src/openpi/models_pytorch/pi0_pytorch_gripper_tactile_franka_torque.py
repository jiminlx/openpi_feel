import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch_gripper_tactile_franka_torque import PaliGemmaWithExpertAndGripperTactileFrankaTorqueModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0PytorchWithGripperTactileFrankaTorque(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config # PI0Config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        tactile_expert_config = _gemma.get_config(config.tactile_expert_variant)
        torque_expert_config = _gemma.get_config(config.torque_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertAndGripperTactileFrankaTorqueModel(
            paligemma_config,
            action_expert_config,
            tactile_expert_config,
            torque_expert_config,
            use_adarms=[False, True, True, True] if self.pi05 else [False, False, False, False],
            precision=config.dtype,
        )
        
        # Tactile Encoder 
        # tactile dim: 30 
        self.tactile_dim = self.config.tactile_input_dim
        self.tactile_in_proj = nn.Linear(self.tactile_dim, tactile_expert_config.width)
        self.tactile_out_proj = nn.Linear(tactile_expert_config.width, self.tactile_dim)

        # Torque Encoder 
        # torque dim: 7 
        self.torque_dim = self.config.torque_input_dim
        self.torque_in_proj = nn.Linear(self.torque_dim, torque_expert_config.width)
        self.torque_out_proj = nn.Linear(torque_expert_config.width, self.torque_dim)

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

            self.tactile_time_mlp_in = nn.Linear(tactile_expert_config.width, tactile_expert_config.width)
            self.tactile_time_mlp_out = nn.Linear(tactile_expert_config.width, tactile_expert_config.width)

            self.torque_time_mlp_in = nn.Linear(torque_expert_config.width, torque_expert_config.width)
            self.torque_time_mlp_out = nn.Linear(torque_expert_config.width, torque_expert_config.width)

        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def set_action_stream_trainable(self, mode: bool = True):
        """
        Action Stream 관련 모듈들의 requires_grad를 제어합니다.
        mode=False 이면 Freeze(학습 안함), mode=True 이면 Unfreeze(학습 함)
        """
        requires_grad = mode
        
        # 1. Action Expert (Gemma)
        for param in self.paligemma_with_expert.gemma_expert.parameters():
            param.requires_grad = requires_grad
            
        # 2. Action Input/Output Projections
        for param in self.action_in_proj.parameters():
            param.requires_grad = requires_grad
        for param in self.action_out_proj.parameters():
            param.requires_grad = requires_grad
            
        # 3. Time MLPs (pi05 여부에 따라 다름)
        if self.pi05:
            # pi05인 경우 time_mlp_in/out이 Action 쪽에 관여
            for param in self.time_mlp_in.parameters():
                param.requires_grad = requires_grad
            for param in self.time_mlp_out.parameters():
                param.requires_grad = requires_grad
        else:
            # pi05가 아닌 경우
            for param in self.state_proj.parameters():
                param.requires_grad = requires_grad
            for param in self.action_time_mlp_in.parameters():
                param.requires_grad = requires_grad
            for param in self.action_time_mlp_out.parameters():
                param.requires_grad = requires_grad

        state_str = "Unfrozen" if mode else "Frozen"
        logging.info(f"Action Stream is now {state_str}")
        

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

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
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
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

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def embed_tactile_history(self, tactile_hist):
        """
        tactile_hist: (Batch, Hist_Len, Dim)
        """
        emb = self.tactile_in_proj(tactile_hist) # (B, T, Hidden)
        
        bsize = emb.shape[0]
        seq_len = emb.shape[1]
        
        pad_mask = torch.ones(bsize, seq_len, dtype=torch.bool, device=emb.device)
        
        # Attention Mask: [0] -> Context : Full Attention
        att_mask_val = [0] * seq_len
        
        return emb, pad_mask, att_mask_val


    def embed_tactile_future(self, tactile_future, timestep):

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.tactile_in_proj.out_features, 
            min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=tactile_future.dtype)

        tactile_emb = self.tactile_in_proj(tactile_future)  # (Batch, Tactile_Seq, Dim)

        def time_mlp_func(t_emb):
            x = self.tactile_time_mlp_in(t_emb)
            x = F.silu(x)
            x = self.tactile_time_mlp_out(x)
            return F.silu(x)
        
        adarms_cond = self._apply_checkpoint(time_mlp_func, time_emb)
        
        bsize = tactile_emb.shape[0]
        seq_len = tactile_emb.shape[1]
        
        pad_mask = torch.ones(bsize, seq_len, dtype=torch.bool, device=tactile_emb.device)
        
        # Attention Mask: [1] -> Generation 
        att_mask_val = [1] + [0] * (seq_len - 1)
        
        return tactile_emb, pad_mask, att_mask_val, adarms_cond

    
    def embed_torque_history(self, torque_hist):
        """
        torque_hist: (Batch, Hist_Len, Dim)
        """
        emb = self.torque_in_proj(torque_hist) # (B, T, Hidden)
        
        bsize = emb.shape[0]
        seq_len = emb.shape[1]
        
        pad_mask = torch.ones(bsize, seq_len, dtype=torch.bool, device=emb.device)
        
        # Attention Mask: [0] -> Context : Full Attention
        att_mask_val = [0] * seq_len
        
        return emb, pad_mask, att_mask_val


    def embed_torque_future(self, torque_future, timestep):

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.torque_in_proj.out_features, 
            min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=torque_future.dtype)

        torque_emb = self.torque_in_proj(torque_future)  # (Batch, Torque_Seq, Dim)

        def time_mlp_func(t_emb):
            x = self.torque_time_mlp_in(t_emb)
            x = F.silu(x)
            x = self.torque_time_mlp_out(x)
            return F.silu(x)
        
        adarms_cond = self._apply_checkpoint(time_mlp_func, time_emb)
        
        bsize = torque_emb.shape[0]
        seq_len = torque_emb.shape[1]
        
        pad_mask = torch.ones(bsize, seq_len, dtype=torch.bool, device=torque_emb.device)
        
        # Attention Mask: [1] -> Generation 
        att_mask_val = [1] + [0] * (seq_len - 1)
        
        return torque_emb, pad_mask, att_mask_val, adarms_cond

    def forward(self, observation, actions, tactile_history, tactile_future, torque_history, torque_future, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        noise_action = None
        noise_tactile = None
        noise_torque = None
    

        action_time = None
        tactile_time = None
        torque_time = None

        if noise is None:
            noise_action = self.sample_noise(actions.shape, actions.device)
            noise_tactile = self.sample_noise(tactile_future.shape, tactile_future.device)
            noise_torque = self.sample_noise(torque_future.shape, torque_future.device)

        if time is None:
            bsize = actions.shape[0]
            common_time = self.sample_time(bsize, actions.device)
            action_time = common_time
            tactile_time = common_time
            torque_time = common_time

        action_time_expanded = action_time[:, None, None]
        tactile_time_expanded = tactile_time[:, None, None]
        torque_time_expanded = torque_time[:, None, None]
        
        # Noisy Actions & Tactile
        t_action = action_time_expanded * noise_action + (1 - action_time_expanded) * actions
        u_action = noise_action - actions

        t_tactile = tactile_time_expanded * noise_tactile + (1 - tactile_time_expanded) * tactile_future
        u_tactile = noise_tactile - tactile_future

        t_torque = torque_time_expanded * noise_torque + (1 - torque_time_expanded) * torque_future
        u_torque = noise_torque - torque_future

        # (Stream 1) : Image + Text 
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        
        # (Stream 2-A) : Tactile History 
        tactile_hist_emb, tactile_hist_pad_masks, tactile_hist_att_masks = self.embed_tactile_history(tactile_history)

        # (Stream 2-B) : Tactile Future 
        tactile_future_embs, tactile_future_pad_masks, tactile_future_att_masks, tactile_future_adarms = self.embed_tactile_future(t_tactile, tactile_time)

        # (Stream 3-A) : Torque History 
        torque_hist_emb, torque_hist_pad_masks, torque_hist_att_masks = self.embed_torque_history(torque_history)

        # (Stream 3-B) : Torque Future 
        torque_future_embs, torque_future_pad_masks, torque_future_att_masks, torque_future_adarms = self.embed_torque_future(t_torque, torque_time)
       
        # (Stream 4) : Action 
        action_embs, action_pad_masks, action_att_masks, action_adarms = self.embed_suffix(state, t_action, action_time)
       
        
        # Merging Tactile Stream (History + Future)
        tactile_full_embs = torch.cat([tactile_hist_emb, tactile_future_embs], dim=1)
        tactile_pad_masks = torch.cat([tactile_hist_pad_masks, tactile_future_pad_masks], dim=1)

        # Merging Torque Stream (History + Future)
        torque_full_embs = torch.cat([torque_hist_emb, torque_future_embs], dim=1)
        torque_pad_masks = torch.cat([torque_hist_pad_masks, torque_future_pad_masks], dim=1)


        # Attention Mask List concat 
        tactile_attn_masks = tactile_hist_att_masks + tactile_future_att_masks
        tactile_attn_masks = torch.tensor(tactile_attn_masks, dtype=torch.bool, device=tactile_pad_masks.device)
        tactile_attn_masks = tactile_attn_masks[None, :].expand(tactile_pad_masks.shape[0], len(tactile_attn_masks))

        # Attention Mask List concat 
        torque_attn_masks = torque_hist_att_masks + torque_future_att_masks
        torque_attn_masks = torch.tensor(torque_attn_masks, dtype=torch.bool, device=torque_pad_masks.device)
        torque_attn_masks = torque_attn_masks[None, :].expand(torque_pad_masks.shape[0], len(torque_attn_masks))
        
        # Global Mask Construction 
        # Order : [Image/Text] -> [Torque full] -> [Action]
        pad_masks = torch.cat([prefix_pad_masks, tactile_pad_masks, torque_pad_masks, action_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, tactile_attn_masks, torque_attn_masks, action_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Casting 
        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            tactile_full_embs = tactile_full_embs.to(dtype=torch.bfloat16)
            torque_full_embs = torque_full_embs.to(dtype=torch.bfloat16)
            action_embs = action_embs.to(dtype=torch.bfloat16)

        # Forward Pass 
        def forward_func(prefix_embs, tactile_embs, torque_embs, action_embs, att_2d_masks_4d, position_ids, adarms_cond_tactile, adarms_cond_torque, adarms_cond_action):
            outputs, _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, tactile_embs, torque_embs, action_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond_tactile, adarms_cond_torque, adarms_cond_action], 
            )
            # outputs: [prefix_out, tactile_out, torque_out, action_out]
            return outputs[1],outputs[2],outputs[3]

        tactile_out_full, torque_out_full, action_out = self._apply_checkpoint(
            forward_func, 
            prefix_embs, tactile_full_embs, torque_full_embs, action_embs, 
            att_2d_masks_4d, position_ids, 
            tactile_future_adarms, torque_future_adarms, action_adarms
        )

        # Extraction & Projection & Loss 

        # Extraction Tactile (for future only)
        future_len = tactile_future_embs.shape[1]
        tactile_out_future = tactile_out_full[:, -future_len:] # (B, Future_Len, Hidden)
        tactile_out_future = tactile_out_future.to(dtype=torch.float32)

        # Extraction Torque (for future only)
        future_len = torque_future_embs.shape[1]
        torque_out_future = torque_out_full[:, -future_len:] # (B, Future_Len, Hidden)
        torque_out_future = torque_out_future.to(dtype=torch.float32)

        action_out = action_out[:, -self.config.action_horizon:]
        action_out = action_out.to(dtype=torch.float32)

        def projection_func(tactile_out, torque_out, act_out):
            return self.tactile_out_proj(tactile_out), self.torque_out_proj(torque_out), self.action_out_proj(act_out)

        tactile_out_future, torque_out_future, action_out = self._apply_checkpoint(projection_func, tactile_out_future, torque_out_future, action_out)

        # loss
        loss_tactile = F.mse_loss(u_tactile, tactile_out_future, reduction="none").mean()
        loss_torque = F.mse_loss(u_torque, torque_out_future, reduction="none").mean()
        loss_action = F.mse_loss(u_action, action_out, reduction="none").mean()

        w_torque = getattr(self.config, "loss_torque_weight", 0.1)
        w_tactile = getattr(self.config, "loss_tactile_weight", 0.1)
        
        if self.action_out_proj.weight.requires_grad:
            loss = loss_tactile * w_tactile + loss_torque * w_torque + loss_action

            return {"loss_tactile": loss_tactile, "loss_torque": loss_torque, "loss_action": loss_action, "loss": loss, "w_tactile": w_tactile, "w_torque": w_torque}
        else:
            loss = loss_tactile * w_tactile + loss_torque * w_torque

            return {"loss_tactile": loss_tactile, "loss_torque": loss_torque, "loss": loss, "w_tactile": w_tactile, "w_torque": w_torque}
    


    #### UPDATED CODE ####
    @torch.no_grad()
    def sample_actions(self, device, observation, tactile_history, tactile_future, torque_history, torque_future, noise_action=None, noise_tactile=None, noise_torque=None, num_steps=10) -> tuple[Tensor, Tensor]:
        """
        Do a full inference forward and compute the action and future tactile, torque.
        Returns: (actions, tactile_future, torque_future)
        """
        bsize = observation.state.shape[0]
        
        # 1. Initialize Noise for Action, Tactile, and Torque Future
        if noise_action is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise_action = self.sample_noise(actions_shape, device)
            
        if noise_tactile is None:
            tactile_shape = (bsize, self.config.action_horizon, self.config.tactile_input_dim) # same horizon as action
            noise_tactile = self.sample_noise(tactile_shape, device)

        if noise_torque is None:
            torque_shape = (bsize, self.config.action_horizon, self.config.torque_input_dim)
            noise_torque = self.sample_noise(torque_shape, device)

        # 2. Preprocess and Cache Prefix (Image + Text)
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        # Prefix Forward to get KV Cache
        # Note: We pass None for Torque and Action slots to match the [Prefix, Torque, Action] structure
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None, None, None], 
            use_cache=True,
        )

        # 3. Pre-compute Static Embeddings (Tactile History, Torque History)
        tactile_hist_emb, tactile_hist_pad_masks, tactile_hist_att_masks = self.embed_tactile_history(tactile_history)
        # Torque History does not change during the diffusion loop
        torque_hist_emb, torque_hist_pad_masks, torque_hist_att_masks = self.embed_torque_history(torque_history)

        # 4. Denoising Loop
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t_action = noise_action
        x_t_tactile = noise_tactile
        x_t_torque = noise_torque
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            
            # Compute velocity (v_t) for both streams
            v_t_tactile, v_t_torque, v_t_action = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t_tactile,
                x_t_torque,
                x_t_action,
                tactile_hist_emb,
                tactile_hist_pad_masks,
                tactile_hist_att_masks,
                torque_hist_emb,
                torque_hist_pad_masks,
                torque_hist_att_masks,
                expanded_time,
            )

            # Euler step
            x_t_action = x_t_action + dt * v_t_action
            x_t_tactile = x_t_tactile + dt * v_t_tactile
            x_t_torque = x_t_torque + dt * v_t_torque
            time += dt

        return x_t_action, x_t_tactile, x_t_torque

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t_tactile,
        x_t_torque,
        x_t_action,
        tactile_hist_emb,
        tactile_hist_pad_masks,
        tactile_hist_att_masks,
        torque_hist_emb,
        torque_hist_pad_masks,
        torque_hist_att_masks,
        timestep,
    ):
        """Apply one denoising step for both Torque and Action streams."""
        
        # --- 1. Embed Dynamic Parts (Torque Future & Action) ---
        
        # Torque Future (Noisy)
        tor_fut_embs, tor_fut_pad_masks, tor_fut_att_masks, adarms_cond_torque = self.embed_torque_future(x_t_torque, timestep)
        
        # Tactile Future (Noisy)
        tac_fut_embs, tac_fut_pad_masks, tac_fut_att_masks, adarms_cond_tactile = self.embed_tactile_future(x_t_tactile, timestep)
        
        # Action (Noisy)
        act_embs, act_pad_masks, act_att_masks, adarms_cond_action = self.embed_suffix(state, x_t_action, timestep)

        # --- 2. Construct Full Inputs ---
        
        # Merge Tactile History + Future
        tactile_full_embs = torch.cat([tactile_hist_emb, tac_fut_embs], dim=1)
        tactile_pad_masks = torch.cat([tactile_hist_pad_masks, tac_fut_pad_masks], dim=1)
        
        # Merge Torque History + Future
        torque_full_embs = torch.cat([torque_hist_emb, tor_fut_embs], dim=1)
        torque_pad_masks = torch.cat([torque_hist_pad_masks, tor_fut_pad_masks], dim=1)
        
        # Merge Attention Masks for Tactile Stream
        tactile_attn_masks_list = tactile_hist_att_masks + tac_fut_att_masks
        tactile_attn_masks = torch.tensor(tactile_attn_masks_list, dtype=torch.bool, device=tactile_pad_masks.device)
        bsize = prefix_pad_masks.shape[0]
        tactile_attn_masks = tactile_attn_masks[None, :].expand(bsize, len(tactile_attn_masks_list))
        
        # Merge Attention Masks for Torque Stream
        torque_attn_masks_list = torque_hist_att_masks + tor_fut_att_masks
        torque_attn_masks = torch.tensor(torque_attn_masks_list, dtype=torch.bool, device=torque_pad_masks.device)
        bsize = prefix_pad_masks.shape[0]
        torque_attn_masks = torque_attn_masks[None, :].expand(bsize, len(torque_attn_masks_list))

        # --- 3. Construct Masks for Attention (Prefix + Tactile + Torque + Action) ---
        
        # The "Generation" block consists of [Tactile Full, Torque Full, Action]
        gen_pad_masks = torch.cat([tactile_pad_masks, torque_pad_masks, act_pad_masks], dim=1)
        gen_att_masks = torch.cat([tactile_attn_masks, torque_attn_masks, act_att_masks], dim=1)
        
        gen_len = gen_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]

        # Allow Generation block to attend to Prefix (via KV Cache)
        # Shape: (Batch, Gen_Len, Prefix_Len)
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, gen_len, prefix_len)

        # Allow Generation block to attend to itself
        # Shape: (Batch, Gen_Len, Gen_Len)
        gen_att_2d_masks = make_att_2d_masks(gen_pad_masks, gen_att_masks)

        # Concatenate to get mask for [Gen queries] x [Prefix keys, Gen keys]
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, gen_att_2d_masks], dim=2)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        # --- 4. Position IDs ---
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        # Position IDs for the generation block start after the prefix
        position_ids = prefix_offsets + torch.cumsum(gen_pad_masks, dim=1) - 1

        # --- 5. Forward Pass ---
        
        # Precision handling
        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            tactile_full_embs = tactile_full_embs.to(dtype=torch.bfloat16)
            torque_full_embs = torque_full_embs.to(dtype=torch.bfloat16)
            act_embs = act_embs.to(dtype=torch.bfloat16)

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        # inputs_embeds structure: [Prefix(None), Tactile, Torque, Action]
        # We pass None for prefix as we rely on past_key_values
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, tactile_full_embs, torque_full_embs, act_embs],
            use_cache=False, 
            adarms_cond=[None, adarms_cond_tactile, adarms_cond_torque, adarms_cond_action],
        )

        # --- 6. Extract and Project Outputs ---
        
        # Outputs: [None, tactile_out, torque_out, action_out]
        tactile_out_full = outputs_embeds[1]
        torque_out_full = outputs_embeds[2]
        action_out = outputs_embeds[2]

        # Extract only the Future part of Tactile
        future_len = tac_fut_embs.shape[1]
        tactile_out_future = tactile_out_full[:, -future_len:]

        # Extract only the Future part of Torque
        future_len = tor_fut_embs.shape[1]
        torque_out_future = torque_out_full[:, -future_len:]
        
        # Extract Action (Action horizon)
        action_out = action_out[:, -self.config.action_horizon:]

        # Cast to float32 for diffusion step
        tactile_out_future = tactile_out_future.to(dtype=torch.float32)
        torque_out_future = torque_out_future.to(dtype=torch.float32)
        action_out = action_out.to(dtype=torch.float32)

        # Project to original space
        v_t_tactile = self.tactile_out_proj(tactile_out_future)
        v_t_torque = self.torque_out_proj(torque_out_future)
        v_t_action = self.action_out_proj(action_out)

        return v_t_tactile, v_t_action, v_t_torque