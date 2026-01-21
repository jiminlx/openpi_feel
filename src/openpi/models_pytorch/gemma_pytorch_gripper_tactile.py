from typing import Literal
import torch
from torch import nn
from transformers import GemmaForCausalLM, PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

class PaliGemmaWithExpertAndGripperTactileModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        tactile_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False, False] # [VLM, , Action]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        # Action Expert Config 
        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[2], # Index 2 for Action
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        # Tactile Expert Config (Gemma)
        tactile_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=tactile_expert_config.head_dim,
            hidden_size=tactile_expert_config.width,
            intermediate_size=tactile_expert_config.mlp_dim,
            num_attention_heads=tactile_expert_config.num_heads,
            num_hidden_layers=tactile_expert_config.depth,
            num_key_value_heads=tactile_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1], # Index 1 for Tactile
            adarms_cond_dim=tactile_expert_config.width if use_adarms[1] else None,
        )

        # model init
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.tactile_expert = GemmaForCausalLM(config=tactile_expert_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)

        # Embedding layer remove (use external embedding)
        self.tactile_expert.model.embed_tokens = None
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None, # [VLM, Tactile, Action] 순서 예상
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        # inputs_embeds = [vlm_emb, tactile_emb, action_emb]
        if adarms_cond is None:
            adarms_cond = [None, None, None]
        
        # Models: [VLM, Tactile, Action]
        models = [self.paligemma.language_model, self.tactile_expert.model, self.gemma_expert.model]
        num_layers = self.paligemma.config.text_config.num_hidden_layers

        # Gradient Checkpointing setup
        use_gradient_checkpointing = (
            hasattr(self.gemma_expert.model, "gradient_checkpointing")
            and self.gemma_expert.model.gradient_checkpointing
            and self.training
        )

        
        # Core Logic: 3-Stream Joint Attention Layer
        def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
            # inputs_embeds : [Tensor(VLM), Tensor(Tactile), Tensor(Action)] list
            
            query_states = []
            key_states = []
            value_states = []
            gates = []
            
            # 1. Independent Projection (Q, K, V) for each modality
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None: continue # skip if the modality is not present

                layer = models[i].layers[layer_idx]
                
                hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])
                gates.append(gate)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                
                # Independent Linear Layers for each modality (Q, K, V)
                q = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                k = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                v = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                query_states.append(q)
                key_states.append(k)
                value_states.append(v)

            # 2. Concatenate for Joint Attention
            query_states = torch.cat(query_states, dim=2)
            key_states = torch.cat(key_states, dim=2)
            value_states = torch.cat(value_states, dim=2)

            # Apply RoPE
            dummy_tensor = torch.zeros(
                query_states.shape[0], 
                query_states.shape[2], 
                query_states.shape[-1],
                device=query_states.device, 
                dtype=query_states.dtype,
            )

            # Note: position_ids must be calculated outside for all modalities
            cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )

            scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

            # 3. Global Attention
            att_output, _ = modeling_gemma.eager_attention_forward(
                self.paligemma.language_model.layers[layer_idx].self_attn,
                query_states, key_states, value_states, attention_mask, scaling,
            )
            
            # Get head_dim from the current layer, not from the model
            head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
            att_output = att_output.reshape(att_output.shape[0], -1, 1 * 8 * head_dim)

            # Process layer outputs
            outputs_embeds = []
            start_pos = 0

            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None: 
                    outputs_embeds.append(None)
                    continue

                layer = models[i].layers[layer_idx]
                end_pos = start_pos + hidden_states.shape[1]

                # Attention Output Projection (use o_proj for each modality)
                curr_att_out = att_output[:, start_pos:end_pos]
                if curr_att_out.dtype != layer.self_attn.o_proj.weight.dtype:
                    curr_att_out = curr_att_out.to(layer.self_attn.o_proj.weight.dtype)
                out_emb = layer.self_attn.o_proj(curr_att_out)

                # First Residual (Gated Residual)
                out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])
                after_first_residual = out_emb.clone()
                out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                
                # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                    out_emb = out_emb.to(dtype=torch.bfloat16)

                out_emb = layer.mlp(out_emb)

                # Second Residual (Gated Residual)
                out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)
                
                outputs_embeds.append(out_emb)
                start_pos = end_pos

            return outputs_embeds

        # Layer Loop
        for layer_idx in range(num_layers):
            if use_gradient_checkpointing:
                inputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_layer_complete,
                    layer_idx, 
                    inputs_embeds, 
                    attention_mask, 
                    position_ids, 
                    adarms_cond,
                    use_reentrant=False, 
                    preserve_rng_state=False,
                )
            else:
                inputs_embeds = compute_layer_complete(
                    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                )

        # Final Norm
        final_outputs = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                final_outputs.append(out)
            else:
                final_outputs.append(None)

        return final_outputs, None # (past_key_values는 구현 생략 - training 위주)