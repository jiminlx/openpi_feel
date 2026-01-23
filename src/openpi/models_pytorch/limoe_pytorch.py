"""
LIMoE (Lightweight Mixture of Experts) implementation in PyTorch.
Ported from ForceVLA's limoe_simple.py (JAX/Flax implementation).

This module provides:
- MlpBlock: Standard transformer MLP block
- Encoder1DBlock: Transformer encoder layer with self-attention and MLP
- MoeLayer: Sparse Mixture of Experts layer with token routing
- LIMoEBlock: Complete LIMoE block combining attention and MoE
"""

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(
        self,
        in_features: int,
        mlp_dim: int,
        out_dim: Optional[int] = None,
        dropout_rate: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.out_dim = out_dim if out_dim is not None else in_features
        self.dtype = dtype

        self.fc1 = nn.Linear(in_features, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, self.out_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        if not deterministic:
            x = self.dropout(x)
        x = self.fc2(x)
        if not deterministic:
            x = self.dropout(x)
        return x


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer with self-attention and MLP."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dtype = dtype

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MlpBlock(
            in_features=hidden_dim,
            mlp_dim=mlp_dim,
            out_dim=hidden_dim,
            dropout_rate=dropout_rate,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        # Self-attention block
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        if not deterministic:
            x = self.dropout1(x)
        x = residual + x

        # MLP block
        residual = x
        x = self.norm2(x)
        x = self.mlp(x, deterministic=deterministic)
        x = residual + x

        return x


class RouterWeights(nn.Module):
    """Router weights for token-to-expert assignment."""

    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        nn.init.xavier_uniform_(self.router.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.router(x)


class MoeLayer(nn.Module):
    """Sparse Mixture of Experts layer with per-token routing.
    
    This is a simplified implementation that uses "tokens choose experts" routing.
    Each token selects top-k experts based on router logits.
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int,
        num_experts: int = 4,
        num_selected_experts: int = 1,
        dropout_rate: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        # Router
        self.router = RouterWeights(hidden_dim, num_experts)

        # Expert MLPs
        self.experts = nn.ModuleList([
            MlpBlock(
                in_features=hidden_dim,
                mlp_dim=mlp_dim,
                out_dim=hidden_dim,
                dropout_rate=dropout_rate,
                dtype=dtype,
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            deterministic: If True, disable dropout
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Compute router logits and probabilities
        router_logits = self.router(x)  # (batch, seq, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts for each token
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.num_selected_experts, dim=-1
        )  # (batch, seq, k)

        # Normalize the selected probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        # For efficiency, we process all tokens through all experts and mask
        expert_outputs = torch.zeros_like(x)

        for expert_idx in range(self.num_experts):
            # Get mask for tokens routed to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # (batch, seq)
            
            if expert_mask.any():
                # Get the weight for this expert
                expert_weight = torch.where(
                    top_k_indices == expert_idx,
                    top_k_probs,
                    torch.zeros_like(top_k_probs)
                ).sum(dim=-1)  # (batch, seq)

                # Process through expert
                expert_out = self.experts[expert_idx](x, deterministic=deterministic)
                
                # Weighted addition
                expert_outputs = expert_outputs + expert_out * expert_weight.unsqueeze(-1)

        return expert_outputs


class LIMoEBlock(nn.Module):
    """LIMoE Block: Combines Transformer attention with Mixture of Experts.
    
    Architecture:
    1. Encoder1DBlock (Self-Attention + MLP)
    2. MoeLayer (Sparse MoE)
    3. Output projection to target dimension
    
    This matches ForceVLA's LIMoEBlock implementation.
    """

    def __init__(
        self,
        mlp_dim: int,
        num_experts: int = 4,
        num_top_k: int = 1,
        num_heads: int = 8,
        out_dim: int = 1024,
        dropout_rate: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.dtype = dtype

        # Encoder block (self-attention + MLP)
        self.encoder_block = Encoder1DBlock(
            hidden_dim=mlp_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            dtype=dtype,
        )

        # MoE layer
        self.moe = MoeLayer(
            hidden_dim=mlp_dim,
            mlp_dim=mlp_dim,
            num_experts=num_experts,
            num_selected_experts=num_top_k,
            dropout_rate=dropout_rate,
            dtype=dtype,
        )

        # Output projection
        self.out_proj = nn.Linear(mlp_dim, out_dim)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.normal_(self.out_proj.bias, std=1e-6)

    def forward(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, mlp_dim)
            deterministic: If True, disable dropout
            
        Returns:
            Output tensor of shape (batch_size, seq_len, out_dim)
        """
        # Encoder block
        x = self.encoder_block(x, deterministic=deterministic)

        # MoE layer with residual
        moe_out = self.moe(x, deterministic=deterministic)
        x = x + moe_out

        # Output projection
        x = self.out_proj(x)

        return x
