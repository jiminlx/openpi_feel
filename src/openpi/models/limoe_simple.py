# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LIMoE implementation for TactilePi05 JAX training."""

from collections.abc import Callable
from typing import Any
from typing import Optional

import flax.linen as nn
import flax
from flax.linen import partitioning as flax_partitioning
import jax
import jax.numpy as jnp

from flaxformer.architectures.moe import routing
from flaxformer.architectures.moe import scatter_utils
from flaxformer.components import dense
from flaxformer.types import Array
from flaxformer.types import DType

Array = Any
PRNGKey = Any
Shape = tuple[int]
Dtype = Any


class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self, x):
        return x


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    out_dim: int | None = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        return nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer."""

    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, deterministic):
        """Applies Encoder1DBlock module."""
        # Attention block.
        y = nn.LayerNorm(name="encoder_in_norm", dtype=self.dtype)(inputs)
        y = nn.MultiHeadDotProductAttention(
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            num_heads=self.num_heads,
            force_fp32_for_softmax=True,
        )(y, y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        inputs = y + inputs

        # MLP block.
        y = nn.LayerNorm(name="encoder_out_norm", dtype=self.dtype)(inputs)
        y = MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic
        )

        return inputs + y, None


@flax.struct.dataclass
class DiversityMetrics:
    """Metrics for analyzing diversity among experts in mixture of experts models."""
    auxiliary_loss: float
    router_z_loss: float
    fraction_tokens_left_behind: float
    expert_usage: float
    router_confidence: float

    def __add__(self, other):
        return DiversityMetrics(
            self.auxiliary_loss + other.auxiliary_loss,
            self.router_z_loss + other.router_z_loss,
            self.fraction_tokens_left_behind + other.fraction_tokens_left_behind,
            self.expert_usage + other.expert_usage,
            self.router_confidence + other.router_confidence,
        )


class MoeLayer(nn.Module):
    """Sparse MoE SPMD layer with per-token routing."""
    num_experts: int
    max_group_size: int
    train_capacity_factor: float
    eval_capacity_factor: float
    expert: dense.MlpBlock
    router: routing.Router
    min_expert_capacity: int = 4
    dropout_rate: float = 0.1
    dtype: DType = jnp.bfloat16
    split_params: bool = True
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT
    num_model_partitions: Optional[int] = None

    def setup(self):
        """Verifies that the MoeLayer is correctly configured."""
        if self.num_model_partitions is not None and self.num_model_partitions <= 1:
            raise ValueError(f'num_model_partitions={self.num_model_partitions} has '
                            'no effect; please set it to None instead.')

    @nn.compact
    def __call__(self,
                inputs,
                decode: bool = False,
                prefill: bool = False,
                prefill_lengths: Optional[Array] = None,
                *,
                enable_dropout: bool = True) -> Array:
        """Applies MoeLayer module."""
        batch_size, seq_length, hidden_dim = inputs.shape
        num_tokens = batch_size * seq_length

        num_groups = _num_groups(num_tokens, self.max_group_size, self.num_experts)
        tokens_per_group = num_tokens // num_groups

        if enable_dropout:
            capacity_factor = self.train_capacity_factor
        else:
            capacity_factor = self.eval_capacity_factor
        expert_capacity = int(
            round(capacity_factor * tokens_per_group / self.num_experts))
        expert_capacity = max(expert_capacity, self.min_expert_capacity)

        token_inputs = jnp.reshape(inputs,
                                (num_groups, tokens_per_group, hidden_dim))

        if isinstance(self.router, routing.ScatterRouter):
            outputs = self._scatter_to_experts(
                token_inputs,
                enable_dropout,
                expert_capacity,
                decode=decode,
                prefill=prefill,
                prefill_lengths=prefill_lengths)
        elif isinstance(self.router, routing.MaskedRouter):
            outputs = self._mask_and_dispatch_to_experts(
                token_inputs,
                enable_dropout,
                expert_capacity,
                decode=decode,
                prefill=prefill,
                prefill_lengths=prefill_lengths)
        else:
            raise ValueError(f'Unrecognized router type: {self.router}')

        result = outputs.reshape((batch_size, seq_length, hidden_dim))
        return result

    def _scatter_to_experts(self, token_inputs: Array, enable_dropout: bool,
                            expert_capacity: int, **kwargs) -> Array:
        """Wraps expert scatter routing and dispatching algorithm."""
        num_groups, tokens_per_group, hidden_dim = token_inputs.shape
        num_tokens = num_groups * tokens_per_group

        router_indices = self.router(
            token_inputs,
            self.num_experts,
            expert_capacity,
            apply_jitter=enable_dropout)
        num_selected_experts = self.router.num_selected_experts

        token_inputs = jnp.repeat(token_inputs, num_selected_experts, axis=1)

        successfully_routed = jnp.logical_and(
            router_indices.dispatch_indices[..., 0] < self.num_experts,
            router_indices.dispatch_indices[..., 1] < expert_capacity)
        successfully_routed = successfully_routed.reshape((num_groups, -1))
        masked_inputs = jnp.einsum(
            '...th,...t->...th',
            token_inputs,
            successfully_routed,
            precision=self.precision)

        flattened_dispatch_indices = router_indices.dispatch_indices.reshape(
            num_groups, -1, 2)

        shape = (self.num_experts, expert_capacity, hidden_dim)
        expert_inputs = jax.vmap(
            lambda i, x: scatter_utils.scatter_nd(i, x, shape))(
                flattened_dispatch_indices, masked_inputs)

        expert_outputs = self._call_experts(expert_inputs, enable_dropout, **kwargs)

        expert_outputs = jax.vmap(lambda i, x: x[i[:, 0], i[:, 1]])(
            flattened_dispatch_indices, expert_outputs)
        expert_outputs = expert_outputs.reshape(
            (num_groups, tokens_per_group, num_selected_experts, hidden_dim))

        combined_outputs = jnp.einsum(
            '...tkh,...tk->...th',
            expert_outputs,
            router_indices.combine_weights,
            precision=self.precision)

        successfully_routed = successfully_routed.reshape(
            (num_groups, tokens_per_group, num_selected_experts))
        num_tokens_dispatched_somewhere = jnp.max(
            successfully_routed, axis=-1).sum()
        fraction_tokens_left_behind = 1.0 - num_tokens_dispatched_somewhere / float(
            num_tokens)
        num_tokens_dispatched = successfully_routed.sum()
        total_expert_capacity = self.num_experts * expert_capacity * num_groups
        expert_usage = num_tokens_dispatched / total_expert_capacity
        router_confidence = (
            router_indices.combine_weights.sum() / num_tokens_dispatched)

        self._sow_expert_metrics(router_indices.auxiliary_loss,
                                router_indices.router_z_loss,
                                fraction_tokens_left_behind, router_confidence,
                                expert_usage)

        return combined_outputs

    def _mask_and_dispatch_to_experts(self, token_inputs: Array,
                                        enable_dropout: bool, expert_capacity: int,
                                        **kwargs) -> Array:
        """Wraps expert masked routing and dispatching algorithm."""
        num_groups, tokens_per_group, _ = token_inputs.shape
        num_tokens = num_groups * tokens_per_group

        router_mask = self.router(
            token_inputs,
            self.num_experts,
            expert_capacity,
            apply_jitter=enable_dropout)

        expert_inputs = jnp.einsum(
            '...th,...tec->...ech',
            token_inputs,
            router_mask.dispatch_mask,
            precision=self.precision)

        expert_outputs = self._call_experts(expert_inputs, enable_dropout, **kwargs)

        combined_outputs = jnp.einsum(
            '...ech,...tec->...th',
            expert_outputs,
            router_mask.combine_array,
            precision=self.precision)

        num_tokens_dispatched_somewhere = jnp.max(
            router_mask.dispatch_mask, axis=(-1, -2)).sum()
        fraction_tokens_left_behind = 1.0 - num_tokens_dispatched_somewhere / float(
            num_tokens)
        num_tokens_dispatched = router_mask.dispatch_mask.sum()
        router_confidence = router_mask.combine_array.sum() / num_tokens_dispatched

        if isinstance(self.router, routing.ExpertsChooseMaskedRouter):
            expert_usage = 1.
        else:
            total_expert_capacity = self.num_experts * expert_capacity * num_groups
            expert_usage = num_tokens_dispatched / total_expert_capacity

        self._sow_expert_metrics(router_mask.auxiliary_loss,
                                router_mask.router_z_loss,
                                fraction_tokens_left_behind, router_confidence,
                                expert_usage)

        return combined_outputs

    def _call_experts(self, inputs: Array, enable_dropout: bool,
                        **kwargs) -> Array:
        """Sends and receives inputs to experts."""
        num_groups, num_experts, capacity, hidden_dim = inputs.shape
        inputs_dtype = inputs.dtype
        inputs = jax.lax.convert_element_type(inputs, self.dtype)

        inputs = flax_partitioning.with_sharding_constraint(
            inputs, ('expert', 'unmodeled', 'length', 'embed'))
        inputs = inputs.reshape(num_experts, num_groups // num_experts, num_experts,
                                capacity, hidden_dim)

        if self.num_model_partitions is not None and self.num_model_partitions > 1:
            inputs = self._swapaxes_with_sharding_constraint(inputs, 0, 2, capacity,
                                                            hidden_dim)
        else:
            inputs = jnp.swapaxes(inputs, 0, 2)
            inputs = inputs.reshape(-1, hidden_dim)
            inputs = flax_partitioning.with_sharding_constraint(inputs,
                                                                ('expert', 'embed'))
            inputs = inputs.reshape(num_experts, num_groups * capacity, hidden_dim)

        def layer_fn(mapped_expert, expert_inputs):
            return mapped_expert(
                expert_inputs, enable_dropout=enable_dropout, **kwargs)

        outputs = flax_partitioning.vmap_with_axes(
            layer_fn,
            in_axes=(0,),
            out_axes=0,
            variable_axes={'params': 0},
            split_rngs={
                'params': self.split_params,
                'dropout': True
            },
            partitioning_axis_names={'params': 'expert'})(self.expert, inputs)

        outputs = outputs.reshape(num_experts * num_groups, capacity, hidden_dim)
        outputs = flax_partitioning.with_sharding_constraint(
            outputs, ('expert', 'length', 'embed'))
        outputs = outputs.reshape(num_experts, num_groups // num_experts,
                                num_experts, capacity, hidden_dim)

        if self.num_model_partitions is not None and self.num_model_partitions > 1:
            outputs = self._swapaxes_with_sharding_constraint(outputs, 0, 2, capacity,
                                                                hidden_dim)
        else:
            outputs = jnp.swapaxes(outputs, 0, 2)

        outputs = outputs.reshape(num_groups, num_experts, capacity, hidden_dim)
        outputs = flax_partitioning.with_sharding_constraint(
            outputs, ('expert', 'unmodeled', 'length', 'embed'))

        return jax.lax.convert_element_type(outputs, inputs_dtype)

    def _sow_expert_metrics(self, auxiliary_loss: float, router_z_loss: float,
                            fraction_tokens_left_behind: float,
                            router_confidence: float, expert_usage: float):
        """Sows metrics to analyze expert routing."""
        self.sow(
            'intermediates',
            'diversity_metrics',
            DiversityMetrics(auxiliary_loss, router_z_loss,
                            fraction_tokens_left_behind, expert_usage,
                            router_confidence),
            init_fn=lambda: DiversityMetrics(0., 0., 0., 0., 0.),
            reduce_fn=lambda a, b: a + b)

    def _swapaxes_with_sharding_constraint(self, array: Array, axis1: int,
                                            axis2: int, expert_capacity: int,
                                            hidden_dim: int) -> Array:
        """Interchanges two array axes under model-parallel sharding constraints."""
        if self.num_model_partitions is None or self.num_model_partitions <= 1:
            raise ValueError('Expected num_model_partitions to be > 1 but got: '
                            f'{self.num_model_partitions}')
        array = array.reshape(self.num_experts, -1, self.num_experts,
                            expert_capacity,
                            hidden_dim // self.num_model_partitions,
                            self.num_model_partitions)
        array = flax_partitioning.with_sharding_constraint(
            array, ('expert', 'expert_group', 'unmodeled', 'length', 'embed',
                    'expert_mlp'))
        array = jnp.swapaxes(array, axis1, axis2)
        return flax_partitioning.with_sharding_constraint(
            array, ('expert', 'expert_group', 'unmodeled', 'length', 'embed',
                    'expert_mlp'))


def _num_groups(num_tokens: int, max_group_size: int, num_experts: int) -> int:
    """Returns the number of token routing groups."""
    min_num_groups = num_tokens // max_group_size
    min_num_groups = max(min_num_groups, num_experts)

    def viable(n):
        return num_tokens % n == 0 and n % num_experts == 0

    num_groups = min_num_groups
    while num_groups < num_tokens and not viable(num_groups):
        num_groups += 1

    if num_tokens % num_groups > 0:
        raise ValueError(
            'Group size and the number of experts must divide evenly into the '
            f'global number of tokens, but num_tokens={num_tokens}, while '
            f'num_groups={num_groups} for max_group_size={max_group_size} '
            f'and num_experts={num_experts}')

    return num_groups


class LIMoEBlock(nn.Module):
    """LIMoE Block for tactile fusion.

    Attributes:
      mlp_dim: dimension of the mlp on top of attention block.
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
      out_dim: dimension of the output.
      dtype: the dtype of the computation (default: float32).
      num_experts: number of experts
      num_top_k: number of experts to use for each token
      dropout_rate: dropout rate.
    """

    mlp_dim: int
    num_heads: int
    out_dim: int 
    dtype: Dtype = jnp.float32
    num_experts: int = 4
    num_top_k: int = 1
    param_dtype: Dtype = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, deterministic=True):
        """Applies LIMoEBlock module.

        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after LIMoE block.
        """
        expert = dense.MlpBlock(
            use_bias=True,
            dtype=self.dtype
        )

        router = routing.TokensChooseScatterRouter(
            router_weights=routing.RouterWeights(),
            jitter_noise=0.01,
            dtype=jnp.float32,
            ignore_padding_tokens=True,
            num_selected_experts=self.num_top_k,
            batch_prioritized_routing=True
        )

        x, _ = Encoder1DBlock(
            name="encoderblock",
            dtype=self.dtype,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )(inputs, deterministic)

        moe_out = MoeLayer(
            num_experts=self.num_experts,
            max_group_size=16,
            train_capacity_factor=1.0,
            eval_capacity_factor=1.0,
            expert=expert,
            router=router,
        )(x, enable_dropout=not deterministic)

        x = x + moe_out

        x = nn.Dense(
            features=self.out_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)

        return x, None
