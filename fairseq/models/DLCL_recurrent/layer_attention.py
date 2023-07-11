import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


"""
    for self attention:
    for not incremental case, attn_weights(B, T, T) will apply attn_mask(B, T, T),
    like [[0, -inf, -inf, -inf],
            [0,    0, -inf, -inf],
            [0,    0,    0, -inf],
            [0,    0,    0,    0]]
    and then apply key_padding_mask(B, 1, T), assume the sentence len = 3,
    like [[0,    0,    0, -inf],
            [0,    0,    0, -inf],
            [0,    0,    0, -inf],
            [0,    0,    0, -inf]]
    so the result like [[0, -inf, -inf, -inf],
                        [0,    0, -inf, -inf],
                        [0,    0,    0, -inf],
                        [0,    0,    0, -inf]]
    which means the last row is deprecated
    finally apply the value(B, T, C),assume is likes[v1; v2; v3; v4]
    so get the result(B, T, C),
    like [[a11*v1],
            [a21*v1+a22*v2],
            [a31*v1+a32*v2+a33*v3],
            [a31*v1+a32*v2+a33*v3]]

    for the incremental case, the q(B, 1, C), and get the k,v with saved k'(B, T-1, C) concatenate
    q(B, 1, C), then save the new k,v.
    attn_weights(B, 1, T), and the "1" means the last row,
    then apply attn_mask, which is None, beacause it is always [0,    0,    0,    0]
    (the last row), then apply key_padding_mask(B, 1, T), so get the last row of the
    result above (B, 1, T), e.g.[0,    0,    0, -inf].finally apply the value(B, T, C)
    and the final result will be (B, 1, C), e.g. [a31*v1+a32*v2+a33*v3].


    for encoder decoder attention:
    for not incremental case, attn_weights(B, T, S), there is no attn_mask.
    then apply key_padding_mask(B, 1, S), assume the sentence len = 3, T = 2, S = 4,
    like [[0,    0,    0, -inf],
            [0,    0,    0, -inf]]
    so the result like [[0,    0,    0, -inf],
                        [0,    0,    0, -inf]]
    finally apply the value(B, S, C),assume is likes[v1; v2; v3; v4]
    so get the result(B, T, C),
    like [[a31*v1+a32*v2+a33*v3],
            [a31*v1+a32*v2+a33*v3]]

    for the incremental case, the T = 1, and get the k,v(B, S, C) by saved_state
    
"""

"""
    in this attention block, we have a similar process with the encoder decoder attention
    above, but we input the q(B, T, C), and the k,v(L, B, S, C), we first transfer k to
    (L, B, C) by mask and mean the S dim value, so we can treat the L as S. transpose k to
    (B, L, C), calculate the attn_weights(B, T, L) without any mask, tranpose v to (S, B, L, C),
    we then have attn_weights(1, B, T, L) * v(S, B, L, C), get the result(S, B, T, C), finally 
    transpose result to (T, S, B, C)
"""


@with_incremental_state
class LayerAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        kdim,
        vdim,
        num_heads,
        layer_num,
        dropout=0.0,
        bias=True,
        reduction="mean",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.layer_num = layer_num

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)

        self.reduction_mean = reduction == "mean"
        if not self.reduction_mean:
            assert (
                reduction == "max"
            ), "Layer attention only support mean reduction or max reduction."

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """query shape: Time x Batch x Channel
           key shape: Layer x Time_S x Batch x Channel
           key shape: Layer x Time_S x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
                assert prev_key_padding_mask is not None
                key_padding_mask = prev_key_padding_mask
            else:
                saved_state["prev_key_padding_mask"] = key_padding_mask
                incremental_state = self._set_input_buffer(incremental_state, saved_state)
        else:
            saved_state = None

        q = self.q_proj(query)
        if key is None:
            assert value is None
            k = v = None
        else:
            assert key is not None and value is not None
            layer_key_padding_mask = (
                key_padding_mask.transpose(0, 1).unsqueeze(2).unsqueeze(0)
            ).to(
                torch.bool
            )  # 1 x Time_S x Batch x 1
            count_key_padding_mask = (
                (~key_padding_mask).sum(dim=1, keepdim=False).to(key)
            )  # batch
            key = key.masked_fill(layer_key_padding_mask, 0.0)

            if self.reduction_mean:
                key = torch.sum(key, dim=1, keepdim=False).div(
                    count_key_padding_mask.unsqueeze(1).unsqueeze(0)
                )  # Layer x Batch x Channel
            else:
                key, _ = torch.max(key, dim=1)  # Layer x Batch x Channel

            k = self.k_proj(key)  # Layer x Batch x Channel
            v = self.v_proj(value)  # Layer x Time_S x Batch x Channel
            assert k.size(0) == self.layer_num
        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )  # Batch*heads x Time x Channel/heads
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )  # Batch*heads x Layer x Channel/heads
        if v is not None:
            v = (
                v.contiguous()
                .view(self.layer_num, -1, bsz * self.num_heads, self.head_dim)
                .permute(1, 2, 0, 3)
            )  #  Time_s x Batch*heads x layer x Channel/heads

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                k = prev_key

            # saved states are stored with shape (seq_len, bsz, num_heads, layer, head_dim)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(
                    -1, bsz * self.num_heads, self.layer_num, self.head_dim
                )
                v = prev_value
            # In this branch incremental_state is never None
        elif incremental_state is not None:
            assert saved_state is not None
            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(
                -1, bsz, self.num_heads, self.layer_num, self.head_dim
            )
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = v.size(0)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))  # Batch*heads x Time x Layer

        assert list(attn_weights.size()) == [
            bsz * self.num_heads,
            tgt_len,
            self.layer_num,
        ]

        attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=False)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None  #  Time_s x Batch*heads x layer x Channel/heads
        attn = torch.matmul(
            attn_probs.unsqueeze(0), v
        )  #  Time_s x Batch*heads x Time x Channel/heads
        assert list(attn.size()) == [
            src_len,
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ]
        attn = (
            attn.permute(2, 0, 1, 3).contiguous().view(tgt_len, src_len, bsz, embed_dim)
        )
        attn = self.out_proj(attn)  # (T, S, B, C)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, self.layer_num
            ).transpose(
                1, 0
            )  # heads x Batch x Time x Layer (H, B, T, L)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @torch.jit.export
    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if k == "prev_value":
                        input_buffer[k] = input_buffer_k.index_select(1, new_order)
                    else:
                        input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "layer_attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "layer_attn_state", buffer)


"""
    note: for this layer attention implementation, we do not apply key_padding_mask to attention
    output anymore, considering that the successive attention(encoder-decoder attention) will
    apply it.
"""


@with_incremental_state
class TimeWiseLayerAttention(nn.Module):
    def __init__(
        self, embed_dim, kdim, vdim, num_heads, layer_num, dropout=0.0, bias=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.layer_num = layer_num

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """query shape: Time x Batch x Channel
           key shape: Layer x Time_S x Batch x Channel
           key shape: Layer x Time_S x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        q = self.q_proj(query)
        if key is None:
            assert value is None
            k = v = None
        else:
            assert key is not None and value is not None

            k = self.k_proj(key)  # Layer x Time_S x Batch x Channel
            v = self.v_proj(value)  # Layer x Time_S x Batch x Channel
            assert k.size(0) == self.layer_num
        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )  # Batch*heads x Time x Channel/heads
        if k is not None:
            k = (
                k.contiguous()
                .view(self.layer_num, -1, bsz * self.num_heads, self.head_dim)
                .permute(1, 2, 0, 3)
            )  #  Time_s x Batch*heads x layer x Channel/heads
        if v is not None:
            v = (
                v.contiguous()
                .view(self.layer_num, -1, bsz * self.num_heads, self.head_dim)
                .permute(1, 2, 0, 3)
            )  #  Time_s x Batch*heads x layer x Channel/heads

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(
                    -1, bsz * self.num_heads, self.layer_num, self.head_dim
                )
                k = prev_key

            # saved states are stored with shape (seq_len, bsz, num_heads, layer, head_dim)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(
                    -1, bsz * self.num_heads, self.layer_num, self.head_dim
                )
                v = prev_value
            # In this branch incremental_state is never None
        elif incremental_state is not None:
            assert saved_state is not None
            saved_state["prev_key"] = k.view(
                -1, bsz, self.num_heads, self.layer_num, self.head_dim
            )
            saved_state["prev_value"] = v.view(
                -1, bsz, self.num_heads, self.layer_num, self.head_dim
            )
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = v.size(0)

        attn_weights = torch.matmul(
            q, k.transpose(2, 3)
        )  # Time_s x Batch*heads x Time x Layer

        assert list(attn_weights.size()) == [
            src_len,
            bsz * self.num_heads,
            tgt_len,
            self.layer_num,
        ]

        attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=False)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None  #  Time_s x Batch*heads x layer x Channel/heads
        attn = torch.matmul(attn_probs, v)  #  Time_s x Batch*heads x Time x Channel/heads
        assert list(attn.size()) == [
            src_len,
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ]
        attn = (
            attn.permute(2, 0, 1, 3).contiguous().view(tgt_len, src_len, bsz, embed_dim)
        )
        attn = self.out_proj(attn)  # (T, S, B, C)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, self.layer_num
            ).transpose(
                1, 0
            )  # heads x Batch x Time x Layer (H, B, T, L)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @torch.jit.export
    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if k == "prev_value":
                        input_buffer[k] = input_buffer_k.index_select(1, new_order)
                    else:
                        input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "layer_attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "layer_attn_state", buffer)


@with_incremental_state
class LayerEncoderDeocderAttention(nn.Module):
    def __init__(
        self, embed_dim, kdim, vdim, num_heads, dropout=0.0, bias=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """query shape: T x B x C
           key shape: T x S x B x C
           value shape: T x S x B x C
           key_padding_mask: B x S

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key is not None and value is not None

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
                assert prev_key_padding_mask is not None
                key_padding_mask = prev_key_padding_mask
            else:
                saved_state["prev_key_padding_mask"] = key_padding_mask
                incremental_state = self._set_input_buffer(incremental_state, saved_state)
        else:
            saved_state = None

        q = self.q_proj(query)
        k = self.k_proj(key)  # T x S x B x C
        v = self.v_proj(value)  # T x S x B x C
        q *= self.scaling
        assert k.size(0) == tgt_len

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )  # Batch*heads x Time x Channel/heads

        k = (
            k.contiguous()
            .view(tgt_len, -1, bsz * self.num_heads, self.head_dim)
            .permute(2, 0, 1, 3)
        )  # Batch*heads x T x S x Channel/heads

        v = (
            v.contiguous()
            .view(tgt_len, -1, bsz * self.num_heads, self.head_dim)
            .permute(2, 0, 1, 3)
        )  # Batch*heads x T x S x Channel/heads

        src_len = v.size(2)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.matmul(q.unsqueeze(2), k.transpose(2, 3)).squeeze(
            2
        )  # Batch*heads x Time x S

        assert list(attn_weights.size()) == [
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"),
            )

            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=False)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None  #  Batch*heads x Time_s x layer x Channel/heads
        attn = torch.matmul(attn_probs.unsqueeze(2), v).squeeze(
            2
        )  #  Batch*heads x Time x Channel/heads
        assert list(attn.size()) == [
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)  # (T, B, C)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(
                1, 0
            )  # heads x Batch x Time x Layer (H, B, T, L)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @torch.jit.export
    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "layer_attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "layer_attn_state", buffer)
