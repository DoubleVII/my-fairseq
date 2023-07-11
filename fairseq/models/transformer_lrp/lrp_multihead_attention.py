# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch
from torch import autograd
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from .lrp_utils import LRPWrapper, LinearWrapper, LRP
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


class AttentionCoreWrapper(LRPWrapper):
    def __init__(
        self, embed_dim, num_heads, decoder_self_attn=False, dropout=0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.decoder_self_attn = decoder_self_attn
        self._future_mask = torch.empty(0)
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.scaling = self.head_dim ** -0.5

        self.onnx_trace = False

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def forward(self, q, k, v, record=False):

        bsz_mul_head, tgt_len, head_dim = q.size()
        bsz = int(bsz_mul_head / self.num_heads)
        embed_dim = self.embed_dim

        if self.decoder_self_attn:
            attn_mask = self.buffered_future_mask(q.transpose(0, 1))
        else:
            attn_mask = None

        if record:
            self.record("q", q)
            self.record("k", k)
            self.record("v", v)

        q = q * self.scaling
        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        key_padding_mask = None

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # attn_weights = AttentionCoreWrapper.apply_sparse_mask(
        #     attn_weights, tgt_len, src_len, bsz
        # )

        assert list(attn_weights.size()) == [bsz_mul_head, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        # if key_padding_mask is not None:
        #     # don't attend to padding symbols
        #     attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        #     if not self.tpu:
        #         attn_weights = attn_weights.masked_fill(
        #             key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
        #             float("-inf"),
        #         )
        #     else:
        #         attn_weights = attn_weights.transpose(0, 2)
        #         attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
        #         attn_weights = attn_weights.transpose(0, 2)
        #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # if before_softmax:
        #     return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        return attn
        assert list(attn.size()) == [bsz_mul_head, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        return attn

    @classmethod
    def apply_sparse_mask(cls, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def _attn_head_jacobian(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """ same as lib.layers.lrp.jacobian(self.attention_core(q, k, v), [q, k, v]), but faster """
        # input shapes: (q, k, v) - [batch_size, n_q or n_kv, dim per head]
        # attn_head_mask: [batch_size, n_q, n_kv]
        assert q.dim() == 3
        bsz_mul_head, tgt_len, _ = q.size()

        # q_ = q
        q = q * self.scaling
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz_mul_head, tgt_len, src_len]

        if self.decoder_self_attn:
            attn_mask = self.buffered_future_mask(q)
        else:
            attn_mask = None

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz_mul_head, tgt_len, self.head_dim]

        # compute jacobian w.r.t. values

        diag_flat_weights = torch.einsum(
            "ij,jqk->iqjk",
            torch.eye(attn_weights.size(0)).to(attn_weights),
            attn_weights,
        )  # [b, n_q, b, n_kv]
        flat_jac_v = (
            diag_flat_weights[:, :, None, :, :, None]
            * torch.eye(self.head_dim).to(attn_weights)[None, None, :, None, None, :]
        )
        # ^-- shape: [batch_size, n_q, dim/h, batch_size, n_kv, dim/h]

        jac_out_wrt_weights = torch.tile(v[:, None], [1, attn.size(1), 1, 1]).transpose(
            2, 3
        )
        # ^-- [batch_size, n_q, (dim), (n_kv)]
        softmax_jac = (
            attn_weights[..., None] * torch.eye(attn_weights.size(-1)).to(attn_weights)
            - attn_weights[..., None, :] * attn_weights[..., :, None]
        )  # <-- [batch_size, n_q, n_kv, n_kv]
        jac_out_wrt_logits = (
            jac_out_wrt_weights @ softmax_jac
        )  # [batch_size, n_q, (dim), (n_kv)]

        jac_out_wrt_k = (
            jac_out_wrt_logits[..., None] * q[:, :, None, None, :]
        )  # [b, (n_q, dim), (n_kv, dim)]

        # product axes:                    b  q  d  kv   d       b  q      d    kv d
        jac_out_wrt_q = jac_out_wrt_logits[:, :, :, :, None] * k[:, None, None, :, :]
        jac_out_wrt_q = torch.sum(jac_out_wrt_q, dim=3, keepdim=True)
        jac_out_wrt_q = jac_out_wrt_q * self.scaling
        jac_out_wrt_q = (
            jac_out_wrt_q
            * torch.eye(jac_out_wrt_q.size(1)).to(attn_weights)[None, :, None, :, None]
        )

        flat_jac_k = (
            jac_out_wrt_k[..., None, :, :]
            * torch.eye(q.size(0)).to(attn_weights)[:, None, None, :, None, None]
        )

        flat_jac_q = (
            jac_out_wrt_q[..., None, :, :]
            * torch.eye(q.size(0)).to(attn_weights)[:, None, None, :, None, None]
        )
        # final shape of flat_jac_{q, k}: [(batch_size, n_q, dim), (batch_size, n_kv, dim)]
        return flat_jac_q.contiguous(), flat_jac_k.contiguous(), flat_jac_v.contiguous()

    def _attn_head_jacobian_simple(self, q, k, v):
        jq, jk, jv = self._attn_head_jacobian(q, k, v)
        return torch.zeros_like(jq), torch.zeros_like(jk), jv

    def relprop(self, out_relevance):
        q = self.get_record("q").contiguous()
        k = self.get_record("k").contiguous()
        v = self.get_record("v").contiguous()
        # input: [*dims, inp_size], out: [*dims, out_size]

        # note: we apply relprop for each independent sample in order to avoid quadratic memory requirements

        in_relevances = [
            LRP.relprop(
                self,
                out_relevance[i, None],
                (q[i, None], k[i, None], v[i, None]),
                jacobians=self._attn_head_jacobian(q[i, None], k[i, None], v[i, None],),
            )
            for i in range(out_relevance.size(0))
        ]
        q_relevance = torch.cat([items[0] for items in in_relevances], dim=1)
        k_relevance = torch.cat([items[1] for items in in_relevances], dim=1)
        v_relevance = torch.cat([items[2] for items in in_relevances], dim=1)

        q_relevance = q_relevance.view_as(q)  # bsz*head_num x tgt_len x head_dim
        k_relevance = k_relevance.view_as(k)
        v_relevance = v_relevance.view_as(v)
        # q_relevance, k_relevance, v_relevance = LRP.relprop(
        #     self, out_relevance, (q, k, v), batch_dim=1
        # )

        # q_relevance = q_relevance.
        # print(
        #     torch.norm(q_relevance, dim=-1).sum() / torch.norm(v_relevance, dim=-1).sum()
        # )
        # print(
        #     torch.norm(k_relevance, dim=-1).sum() / torch.norm(v_relevance, dim=-1).sum()
        # )
        return q_relevance, k_relevance, v_relevance


@with_incremental_state
class MultiheadAttentionWrapper(LRPWrapper):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        decoder_self_attn=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = LinearWrapper(
            quant_noise(
                nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
            )
        )
        self.v_proj = LinearWrapper(
            quant_noise(
                nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
            )
        )
        self.q_proj = LinearWrapper(
            quant_noise(
                nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
            )
        )

        self.out_proj = LinearWrapper(
            quant_noise(
                nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
            )
        )

        assert not add_bias_kv
        assert not add_zero_attn

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.decoder_self_attn = decoder_self_attn

        self.attn_core = AttentionCoreWrapper(
            embed_dim, num_heads, decoder_self_attn, dropout
        )

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

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
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        record: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

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
        assert key_padding_mask is None
        if not self.decoder_self_attn:
            assert attn_mask is None
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if (
            False
            and not self.onnx_trace
            and not self.tpu  # don't use PyTorch version on TPUs
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
        ):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training or self.dropout_module.apply_during_inference,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )

        assert incremental_state is None

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query, record=record)
            k = self.k_proj(query, record=record)
            v = self.v_proj(query, record=record)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query, record=record)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key, record=record)
                v = self.v_proj(key, record=record)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query, record=record)
            k = self.k_proj(key, record=record)
            v = self.v_proj(value, record=record)

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        attn = self.attn_core(q, k, v, record=record)

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        attn = self.out_proj(attn, record=record)
        # attn_weights: Optional[Tensor] = None
        assert not need_weights
        # if need_weights:
        #     attn_weights = attn_weights_float.view(
        #         bsz, self.num_heads, tgt_len, src_len
        #     ).transpose(1, 0)
        #     if not need_head_weights:
        #         # average attention weights over heads
        #         attn_weights = attn_weights.mean(dim=0)

        return attn

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
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
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

    def relprop(self, out_relevance):
        out_relevance = self.out_proj.relprop(out_relevance)
        tgt_len = out_relevance.size(0)
        out_relevance = (
            out_relevance.view(tgt_len, -1, self.head_dim).transpose(0, 1).contiguous()
        )

        q_relevance, k_relevance, v_relevance = self.attn_core.relprop(out_relevance)

        q_relevance = (
            q_relevance.transpose(0, 1)
            .contiguous()
            .view(q_relevance.size(1), -1, self.num_heads * self.head_dim)
        )

        k_relevance = (
            k_relevance.transpose(0, 1)
            .contiguous()
            .view(k_relevance.size(1), -1, self.num_heads * self.head_dim)
        )

        v_relevance = (
            v_relevance.transpose(0, 1)
            .contiguous()
            .view(v_relevance.size(1), -1, self.num_heads * self.head_dim)
        )
        query_relevance = self.q_proj.relprop(q_relevance)
        key_relevance = self.k_proj.relprop(k_relevance)
        value_relevance = self.v_proj.relprop(v_relevance)
        kv_relevance = key_relevance + value_relevance

        return query_relevance, kv_relevance
