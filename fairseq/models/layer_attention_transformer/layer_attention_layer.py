from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import Tensor
from .layer_attention import (
    LayerAttention,
    TimeWiseLayerAttention,
    LayerEncoderDeocderAttention,
)
from .layer_linear_aggregation import LayerLinearAggregation

LAYER_ROUTE_TYPE = ["low", "high", "all", "time-wise", "time-wise-ind"]


class TransformerRecurrentLayer(nn.Module):
    def __init__(self, embed_dim, layer_route, layer_recurrent, args):
        super().__init__()
        self.layer_route = layer_route
        self.layer_recurrent = layer_recurrent

        if not self.layer_route:
            return

        assert self.layer_route in LAYER_ROUTE_TYPE

        if self.layer_route == "all":
            self.route_weight = nn.Parameter(
                torch.zeros((self.layer_recurrent), requires_grad=True),
                requires_grad=True,
            )
        elif self.layer_route == "time-wise":
            self.route_weight = nn.Linear(
                self.layer_recurrent * embed_dim, self.layer_recurrent, bias=False
            )
        elif self.layer_route == "time-wise-ind":
            self.route_weight = nn.Parameter(
                torch.zeros((self.layer_recurrent, embed_dim), requires_grad=True),
                requires_grad=True,
            )

    def aggregation_layer_output(self, last_layer_state):
        if self.layer_route == "low":
            return last_layer_state[0]
        elif self.layer_route == "high":
            return last_layer_state[-1]
        elif self.layer_route == "all":
            norm_route_weight_float = utils.softmax(self.route_weight, dim=-1)
            norm_route_weight = norm_route_weight_float.type_as(self.route_weight)
            last_layer_state_tensor = torch.stack(last_layer_state, dim=-1)
            return torch.matmul(last_layer_state_tensor, norm_route_weight)
        elif self.layer_route == "time-wise":
            (len_size, batch_size, _) = last_layer_state[-1].size()
            input_features = torch.stack(last_layer_state, dim=-2).view(
                len_size, batch_size, -1
            )  # T x B x L*C
            output_features = self.route_weight(input_features)  # T x B x L
            norm_route_weight_float = utils.softmax(output_features, dim=-1)
            norm_route_weight = norm_route_weight_float.type_as(output_features)
            last_layer_state_tensor = torch.stack(
                last_layer_state, dim=-1
            )  # T x B x C x L
            return torch.matmul(
                last_layer_state_tensor, norm_route_weight.unsqueeze(-1)
            ).squeeze(-1)
        elif self.layer_route == "time-wise-ind":
            (len_size, batch_size, channel) = last_layer_state[-1].size()
            input_features = torch.stack(last_layer_state, dim=-2).view(
                len_size * batch_size, -1, channel
            )  # T*B x L x C
            output_features = (
                torch.matmul(
                    input_features.unsqueeze(-2), self.route_weight.unsqueeze(-1)
                )
                .squeeze(-1)
                .squeeze(-1)
                .view(len_size, batch_size, -1)
            )  # T x B x L
            norm_route_weight_float = utils.softmax(output_features, dim=-1)
            norm_route_weight = norm_route_weight_float.type_as(output_features)
            last_layer_state_tensor = torch.stack(
                last_layer_state, dim=-1
            )  # T x B x C x L
            return torch.matmul(
                last_layer_state_tensor, norm_route_weight.unsqueeze(-1)
            ).squeeze(-1)


class LATransformerEncoderLayer(TransformerRecurrentLayer):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, first_layer=False):
        super().__init__(
            args.encoder_embed_dim,
            None if first_layer else args.encoder_layer_route,
            args.encoder_recurrent,
            args,
        )
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.fc1 = self.build_fc1(self.embed_dim, args.encoder_ffn_embed_dim,)
        self.fc2 = self.build_fc2(args.encoder_ffn_embed_dim, self.embed_dim,)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.normalize_before = args.encoder_normalize_before

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x, key=x, value=x, key_padding_mask=encoder_padding_mask
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class AggregationTransformerDecoderLayer(TransformerRecurrentLayer):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, first_layer=False):
        super().__init__(
            args.decoder_embed_dim,
            None if first_layer else args.decoder_layer_route,
            args.decoder_recurrent,
            args,
        )
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

        self.self_attn = self.build_self_attention(self.embed_dim, args)

        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.layer_aggregation = args.layer_aggregation

        self.layer_attn = None
        self.layer_linear = None
        if self.layer_aggregation == "attn":
            self.layer_attn = self.build_layer_attention(self.embed_dim, args)
        if self.layer_aggregation == "linear":
            self.layer_linear = self.build_layer_linear(args)

        self.layer_attn_norm = LayerNorm(self.embed_dim, export=export)

        self.layer_attn_fc1 = None
        self.layer_attn_fc2 = None
        if args.layer_attn_ffn:
            self.layer_attn_fc1 = self.build_fc1(
                self.embed_dim, args.decoder_ffn_embed_dim,
            )
            self.layer_attn_fc2 = self.build_fc2(
                args.decoder_ffn_embed_dim, self.embed_dim,
            )

        self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(self.embed_dim, args.decoder_ffn_embed_dim,)
        self.fc2 = self.build_fc2(args.decoder_ffn_embed_dim, self.embed_dim,)

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.normalize_before = args.decoder_normalize_before

        self.need_attn = True

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

    def build_layer_attention(self, embed_dim, args):
        if args.time_wise_layer_attn:
            return TimeWiseLayerAttention(
                embed_dim,
                kdim=embed_dim,
                vdim=embed_dim,
                num_heads=args.decoder_attention_heads,
                layer_num=args.layer_attention_num,
                dropout=args.attention_dropout,
            )
        else:
            return LayerAttention(
                embed_dim,
                kdim=embed_dim,
                vdim=embed_dim,
                num_heads=args.decoder_attention_heads,
                layer_num=args.layer_attention_num,
                dropout=args.attention_dropout,
                reduction=args.layer_attn_reduction,
            )

    def build_layer_linear(self, args):
        return LayerLinearAggregation(
            args.layer_attention_num, dropout=args.attention_dropout
        )

    def build_encoder_attention(self, embed_dim, args):
        if self.layer_aggregation == "attn":
            return LayerEncoderDeocderAttention(
                embed_dim,
                kdim=embed_dim,
                vdim=embed_dim,
                num_heads=args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
        else:
            return MultiheadAttention(
                embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, "encoder_embed_dim", None),
                vdim=getattr(args, "encoder_embed_dim", None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_states: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_states (Tensor): (layer , batch, src_len, channel)
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        """
            self_attn_mask: triangle up matrix T * T
            self_attn_padding_mask: pading mask B * T
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.layer_attn is not None:
            attn_encoder_out, _ = self.layer_attn(
                query=x,
                key=encoder_states,
                value=encoder_states,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
            )
        else:
            attn_encoder_out = self.layer_linear(encoder_states)

        attn_encoder_out = self.dropout_module(attn_encoder_out)
        # TODO:residual
        # attn_encoder_out = attn_encoder_out + encoder_out.unsqueeze(0)
        attn_encoder_out = self.layer_attn_norm(attn_encoder_out)

        if self.layer_attn_fc1 is not None:
            residual_layer_attn = attn_encoder_out
            attn_encoder_out = self.activation_fn(self.layer_attn_fc1(attn_encoder_out))
            attn_encoder_out = self.layer_attn_fc2(attn_encoder_out)
            attn_encoder_out = self.dropout_module(attn_encoder_out)
            attn_encoder_out = residual_layer_attn + attn_encoder_out
            attn_encoder_out = self.layer_attn_norm(attn_encoder_out)

        # TODO: apply ffn to attn_encoder_out

        residual = x
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        if self.layer_attn is not None:
            x, attn = self.encoder_attn(
                query=x,
                key=attn_encoder_out,
                value=attn_encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                need_weights=need_attn or not self.training,
                need_head_weights=need_head_weights,
            )
        else:
            x, attn = self.encoder_attn(
                query=x,
                key=attn_encoder_out,
                value=attn_encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or not self.training,
                need_head_weights=need_head_weights,
            )
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


from fairseq.modules.quant_noise import quant_noise


class TransformerDecoderLayer(TransformerRecurrentLayer):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        args,
        first_layer=False,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False,
    ):
        super().__init__(
            args.decoder_embed_dim,
            None if first_layer else args.decoder_layer_route,
            args.decoder_recurrent,
            args,
        )
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim, args, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        """
            self_attn_mask: triangle up matrix T * T
            self_attn_padding_mask: pading mask T * B * C
        """
        # if prev_self_attn_state is not None:
        #     print("\n\nprev_self_attn_state is not None!!\n\n", prev_self_attn_state)

        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn
