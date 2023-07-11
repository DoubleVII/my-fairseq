import torch
from torch import Tensor

from fairseq.models.transformer_attribution import (
    TransformerModel as BaseTransformerModel,
    TransformerEncoder as BaseTransformerEncoder,
    TransformerDecoder as BaseTransformerDecoder,
)
from .transformer_layer import TransformerEncoderLayer, TransformerDecoderLayer
from fairseq.models import register_model
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Optional[Tensor]),  # B x T
        ("encoder_embedding", Optional[Tensor]),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1
        ("encoder_compositions", Optional[Tensor]),  # T x T x B x C
        ("compositions_states", Optional[List[Tensor]]),  # List[T x T x B x C]
        ("attn_states", Optional[List[Tensor]]),  # List[B x T x T]
    ],
)


@register_model("attribution_with_position_transformer")
class TransformerModel(BaseTransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, *args, **kargs):
        return TransformerEncoder(*args, **kargs)

    @classmethod
    def build_decoder(cls, *args, **kargs):
        return TransformerDecoder(*args, **kargs)


class TransformerEncoder(BaseTransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def build_encoder_layer(self, args):
        return TransformerEncoderLayer(args)

    def forward(
        self,
        src_tokens: torch.LongTensor,
        src_lengths,
        return_all_hiddens: bool = False,
        return_compositions=False,
        dropout_tokens=None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if dropout_tokens is not None:
            assert not return_compositions
            dropout_tokens = dropout_tokens.to(src_tokens)
            assert src_tokens.size(0) == 1
            dropped_src_tokens = src_tokens.index_fill(
                1, dropout_tokens[0], self.padding_idx
            )
            x, encoder_embedding = self.forward_embedding(dropped_src_tokens)
        else:
            x, encoder_embedding = self.forward_embedding(src_tokens)

        # x = x - encoder_embedding

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        src_len, batch_size, hidden_dim = x.size()

        if return_compositions:
            src_len, batch_size, hidden_dim = x.size()

            compositions = torch.zeros(
                (src_len, src_len, 2, batch_size, hidden_dim)
            ).to(x)
            index = torch.arange(0, src_len).to(src_tokens)
            position_embed = self.embed_positions(src_tokens)
            compositions[index, index, 0, :, :] = encoder_embedding.transpose(0, 1)
            compositions[index, index, 1, :, :] = position_embed.transpose(0, 1)

            compositions = torch.cat(
                [
                    torch.zeros((src_len, 1, 2, batch_size, hidden_dim)).to(x),
                    compositions,
                ],
                dim=1,
            )

            compositions = compositions.view(src_len, -1, batch_size, hidden_dim)

        else:
            compositions = None

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()
        encoder_padding_mask = encoder_padding_mask if has_pads else None

        encoder_states = [] if return_all_hiddens else None
        attn_states = []
        compositions_states = []
        # if compositions is not None:
        #     compositions_states.append(compositions)

        # encoder layers
        for layer in self.layers:
            x, compositions, attn_weights = layer(
                x, encoder_padding_mask=encoder_padding_mask, compositions=compositions,
            )
            if compositions is not None:
                compositions_states.append(compositions)
            attn_states.append(attn_weights)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
            encoder_compositions=compositions,
            compositions_states=compositions_states,  # compositions_states,
            attn_states=attn_states,
        )


class TransformerDecoder(BaseTransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return TransformerDecoderLayer(args, no_encoder_attn)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        return_compositions=False,
        dropout_tokens=None,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            return_compositions=return_compositions,
            dropout_tokens=dropout_tokens,
        )

        compositions = extra["compositions"]
        if compositions is not None:
            compositions = compositions.permute(2, 0, 1, 3)
            compositions_logits = self.output_layer(compositions)
            extra["compositions_logits"] = compositions_logits
        else:
            extra["compositions_logits"] = None

        if not features_only:
            x = self.output_layer(x)
            if compositions is not None:
                # analysis_info = self.compute_info(x, compositions_logits, compositions)
                analysis_info = {}
                analysis_info["compositions_states"] = extra["compositions_states"]
                extra["analysis_info"] = analysis_info
            else:
                extra["analysis_info"] = None
        return x, extra

    def compute_info(self, x, compositions_logits, compositions):
        analysis_info = {}
        assert x.size()[:2] == (1, 1)
        pred_index = torch.argmax(x, dim=-1)[0, 0].item()
        # pred_logits = (compositions_logits[0, 0, 1:, pred_index] * 10).round() / 10
        # pred_norms = (torch.norm(compositions[0, 0, 1:], dim=-1) * 10).round() / 10
        # import pdb
        # pdb.set_trace()
        composition_norms = torch.norm(compositions[0, 0, 1:], dim=-1)
        analysis_info["composition_norms"] = composition_norms
        analysis_info["compositions_logits"] = compositions_logits

        return analysis_info

    def cal_corrcoef(self, x, compositions_logits, compositions):

        assert x.size()[:2] == (1, 1)
        pred_index = torch.argmax(x, dim=-1)[0, 0].item()
        pred_logits = (compositions_logits[0, 0, 1:, pred_index] * 10).round() / 10
        pred_norms = (torch.norm(compositions[0, 0, 1:], dim=-1) * 10).round() / 10
        print(pred_logits)
        print(pred_norms)
        import pdb

        pdb.set_trace()
        analysis_info = torch.corrcoef(torch.stack([pred_logits, pred_norms], dim=0))[
            0, 1
        ]
        return analysis_info

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        return_compositions=False,
        dropout_tokens=None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # print(prev_output_tokens)
        # if prev_output_tokens.size(1) == 3:
        #     prev_output_tokens[:,1] = 2471
        #     prev_output_tokens[:,2] = 80 #29

        # # print(encoder_out.encoder_out.size())
        # rand_encoder_out = torch.zeros(encoder_out.encoder_out.size()).to(encoder_out.encoder_out)

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        batch_size, prev_token_len = prev_output_tokens.size()

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        if dropout_tokens is not None:
            assert not return_compositions
            assert incremental_state is None
            dropout_tokens = dropout_tokens.to(prev_output_tokens)
            assert prev_output_tokens.size(0) == 1
            dropped_src_tokens = prev_output_tokens.index_fill(
                1, dropout_tokens[0], self.padding_idx
            )
            x = self.embed_scale * self.embed_tokens(dropped_src_tokens)
        else:
            # embed tokens and positions
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        embeds = x
        if positions is not None:
            x = embeds + positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        if return_compositions:
            assert incremental_state is not None
            assert x.size(0) == 1
            hidden_dim = x.size(-1)
            # compositions = torch.zeros((1, encoder_out.encoder_out.size(0)+prev_token_len, batch_size, hidden_dim)).to(x)
            compositions = torch.zeros(
                (
                    1,
                    encoder_out.encoder_out.size(0) + prev_token_len + 1,
                    2,
                    batch_size,
                    hidden_dim,
                )
            ).to(x)
            compositions[:, -1, 0, :, :] = embeds.transpose(0, 1)
            compositions[:, -1, 1, :, :] = positions.transpose(0, 1)
            compositions = compositions.view(1, -1, batch_size, hidden_dim,)

        else:
            compositions = None

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        compositions_states: List[Optional[Tensor]] = [compositions]
        attn_states: List[Optional[Tensor]] = []

        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # x, layer_attn, compositions = layer(
            #     x,
            #     encoder_out.encoder_out if encoder_out is not None else None,
            #     encoder_out.encoder_padding_mask if encoder_out is not None else None,
            #     incremental_state,
            #     self_attn_mask=self_attn_mask,
            #     self_attn_padding_mask=self_attn_padding_mask,
            #     need_attn=bool((idx == alignment_layer)),
            #     need_head_weights=bool((idx == alignment_layer)),
            #     encoder_compositions=encoder_out.encoder_compositions,
            #     compositions=compositions,
            # )
            x, layer_attn, compositions = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=True,
                need_head_weights=False,
                encoder_compositions=encoder_out.encoder_compositions,
                compositions=compositions,
            )
            inner_states.append(x)
            compositions_states.append(compositions)
            attn_states.append(layer_attn)

            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            # attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            assert False
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return (
            x,
            {
                "attn": [attn],
                "inner_states": inner_states,
                "compositions": compositions,
                "compositions_states": compositions_states,
                "attn_states": attn_states,
            },
        )
