from fairseq.models import (
    register_model,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    FairseqEncoder,
    FairseqDecoder,
)
import math

from fairseq import utils

import torch
import torch.nn as nn
from torch import Tensor

from fairseq.data.dictionary import Dictionary


from fairseq.modules import FairseqDropout, LayerNorm

from typing import Any, Dict, List, Optional, Tuple, NamedTuple


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Optional[Tensor]),  # B x T
        ("encoder_embedding", Optional[Tensor]),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1
    ],
)

# @register_model("ESD_model")
class ESDModel(FairseqEncoderDecoderModel):
    def __init__(
        self, args, encoder, decoder,
    ):
        super().__init__(encoder, decoder)
        self.args = args

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--word-embed-dim", type=int, metavar="N", help="word embed dim",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        # base_architecture(args)

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        encoder_embed_tokens = cls.build_embedding(args, src_dict, args.encoder_embed_dim)
        decoder_embed_tokens = cls.build_embedding(args, tgt_dict, args.decoder_embed_dim)

        # TODO: make sure component_embed_dim equal to decoder hidden dim
        component_embed_tokens = cls.build_embedding(args, None, args.component_embed_dim)

        encoder = cls.build_encoder(
            args, src_dict, encoder_embed_tokens, component_embed_tokens
        )
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(
        cls, args, dictionary: Optional[Dictionary], embed_dim, num_embeddings=None
    ):
        if dictionary is not None:
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            assert num_embeddings is not None
            emb = Embedding(num_embeddings, embed_dim)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens, component_embed_tokens):
        pass
        # return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        pass
        # return TransformerDecoder(args, tgt_dict, embed_tokens)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """

        return_all_hiddens = True
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, src_lengths=src_lengths,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class ESDEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens, component_embed_tokens):
        super().__init__(dictionary)
        self.context_encoder = ESDEncoder.build_context_encoder(
            args, dictionary, embed_tokens
        )
        self.word_encoder = ESDEncoder.build_word_encoder(args, dictionary, embed_tokens)
        self.definition_encoder = ESDEncoder.build_definition_encoder(
            args, dictionary, embed_tokens
        )
        self.semantic_component_prior_predictor = ESDEncoder.build_semantic_component_prior_predictor(
            args, dictionary
        )
        self.semantic_component_posterior_predictor = ESDEncoder.build_semantic_component_posterior_predictor(
            args, dictionary
        )

        self.component_embed_tokens = component_embed_tokens

    @classmethod
    def build_context_encoder(cls, args, src_dict, embed_tokens):
        pass

    @classmethod
    def build_word_encoder(cls, args, src_dict, embed_tokens):
        pass

    @classmethod
    def build_definition_encoder(cls, args, src_dict, embed_tokens):
        pass

    @classmethod
    def build_semantic_component_prior_predictor(cls, args, src_dict):
        pass

    @classmethod
    def build_semantic_component_posterior_predictor(cls, args, src_dict):
        pass

    def forward(
        self,
        word_tokens,
        word_lengths,
        char_tokens,
        char_lengths,
        context_tokens,
        context_lengths,
        definition_tokens=None,
        definition_lengths=None,
    ):

        word_hidden = self.word_encoder(
            word_tokens, word_lengths, char_tokens, char_lengths
        )

        context_hiddens = self.context_encoder(context_tokens, context_lengths)

        q = None
        components = None
        definition_hiddens = None
        if definition_tokens is not None:
            definition_hiddens = self.definition_encoder(
                definition_tokens, definition_lengths
            )
            q = self.semantic_component_posterior_predictor(
                context_hiddens,
                context_lengths,
                definition_hiddens,
                definition_lengths,
                word_hidden,
            )
            components = self.component_embed_tokens(q)

        p = self.semantic_component_prior_predictor(
            context_hiddens, context_lengths, word_hidden
        )
        if not components:
            components = self.component_embed_tokens(p)

        return EncoderOut(
            word_hidden=word_hidden,  # T x B x C
            context_hiddens=context_hiddens,  # B x T
            components=components,
            definition_hiddens=definition_hiddens,
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        new_word_hidden = (
            encoder_out.word_hidden
            if encoder_out.word_hidden is None
            else encoder_out.word_hidden.index_select(0, new_order)
        )

        new_context_hiddens = (
            encoder_out.context_hiddens
            if encoder_out.context_hiddens is None
            else encoder_out.context_hiddens.index_select(0, new_order)
        )

        new_components = (
            encoder_out.components
            if encoder_out.components is None
            else encoder_out.components.index_select(0, new_order)
        )

        new_definition_hiddens = (
            encoder_out.definition_hiddens
            if encoder_out.definition_hiddens is None
            else encoder_out.definition_hiddens.index_select(0, new_order)
        )

        return EncoderOut(
            word_hidden=new_word_hidden,
            context_hiddens=new_context_hiddens,
            components=new_components,
            definition_hiddens=new_definition_hiddens,
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)


class ESDDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)

        self.f_hidden_dim = (
            args.context_encoder_hidden * 2
            + args.word_embed_dim
            + args.char_conv_feature_size
            + args.component_embed_dim
        )

        self.decoder_hidden_dim = args.decoder_hidden_dim

        self.proj_u = nn.Linear(
            self.f_hidden_dim + self.decoder_hidden_dim, self.decoder_hidden_dim
        )
        self.proj_v = nn.Linear(self.f_hidden_dim, self.decoder_hidden_dim)
        self.sigmoid_activation = utils.get_activation_fn(activation="sigmoid")
        self.tanh_activation = utils.get_activation_fn(activation="tanh")

        self.proj_s = nn.Liner(
            self.f_hidden_dim + self.decoder_hidden_dim, self.decoder_hidden_dim
        )

    @classmethod
    def build_context_encoder(cls, args, src_dict, embed_tokens):
        pass

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs
    ):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict, optional): dictionary used for storing
                state during :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def get_attn_hidden(self, querry_hidden, keys):
        attn_weights = torch.bmm(keys, querry_hidden.unsquezee(dim=2)).squeeze(
            dim=2
        )  # B x T
        attn_weights = utils.softmax(attn_weights, dim=-1)
        attn = torch.bmm(attn_weights.unsqueeze(dim=1), keys).squeeze(dim=1)  # B x C
        return attn

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        else:
            s0
            context_hidden_attn = self.get_attn_hidden()

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder incremental state.

        This will be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        pass


class ContextEncoder(nn.Module):
    """
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, embed_tokens):
        super().__init__()

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

        self.embed_dim = embed_tokens.embedding_dim
        self.hidden_size = args.context_encoder_hidden
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_context_positions = args.max_source_context_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = math.sqrt(self.embed_dim)

        self.num_layers = args.context_encoder_layers

        self.dropout_rate = args.dropout

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(self.embed_dim)
        else:
            self.layernorm_embedding = None

        self.lstm = nn.LSTM(
            self.embed_dim,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate,
            bidirectional=True,
        )

        self.output_norm = LayerNorm(self.embed_dim)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        return x, embed

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            output (FloatTensor): shape `(batch, src_len, 2 * hidden_size)`
        """
        x, _ = self.forward_embedding(src_tokens)

        print("check is sorted: ", src_lengths)
        assert False
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, src_lengths, batch_first=True, enforce_sorted=False
        )

        packed_output, (h_n, c_n) = self.lstm(packed_x)

        output = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        output = self.output_norm(output)

        return output

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """

        return encoder_out.index_select(0, new_order)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_context_positions
        return min(self.max_source_context_positions, self.embed_positions.max_positions)


class WordEncoder(nn.Module):
    """
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        word_embed_tokens (torch.nn.Embedding): input word embedding
        char_embed_tokens (torch.nn.Embedding): input character embedding
    """

    def __init__(self, args, word_embed_tokens, char_embed_tokens):
        super().__init__()

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

        self.word_embed_dim = word_embed_tokens.embedding_dim
        self.char_embed_dim = char_embed_tokens.embedding_dim
        self.char_conv_feature_size = args.char_conv_feature_size
        self.char_conv_kernal_size = args.char_conv_kernal_size
        self.char_padding_idx = char_embed_tokens.padding_idx
        self.max_source_word_positions = args.max_source_word_positions
        self.max_source_char_positions = args.max_source_char_positions

        self.word_embed_tokens = word_embed_tokens
        self.char_embed_tokens = char_embed_tokens

        self.word_embed_scale = math.sqrt(self.word_embed_dim)
        self.char_embed_scale = math.sqrt(self.char_embed_dim)

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(self.embed_dim)
        else:
            self.layernorm_embedding = None

        self.conv1d = nn.Conv1d(
            self.char_embed_dim, self.char_conv_feature_size, self.char_conv_kernal_size
        )

        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )

        self.output_norm = LayerNorm(self.word_embed_dim + self.char_conv_feature_size)

    def forward_word_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.word_embed_scale * self.word_embed_tokens(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        return x, embed

    def forward_char_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.char_embed_scale * self.char_embed_tokens(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        return x, embed

    def forward(
        self, src_word_tokens, src_word_lengths, src_char_tokens, src_char_lengths
    ):
        """
        Args:
            src_word_tokens (LongTensor): word tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): word lengths of each source sentence of
                shape `(batch)`
            src_char_tokens (LongTensor): char tokens of the given word
                `(batch, src_len)`
            src_lengths (torch.LongTensor): char lengths of the given word
                `(batch)`

        Returns:
        """
        assert torch.sum(src_word_lengths).item() == len(src_word_lengths)

        char_padding_mask = src_char_tokens.eq(self.char_padding_idx)

        x_word, _ = self.forward_word_embedding(src_word_tokens)  # B x 1 x C

        x_word = x_word.squeeze(1)  # B x C

        x_char, _ = self.forward_char_embedding(src_char_tokens)  # B x T X C

        x_char = x_char.transpose(1, 2)  # B x C x T

        x_char = self.conv1d(x_char)

        x_char = self.activation_fn(x_char)

        conv_char_padding_mask = char_padding_mask[
            self.char_conv_kernal_size - 1 :,
        ]
        assert conv_char_padding_mask.size() == x_char.size()
        x_char = x_char.masked_fill(conv_char_padding_mask, 0.0)

        x_char = torch.max(x_char, dim=2)  # B x C

        x_char = self.dropout_module(x_char)

        output = torch.cat((x_word, x_char), dim=1)

        output = self.output_norm(output)

        return output

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """

        return encoder_out.index_select(0, new_order)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        assert self.max_source_char_positions is None
        return self.max_source_char_positions


class DefinitionEncoder(nn.Module):
    """
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, embed_tokens):
        super().__init__()

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

        self.embed_dim = embed_tokens.embedding_dim
        self.hidden_size = args.definition_encoder_hidden
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_definition_positions = args.max_source_definition_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = math.sqrt(self.embed_dim)

        self.num_layers = args.definition_encoder_layers

        self.dropout_rate = args.dropout

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(self.embed_dim)
        else:
            self.layernorm_embedding = None

        self.lstm = nn.LSTM(
            self.embed_dim,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate,
            bidirectional=True,
        )

        self.output_norm = LayerNorm(self.embed_dim)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        return x, embed

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            output (FloatTensor): shape `(batch, src_len, 2 * hidden_size)`
        """
        x, _ = self.forward_embedding(src_tokens)

        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, src_lengths, batch_first=True, enforce_sorted=False
        )

        packed_output, (h_n, c_n) = self.lstm(packed_x)

        output = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        output = self.output_norm(output)

        return output

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """

        return encoder_out.index_select(0, new_order)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        assert self.max_source_definition_positions is not None
        return self.max_source_definition_positions


class SemanticComponentPriorPredictor(nn.Module):
    """
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args):
        super().__init__()

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

        self.input_feature = (
            args.context_encoder_hidden * 2
            + args.word_embed_dim
            + args.char_conv_feature_size
        )

        self.component_num = args.component_num

        self.proj_p = nn.Linear(self.input_feature, self.component_num)

        self.output_norm = LayerNorm(self.component_num)

    def forward(self, context_hiddens, context_lengths, word_hidden):
        """
        Args:
            context_hiddens (FloatTensor): 
                `(batch, src_len, context_hidden_dim)`
            context_lengths (torch.LongTensor): 
                shape `(batch)`
            word_hidden (torch.FloatTensor): shape `(batch, word_hidden_dim)`

        Returns:
            output (FloatTensor): shape `(batch, src_len, 2 * hidden_size)`
        """

        context_hidden = torch.max(context_hiddens, dim=1)  # B x context_hidden_dim

        x = torch.cat((word_hidden, context_hidden), dim=1)

        p = self.proj_p(x)

        p = self.dropout_module(p)

        p = self.output_norm(p)

        return p

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """

        return encoder_out.index_select(0, new_order)


class SemanticComponentPosteriorPredictor(nn.Module):
    """
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args):
        super().__init__()

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

        self.input_feature = (
            args.context_encoder_hidden * 2
            + args.word_embed_dim
            + args.char_conv_feature_size
            + args.definition_encoder_hidden
        )

        self.component_num = args.component_num

        self.proj_q = nn.Linear(self.input_feature, self.component_num)

        self.output_norm = LayerNorm(self.component_num)

    def forward(
        self,
        context_hiddens,
        context_lengths,
        definition_hiddens,
        definition_lengths,
        word_hidden,
    ):
        """
        Args:
            context_hiddens (FloatTensor): 
                `(batch, src_len, context_hidden_dim)`
            context_lengths (torch.LongTensor): 
                shape `(batch)`
            word_hidden (torch.FloatTensor): shape `(batch, word_hidden_dim)`

        Returns:
            output (FloatTensor): shape `(batch, src_len, 2 * hidden_size)`
        """

        context_hidden = torch.max(context_hiddens, dim=1)  # B x context_hidden_dim
        definition_hidden = torch.max(
            definition_hiddens, dim=1
        )  # B x definition_hidden_dim

        x = torch.cat((word_hidden, context_hidden, definition_hidden), dim=1)

        q = self.proj_q(x)

        q = self.dropout_module(q)

        q = self.output_norm(q)

        return q

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """

        return encoder_out.index_select(0, new_order)
