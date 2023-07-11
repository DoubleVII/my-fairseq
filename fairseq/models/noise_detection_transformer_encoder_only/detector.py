
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import torch

from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
import math

class DetectorModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embed_dim = args.decoder_embed_dim * 2
        self.embed_scale = math.sqrt(args.decoder_embed_dim)
        self.dropout_module = FairseqDropout(
            args.detector_dropout, module_name=self.__class__.__name__
        )


        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )

        self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.detector_ffn_embed_dim,
        )
        self.fc2 = self.build_fc2(
            args.detector_ffn_embed_dim,
            self.embed_dim,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

        self.cls_embedding = torch.randn((1,1,self.embed_dim), dtype=torch.float) / 2
        self.cls_embedding = nn.Parameter(self.cls_embedding)

        self.output_layer1 = self.build_fc1(
            self.embed_dim,
            args.detector_output_hidden_dim,
        )
        self.output_layer2 = self.build_fc2(
            args.detector_output_hidden_dim,
            2,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.detector_attention_heads,
            kdim=embed_dim,
            vdim=embed_dim,
            dropout=args.detector_attention_dropout,
            encoder_decoder_attention=True,
        )

    def build_fc1(self, input_dim, output_dim, q_noise=0, qn_block_size=8):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise=0, qn_block_size=8):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)
        
    def forward(self,
        encoder_hidden: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        features_only = False,
    ):
        batch_size = encoder_hidden.size(1)
        cls_embedding = self.embed_scale * self.cls_embedding
        x = cls_embedding.expand(-1,batch_size,-1)

        residual = x

        x, attn = self.encoder_attn(
            query=x,
            key=encoder_hidden,
            value=encoder_hidden,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            need_head_weights=False,
        )
        x = self.dropout_module(x)
        x = residual + x
        x = self.encoder_attn_layer_norm(x)

        x = x.squeeze(dim=0)

        residual = x

        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.final_layer_norm(x)
        if not features_only:
            x = self.dropout_module(x)
            x = self.output_layer1(x)
            x = self.activation_fn(x)
            x = self.dropout_module(x)
            x = self.output_layer2(x)
        return x