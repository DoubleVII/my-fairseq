from fairseq.models import register_model_architecture
from fairseq.models.transformer import base_architecture

# The first argument to ``register_model_architecture()`` should be the name
# of the model we registered above (i.e., 'rnn_classifier'). The function we
# register here should take a single argument *args* and modify it in-place
# to match the desired architecture.


@register_model_architecture("dlcl_rec_transformer", "dlcl_rec_transformer_arch")
def dlcl_rec_transformer_arch(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.encoder_recurrent = getattr(args, "encoder_recurrent", 1)
    args.decoder_recurrent = getattr(args, "decoder_recurrent", 1)
    args.layer_attn_reduction = getattr(args, "layer_attn_reduction", "mean")
    args.no_layer_output_residual = getattr(args, "no_layer_output_residual", False)
    args.only_last_recurrent = getattr(args, "only_last_recurrent", False)
    args.layer_attention_num = 2 + args.encoder_layers * (
        1 if args.only_last_recurrent else args.encoder_recurrent
    )
    args.layer_attn_ffn = getattr(args, "layer_attn_ffn", False)
    args.encoder_layer_route = getattr(args, "encoder_layer_route", "low")
    args.decoder_layer_route = getattr(args, "decoder_layer_route", "high")
    args.layer_aggregation = getattr(args, "layer_aggregation", "attn")
    args.aggregation_in_last_layer = getattr(args, "aggregation_in_last_layer", False)
    args.time_wise_layer_attn = getattr(args, "time_wise_layer_attn", False)
    base_architecture(args)
