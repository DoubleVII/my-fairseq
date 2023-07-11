from fairseq.models import register_model_architecture
from fairseq.models.transformer import base_architecture

# The first argument to ``register_model_architecture()`` should be the name
# of the model we registered above (i.e., 'rnn_classifier'). The function we
# register here should take a single argument *args* and modify it in-place
# to match the desired architecture.


@register_model_architecture(
    "hybrid_teacher_recurrent_model", "hybrid_teacher_recurrent_arch"
)
def teacher_recurrent_transformer_arch(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.teacher_encoder_layers = getattr(args, "teacher_encoder_layers", 6)
    args.recurrent_encoder_layers = getattr(args, "recurrent_encoder_layers", 1)
    args.encoder_recurrent = getattr(args, "encoder_recurrent", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.start_teach_layer = getattr(args, "start_teach_layer", 0)
    args.end_teach_layer = getattr(args, "end_teach_layer", args.teacher_encoder_layers)
    base_architecture(args)
