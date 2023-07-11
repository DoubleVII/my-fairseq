from fairseq.models import register_model_architecture


@register_model_architecture("example_model", "example_arch")
def example_arch(args):
    args.layer_num = getattr(args, "layer_num", 3)
    args.linear_dim = getattr(args, "linear_dim", 16)
