from fairseq.models import register_model, BaseFairseqModel
from torch import nn


@register_model("example_model")
class ExampleModel(BaseFairseqModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.layer_num = args.layer_num
        self.linear_dim = args.linear_dim

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                nn.Linear(
                    1 if idx == 0 else self.linear_dim,
                    1 if idx == self.layer_num - 1 else self.linear_dim,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.activation = nn.ReLU()

    @staticmethod
    def add_args(parser):
        parser.add_argument("--linear-dim", type=int, help="linear dim")
        parser.add_argument("--layer-num", type=int, help="layer number")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args)

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

