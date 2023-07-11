import torch

m = torch.load("roberta.base/model_fitting_gelu.pt")
gelu_param = torch.load("roberta.base/fitting_gelu_paras.pt")

for i in range(12):
    m["model"][
        f"encoder.sentence_encoder.layers.{i}.activation_fn.fc1.weight"
    ] = gelu_param["fc1.weight"]
    m["model"][
        f"encoder.sentence_encoder.layers.{i}.activation_fn.fc1.bias"
    ] = gelu_param["fc1.bias"]
    m["model"][
        f"encoder.sentence_encoder.layers.{i}.activation_fn.fc2.weight"
    ] = gelu_param["fc2.weight"]
    m["model"][
        f"encoder.sentence_encoder.layers.{i}.activation_fn.fc2.bias"
    ] = gelu_param["fc2.bias"]
# m = torch.load('checkpoints/checkpoint_best.acd.pt')
# import pdb

# pdb.set_trace()

torch.save(m, "roberta.base/model_fitting_gelu.pt")
