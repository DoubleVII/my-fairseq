import torch
import torch.nn as nn
import torch.nn.functional as F

base_checkpoint_dir = (
    "/home/yangs/code/fairseq_models/rectification_transformer/checkpoint.avg.pt"
)
len_checkpoint_dir = "/home/yangs/code/fairseq_models/rectification_transformer/checkpoints_base/checkpoint.avg.pt"
dest_checkpoint_dir = "/home/yangs/code/fairseq_models/rectification_transformer/checkpoints_base/checkpoint_merged.pt"

base_ckpt = torch.load(base_checkpoint_dir)
rec_ckpt = torch.load(len_checkpoint_dir)
# print(base_ckpt["model"].keys())
# print(rec_ckpt["model"].keys())

# len_ckpt["model"].update(base_ckpt["model"])
# torch.save(len_ckpt, dest_checkpoint_dir)

base_model = base_ckpt["model"]
rec_model = rec_ckpt["model"]
# temp_model = temp_ckpt["model"]
merged_param = {}

for k, v in base_model.items():
    k_split = k.split(".")
    if k_split[1] != "version":
        merged_param["base_" + k] = v
rec_ckpt["model"].update(merged_param)
torch.save(rec_ckpt, dest_checkpoint_dir)

# for k, v in rec_model.items():
#     k_split = k.split(".")
#     if k_split[0] == "recurrent_encoder" and k_split[1] == "layers":
#         merged_param["encoder.recurrent_layer" + k[26:]] = v


# temp_ckpt["model"] = merged_param
# # print(teacher_keys.keys())
# # ckpt_new = {"args": ckpt["args"], "model": ckpt["model"]}
# torch.save(temp_ckpt, merged_checkpoint_dir)
# # print(ckpt.keys())
