import torch
import torch.nn as nn
import torch.nn.functional as F

# x = torch.FloatTensor([1, 2, 3])
# y = torch.FloatTensor([2, 2, 1])

# x = torch.randn((12, 6, 1, 5))  # T x B x L x C
# y = torch.randn((6, 5, 2))  # L x C

da = {"a": 1, "b": 2, "d":12}
db = {"b": 3}
da.update(db)
print(da)

# checkpoint_dir = (
#     r"E:\code\fairseq\fairseq\models\teacher_for_recurrent\checkpoints\checkpoint_last.pt"
# )

# ckpt = torch.load(checkpoint_dir)
# model = ckpt["model"]
# print(model.keys())
# exit()
# del_keys = []
# for k, v in model.items():
#     k_split = k.split(".")
#     if k_split[0] == "recurrent_encoder":
#         del_keys.append(k)

# for k in del_keys:
#     model.pop(k)
# print(ckpt["model"].keys())
# # ckpt_new = {"args": ckpt["args"], "model": ckpt["model"]}
# torch.save(ckpt, checkpoint_dir)
# # print(ckpt.keys())


# dict_keys(['teacher_encoder.version', 'teacher_encoder.embed_tokens.weight', 'teacher_encoder.embed_positions._float_tensor', 'teacher_encoder.layers.0.self_attn.k_proj.weight', 'teacher_encoder.layers.0.self_attn.k_proj.bias', 'teacher_encoder.layers.0.self_attn.v_proj.weight', 'teacher_encoder.layers.0.self_attn.v_proj.bias', 'teacher_encoder.layers.0.self_attn.q_proj.weight', 'teacher_encoder.layers.0.self_attn.q_proj.bias', 'teacher_encoder.layers.0.self_attn.out_proj.weight', 'teacher_encoder.layers.0.self_attn.out_proj.bias', 'teacher_encoder.layers.0.self_attn_layer_norm.weight', 'teacher_encoder.layers.0.self_attn_layer_norm.bias', 'teacher_encoder.layers.0.fc1.weight', 'teacher_encoder.layers.0.fc1.bias', 'teacher_encoder.layers.0.fc2.weight', 'teacher_encoder.layers.0.fc2.bias', 'teacher_encoder.layers.0.final_layer_norm.weight', 'teacher_encoder.layers.0.final_layer_norm.bias', 'teacher_encoder.layers.1.self_attn.k_proj.weight', 'teacher_encoder.layers.1.self_attn.k_proj.bias', 'teacher_encoder.layers.1.self_attn.v_proj.weight', 'teacher_encoder.layers.1.self_attn.v_proj.bias', 'teacher_encoder.layers.1.self_attn.q_proj.weight', 'teacher_encoder.layers.1.self_attn.q_proj.bias', 'teacher_encoder.layers.1.self_attn.out_proj.weight', 'teacher_encoder.layers.1.self_attn.out_proj.bias', 'teacher_encoder.layers.1.self_attn_layer_norm.weight', 'teacher_encoder.layers.1.self_attn_layer_norm.bias', 'teacher_encoder.layers.1.fc1.weight', 'teacher_encoder.layers.1.fc1.bias', 'teacher_encoder.layers.1.fc2.weight', 'teacher_encoder.layers.1.fc2.bias', 'teacher_encoder.layers.1.final_layer_norm.weight', 'teacher_encoder.layers.1.final_layer_norm.bias', 'teacher_encoder.layers.2.self_attn.k_proj.weight', 'teacher_encoder.layers.2.self_attn.k_proj.bias', 'teacher_encoder.layers.2.self_attn.v_proj.weight', 'teacher_encoder.layers.2.self_attn.v_proj.bias', 'teacher_encoder.layers.2.self_attn.q_proj.weight', 'teacher_encoder.layers.2.self_attn.q_proj.bias', 'teacher_encoder.layers.2.self_attn.out_proj.weight', 'teacher_encoder.layers.2.self_attn.out_proj.bias', 'teacher_encoder.layers.2.self_attn_layer_norm.weight', 'teacher_encoder.layers.2.self_attn_layer_norm.bias', 'teacher_encoder.layers.2.fc1.weight', 'teacher_encoder.layers.2.fc1.bias', 'teacher_encoder.layers.2.fc2.weight', 'teacher_encoder.layers.2.fc2.bias', 'teacher_encoder.layers.2.final_layer_norm.weight', 'teacher_encoder.layers.2.final_layer_norm.bias', 'teacher_encoder.layers.3.self_attn.k_proj.weight', 'teacher_encoder.layers.3.self_attn.k_proj.bias', 'teacher_encoder.layers.3.self_attn.v_proj.weight', 'teacher_encoder.layers.3.self_attn.v_proj.bias', 'teacher_encoder.layers.3.self_attn.q_proj.weight', 'teacher_encoder.layers.3.self_attn.q_proj.bias', 'teacher_encoder.layers.3.self_attn.out_proj.weight', 'teacher_encoder.layers.3.self_attn.out_proj.bias', 'teacher_encoder.layers.3.self_attn_layer_norm.weight', 'teacher_encoder.layers.3.self_attn_layer_norm.bias', 'teacher_encoder.layers.3.fc1.weight', 'teacher_encoder.layers.3.fc1.bias', 'teacher_encoder.layers.3.fc2.weight', 'teacher_encoder.layers.3.fc2.bias', 'teacher_encoder.layers.3.final_layer_norm.weight', 'teacher_encoder.layers.3.final_layer_norm.bias', 'teacher_encoder.layers.4.self_attn.k_proj.weight', 'teacher_encoder.layers.4.self_attn.k_proj.bias', 'teacher_encoder.layers.4.self_attn.v_proj.weight', 'teacher_encoder.layers.4.self_attn.v_proj.bias', 'teacher_encoder.layers.4.self_attn.q_proj.weight', 'teacher_encoder.layers.4.self_attn.q_proj.bias', 'teacher_encoder.layers.4.self_attn.out_proj.weight', 'teacher_encoder.layers.4.self_attn.out_proj.bias', 'teacher_encoder.layers.4.self_attn_layer_norm.weight', 'teacher_encoder.layers.4.self_attn_layer_norm.bias', 'teacher_encoder.layers.4.fc1.weight', 'teacher_encoder.layers.4.fc1.bias', 'teacher_encoder.layers.4.fc2.weight', 'teacher_encoder.layers.4.fc2.bias', 'teacher_encoder.layers.4.final_layer_norm.weight', 'teacher_encoder.layers.4.final_layer_norm.bias', 'teacher_encoder.layers.5.self_attn.k_proj.weight', 'teacher_encoder.layers.5.self_attn.k_proj.bias', 'teacher_encoder.layers.5.self_attn.v_proj.weight', 'teacher_encoder.layers.5.self_attn.v_proj.bias', 'teacher_encoder.layers.5.self_attn.q_proj.weight', 'teacher_encoder.layers.5.self_attn.q_proj.bias', 'teacher_encoder.layers.5.self_attn.out_proj.weight', 'teacher_encoder.layers.5.self_attn.out_proj.bias', 'teacher_encoder.layers.5.self_attn_layer_norm.weight', 'teacher_encoder.layers.5.self_attn_layer_norm.bias', 'teacher_encoder.layers.5.fc1.weight', 'teacher_encoder.layers.5.fc1.bias', 'teacher_encoder.layers.5.fc2.weight', 'teacher_encoder.layers.5.fc2.bias', 'teacher_encoder.layers.5.final_layer_norm.weight', 'teacher_encoder.layers.5.final_layer_norm.bias'])
# dict_keys(['teacher_encoder.version', 'teacher_encoder.embed_tokens.weight', 'teacher_encoder.embed_positions._float_tensor', 'teacher_encoder.layers.0.self_attn.k_proj.weight', 'teacher_encoder.layers.0.self_attn.k_proj.bias', 'teacher_encoder.layers.0.self_attn.v_proj.weight', 'teacher_encoder.layers.0.self_attn.v_proj.bias', 'teacher_encoder.layers.0.self_attn.q_proj.weight', 'teacher_encoder.layers.0.self_attn.q_proj.bias', 'teacher_encoder.layers.0.self_attn.out_proj.weight', 'teacher_encoder.layers.0.self_attn.out_proj.bias', 'teacher_encoder.layers.0.self_attn_layer_norm.weight', 'teacher_encoder.layers.0.self_attn_layer_norm.bias', 'teacher_encoder.layers.0.fc1.weight', 'teacher_encoder.layers.0.fc1.bias', 'teacher_encoder.layers.0.fc2.weight', 'teacher_encoder.layers.0.fc2.bias', 'teacher_encoder.layers.0.final_layer_norm.weight', 'teacher_encoder.layers.0.final_layer_norm.bias', 'teacher_encoder.layers.1.self_attn.k_proj.weight', 'teacher_encoder.layers.1.self_attn.k_proj.bias', 'teacher_encoder.layers.1.self_attn.v_proj.weight', 'teacher_encoder.layers.1.self_attn.v_proj.bias', 'teacher_encoder.layers.1.self_attn.q_proj.weight', 'teacher_encoder.layers.1.self_attn.q_proj.bias', 'teacher_encoder.layers.1.self_attn.out_proj.weight', 'teacher_encoder.layers.1.self_attn.out_proj.bias', 'teacher_encoder.layers.1.self_attn_layer_norm.weight', 'teacher_encoder.layers.1.self_attn_layer_norm.bias', 'teacher_encoder.layers.1.fc1.weight', 'teacher_encoder.layers.1.fc1.bias', 'teacher_encoder.layers.1.fc2.weight', 'teacher_encoder.layers.1.fc2.bias', 'teacher_encoder.layers.1.final_layer_norm.weight', 'teacher_encoder.layers.1.final_layer_norm.bias', 'teacher_encoder.layers.2.self_attn.k_proj.weight', 'teacher_encoder.layers.2.self_attn.k_proj.bias', 'teacher_encoder.layers.2.self_attn.v_proj.weight', 'teacher_encoder.layers.2.self_attn.v_proj.bias', 'teacher_encoder.layers.2.self_attn.q_proj.weight', 'teacher_encoder.layers.2.self_attn.q_proj.bias', 'teacher_encoder.layers.2.self_attn.out_proj.weight', 'teacher_encoder.layers.2.self_attn.out_proj.bias', 'teacher_encoder.layers.2.self_attn_layer_norm.weight', 'teacher_encoder.layers.2.self_attn_layer_norm.bias', 'teacher_encoder.layers.2.fc1.weight', 'teacher_encoder.layers.2.fc1.bias', 'teacher_encoder.layers.2.fc2.weight', 'teacher_encoder.layers.2.fc2.bias', 'teacher_encoder.layers.2.final_layer_norm.weight', 'teacher_encoder.layers.2.final_layer_norm.bias', 'teacher_encoder.layers.3.self_attn.k_proj.weight', 'teacher_encoder.layers.3.self_attn.k_proj.bias', 'teacher_encoder.layers.3.self_attn.v_proj.weight', 'teacher_encoder.layers.3.self_attn.v_proj.bias', 'teacher_encoder.layers.3.self_attn.q_proj.weight', 'teacher_encoder.layers.3.self_attn.q_proj.bias', 'teacher_encoder.layers.3.self_attn.out_proj.weight', 'teacher_encoder.layers.3.self_attn.out_proj.bias', 'teacher_encoder.layers.3.self_attn_layer_norm.weight', 'teacher_encoder.layers.3.self_attn_layer_norm.bias', 'teacher_encoder.layers.3.fc1.weight', 'teacher_encoder.layers.3.fc1.bias', 'teacher_encoder.layers.3.fc2.weight', 'teacher_encoder.layers.3.fc2.bias', 'teacher_encoder.layers.3.final_layer_norm.weight', 'teacher_encoder.layers.3.final_layer_norm.bias', 'teacher_encoder.layers.4.self_attn.k_proj.weight', 'teacher_encoder.layers.4.self_attn.k_proj.bias', 'teacher_encoder.layers.4.self_attn.v_proj.weight', 'teacher_encoder.layers.4.self_attn.v_proj.bias', 'teacher_encoder.layers.4.self_attn.q_proj.weight', 'teacher_encoder.layers.4.self_attn.q_proj.bias', 'teacher_encoder.layers.4.self_attn.out_proj.weight', 'teacher_encoder.layers.4.self_attn.out_proj.bias', 'teacher_encoder.layers.4.self_attn_layer_norm.weight', 'teacher_encoder.layers.4.self_attn_layer_norm.bias', 'teacher_encoder.layers.4.fc1.weight', 'teacher_encoder.layers.4.fc1.bias', 'teacher_encoder.layers.4.fc2.weight', 'teacher_encoder.layers.4.fc2.bias', 'teacher_encoder.layers.4.final_layer_norm.weight', 'teacher_encoder.layers.4.final_layer_norm.bias', 'teacher_encoder.layers.5.self_attn.k_proj.weight', 'teacher_encoder.layers.5.self_attn.k_proj.bias', 'teacher_encoder.layers.5.self_attn.v_proj.weight', 'teacher_encoder.layers.5.self_attn.v_proj.bias', 'teacher_encoder.layers.5.self_attn.q_proj.weight', 'teacher_encoder.layers.5.self_attn.q_proj.bias', 'teacher_encoder.layers.5.self_attn.out_proj.weight', 'teacher_encoder.layers.5.self_attn.out_proj.bias', 'teacher_encoder.layers.5.self_attn_layer_norm.weight', 'teacher_encoder.layers.5.self_attn_layer_norm.bias', 'teacher_encoder.layers.5.fc1.weight', 'teacher_encoder.layers.5.fc1.bias', 'teacher_encoder.layers.5.fc2.weight', 'teacher_encoder.layers.5.fc2.bias', 'teacher_encoder.layers.5.final_layer_norm.weight', 'teacher_encoder.layers.5.final_layer_norm.bias'])