import torch
import torch.nn as nn
import torch.nn.functional as F

# l = [0.0064, 0.0060, 0.0238, 0.0034, 0.0033, 0.0079, 0.0092, 0.0062, 0.0104,
#          0.0097, 0.0046]
# sum = 0
# for e in l:
#     sum = sum + e
# print(sum)

# l=[e/sum*0.8403999999999998 for e in l]
# print(l)
attn_weights = torch.randn(2 * 3, 3, 4)
bsz = 2
num_heads = 3
tgt_len = 3
src_len = 4
attn_weights_topk = attn_weights.view(bsz, num_heads, tgt_len, src_len).mean(
    dim=1, keepdim=True
)
k = min(1000, attn_weights_topk.size(2))
_, index = attn_weights_topk.topk(k, -1)
print(index)
print(index.size())
z_mask = torch.full_like(attn_weights_topk, float(-100))
z_mask = (
    z_mask.scatter(-1, index, 0.0)
    .repeat(1, num_heads, 1, 1)
    .view(bsz * num_heads, tgt_len, src_len)
)
print(z_mask)

# a = torch.randn(2, 3, 4)
# for i1, k in enumerate(a):
#     for i2, k1 in enumerate(k):
#         print(i2)
#         k1[0] = 12
# print(a)
# def sort_d(d):
#     d.sort(key=lambda elem: elem["name"])


# # print(torch.__version__)
# # print(0 % 5)
# d = {"a": 12, "b": "asd", "c": 222}
# print(d)
# sort_d(d)
# print(d)

# x = torch.randn(3, 4)
# y = torch.randn(3)

# print(torch.div(x, y.unsqueeze(-1)))

# x = torch.randn((12, 6, 1, 5))  # T x B x L x C
# y = torch.randn((6, 5, 2))  # L x C

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


# import keyword
# import torch
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()

# meta = []
# while len(meta) < 100:
#     meta = meta + keyword.kwlist  # get some strings
# meta = meta[:100]

# for i, v in enumerate(meta):
#     meta[i] = i // 20

# label_img = torch.rand(100, 3, 10, 32)
# for i in range(100):
#     label_img[i] *= i / 100.0

# writer.add_embedding(
#     torch.randn(100, 5), metadata=meta, label_img=label_img, global_step=0
# )
# writer.add_embedding(torch.randn(100, 5), label_img=label_img, global_step=1)
# writer.add_embedding(torch.randn(100, 5), metadata=meta, global_step=2)
