import torch
import torch.nn as nn
import torch.nn.functional as F

score =torch.FloatTensor([ 0.6, 4.0, -1.0, 1.2, 6.4, 4.1, 0.1, 0.7, 1.7, -1.2])
ratio = score / torch.sum(score)
print(torch.sum(ratio))
print(score + -6.6*ratio)


exit()

range_vec = torch.arange(4)
range_mat = range_vec.repeat(4, 1)
distance_mat = range_mat - range_mat.transpose(0, 1)
print(distance_mat)
exit()

# l = [0.0064, 0.0060, 0.0238, 0.0034, 0.0033, 0.0079, 0.0092, 0.0062, 0.0104,
#          0.0097, 0.0046]
# sum = 0
# for e in l:
#     sum = sum + e
# print(sum)

# l=[e/sum*0.8403999999999998 for e in l]
# print(l)
torch.manual_seed(200)


def norm(mat):
    mat = mat.abs()
    for i, row in enumerate(mat):
        mat[i] = row / row.sum()
    return mat


attn = norm(torch.randn((4, 4)))
attn = torch.FloatTensor(
    [
        [0.3361, 0.1653, 0.1441, 0.3546],
        [0.0304, 0.1847, 0.2971, 0.4879],
        [0.3483, 0.1485, 0.3802, 0.1230],
        [0.0853, 0.6475, 0.0877, 0.1795],
    ]
)
print(attn)
print(attn[:, 0].mean())
print(attn[:, 1].mean())
print(attn[:, 2].mean())
print(attn[:, 3].mean())
print("----------------")

# tem = torch.ones((4, 4))
# tem[:, 1:] = 0
for i in range(20):
    attn = torch.matmul(norm(torch.randn((4, 4))), attn)
    # attn = torch.matmul(attn, attn)
    print(attn[0, :].sum())

print(attn)
# attn = torch.matmul(tem, attn)
# print(attn)


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



# base encoder out
# tensor([[[ 1.0176,  0.3262, -1.0557,  ...,  1.8906,  1.7939,  0.2299],
#          [ 2.3184,  1.3008,  0.5063,  ...,  0.3569,  1.3330, -0.1011],
#          [ 2.9551,  0.1799, -0.4097,  ...,  2.9961,  0.9844, -0.4714],
#          ...,
#          [ 1.9600,  0.1572, -0.6738,  ...,  1.9395,  0.6616,  0.1858],
#          [ 0.8501,  0.6348, -0.2559,  ...,  1.1836,  0.8950,  0.3176],
#          [ 2.4824,  1.1748, -0.5054,  ...,  0.9922,  1.4053,  0.1130]],

#         [[ 1.0820,  1.4834, -0.2876,  ...,  1.6465,  0.1237, -0.7603],
#          [ 0.2468,  0.3340, -0.1442,  ...,  2.3008,  1.5293,  0.2520],
#          [ 1.9756,  0.4788, -1.3311,  ...,  1.9111,  1.2412, -0.2443],
#          ...,
#          [ 2.0820, -0.3062, -1.4951,  ...,  1.6230,  0.1268,  0.1758],
#          [ 2.9629,  0.8877, -0.4375,  ...,  1.3691,  2.4883,  1.1797],
#          [ 2.2637,  0.9634, -1.1523,  ...,  1.0322,  0.7251,  1.0576]],

#         [[ 0.5142,  0.7622, -0.7900,  ...,  2.8887,  1.3564, -0.7104],
#          [ 1.5742, -0.3254, -1.2939,  ...,  2.4141,  1.9814, -0.4580],
#          [ 1.3135,  0.7822, -1.3086,  ...,  1.9297,  1.3037,  1.5479],
#          ...,
#          [-0.2048, -0.9561, -1.4668,  ...,  1.1660,  0.6201,  0.0370],
#          [ 1.8564, -0.4060, -0.9902,  ...,  2.1602,  1.5127,  1.7451],
#          [ 1.0967,  0.3723, -1.0771,  ...,  0.6484,  2.4551,  0.2847]],

#         ...,

#         [[ 0.5645,  0.4363, -0.3931,  ...,  1.8359,  1.5127, -0.0429],
#          [ 0.7388, -0.1621, -0.3530,  ...,  1.9580,  1.1816,  0.2993],
#          [ 1.8750, -0.1771, -1.7393,  ...,  2.6504,  0.6606,  0.7383],
#          ...,
#          [ 1.8770,  0.2499, -0.2754,  ...,  2.0605,  0.9136,  0.7002],
#          [ 2.0137,  0.9585, -0.3945,  ...,  2.5312,  0.7788,  0.5283],
#          [ 1.5859,  0.0942, -0.6548,  ...,  1.5381, -0.3962,  0.3162]],

#         [[ 1.0762,  1.5029, -1.4111,  ...,  1.2158,  1.4473, -0.7974],
#          [ 1.7295,  1.0635, -0.6406,  ...,  0.8247,  0.9321, -0.8755],
#          [ 2.4785,  1.1289, -0.5103,  ...,  1.5508,  1.2705,  0.3174],
#          ...,
#          [ 1.7168,  1.0498, -1.6016,  ...,  1.3408,  0.8975,  0.3008],
#          [ 2.5938,  0.8027, -0.7876,  ...,  0.7754,  1.1074, -0.0837],
#          [ 2.9590,  1.0371, -1.0176,  ..., -0.0716,  1.1387, -0.0637]],

#         [[ 2.0938,  0.6494, -0.3228,  ...,  1.5830,  0.9302,  0.3713],
#          [ 1.7734,  1.0264,  0.0668,  ...,  1.6182,  2.0098,  0.5435],
#          [ 2.0898,  0.8599, -1.1211,  ...,  1.7090,  1.4707,  0.9604],
#          ...,
#          [ 1.4834,  0.3572, -1.2793,  ...,  1.2539,  1.4141,  1.1221],
#          [ 2.6406,  1.1270, -0.6904,  ...,  1.3564,  1.4961,  0.9189],
#          [ 2.9277, -0.0203, -0.4072,  ...,  1.6289,  1.5781,  0.6475]]],
#        device='cuda:0', dtype=torch.float16, grad_fn=<NativeLayerNormBackward>)


# adaptive encoder out

# tensor([[[ 2.4316,  0.8823, -0.8208,  ...,  1.9023,  1.3379, -0.2522],
#          [ 2.4395,  1.4326,  0.0141,  ...,  0.2441,  0.6470, -0.3721],
#          [ 2.5293,  0.9805, -0.8262,  ...,  2.5488,  1.3262, -0.6274],
#          ...,
#          [ 1.1543,  0.4661, -0.0493,  ...,  2.0449,  0.3157,  0.6162],
#          [ 2.4512,  0.4783, -0.5889,  ...,  1.4473,  1.8584,  0.0105],
#          [ 2.2305,  0.5835, -0.3875,  ...,  1.8184,  1.8320,  0.6343]],

#         [[ 1.9414,  1.5059,  0.0185,  ...,  1.6562,  1.1650, -0.7583],
#          [ 1.1406,  0.1904, -0.1776,  ...,  1.9697,  1.8604, -0.3096],
#          [ 0.3621, -0.1479, -1.1064,  ...,  1.6016,  0.6548,  0.2776],
#          ...,
#          [ 2.0977, -0.0735, -1.0068,  ...,  1.0146,  0.3352,  0.2159],
#          [ 1.2090,  1.2051, -0.4131,  ...,  1.8467,  1.5645,  1.1914],
#          [ 0.7178,  0.3752, -0.5830,  ...,  1.9238,  1.9482,  1.3184]],

#         [[ 1.5381,  0.6030, -1.5020,  ...,  1.7217,  1.4307, -0.9668],
#          [ 1.1357, -0.4114, -1.0850,  ...,  2.4844,  1.8057, -0.7202],
#          [ 1.1973,  0.3042, -1.1865,  ...,  1.8154,  1.2881,  1.1006],
#          ...,
#          [ 1.2910, -0.6216, -1.0508,  ...,  1.2002,  0.9434,  0.2102],
#          [ 1.6113, -0.1539, -0.6313,  ...,  1.9180,  1.1621,  1.3496],
#          [-0.3933,  0.4956, -1.2041,  ...,  1.1582,  2.7129,  0.1545]],

#         ...,

#         [[ 1.6416,  0.6689, -0.8086,  ...,  1.5439,  1.8506, -0.0812],
#          [ 1.5449, -0.6890, -0.6289,  ...,  2.0293,  1.3213,  0.8833],
#          [ 1.6406, -0.1837, -1.4307,  ...,  2.2949, -0.0091,  0.3289],
#          ...,
#          [ 1.9541,  0.2810,  0.0852,  ...,  1.7783,  0.7573,  0.6436],
#          [ 1.8018,  0.9272, -0.1423,  ...,  2.6445,  0.8057,  0.6523],
#          [ 1.5488,  0.3132, -0.5625,  ...,  1.6348,  2.0645,  0.3433]],

#         [[ 2.8691,  1.6816, -0.8496,  ...,  0.6362,  1.2100, -0.6689],
#          [ 1.6689,  1.1953, -0.3687,  ...,  1.1377,  0.8540, -0.8848],
#          [ 2.2695,  0.8145, -1.3496,  ...,  1.4326,  1.4961,  0.4080],
#          ...,
#          [ 2.4609,  0.9004, -0.8164,  ...,  1.1777,  0.6172, -0.0781],
#          [ 2.6621,  1.2861, -0.7808,  ...,  1.4053,  0.9653,  0.1203],
#          [ 2.9004,  1.0801, -1.1934,  ...,  1.2119,  1.6084, -0.4036]],

#         [[ 1.9385,  0.7847, -0.4399,  ...,  1.5801,  1.7412, -0.2649],
#          [ 1.4414,  0.6074, -0.1820,  ...,  1.8828,  1.4463, -0.1646],
#          [ 1.9023,  0.7358, -1.1113,  ...,  2.1270,  1.5244,  0.6719],
#          ...,
#          [ 1.9688,  0.4822, -0.9834,  ...,  1.5127,  1.6357,  0.7876],
#          [ 0.8057,  0.9229, -0.0378,  ...,  1.8086,  1.2461,  1.3105],
#          [ 2.5234,  0.4351, -0.6411,  ...,  2.0000,  1.7363,  0.2583]]],
#        device='cuda:0', dtype=torch.float16, grad_fn=<NativeLayerNormBackward>)



# base input
# tensor([[[ 3.2344e+00, -2.6270e-01,  2.9297e-02,  ...,  1.6514e+00,
#            1.4189e+00,  2.2129e+00],
#          [ 2.7759e-01,  9.9805e-01,  2.0508e+00,  ...,  2.5488e-01,
#            6.5283e-01,  5.2734e-01],
#          [ 1.7090e-01, -5.6201e-01, -4.4482e-01,  ...,  3.6094e+00,
#            1.4561e+00,  3.9233e-01],
#          ...,
#          [ 0.0000e+00, -1.8047e+00,  0.0000e+00,  ...,  4.0625e-01,
#            1.1982e+00,  0.0000e+00],
#          [ 1.7363e+00,  1.2539e+00, -1.8877e+00,  ...,  0.0000e+00,
#            4.0680e-02,  0.0000e+00],
#          [ 2.0840e+00,  9.1699e-01, -3.8477e-01,  ...,  1.0996e+00,
#            2.2305e+00,  1.2832e+00]],

#         [[ 3.2637e+00,  2.3301e+00,  1.7100e+00,  ..., -9.1895e-01,
#            4.6436e-01,  1.1553e+00],
#          [ 6.4893e-01,  0.0000e+00,  1.2881e+00,  ...,  1.4824e+00,
#            1.1025e+00,  1.9365e+00],
#          [ 3.7769e-01, -2.7109e+00, -1.8623e+00,  ...,  3.3281e+00,
#            1.9854e+00,  0.0000e+00],
#          ...,
#          [ 0.0000e+00, -6.3965e-01, -1.9336e+00,  ...,  3.6523e+00,
#            0.0000e+00,  2.3496e+00],
#          [ 1.7363e+00,  1.2539e+00, -1.8877e+00,  ..., -4.5898e-01,
#            4.0680e-02,  1.2482e-02],
#          [ 2.0840e+00,  9.1699e-01, -3.8477e-01,  ...,  1.0996e+00,
#            2.2305e+00,  1.2832e+00]],

#         [[ 1.5986e+00,  1.0254e-01,  1.4785e+00,  ...,  1.9688e+00,
#            0.0000e+00,  1.7139e-01],
#          [ 5.9570e-01,  0.0000e+00,  0.0000e+00,  ...,  6.9727e-01,
#            7.0654e-01,  4.7852e-01],
#          [-1.5405e-01,  0.0000e+00,  2.6196e-01,  ...,  1.4482e+00,
#            1.7373e+00,  3.2988e+00],
#          ...,
#          [ 1.0137e+00, -1.9736e+00, -2.6582e+00,  ...,  2.1855e+00,
#           -3.2558e-03,  2.3008e+00],
#          [ 1.7363e+00,  1.2539e+00,  0.0000e+00,  ...,  0.0000e+00,
#            4.0680e-02,  1.2482e-02],
#          [ 2.0840e+00,  9.1699e-01, -3.8477e-01,  ...,  1.0996e+00,
#            2.2305e+00,  1.2832e+00]],

#         ...,

#         [[ 2.3789e+00,  6.8701e-01,  4.4653e-01,  ...,  1.0127e+00,
#            1.4758e-01,  0.0000e+00],
#          [ 7.5977e-01, -7.4756e-01, -2.4927e-01,  ...,  2.3086e+00,
#            0.0000e+00,  0.0000e+00],
#          [-4.0210e-01, -2.7559e+00, -1.3584e+00,  ...,  0.0000e+00,
#            7.0654e-01,  4.7852e-01],
#          ...,
#          [ 5.4541e-01,  1.1406e+00,  1.8506e-01,  ...,  2.8496e+00,
#            1.4053e+00,  1.5557e+00],
#          [ 0.0000e+00,  1.2539e+00, -1.8877e+00,  ..., -4.5898e-01,
#            4.0680e-02,  1.2482e-02],
#          [ 0.0000e+00,  9.1699e-01, -3.8477e-01,  ...,  0.0000e+00,
#            2.2305e+00,  1.2832e+00]],

#         [[ 1.0117e+00,  6.3037e-01, -6.8311e-01,  ...,  4.0039e-01,
#            0.0000e+00,  3.4717e-01],
#          [ 2.0527e+00,  0.0000e+00,  6.5967e-01,  ...,  3.5547e-01,
#            3.4297e+00,  2.5762e+00],
#          [-7.1582e-01, -2.5957e+00, -4.1040e-01,  ...,  8.6475e-01,
#            1.7949e+00,  3.5039e+00],
#          ...,
#          [ 1.1904e+00,  8.4326e-01, -3.3032e-01,  ...,  1.8398e+00,
#            0.0000e+00,  2.2021e-01],
#          [ 1.7363e+00,  1.2539e+00, -1.8877e+00,  ..., -4.5898e-01,
#            4.0680e-02,  1.2482e-02],
#          [ 2.0840e+00,  9.1699e-01, -3.8477e-01,  ...,  1.0996e+00,
#            2.2305e+00,  1.2832e+00]],

#         [[ 9.2236e-01,  1.8203e+00,  6.0107e-01,  ...,  2.1992e+00,
#            0.0000e+00,  1.3838e+00],
#          [ 1.1221e+00,  1.0225e+00, -2.6611e-01,  ...,  1.6025e+00,
#            0.0000e+00,  3.1211e+00],
#          [-1.0654e+00,  3.8477e-01, -7.7051e-01,  ...,  1.2109e+00,
#            3.4805e+00,  5.7959e-01],
#          ...,
#          [ 1.9531e-02, -1.3506e+00, -2.9736e-01,  ...,  2.7563e-01,
#            0.0000e+00,  1.7227e+00],
#          [ 1.7363e+00,  1.2539e+00, -1.8877e+00,  ..., -4.5898e-01,
#            4.0680e-02,  1.2482e-02],
#          [ 2.0840e+00,  0.0000e+00, -3.8477e-01,  ...,  1.0996e+00,
#            2.2305e+00,  1.2832e+00]]], device='cuda:0', dtype=torch.float16,
#        grad_fn=<FusedDropoutBackward>)

# adaptive input
# tensor([[[ 2.9102e+00, -2.3633e-01,  2.6367e-02,  ...,  1.4863e+00,
#            1.2773e+00,  1.9922e+00],
#          [ 2.4976e-01,  8.9941e-01,  1.8457e+00,  ...,  2.2949e-01,
#            5.8740e-01,  4.7461e-01],
#          [ 1.5381e-01, -5.0586e-01, -3.9990e-01,  ...,  3.2480e+00,
#            1.3105e+00,  3.5303e-01],
#          ...,
#          [ 2.3711e+00, -1.6221e+00, -1.3848e+00,  ...,  3.6572e-01,
#            1.0781e+00,  1.5000e+00],
#          [ 1.5625e+00,  1.1348e+00, -1.7041e+00,  ..., -4.1309e-01,
#            3.6621e-02,  1.1230e-02],
#          [ 1.8760e+00,  8.2080e-01, -3.5181e-01,  ...,  9.8975e-01,
#            2.0078e+00,  1.1553e+00]],

#         [[ 2.9375e+00,  2.0977e+00,  1.5391e+00,  ..., -8.2715e-01,
#            4.1797e-01,  1.0400e+00],
#          [ 5.8398e-01,  7.9980e-01,  1.1592e+00,  ...,  1.3340e+00,
#            9.9268e-01,  1.7432e+00],
#          [ 3.3984e-01, -2.4395e+00, -1.6758e+00,  ...,  2.9961e+00,
#            1.7871e+00, -4.6973e-01],
#          ...,
#          [ 1.3926e+00, -5.7373e-01, -1.7285e+00,  ...,  3.2871e+00,
#           -4.5801e-01,  2.1152e+00],
#          [ 1.5625e+00,  1.1348e+00, -1.7041e+00,  ..., -4.1309e-01,
#            3.6621e-02,  1.1230e-02],
#          [ 1.8760e+00,  8.2080e-01, -3.5181e-01,  ...,  9.8975e-01,
#            2.0078e+00,  1.1553e+00]],

#         [[ 1.4385e+00,  9.2285e-02,  1.3311e+00,  ...,  1.7715e+00,
#            1.7441e+00,  1.5430e-01],
#          [ 5.3613e-01, -1.5781e+00, -3.3081e-01,  ...,  6.2744e-01,
#            6.3574e-01,  4.3066e-01],
#          [-1.3867e-01, -1.4746e+00,  2.3633e-01,  ...,  1.3037e+00,
#            1.5635e+00,  2.9688e+00],
#          ...,
#          [ 9.1260e-01, -1.7744e+00, -2.3809e+00,  ...,  1.9668e+00,
#           -2.9297e-03,  2.0703e+00],
#          [ 1.5625e+00,  1.1348e+00, -1.7041e+00,  ..., -4.1309e-01,
#            3.6621e-02,  1.1230e-02],
#          [ 1.8760e+00,  8.2080e-01, -3.5181e-01,  ...,  9.8975e-01,
#            2.0078e+00,  1.1553e+00]],

#         ...,

#         [[ 2.1406e+00,  6.1816e-01,  4.0186e-01,  ...,  9.1113e-01,
#            1.3281e-01,  1.4404e+00],
#          [ 6.8359e-01, -6.7188e-01, -2.2437e-01,  ...,  2.0781e+00,
#            6.6406e-02,  9.1260e-01],
#          [-3.6182e-01, -2.4805e+00, -1.2217e+00,  ...,  6.2744e-01,
#            6.3574e-01,  4.3066e-01],
#          ...,
#          [ 4.9097e-01,  1.0283e+00,  1.7725e-01,  ...,  2.5645e+00,
#            1.2646e+00,  1.4004e+00],
#          [ 1.5625e+00,  1.1348e+00, -1.7041e+00,  ..., -4.1309e-01,
#            3.6621e-02,  1.1230e-02],
#          [ 1.8760e+00,  8.2080e-01, -3.5181e-01,  ...,  9.8975e-01,
#            2.0078e+00,  1.1553e+00]],

#         [[ 9.1064e-01,  5.6738e-01, -6.1475e-01,  ...,  3.6035e-01,
#            1.4541e+00,  3.1250e-01],
#          [ 1.8477e+00,  1.0420e+00,  5.9375e-01,  ...,  3.1982e-01,
#            3.0859e+00,  2.3184e+00],
#          [-6.4404e-01, -2.3359e+00, -3.6890e-01,  ...,  7.7832e-01,
#            1.6152e+00,  3.1543e+00],
#          ...,
#          [ 1.0713e+00,  7.6074e-01, -2.8662e-01,  ...,  1.6562e+00,
#            1.9629e+00,  1.9824e-01],
#          [ 1.5625e+00,  1.1348e+00, -1.7041e+00,  ..., -4.1309e-01,
#            3.6621e-02,  1.1230e-02],
#          [ 1.8760e+00,  8.2080e-01, -3.5181e-01,  ...,  9.8975e-01,
#            2.0078e+00,  1.1553e+00]],

#         [[ 8.3008e-01,  1.6387e+00,  5.4102e-01,  ...,  1.9785e+00,
#            1.2119e+00,  1.2451e+00],
#          [ 1.0098e+00,  9.2139e-01, -2.3950e-01,  ...,  1.4424e+00,
#            2.5039e+00,  2.8086e+00],
#          [-9.5850e-01,  3.4668e-01, -6.9287e-01,  ...,  1.0898e+00,
#            3.1328e+00,  5.2148e-01],
#          ...,
#          [ 1.7578e-02, -1.2139e+00, -2.5684e-01,  ...,  2.4805e-01,
#            2.3320e+00,  1.5508e+00],
#          [ 1.5625e+00,  1.1348e+00, -1.7041e+00,  ..., -4.1309e-01,
#            3.6621e-02,  1.1230e-02],
#          [ 1.8760e+00,  8.2080e-01, -3.5181e-01,  ...,  9.8975e-01,
#            2.0078e+00,  1.1553e+00]]], device='cuda:0', dtype=torch.float16,
#        grad_fn=<AddBackward0>)



# base layer 1
# tensor([[[ 2.3809, -0.5127, -0.3743,  ...,  0.4453, -0.0095,  0.8447],
#          [ 2.3477,  1.7441,  0.5479,  ..., -1.2480,  0.2115,  0.1307],
#          [ 1.2275, -0.1637,  0.0779,  ...,  0.9287, -1.0059, -0.6172],
#          ...,
#          [ 1.3408,  0.3003, -0.1344,  ...,  0.0989, -0.8452, -0.9287],
#          [ 0.7559,  0.4485, -0.4004,  ..., -0.9922, -0.2532, -0.1439],
#          [ 0.4614,  1.4229, -0.0513,  ...,  1.1533, -0.4360,  0.5791]],

#         [[ 0.2898,  0.7563,  0.9551,  ..., -0.4265, -0.4463,  0.2554],
#          [-0.0852, -0.2223,  0.6543,  ...,  0.4631,  0.1096,  0.7217],
#          [ 0.2405,  0.1222, -0.7222,  ..., -0.5317, -0.0183, -0.3542],
#          ...,
#          [ 0.4795, -0.8750, -0.5425,  ...,  0.5645, -0.7754, -0.6245],
#          [ 1.5732, -0.3845,  0.4365,  ..., -0.6631,  2.1133,  1.3105],
#          [ 0.4998,  0.8721, -0.7051,  ...,  0.8237, -0.2212,  2.3438]],

#         [[ 0.2842, -0.3010, -0.6274,  ...,  2.0488,  0.3955, -0.6382],
#          [ 0.3708, -1.8984, -1.9160,  ...,  2.0254,  0.8018, -0.6963],
#          [-0.2791,  0.0634, -0.2637,  ..., -0.0049,  0.8311,  2.2227],
#          ...,
#          [-0.5239, -2.0547, -1.5049,  ..., -0.8950,  0.1927,  0.0783],
#          [-0.7676, -1.8740, -0.4973,  ...,  0.2397,  1.1270,  2.0137],
#          [-1.1514,  0.1227, -0.7144,  ...,  0.3215,  2.1836, -0.2058]],

#         ...,

#         [[-0.3752, -1.1875, -0.0455,  ..., -0.1569,  0.3013, -0.2495],
#          [-0.1270, -0.3882, -1.5020,  ...,  1.7998, -0.6143,  1.1377],
#          [ 0.4055, -1.5176, -2.0547,  ...,  0.9087, -0.5952,  1.7012],
#          ...,
#          [ 0.3408,  0.6602,  0.0561,  ...,  1.3633,  0.8569,  0.6826],
#          [ 0.4568,  0.6074, -0.1542,  ...,  0.3972, -0.1875, -0.0449],
#          [-0.4121, -0.6367, -0.4038,  ...,  0.0798, -0.6328,  0.9971]],

#         [[ 1.2129,  1.1182, -1.5254,  ..., -0.3779, -0.1752,  0.0403],
#          [ 1.0430,  1.1865, -1.6680,  ..., -0.6699,  0.0950, -0.1907],
#          [ 1.1992,  1.1650, -0.1726,  ..., -0.5234, -0.1097, -0.2220],
#          ...,
#          [-0.0411,  1.3975, -1.7656,  ..., -0.8242,  0.0058, -0.0497],
#          [ 1.3105,  1.4277, -1.4902,  ..., -0.9565, -0.3916, -0.2365],
#          [ 1.1787,  1.0547, -1.5488,  ..., -0.7725,  0.1488,  0.1240]],

#         [[ 1.3809,  0.3711, -0.0842,  ...,  0.0891,  0.9502,  0.5757],
#          [ 1.2588,  0.7485, -0.4897,  ...,  0.2561,  1.2168,  0.3518],
#          [ 1.3271,  0.2793, -0.0461,  ...,  0.0060,  0.9629,  0.8408],
#          ...,
#          [-0.1108,  0.5479, -0.3481,  ..., -0.4058,  1.3076,  0.7847],
#          [ 1.3105,  0.5088, -0.2515,  ...,  0.0614,  1.4951,  0.6650],
#          [ 1.2793, -0.3584, -0.1716,  ...,  0.2231,  1.3086,  0.5229]]],
#        device='cuda:0', dtype=torch.float16, grad_fn=<NativeLayerNormBackward>)


# adaptive
# tensor([[[ 2.3809, -0.5132, -0.3738,  ...,  0.4453, -0.0092,  0.8428],
#          [ 2.3457,  1.7432,  0.5483,  ..., -1.2461,  0.2112,  0.1306],
#          [ 1.2275, -0.1635,  0.0779,  ...,  0.9292, -1.0059, -0.6172],
#          ...,
#          [ 1.3408,  0.3005, -0.1348,  ...,  0.0989, -0.8457, -0.9292],
#          [ 0.7568,  0.4487, -0.3994,  ..., -0.9922, -0.2532, -0.1432],
#          [ 0.4614,  1.4229, -0.0512,  ...,  1.1533, -0.4360,  0.5791]],

#         [[ 0.2905,  0.7573,  0.9541,  ..., -0.4268, -0.4465,  0.2556],
#          [-0.0858, -0.2229,  0.6543,  ...,  0.4629,  0.1103,  0.7231],
#          [ 0.2401,  0.1220, -0.7217,  ..., -0.5317, -0.0181, -0.3545],
#          ...,
#          [ 0.4795, -0.8740, -0.5425,  ...,  0.5640, -0.7754, -0.6245],
#          [ 1.5732, -0.3840,  0.4365,  ..., -0.6626,  2.1133,  1.3115],
#          [ 0.4998,  0.8735, -0.7041,  ...,  0.8237, -0.2212,  2.3438]],

#         [[ 0.2852, -0.3015, -0.6270,  ...,  2.0488,  0.3955, -0.6387],
#          [ 0.3711, -1.8984, -1.9150,  ...,  2.0254,  0.8018, -0.6958],
#          [-0.2791,  0.0635, -0.2627,  ..., -0.0049,  0.8311,  2.2227],
#          ...,
#          [-0.5234, -2.0547, -1.5039,  ..., -0.8950,  0.1934,  0.0782],
#          [-0.7676, -1.8740, -0.4963,  ...,  0.2397,  1.1270,  2.0137],
#          [-1.1514,  0.1226, -0.7134,  ...,  0.3215,  2.1836, -0.2056]],

#         ...,

#         [[-0.3745, -1.1875, -0.0451,  ..., -0.1573,  0.3013, -0.2491],
#          [-0.1274, -0.3862, -1.4941,  ...,  1.7998, -0.6152,  1.1367],
#          [ 0.4065, -1.5166, -2.0430,  ...,  0.9082, -0.5957,  1.7021],
#          ...,
#          [ 0.3413,  0.6611,  0.0627,  ...,  1.3623,  0.8564,  0.6821],
#          [ 0.4568,  0.6074, -0.1450,  ...,  0.3972, -0.1891, -0.0459],
#          [-0.4114, -0.6377, -0.3953,  ...,  0.0792, -0.6328,  0.9976]],

#         [[ 1.2129,  1.1221, -1.5312,  ..., -0.3794, -0.1760,  0.0402],
#          [ 1.0430,  1.1895, -1.6738,  ..., -0.6704,  0.0942, -0.1907],
#          [ 1.2002,  1.1699, -0.1730,  ..., -0.5229, -0.1107, -0.2217],
#          ...,
#          [-0.0412,  1.4023, -1.7705,  ..., -0.8247,  0.0064, -0.0501],
#          [ 1.3115,  1.4336, -1.4961,  ..., -0.9565, -0.3923, -0.2355],
#          [ 1.1777,  1.0586, -1.5547,  ..., -0.7729,  0.1478,  0.1248]],

#         [[ 1.3789,  0.3682, -0.0893,  ...,  0.0889,  0.9487,  0.5757],
#          [ 1.2588,  0.7446, -0.4956,  ...,  0.2532,  1.2158,  0.3533],
#          [ 1.3271,  0.2769, -0.0505,  ...,  0.0059,  0.9604,  0.8413],
#          ...,
#          [-0.1107,  0.5459, -0.3533,  ..., -0.4067,  1.3086,  0.7842],
#          [ 1.3115,  0.5059, -0.2566,  ...,  0.0593,  1.4951,  0.6650],
#          [ 1.2803, -0.3582, -0.1766,  ...,  0.2224,  1.3076,  0.5225]]]



# tensor(0.16735839843750000000, device='cuda:0', dtype=torch.float16,
#        grad_fn=<SelectBackward>)
# tensor(0.16650390625000000000, device='cuda:0', dtype=torch.float16,
#        grad_fn=<SelectBackward>)

# tensor(0.69433593750000000000, device='cuda:0', dtype=torch.float16,
#        grad_fn=<SelectBackward>)
# tensor(0.69531250000000000000, device='cuda:0', dtype=torch.float16,
#        grad_fn=<SelectBackward>)