import torch


def forward_position_index(src_tokens, bias):
    padding_idx = 0
    mask = src_tokens.ne(padding_idx).int()
    forward_postion_index = (
        (torch.cumsum(mask, dim=1).type_as(mask) + bias) * mask
    ).long() + padding_idx  # bsz x seqlen
    mask = torch.flip(mask, [-1])
    backward_postion_index = (
        (torch.cumsum(mask, dim=1).type_as(mask) + bias) * mask
    ).long() + padding_idx  # bsz x seqlen
    backward_postion_index = torch.flip(backward_postion_index, [-1])
    postion_index = torch.stack((forward_postion_index, backward_postion_index), dim=2)
    return postion_index


# tgt_tokens = torch.LongTensor([[0, 0, 3, 4], [0, 3, 1, 4]])
# tgt_lengths = torch.LongTensor([2, 3])

# res = forward_position_index(tgt_tokens, 0)
# print(res)
# print(res.size())


import torch.nn as nn


class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 2, False)


class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Model1()
        self.l2 = self.model1.l1


m2 = Model2()
print(m2.model1.l1.weight)
print(m2.l2.weight)
# m2.l2.weight[0, 0] = 233.33
nn.init.normal_(m2.l2.weight)
print(m2.model1.l1.weight)
print(m2.l2.weight)

