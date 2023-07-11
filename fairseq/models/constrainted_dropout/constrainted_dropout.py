import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstraitedDropout(nn.Module):
    def __init__(self, p, sample_num, feature_dim):
        super().__init__()
        self.p = p
        self.sample_num = sample_num

        mask = (torch.rand((sample_num, feature_dim)) > p).to(torch.float) / (1-p)
        self.register_buffer("dropout_mask", mask)

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and self.training:
            mask_index = torch.randint(self.sample_num, (1,))
            return x * self.dropout_mask[mask_index]
        else:
            return x
