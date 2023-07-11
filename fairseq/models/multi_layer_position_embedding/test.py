import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

position_factor = nn.Parameter(torch.FloatTensor([1.0]))
print(position_factor)