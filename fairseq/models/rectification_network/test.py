import torch
import torch.nn as nn
import torch.nn.functional as F

# m1 = nn.Linear(5, 5, False)
# m2 = nn.Linear(5, 5, False)
# m3 = nn.Linear(5, 5, False)
# m4 = torch.zeros((5, 5), dtype=torch.float)

# m_combination = torch.cat(
#     (torch.cat((m1.weight, m4), dim=1), torch.cat((m2.weight, m3.weight), dim=1)), dim=0,
# )

# print(m_combination.size())

lprobs = torch.FloatTensor([[0.1, 0.3, 0.6]])
target = torch.LongTensor([4])
loss = F.nll_loss(lprobs, target, reduction="sum")
print(loss)

