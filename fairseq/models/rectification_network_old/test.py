# import torch
# import torch.nn as nn

# m1 = nn.Linear(5, 5, False)
# m2 = nn.Linear(5, 5, False)
# m3 = nn.Linear(5, 5, False)
# m4 = torch.zeros((5, 5), dtype=torch.float)

# m_combination = torch.cat(
#     (torch.cat((m1.weight, m4), dim=1), torch.cat((m2.weight, m3.weight), dim=1)), dim=0,
# )

# print(m_combination.size())

x = input()
x = x.lower()
punc = [".", "!", ",", "?"]
for p in punc:
    x = x.replace(p, " " + p)
x = x.split()
print(*x)
for i in range(len(x)):
    word = x[i]
    if x.count(word)<=1:
        x[i] = "[unk]"
print(*x)

