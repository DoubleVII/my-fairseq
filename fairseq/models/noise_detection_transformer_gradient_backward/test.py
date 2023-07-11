import torch
x = torch.randn((10,4,32), dtype=torch.float) / 2
y = torch.randn((1,1,32), dtype=torch.float) / 2
print(y)
print(torch.cat((y.expand(-1,4,-1),x), dim=0).size())