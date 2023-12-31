import pydec
from pydec import Composition
import torch

torch.manual_seed(114514)

# # t = torch.randn((3,))
# # # t = t + 1
# # print(t)
# # torch.tensor(t).unsqueeze_(1)
# # t.clone().detach().unsqueeze_(1)
# # # t3.squeeze_(1)
# pydec.set_bias_decomposition_func("abs_decomposition")
# c = Composition((2, 3), 3, dtype=torch.float)
# # c = Composition(torch.empty((3, 2)), torch.empty((4,)))
# # c = Composition(c)
# # print(c)
# t0 = torch.randn((2, 3), requires_grad=True)
# t1 = torch.randn((2, 3), requires_grad=True)
# t2 = torch.randn((2, 3), requires_grad=True)
# index = torch.LongTensor([[2, 1], [2, 0]])

# c[0] = t0
# c[1] = t1
# c[2] = t2
# print(c.size(1))
# # print(c.all(1, 2, keepdim=True))
# # torch.nn.Conv2d
# # print(t0)
# # print(t1)
# # print(t0.scatter(dim=1, index=index, src=t1, reduce="add")+t1+t2)
# # print(c.scatter(dim=1, index=index, src=t1, reduce="add").c_sum())
# # exit()
# # c._composition_tensor.requires_grad_(True)
# # c._composition_tensor += 3
# c = 2 * c
# c = c + 3
# # print(c[0])
# # c += 3
# # print(c)
# # c += 3
# c = c.permute(dims=(-1, 0))
# t0.permute(dims=(-1, 0))
# exit()
# # c = c.contiguous().view((6,)).to(c)

# # c = c.view_as(t0)
# # print(c.c_sum().sum())
# # print(c)
# # c = c.sum((-1,0), keepdim=True)
# # print(c)
# print(c.size())
# print(c)

# loss = c.sum().c_sum()
# loss.backward()
# print(t0.grad)
# # print(c.size())
# # c.unsqueeze_(1)
# # print(c.size())


def init_composition(requires_grad=False):
    c = Composition((2, 3), 3, dtype=torch.float)
    t0 = torch.randn((2, 3), requires_grad=requires_grad)
    t1 = torch.randn((2, 3), requires_grad=requires_grad)
    t2 = torch.randn((2, 3), requires_grad=requires_grad)

    c[0] = t0
    c[1] = t1
    c[2] = t2
    return c


class TestView:
    c = init_composition()

    def test_view1(self):
        assert self.c.view(torch.float16)._composition_tensor.dtype == torch.float16

        assert (
            self.c.view(dtype=torch.float16)._composition_tensor.dtype == torch.float16
        )

    def test_view2(self):
        assert self.c.view((3, 2)).size() == (3, 2)
        assert self.c.view(size=(3, 2)).size() == (3, 2)
        assert self.c.view(3, 2).size() == (3, 2)
