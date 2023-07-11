from numpy.core.fromnumeric import shape
import torch
import numpy as np
import itertools
import torch.nn.functional as F

t = torch.FloatTensor([-1e10, -1e10, 1])
print(t)
print(F.softmax(t, dim=0))


def reverse_hyp(hyp):
    hyp_words = hyp.split()
    if "<middle>" in hyp_words:
        middle_pos = hyp_words.index("<middle>")
        hyp_words = hyp_words[middle_pos + 1 :] + hyp_words[:middle_pos]
        hyp = " ".join(hyp_words)
        return hyp
    else:
        return hyp


# print(reverse_hyp("a b c d <middle> e f g"))
