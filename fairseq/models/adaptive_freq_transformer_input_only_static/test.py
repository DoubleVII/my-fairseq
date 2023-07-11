# from .adaptive_freq_sinusoidal_positional_embedding import (
#     AdaptiveFreqSinusoidalPositionalEmbedding,
# )
from fairseq.modules import SinusoidalPositionalEmbedding
import torch

# from adaptive_freq_transformer_input_only_static.adaptive_freq_sinusoidal_positional_embedding import (
#     AdaptiveFreqSinusoidalPositionalEmbedding,
# )
from fairseq.models.adaptive_freq_transformer_input_only_static import (
    AdaptiveFreqSinusoidalPositionalEmbedding,
)

from fairseq.models.adaptive_freq_transformer_input_only_static import MultiheadAttention
import torch.nn.functional as F


def test0():
    embedding_dim = 8
    padding_idx = 1

    ape = AdaptiveFreqSinusoidalPositionalEmbedding(
        embedding_dim, padding_idx, False, 10000.0
    )
    spe = SinusoidalPositionalEmbedding(embedding_dim, padding_idx)

    p_index = torch.LongTensor([1, 2, 3, 4]).unsqueeze(0)
    w_index = torch.LongTensor([1, 34, 3, 4]).unsqueeze(0)
    torch.set_printoptions(20)
    print(p_index.size())
    print(ape(p_index))
    print(spe(w_index))


def test1():
    t = torch.ones((2, 3, 4), dtype=torch.float)
    m = torch.LongTensor([[0, 0, 1], [1, 1, 0]]).eq(0)
    t = t.masked_fill(m.unsqueeze(-1), 0.0)
    print(t)


def test2():
    half_dim = 128
    freq_weights = torch.linspace(0, 1, half_dim, dtype=torch.float, requires_grad=False)
    freq_weights2 = torch.arange(half_dim, dtype=torch.float) / (half_dim - 1)
    torch.set_printoptions(20)
    print(freq_weights)
    print(freq_weights2)


def test3():
    torch.manual_seed(1)
    embed_dim = 256
    head = 8
    dropout = 0.0
    attn = MultiheadAttention(embed_dim, head, dropout=dropout, self_attention=True,)
    input_tensor = torch.rand((3, 1, embed_dim))  # bsz x len x dim
    encoder_padding_mask = torch.zeros((1, 3)).bool()
    out_attn1, _ = attn(
        query=input_tensor,
        key=input_tensor,
        value=input_tensor,
        key_padding_mask=encoder_padding_mask,
        attn_mask=None,
    )
    out_attn2, _ = F.multi_head_attention_forward(
        input_tensor,
        input_tensor,
        input_tensor,
        attn.embed_dim,
        attn.num_heads,
        torch.empty([0]),
        torch.cat((attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias)),
        attn.bias_k,
        attn.bias_v,
        attn.add_zero_attn,
        attn.dropout_module.p,
        attn.out_proj.weight,
        attn.out_proj.bias,
        attn.training or attn.dropout_module.apply_during_inference,
        key_padding_mask=encoder_padding_mask,
        need_weights=False,
        attn_mask=None,
        use_separate_proj_weight=True,
        q_proj_weight=attn.q_proj.weight,
        k_proj_weight=attn.k_proj.weight,
        v_proj_weight=attn.v_proj.weight,
    )
    print(out_attn1[0, 0, 0])
    print(out_attn2[0, 0, 0])
    print(out_attn1[0, 0, 1])
    print(out_attn2[0, 0, 1])


if __name__ == "__main__":
    test6()
