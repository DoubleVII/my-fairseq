import torch
import numpy as np

# def forward_bias_decomposition(x, bias):
#     """
#         the function torch.linalg.lstsq on CUDA requires X is
#     """
#     token_len, class_len, bsz, _ = x.size()
#     component = torch.zeros((token_len, class_len, bsz)).to(x)
#     for i in range(bsz):
#         for j in range(token_len):
#             basis = x[j,:,i,:] # CT x C
#             zeros_mask = (basis == 0)[:,0]
#             full_rank_basis = basis[~zeros_mask,:] # CT' x C

#             solution = torch.linalg.lstsq(full_rank_basis.transpose(0,1), bias[:, None]).solution # CT' x 1
#             if not solution.isfinite().all() or solution.isnan().any() :
#                 import pdb
#                 pdb.set_trace()
#             component[j,~zeros_mask,i] = solution[:,0]
#     # res = torch.linalg.lstsq(x.permute(0,2,3,1), bias[None,None,:,None])

#     # assert torch.max(component.abs()) < 1.0
#     # error_bound = 10
#     # component[component.abs() > error_bound] = component[component.abs() > error_bound] / component.abs()[component.abs() > error_bound]
#     # component[:] = 0.0
#     component = component.unsqueeze(-1) * x # T x CT x B x C
#     residual = bias - component.sum(dim=1, keepdim=True)
#     x = x + component
#     return x, residual


def forward_bias_decomposition(x, bias):
    """
    x: T x CT x B x C
    bias: T x 1 x B x C
    the parameter 'bias' support boardcast
    """
    assert len(x.size()) == len(bias.size())
    assert x.size(-1) == bias.size(-1) and bias.size(1) == 1
    solution = torch.linalg.lstsq(
        x.permute(0, 2, 3, 1).cpu(),
        bias.permute(0, 2, 3, 1).cpu(),
        rcond=1e-2,
        driver="gelsd",
    ).solution  # T x B x CT x 1
    component = solution.to(x).squeeze(-1).transpose(1, 2)  # T X CT x B
    # if torch.max(component.abs()) > 10:
    #     import pdb
    #     pdb.set_trace()
    # assert torch.max(component.abs()) < 1.0
    error_bound = 10
    component[component.abs() > error_bound] = (
        component[component.abs() > error_bound]
        / component.abs()[component.abs() > error_bound]
    )
    # component[:] = 0.0
    component = component.unsqueeze(-1) * x
    residual = bias - component.sum(dim=1, keepdim=True)
    x = x + component
    return x, residual


def forward_iter_decomposition(x):
    residual = x[:, 0:1, :, :]  # T x 1 x B x C
    _x, residual = forward_bias_decomposition(x[:, 1:], residual)
    x[:, 1:] = _x
    x[:, 0:1, :, :] = residual
    return x


def forward_abs_decomposition(x, eps=1e-6):
    # import pdb
    # pdb.set_trace()
    residual = x[:, 0:1, :, :]  # T x 1 x B x C
    compositions = x[:, 1:]
    abs_compositions = compositions.abs()
    sum_compositions = abs_compositions.sum(dim=1, keepdim=True)
    sum_compositions[sum_compositions == 0] = eps
    weights = abs_compositions / sum_compositions
    x[:, 1:] += weights * residual
    x[:, 0] = 0.0
    return x


def forward_norm_decomposition(x, eps=1e-6):
    residual = x[:, 0:1, :, :]  # T x 1 x B x C
    compositions = x[:, 1:]
    norm_compositions = torch.norm(
        compositions, p=2, dim=-1, keepdim=True
    )  # T x CT x B x 1
    sum_compositions = norm_compositions.sum(dim=1, keepdim=True)  # T x 1 x B x 1
    sum_compositions[sum_compositions == 0] = eps

    weights = norm_compositions / sum_compositions  # T x CT x B x 1
    x[:, 1:] += weights * residual
    x[:, 0] = 0.0
    return x


def forward_softmax_decomposition(x, eps=1e-6):
    # import pdb
    # pdb.set_trace()
    residual = x[:, 0:1, :, :]  # T x 1 x B x C
    compositions = x[:, 1:]
    weights = torch.softmax(compositions, dim=1)
    x[:, 1:] += weights * residual
    x[:, 0] = 0.0
    return x


def forward_norm_softmax_decomposition(x, eps=1e-6):
    residual = x[:, 0:1, :, :]  # T x 1 x B x C
    compositions = x[:, 1:]
    norm_compositions = torch.norm(
        compositions, p=2, dim=-1, keepdim=True
    )  # T x CT x B x 1

    weights = torch.softmax(norm_compositions, dim=1)  # T x CT x B x 1
    x[:, 1:] += weights * residual
    x[:, 0] = 0.0
    return x


def forward_relu(x):
    assert len(x.size()) == 4
    x_sum = x.sum(dim=1, keepdim=True)
    beta_x = torch.relu(x[:, -1, None])
    x_sum = torch.relu(x_sum)
    x[:, -1, None] = beta_x
    x[:, 1, None] = x_sum - beta_x
    return x


def forward_qk_mul(query_compositions, key_compositions, attn_weights=None):
    if attn_weights is None:
        q = query_compositions.sum(1)
        k = key_compositions.sum(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
    # bsz * self.num_heads, 3, query_len, self.head_dim
    assert query_compositions.size(1) == 3
    beta_attn_compositions = torch.matmul(
        query_compositions[:, 2, None], key_compositions[:, 2, None].transpose(2, 3)
    )  # bsz * self.num_heads x 1 x query_len x value_len
    gamma_attn_compositions = attn_weights - beta_attn_compositions[:, 0]
    gamma_attn_compositions = gamma_attn_compositions[:, None]
    # gamma_attn_compositions0 = torch.matmul(
    #     query_compositions[:, 1, None], key_compositions[:, 2, None].transpose(2, 3)
    # )  # bsz * self.num_heads x 1 x query_len x value_len
    # gamma_attn_compositions1 = torch.matmul(
    #     query_compositions[:, 2, None], key_compositions[:, 1, None].transpose(2, 3)
    # )  # bsz * self.num_heads x 1 x query_len x value_len
    # gamma_attn_compositions2 = torch.matmul(
    #     query_compositions[:, 1, None], key_compositions[:, 1, None].transpose(2, 3)
    # )  # bsz * self.num_heads x 1 x query_len x value_len
    # gamma_attn_compositions = (
    #     gamma_attn_compositions0 + gamma_attn_compositions1 + gamma_attn_compositions2
    # )
    attn_compositions = torch.cat(
        (
            torch.zeros_like(beta_attn_compositions).to(query_compositions),
            beta_attn_compositions,
            gamma_attn_compositions,
        ),
        dim=1,
    )

    return attn_compositions


def forward_attn_v_mul(attn_compositions, value_compositions, attn=None):
    if attn is None:
        attn_probs = attn_compositions.sum(1)
        v = value_compositions.sum(1)
        attn = torch.bmm(attn_probs, v)
    # attn_compositions: bsz * self.num_heads, 3, query_len, value_len
    # value_compositions: bsz * self.num_heads, 3, value_len, self.head_dim
    assert attn_compositions.size(1) == 3
    assert value_compositions.size(1) == 3
    beta_attn_compositions = torch.matmul(
        attn_compositions[:, 2, None], value_compositions[:, 2, None]
    )  # bsz * self.num_heads x 1 x query_len x value_len
    gamma_attn_compositions = attn - beta_attn_compositions[:, 0]
    gamma_attn_compositions = gamma_attn_compositions[:, None]
    # gamma_attn_compositions0 = torch.matmul(
    #     attn_compositions[:, 1, None], value_compositions[:, 2, None]
    # )  # bsz * self.num_heads x 1 x query_len x value_len
    # gamma_attn_compositions1 = torch.matmul(
    #     attn_compositions[:, 2, None], value_compositions[:, 1, None]
    # )  # bsz * self.num_heads x 1 x query_len x value_len
    # gamma_attn_compositions2 = torch.matmul(
    #     attn_compositions[:, 1, None], value_compositions[:, 1, None]
    # )  # bsz * self.num_heads x 1 x query_len x value_len
    # gamma_attn_compositions = (
    #     gamma_attn_compositions0 + gamma_attn_compositions1 + gamma_attn_compositions2
    # )
    attn_compositions = torch.cat(
        (
            torch.zeros_like(beta_attn_compositions).to(attn_compositions),
            beta_attn_compositions,
            gamma_attn_compositions,
        ),
        dim=1,
    )

    return attn_compositions


def forward_linear(
    x, weight, bias, key_padding_mask=None, decompose_bias=False, iter_decompose=True
):
    assert (
        len(x.size()) == 4
    )  # T x CT+1 x B X C , CT + 1 means add one residual component
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.to(torch.bool)
        bsz, encoder_len = key_padding_mask.size()
        key_padding_mask = torch.cat(
            [torch.zeros((bsz, 1)).to(key_padding_mask), key_padding_mask], dim=1
        )
        x = torch.masked_fill(
            x, key_padding_mask.transpose(0, 1).unsqueeze(0).unsqueeze(-1), 0.0
        )
    x = x.matmul(weight.t())

    if decompose_bias:
        if iter_decompose:
            x[:, 0, :, :] += bias
            x = forward_iter_decomposition(x)
        else:
            _x, residual = forward_bias_decomposition(
                x[:, 1:], bias[None, None, None, :]
            )
            x[:, 1:] = _x
            x[:, 0:1, :, :] += residual
    else:
        x[:, 0, :, :] += bias
        x = forward_abs_decomposition(x)
    return x


def forward_layer_norm(
    x, weight, bias, key_padding_mask=None, decompose_bias=False, iter_decompose=True
):
    assert len(x.size()) == 4
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.to(torch.bool)
        bsz, encoder_len = key_padding_mask.size()
        key_padding_mask = torch.cat(
            [torch.zeros((bsz, 1)).to(key_padding_mask), key_padding_mask], dim=1
        )
        x = torch.masked_fill(
            x, key_padding_mask.transpose(0, 1).unsqueeze(0).unsqueeze(-1), 0.0
        )

    # x = x.to(torch.float64)
    # weight = weight.to(torch.float64)
    x_mean = x.mean(dim=-1, keepdim=True)
    x_std = torch.sqrt(
        torch.var(x.sum(dim=1), dim=-1, unbiased=False, keepdim=True) + 1e-5
    )
    x = (x - x_mean) * weight / x_std.unsqueeze(dim=1)
    # x = x.to(torch.float32)

    if decompose_bias:
        if iter_decompose:
            x[:, 0, :, :] += bias
            x = forward_iter_decomposition(x)
        else:
            _x, residual = forward_bias_decomposition(
                x[:, 1:], bias[None, None, None, :]
            )
            x[:, 1:] = _x
            x[:, 0:1, :, :] += residual
    else:
        x[:, 0, :, :] += bias
        x = forward_abs_decomposition(x)
    return x


def forward_layer_norm_decoder(
    x, weight, bias, key_padding_mask=None, decompose_bias=False, iter_decompose=True
):
    assert len(x.size()) == 4
    # assert x.size(0) == 1

    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.to(torch.bool)
        bsz, encoder_len = key_padding_mask.size()
        all_len = x.size(1)
        cat_mask = torch.zeros((bsz, all_len - encoder_len)).to(key_padding_mask)
        key_padding_mask = torch.cat(
            [cat_mask[:, 0:1], key_padding_mask, cat_mask[:, 1:]], dim=1
        )
        x = x.masked_fill(
            key_padding_mask.transpose(0, 1).unsqueeze(0).unsqueeze(-1), 0.0
        )

    x_mean = x.mean(dim=-1, keepdim=True)
    x_std = torch.sqrt(
        torch.var(x.sum(dim=1), dim=-1, unbiased=False, keepdim=True) + 1e-5
    )
    x = (x - x_mean) * weight / x_std.unsqueeze(dim=1)

    if decompose_bias:
        if iter_decompose:
            x[:, 0, :, :] += bias
            x = forward_iter_decomposition(x)
        else:
            _x, residual = forward_bias_decomposition(
                x[:, 1:], bias[None, None, None, :]
            )
            x[:, 1:] = _x
            x[:, 0:1, :, :] += residual
    else:
        x[:, 0, :, :] += bias
        x = forward_abs_decomposition(x)
    return x


def check_error(x, ref_x, key_padding_mask=None, error=1e3):
    return
    assert len(x.size()) == 4
    bsz = x.size(2)
    x_sub = x.sum(dim=1) - ref_x
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.to(torch.bool)
        sample_error = (
            torch.sum(
                x_sub.masked_fill(
                    key_padding_mask.transpose(0, 1).unsqueeze(-1), 0.0
                ).abs()
            )
            / bsz
        )
        assert sample_error < error, f"error: {sample_error}"

    else:
        sample_error = torch.sum(x_sub.abs()) / bsz
        assert sample_error < error, f"error: {sample_error}"


def check_error_decoder(x, ref_x, key_padding_mask=None, error=1e3):
    return
    assert len(x.size()) == 4
    # assert x.size(0) == 1
    bsz = x.size(2)

    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.to(torch.bool)
        bsz, encoder_len = key_padding_mask.size()
        all_len = x.size(1)
        cat_mask = torch.zeros((bsz, all_len - encoder_len)).to(key_padding_mask)
        key_padding_mask = torch.cat(
            [cat_mask[:, 0:1], key_padding_mask, cat_mask[:, 1:]], dim=1
        )
        x = x.masked_fill(
            key_padding_mask.transpose(0, 1).unsqueeze(0).unsqueeze(-1), 0.0
        )

    x_sub = x.sum(dim=1) - ref_x
    sample_error = torch.sum(x_sub.abs()) / bsz
    assert sample_error < error, f"error: {sample_error}"

