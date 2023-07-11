from typing import ContextManager, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor


_BIAS_DECOMPOSITION_FUNC_REGISTRY = {}


def register_bias_decomposition_func(name):
    """
    New bias_decomposition_func can be added with the :func:`register_bias_decomposition_func`
    function decorator.

    For example::

        @register_model('forward_norm_decomposition')
        def forward_abs_decomposition(c: Composition, bias: Union[Number, Tensor] = None, *, eps=1e-6):
            (...)

    Args:
        name (str): the name of the funcion
    """

    def register_func(func):
        if name in _BIAS_DECOMPOSITION_FUNC_REGISTRY:
            raise ValueError("Cannot register duplicate function ({})".format(name))
        if name == "none":
            raise ValueError(
                'Cannot register function ({}), the name "none" is reserved.'.format(
                    name
                )
            )

        _BIAS_DECOMPOSITION_FUNC_REGISTRY[name] = func

        return func

    return register_func


class CompositionModual:
    bypass_check = False
    bias_decomposition_name = "norm_decomposition"  # defalut setting

    BIAS_DECOMPOSITION_FUNC_REGISTRY = _BIAS_DECOMPOSITION_FUNC_REGISTRY

    @classmethod
    def no_error_check(cls):
        return no_error_check(cls)

    @classmethod
    def error_check(cls):
        return error_check(cls)

    @classmethod
    def set_bias_decomposition_func(cls, name: str) -> None:
        if name not in _BIAS_DECOMPOSITION_FUNC_REGISTRY:
            raise ValueError("Func ({}) is not registered".format(name))
        cls.bias_decomposition_name = name

    @classmethod
    def using_bias_decomposition_func(cls, name: str):
        if name not in _BIAS_DECOMPOSITION_FUNC_REGISTRY:
            raise ValueError("Func ({}) is not registered".format(name))
        return using_bias_decomposition_func(cls, name)

    @classmethod
    def no_bias_decomposition(cls):
        return no_bias_decomposition(cls)

    @classmethod
    def get_bias_decomposition_func(cls):
        if cls.bias_decomposition_name == "none":
            return None
        else:
            return _BIAS_DECOMPOSITION_FUNC_REGISTRY[cls.bias_decomposition_name]


class no_error_check(ContextManager):
    def __init__(self, cls: CompositionModual) -> None:
        self.rollback_value = None

    def __enter__(self):
        self.rollback_value = CompositionModual.bypass_check
        CompositionModual.bypass_check = True

    def __exit__(
        self,
        __exc_type,
        __exc_value,
        __traceback,
    ):
        CompositionModual.bypass_check = self.rollback_value


class error_check(ContextManager):
    def __init__(self, cls: CompositionModual) -> None:
        self.rollback_value = None

    def __enter__(self):
        self.rollback_value = CompositionModual.bypass_check
        CompositionModual.bypass_check = False

    def __exit__(
        self,
        __exc_type,
        __exc_value,
        __traceback,
    ):
        CompositionModual.bypass_check = self.rollback_value


class using_bias_decomposition_func(ContextManager):
    def __init__(self, cls: CompositionModual, name: str) -> None:
        self.rollback_value = None
        self.using_name = name

    def __enter__(self):
        self.rollback_value = CompositionModual.bias_decomposition_name
        CompositionModual.bias_decomposition_name = self.using_name

    def __exit__(
        self,
        __exc_type,
        __exc_value,
        __traceback,
    ):
        CompositionModual.bias_decomposition_name = self.rollback_value


class no_bias_decomposition(ContextManager):
    def __init__(self, cls: CompositionModual) -> None:
        self.rollback_value = None

    def __enter__(self):
        self.rollback_value = CompositionModual.bias_decomposition_name
        CompositionModual.bias_decomposition_name = "none"

    def __exit__(
        self,
        __exc_type,
        __exc_value,
        __traceback,
    ):
        CompositionModual.bias_decomposition_name = self.rollback_value


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


# deprecated
def forward_bias_decomposition(x, bias):
    """
    deprecated
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


# deprecated
def forward_iter_decomposition(x):
    """
    deprecated
    """
    residual = x[:, 0:1, :, :]  # T x 1 x B x C
    _x, residual = forward_bias_decomposition(x[:, 1:], residual)
    x[:, 1:] = _x
    x[:, 0:1, :, :] = residual
    return x


@register_bias_decomposition_func("norm_decomposition")
def forward_norm_decomposition(x, eps=1e-6):
    residual = x[:, 0:1, :, :]  # T x 1 x B x C
    compositions = x[:, 1:]
    norm_compositions = torch.norm(
        compositions, p=float("inf"), dim=-1, keepdim=True
    )  # T x CT x B x 1
    sum_compositions = norm_compositions.sum(dim=1, keepdim=True)  # T x 1 x B x 1
    sum_compositions[sum_compositions == 0] = eps

    weights = norm_compositions / sum_compositions  # T x CT x B x 1
    x[:, 1:] += weights * residual
    x[:, 0] = 0.0
    return x


# sparse_rec = []


# def forward_sparse_abs_decomposition(x: Tensor, eps=1e-6):
#     # import pdb
#     # pdb.set_trace()
#     residual = x[:, 0:1, :, :]  # T x 1 x B x C
#     compositions = x[:, 1:]
#     abs_compositions = compositions.abs()
#     sum_compositions = abs_compositions.sum(dim=1, keepdim=True)
#     sum_compositions[sum_compositions == 0] = eps
#     topk_abs = abs_compositions.topk(k=abs_compositions.size(1) // 2, dim=1)[0]
#     sparse_ratio = topk_abs.sum(dim=1, keepdim=True) / sum_compositions
#     sparse_rec.append(sparse_ratio.view(-1).mean())
#     if len(sparse_rec) % 1000 == 0:
#         print("avg sparse ratio: ", torch.stack(sparse_rec).mean().item())
#     if len(sparse_rec) == 40000:
#         exit()
#     weights = abs_compositions / sum_compositions
#     x[:, 1:] += weights * residual
#     x[:, 0] = 0.0
#     return x


# def forward_sparse_norm_decomposition(x: Tensor, eps=1e-6):
#     residual = x[:, 0:1, :, :]  # T x 1 x B x C
#     compositions = x[:, 1:]
#     norm_compositions = torch.norm(
#         compositions, p=float("inf"), dim=-1, keepdim=True
#     )  # T x CT x B x 1
#     sum_compositions = norm_compositions.sum(dim=1, keepdim=True)  # T x 1 x B x 1
#     sum_compositions[sum_compositions == 0] = eps
#     topk_abs = norm_compositions.topk(k=norm_compositions.size(1) // 2, dim=1)[0]
#     sparse_ratio = topk_abs.sum(dim=1, keepdim=True) / sum_compositions
#     sparse_rec.append(sparse_ratio.view(-1).mean())
#     if len(sparse_rec) % 1000 == 0:
#         print("avg sparse ratio: ", torch.stack(sparse_rec).mean().item())
#     if len(sparse_rec) == 40000:
#         exit()
#     weights = norm_compositions / sum_compositions  # T x CT x B x 1
#     x[:, 1:] += weights * residual
#     x[:, 0] = 0.0
#     return x


# @register_bias_decomposition_func("softmax_decomposition")
# def forward_abs_decomposition(x, eps=1e-6):
#     residual = x[:, 0:1, :, :]  # T x 1 x B x C
#     compositions = x[:, 1:]
#     exp_compositions = torch.exp(compositions)
#     sum_compositions = exp_compositions.sum(dim=1, keepdim=True)
#     weights = exp_compositions / sum_compositions
#     x[:, 1:] += weights * residual
#     x[:, 0] = 0.0
#     return x


# def forward_hybrid_norm_decomposition(x: Tensor, power_factor=1, eps=1e-6):
#     residual = x[:, 0:1, :, :]  # T x 1 x B x C
#     compositions = x[:, 1:]  # T x CT x B x C
#     norm_index = compositions.abs().max(dim=-1, keepdim=True)[1]
#     signed_norm = torch.gather(compositions, -1, norm_index)  # T x CT x B x 1

#     sum_signed_norm = signed_norm.sum(dim=1, keepdim=True)  # T x 1 x B x 1
#     abs_signed_norm = signed_norm.abs()
#     abs_sum_signed_norm = abs_signed_norm.sum(dim=1, keepdim=True)  # T x 1 x B x 1
#     ratio = sum_signed_norm.abs() / abs_sum_signed_norm  # T x 1 x B x 1

#     sum_signed_norm[sum_signed_norm == 0] = eps
#     abs_sum_signed_norm[abs_sum_signed_norm == 0] = eps

#     # ratio = ratio ** power_factor
#     ratio = 0

#     weights = ratio * signed_norm / sum_signed_norm  # T x CT x B x 1
#     abs_weights = (1 - ratio) * abs_signed_norm / abs_sum_signed_norm  # T x CT x B x 1
#     x[:, 1:] += weights * residual + abs_weights * residual
#     x[:, 0] = 0.0
#     return x


def forward_relu(x: torch.Tensor):
    assert len(x.size()) == 4
    x_sum = x.sum(dim=1, keepdim=True)
    zero_mask = x_sum < 0
    x = torch.masked_fill(x, zero_mask, 0.0)
    # x[zero_mask] = 0.0
    return x


# def forward_relu(x: torch.Tensor):
#     assert len(x.size()) == 4
#     x_sum = x.sum(dim=1, keepdim=True)
#     zero_mask = x_sum < 0
#     x = torch.masked_fill(x, zero_mask, 0.0)
#     # x[zero_mask] = 0.0
#     with CompositionModual.using_bias_decomposition_func("abs_decomposition"):
#         bias_decomposition_func = CompositionModual.get_bias_decomposition_func()
#         if bias_decomposition_func is not None:
#             x = bias_decomposition_func(x)
#     return x


def forward_gelu(x: torch.Tensor, weight1, bias1, weight2, bias2):
    assert len(x.size()) == 4
    assert weight1.size(1) == 1
    assert weight2.size(0) == 1
    s_size, c_size, bsz, h_size = x.size()
    x = forward_linear(x.view(s_size, c_size, -1).unsqueeze(-1), weight1, bias1)
    x = forward_relu(x)
    x = forward_linear(x, weight2, bias2)
    x = x.squeeze(-1).view(s_size, c_size, bsz, h_size)
    return x


def forward_linear(
    x: torch.Tensor,
    weight,
    bias,
    key_padding_mask=None,
    iter_decompose=True,
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

    bias_decomposition_func = CompositionModual.get_bias_decomposition_func()
    if bias_decomposition_func is not None:
        x[:, 0, :, :] += bias
        x = bias_decomposition_func(x)
        # if iter_decompose:
        #     x[:, 0, :, :] += bias
        #     x = forward_iter_decomposition(x)
        # else:
        #     _x, residual = forward_bias_decomposition(x[:, 1:], bias[None, None, None, :])
        #     x[:, 1:] = _x
        #     x[:, 0:1, :, :] += residual
    else:
        x[:, 0, :, :] += bias

    return x


def forward_layer_norm(x, weight, bias, key_padding_mask=None, std: Tensor = None):
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

    x_mean = x.mean(dim=-1, keepdim=True)
    if std is None:
        x_std = torch.sqrt(
            torch.var(x.sum(dim=1), dim=-1, unbiased=False, keepdim=True) + 1e-5
        )
    else:
        x_std = std
        assert x_std.dim() == 3 and x_std.size(-1) == 1
    x = (x - x_mean) * weight / x_std.unsqueeze(dim=1)

    bias_decomposition_func = CompositionModual.get_bias_decomposition_func()
    if bias_decomposition_func is not None:
        x[:, 0, :, :] += bias
        x = bias_decomposition_func(x)
        # if iter_decompose:
        #     x[:, 0, :, :] += bias
        #     x = forward_iter_decomposition(x)
        # else:
        #     _x, residual = forward_bias_decomposition(x[:, 1:], bias[None, None, None, :])
        #     x[:, 1:] = _x
        #     x[:, 0:1, :, :] += residual
    else:
        x[:, 0, :, :] += bias
    return x


def forward_layer_norm_decoder(
    x, weight, bias, key_padding_mask=None, std: Tensor = None
):
    assert len(x.size()) == 4
    assert x.size(0) == 1

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
    if std is None:
        x_std = torch.sqrt(
            torch.var(x.sum(dim=1), dim=-1, unbiased=False, keepdim=True) + 1e-5
        )
    else:
        x_std = std
        assert x_std.dim() == 3 and x_std.size(-1) == 1
    x = (x - x_mean) * weight / x_std.unsqueeze(dim=1)

    bias_decomposition_func = CompositionModual.get_bias_decomposition_func()
    if bias_decomposition_func is not None:
        x[:, 0, :, :] += bias
        x = bias_decomposition_func(x)
        # if iter_decompose:
        #     x[:, 0, :, :] += bias
        #     x = forward_iter_decomposition(x)
        # else:
        #     _x, residual = forward_bias_decomposition(x[:, 1:], bias[None, None, None, :])
        #     x[:, 1:] = _x
        #     x[:, 0:1, :, :] += residual
    else:
        x[:, 0, :, :] += bias

    return x


def check_error(x, ref_x, key_padding_mask=None, error=1e-2):
    if CompositionModual.bypass_check:
        return
    assert len(x.size()) == 4
    bsz = x.size(2)
    x_sub = x.sum(dim=1) - ref_x
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.to(torch.bool)
        x_error = (
            torch.sum(
                x_sub.masked_fill(
                    key_padding_mask.transpose(0, 1).unsqueeze(-1), 0.0
                ).abs()
            )
            / bsz
        )
        assert x_error < error, f"error overflow: {x_error}"
    else:
        x_error = torch.sum(x_sub.abs()) / bsz
        assert x_error < error, f"error overflow: {x_error}"


def check_error_decoder(
    x: Tensor,
    ref_x: Tensor,
    key_padding_mask: Tensor = None,
    sum_dims: Tuple[int] = (1,),
    padding_dim: int = 1,
    batch_dim: int = 2,
    error=1e-2,
):
    if CompositionModual.bypass_check:
        return
    assert len(x.size()) == len(ref_x.size()) + len(sum_dims)
    assert x.size(0) == 1
    bsz = x.size(2)

    if key_padding_mask is not None and padding_dim is not None:
        key_padding_mask = key_padding_mask.to(torch.bool)
        bsz, encoder_len = key_padding_mask.size()
        all_len = x.size(1)
        cat_mask = torch.zeros((bsz, all_len - encoder_len)).to(key_padding_mask)
        key_padding_mask = torch.cat(
            [cat_mask[:, 0:1], key_padding_mask, cat_mask[:, 1:]], dim=1
        )
        first_dim = None
        second_dim = None
        if padding_dim < batch_dim:
            key_padding_mask = key_padding_mask.transpose(0, 1)
            first_dim = padding_dim
            second_dim = batch_dim
        else:
            first_dim = batch_dim
            second_dim = padding_dim
        first_insert_dim = first_dim
        second_insert_dim = second_dim - first_dim - 1
        for _ in range(first_insert_dim):
            key_padding_mask = key_padding_mask.unsqueeze(0)
        for _ in range(second_insert_dim):
            key_padding_mask = key_padding_mask.unsqueeze(second_dim)

        x = x.masked_fill(
            key_padding_mask.transpose(0, 1).unsqueeze(0).unsqueeze(-1), 0.0
        )

    x_sub = x.sum(dim=sum_dims) - ref_x
    x_error = torch.sum(x_sub.abs()) / bsz
    assert x_error < error, f"error overflow: {x_error}"


def init_compositions_align_dim(
    x: Tensor,
    dest_dim: int,
    align_dim: int,
    add_bias_composition: bool = True,
) -> Tensor:
    """
    initialize composition tensor
    x: source tensor
    dest_dim: dim to insert the compositions
    align_dim: do init_compositions with source_dim = align_dim, composition_num = size(align_dim)
    add_bias_composition: add a bias composition
    """
    return init_compositions(
        x,
        source_dim=align_dim,
        dest_dim=dest_dim,
        composition_num=x.size(align_dim),
        add_bias_composition=add_bias_composition,
    )


def init_compositions(
    x: Tensor,
    source_dim: int,
    dest_dim: int,
    composition_num: int,
    add_bias_composition: bool = True,
    left_padding: bool = True,
) -> Tensor:
    """
    initialize composition tensor
    x: source tensor
    source_dim: dim of the source compositions
    dest_dim: dim to insert the compositions
    add_bias_composition: add a bias composition
    composition_num: composition number, must be equal or large than the size of source dim
    left_padding: padding the source composition by left zeros tensor if composition_num
    is large than the size of source dim. if False, padding by right
    """

    source_size = x.size(source_dim)
    assert (
        composition_num >= source_size
    ), f"composition_num [{composition_num}] must be equal or large than the size of source dim [{source_size}]"

    dest_size = list(x.size())
    dest_size.insert(dest_dim, source_size)
    diag_compositions = torch.zeros(dest_size).to(x)

    permute_dims = list(range(x.dim()))
    permute_dims.pop(source_dim)
    permute_dims.append(source_dim)
    permute_x = x.permute(*permute_dims)
    diag_compositions = torch.diagonal_scatter(
        diag_compositions, permute_x, 0, source_dim, dest_dim
    )

    padding_num = composition_num - diag_compositions.size(dest_dim)
    if padding_num > 0:
        padding_size = list(x.size())
        padding_size.insert(dest_dim, padding_num)
        padding_compositions = torch.zeros(padding_size).to(x)
        if left_padding:
            compositions = torch.cat(
                (padding_compositions, diag_compositions), dim=dest_dim
            )
        else:
            compositions = torch.cat(
                (diag_compositions, padding_compositions), dim=dest_dim
            )
    else:
        compositions = diag_compositions
    if add_bias_composition:
        bias_size = list(x.size())
        bias_size.insert(dest_dim, 1)
        bias_compositions = torch.zeros(bias_size).to(x)
        compositions = torch.cat((bias_compositions, compositions), dim=dest_dim)

    return compositions


# bias_decomposition.forward_abs_decomposition(None)


# def register_get_bias_decomposition_func(cls):
#     cls.get_bias_decomposition_func = CompositionModual.get_bias_decomposition_func
#     return cls

# CompositionModual.set_bias_decomposition_func("abs_decomposition")

# Composition.get_bias_decomposition_func = CompositionModual.get_bias_decomposition_func

# c = Composition((3, 4), 3, dtype=torch.float)


# print(torch.tensor([0,0,0,0], dtype=torch.float) in c[0])
# print(c[1])
# print(c[1])
# c._composition_tensor[:] = 1
# print(c.residual())
# c.residual()[:] = 1
# print(c)
# t1 = torch.randn(
#     (
#         3,
#         3,
#         4,
#     )
# )
# t2 = torch.randn(
#     (
#         3,
#         4,
#         2,
#     )
# )
# print((t1 @ t2).size())
# c = 3 + c
# t1 += "asd"
# t1.__matmul__
# CompositionModual.get_bias_decomposition_func()(c)
# print(c)
# print(c.__repr__())
# c.__str__

from typing import Optional


# use for experiment of testing layernorm std influence
def layer_norm_1d(
    input: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    r"""Applies Layer Normalization for last dimension."""
    input_mean = input.mean(dim=-1, keepdim=True)
    input_std = torch.sqrt(torch.var(input, dim=-1, unbiased=False, keepdim=True) + eps)
    # out = (input - input_mean) * weight / input_std.detach()
    out = ((input - input_mean) * weight).detach() / input_std

    if bias is not None:
        out += bias
        return out
    else:
        return out
