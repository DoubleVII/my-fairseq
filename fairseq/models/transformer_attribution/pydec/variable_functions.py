import torch
from torch import Tensor

from pydec.composition import Composition

def void() -> Composition:
    return Composition(tuple(), 0)


def from_tensor(
    composition_tensor: Tensor, residual_tensor: Tensor = None
) -> Composition:
    out = void()
    # out._composition_tensor = composition_tensor.clone().detach()
    out._composition_tensor = torch.tensor(composition_tensor).to(composition_tensor)
    if residual_tensor is not None:
        assert (
            composition_tensor.size()[1:] == residual_tensor.size()
        ), f"sizes of composition and residual not match, [{composition_tensor.size()[1:]}] for composition and [{residual_tensor.size()}] for residual."
        # out._residual_tensor = residual_tensor.clone().detach()
        out._residual_tensor = torch.tensor(residual_tensor).to(residual_tensor)
    else:
        out._residual_tensor = torch.zeros_like(composition_tensor[0]).to(
            composition_tensor
        )
    return out


def from_composition(composition: Composition) -> Composition:
    return from_tensor(composition.composition_tensor(), composition.residual())


def _from_replce(
    composition_tensor: Tensor, residual_tensor: Tensor = None
) -> Composition:
    out = void()
    out._composition_tensor = composition_tensor
    out._residual_tensor = residual_tensor
    return out


def cat():
    ...


def c_cat():
    ...


def stack():
    ...
