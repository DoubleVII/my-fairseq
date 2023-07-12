from typing import Type, Union
import torch
from torch.types import _int


def args_error(
    funcion_name: str, args: Union[list, tuple], kwargs: dict = {}
) -> TypeError:
    arg_types = tuple(type(arg).__name__ for arg in args)
    kwarg_types = tuple(
        f"{key}={type(value).__name__}" for key, value in kwargs.items()
    )
    type_str = ", ".join(arg_types + kwarg_types)
    return TypeError(
        f"{funcion_name}() received an invalid combination of arguments - got ({type_str})"
    )


def arg_value_error(msg: str) -> RuntimeError:
    return RuntimeError(msg)


def size_error(
    s1: torch.Size, s2: torch.Size, t1_name="tensor1", t2_name="tensor2"
) -> RuntimeError:
    return RuntimeError(
        f"The size of {t1_name} {s1} doesn't match the size of {t2_name} {s2}."
    )


def composition_num_error(
    num1: _int, num2: _int, c1_name="composition1", c2_name="composition2"
) -> RuntimeError:
    return RuntimeError(
        f"The composition number of {c1_name} ({num1}) doesn't match the composition number of {c2_name} ({num2})."
    )


def unsupported_operand_error(
    operation_name: str, operand_type1: Type, operand_type2: Type
) -> TypeError:
    return TypeError(
        f"unsupported operand type(s) for {operation_name}: '{operand_type1.__name__}' and '{operand_type2.__name__}'"
    )
