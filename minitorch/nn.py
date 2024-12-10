from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError("Need to implement for Task 4.3")
    input = (
        input.contiguous()
        .view(batch, channel, height, int(width / kw), kw)
        .contiguous()
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    input = (
        input.view(batch, channel, int(width / kw), int(height / kh), kw * kh)
        .contiguous()
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    return input, int(height / kh), int(width / kw)


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
    -------
        Pooled tensor

    """
    batch, channel, height, width = input.shape
    tiled_input, tiled_height, tiled_width = tile(input, kernel)
    tiled_input = tiled_input.sum(4) / (kernel[0] * kernel[1])
    return tiled_input.view(batch, channel, tiled_height, tiled_width)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input : input tensor
        dim : dimension to apply argmax

    Returns:
    -------
        Tensor with 1 on highest cell in dim, 0 otherwise

    """
    max_vals = max_reduce(input, dim)
    return input == max_vals


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward of max should be max reduction"""
        output = max_reduce(input, int(dim.item()))
        ctx.save_for_backward(input, dim)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max should be argmax (see above)"""
        input, dim = ctx.saved_values
        max_idx = argmax(input, int(dim.item()))
        return grad_output * max_idx, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the softmax as a tensor.

    Args:
    ----
        input : input tensor
        dim : dimension to apply softmax

    Returns:
    -------
        softmax tensor

    """
    input = input.exp()
    return input / input.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the log of the softmax as a tensor.

    Args:
    ----
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
    -------
         log of softmax tensor

    """
    max_val = max(input, dim)
    log_sum_exp = (input - max_val).exp().sum(dim).log()
    return input - log_sum_exp - max_val


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor : pooled tensor

    """
    batch, channel, height, width = input.shape
    tiled_input, tiled_height, tiled_width = tile(input, kernel)
    new_input = max(tiled_input, 4)
    return new_input.view(batch, channel, tiled_height, tiled_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
    -------
        tensor with randoom positions dropped out

    """
    if ignore:
        return input
    else:
        return input * (rand(input.shape, input.backend) > rate)
