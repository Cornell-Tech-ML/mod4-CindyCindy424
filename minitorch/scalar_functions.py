from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """The function is used to calculate the gradient of the output with respect to the input."""
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """The function is used to calculate the forward pass of the scalar function."""
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """The function is used to apply the scalar function to the input values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(float(v))

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """The function is used to calculate the forward pass of the addition function."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """The function is used to calculate the gradient of the output with respect to the input."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """The function is used to calculate the forward pass of the logarithm function."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The function is used to calculate the gradient of the output with respect to the input."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """The function is used to calculate the forward pass of the multiplication function."""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """The function is used to calculate the gradient of the output with respect to the input."""
        a, b = ctx.saved_values
        return operators.mul(b, d_output), operators.mul(a, d_output)
        # return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """The function is used to calculate the forward pass of the inverse function."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The function is used to calculate the gradient of the output with respect to the input."""
        a = ctx.saved_values[0]
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """The function is used to calculate the forward pass of the negation function."""
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The function is used to calculate the gradient of the output with respect to the input."""
        return operators.neg(d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """The function is used to calculate the forward pass of the sigmoid function."""
        output = operators.sigmoid(a)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The function is used to calculate the gradient of the output with respect to the input."""
        sigma = ctx.saved_values[0]
        return operators.mul(d_output, operators.mul(sigma, 1.0 - sigma))


class ReLU(ScalarFunction):
    """Rectified Linear Unit function $f(x) = max(0, x)$$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """The function is used to calculate the forward pass of the rectified linear unit function."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The function is used to calculate the gradient of the output with respect to the input."""
        a = ctx.saved_values[0]
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """The function is used to calculate the forward pass of the exponential function."""
        output = operators.exp(a)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The function is used to calculate the gradient of the output with respect to the input."""
        output = ctx.saved_values[0]
        return operators.mul(d_output, output)


class LT(ScalarFunction):
    """Less than function $f(x, y) = 1.0$ if $x < y$ else $0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """The function is used to calculate the forward pass of the less than function."""
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """The function is used to calculate the gradient of the output with respect to the input."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x, y) = 1.0$ if $x == y$ else $0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """The function is used to calculate the forward pass of the equal function."""
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """The function is used to calculate the gradient of the output with respect to the input."""
        return 0.0, 0.0
