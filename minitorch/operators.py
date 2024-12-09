"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def id(a: float) -> float:
    """Identity function."""
    return a


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def neg(a: float) -> float:
    """Negate a number."""
    return -a


def lt(a: float, b: float) -> float:
    """Less than."""
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Equal."""
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Maximum."""
    return a if a > b else b


def is_close(a: float, b: float) -> float:
    """Close."""
    return (a - b < 1e-2) and (b - a < 1e-2)


def sigmoid(a: float) -> float:
    """Sigmoid."""
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        return math.exp(a) / (1.0 + math.exp(a))


def relu(a: float) -> float:
    """ReLU."""
    return a if a > 0 else 0.0


def log(a: float) -> float:
    """Logarithm."""
    return math.log(a + 1e-6)


def exp(a: float) -> float:
    """Exponentiation."""
    return math.exp(a)


def log_back(a: float, d: float) -> float:
    """Logarithm back."""
    # return d / (a + 1e-6)
    return d / a


def inv(a: float) -> float:
    """Inverse."""
    return 1.0 / a


def inv_back(a: float, d: float) -> float:
    """Inverse back."""
    # return -d / (a * a)
    return -(1.0 / a**2) * d


def relu_back(a: float, d: float) -> float:
    """ReLU back."""
    return d if a > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

# T = TypeVar("T")
# U = TypeVar("U")

# Core higher-order functions


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """The function is used to map a function to a list of floats."""

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """The function is used to zip two lists of floats and apply a function to each pair of elements."""

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """The function is used to reduce a list of floats to a single float."""

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


# Functions implemented using the core higher-order functions


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list."""
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add two lists element-wise."""
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum a list of floats."""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product of a list of floats."""
    return reduce(mul, 1.0)(ls)
