from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # DONE: Implement for Task 1.1.
    valsl = [v for v in vals]
    valsl[arg] = valsl[arg] + epsilon
    valsr = [v for v in vals]
    valsr[arg] = valsr[arg] - epsilon
    delta = f(*valsl) - f(*valsr)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """The function is used to accumulate the derivative of the variable."""
        ...

    @property
    def unique_id(self) -> None:
        """The function is used to get the unique id of the variable."""
        ...

    def is_leaf(self) -> None:
        """The function is used to check if the variable is a leaf."""
        ...

    def is_constant(self) -> None:
        """The function is used to check if the variable is a constant."""
        ...

    @property
    def parents(self) -> None:
        """The function is used to get the parents of the variable."""
        ...

    def chain_rule(self, d_output: Any) -> None:
        """The function is used to get the chain rule of the variable."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")
    order: List[Variable] = []
    seen = set()

    def visit(v: Variable) -> None:
        if v.unique_id in seen or v.is_constant():
            return
        if not v.is_leaf():
            for m in v.parents:
                if not m.is_constant():
                    visit(m)
        order.insert(0, v)
        seen.add(v.unique_id)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable.
        deriv: Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        None. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")
    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    for v in queue:
        deriv = derivatives[v.unique_id]
        if v.is_leaf():
            v.accumulate_derivative(deriv)
        else:
            if v.parents is not None:
                for parent, grad in v.chain_rule(deriv):
                    if parent.is_constant():
                        continue
                    derivatives.setdefault(parent.unique_id, 0)
                    derivatives[parent.unique_id] += grad


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """The function is used to get the saved tensors."""
        return self.saved_values
