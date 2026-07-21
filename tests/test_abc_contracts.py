"""Locks on the contracts: these are the ones that cost the most.

Python only checks the **name** of abstract methods, never their
signature: the three ABCs thus carried, for months, signatures that no
implementation respected (`Prior.log_prior(y, theta)` versus
`GaussianPrior.log_prior(theta)`, `InferenceModel._mu(theta, design)`
versus `LinearModel._mu(y, theta, design)`). These tests replace vigilance.
"""

import ast
import inspect
import textwrap

import jax.numpy as jnp
import pytest  # type: ignore

from cboed.criteria.optimality import EIG, AOptimal, DOptimal
from cboed.inference.base import InferenceModel
from cboed.inference.goal_oriented import GoalOrientedModel
from cboed.inference.linear_model import LinearModel
from cboed.likelihood.base import Likelihood
from cboed.likelihood.gaussian_likelihood import GaussianLikelihood
from cboed.priors.base import Prior
from cboed.priors.gaussian_process import GaussianPrior
from cboed.priors.kernel import Gaussian, Matern12, Matern32, Matern52


def _called_names(cls) -> set[str]:
    """Names of functions actually called within the body of `cls`.

    Via the AST rather than the text: docstrings are `Constant` nodes, so
    they are ignored. A grep on `inspect.getsource` would confuse an
    explanation with a call -- exactly what must be avoided.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


FORBIDDEN_IN_CRITERIA = frozenset(
    {"eigvalsh", "eigh", "eig", "cho_factor", "cho_solve", "cholesky", "slogdet", "inv"}
)


@pytest.mark.parametrize("cls", [EIG, DOptimal, AOptimal])
def test_criteria_do_no_linear_algebra(cls):
    """Criteria consume the contract, they do not factorize themselves."""
    offending = _called_names(cls) & FORBIDDEN_IN_CRITERIA
    assert not offending, f"{cls.__name__} calls {sorted(offending)}"


ABC_IMPLEMENTATIONS = [
    (Prior, GaussianPrior),
    (InferenceModel, LinearModel),
    (InferenceModel, GoalOrientedModel),
    (Likelihood, GaussianLikelihood),
]


def _abstract_methods(abc_cls):
    return sorted(getattr(abc_cls, "__abstractmethods__", frozenset()))


@pytest.mark.parametrize(("abc_cls", "impl_cls"), ABC_IMPLEMENTATIONS)
def test_abstract_signatures_match_implementation(abc_cls, impl_cls):
    """Each abstract method has the same signature in the implementation."""
    for name in _abstract_methods(abc_cls):
        abc_attr = getattr(abc_cls, name)
        impl_attr = getattr(impl_cls, name)
        if isinstance(abc_attr, property):
            assert isinstance(impl_attr, property), f"{impl_cls.__name__}.{name}"
            continue
        expected = list(inspect.signature(abc_attr).parameters)
        actual = list(inspect.signature(impl_attr).parameters)
        assert actual == expected, (
            f"{impl_cls.__name__}.{name}{tuple(actual)} "
            f"!= {abc_cls.__name__}.{name}{tuple(expected)}"
        )


def test_goal_oriented_inherits_the_contract():
    """Duck typing is dead: the wrapper implements the contract."""
    assert issubclass(GoalOrientedModel, InferenceModel)


def test_inference_contract_declares_only_what_is_consumed():
    """`posterior`, `_mu`, `_cov` are no longer interface promises."""
    declared = set(_abstract_methods(InferenceModel))
    assert declared == {
        "log_det_posterior_precision",
        "log_det_prior_precision",
        "posterior_cov_matmul",
    }


def test_no_design_in_prior_contract():
    """Design concerns the data, never anything that only touches theta."""
    for name in _abstract_methods(Prior):
        attr = getattr(Prior, name)
        if isinstance(attr, property):
            continue
        params = inspect.signature(attr).parameters
        assert "design" not in params, f"Prior.{name} takes a design"
        assert "y" not in params, f"Prior.{name} takes a y"


@pytest.mark.parametrize("kernel_cls", [Gaussian, Matern12, Matern32, Matern52])
def test_no_redundant_kernel_init(kernel_cls):
    """Simple kernels have no `__init__` of their own -- the base suffices."""
    assert "__init__" not in vars(kernel_cls)


@pytest.mark.parametrize("kernel_cls", [Gaussian, Matern12, Matern32, Matern52])
def test_no_kernel_overrides_shared_hyperparameters(kernel_cls):
    """length_scale / sigma live in KernelBase, nowhere else."""
    assert "length_scale" not in vars(kernel_cls)
    assert "sigma" not in vars(kernel_cls)


def test_unknown_hyperparameter_rejected():
    with pytest.raises(TypeError, match="unexpected"):
        Gaussian(length_scale=0.2, sigma=1.0, period=2.0)


@pytest.mark.parametrize("bad", [{"length_scale": 0.0}, {"length_scale": -1.0}, {"sigma": 0.0}])
def test_nonpositive_hyperparameters_rejected(bad):
    with pytest.raises(ValueError):
        Gaussian(**{"length_scale": 1.0, "sigma": 1.0, **bad})


def test_gaussian_prior_does_not_invert_at_construction():
    """The old `_H = -cho_solve(chol, eye(n))` materialized at init."""
    src = inspect.getsource(GaussianPrior.__init__)
    assert "cho_solve" not in src
    assert "_H" not in src


def test_prior_hessian_is_negative_precision():
    """Oracle: hessian() == -Gamma_prior^{-1}, materialized on demand."""
    from cboed.priors.gaussian_process import GaussianProcess

    gp = GaussianProcess(Matern32(length_scale=0.3, sigma=1.0), jnp.zeros(8))
    prior = GaussianPrior(prior=gp)
    identity = prior.Sigma() @ (-prior.hessian())
    assert jnp.allclose(identity, jnp.eye(8), atol=1e-8)
