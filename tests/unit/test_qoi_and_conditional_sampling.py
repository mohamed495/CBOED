import jax
import jax.numpy as jnp

from cboed.benchmarks import N, build_qoi
from cboed.inference.conditional_sampling import sample_eta_given_theta
from cboed.priors.gaussian_process import GaussianPrior, GaussianProcess
from cboed.priors.kernel import Gaussian


def test_build_qoi_nuisance_contract():
    h, sigma_xi, n_qoi = build_qoi("nuisance")

    assert n_qoi == N // 2
    assert sigma_xi.shape == (N // 2, N // 2)
    assert h(jnp.zeros(N)).shape == (N // 2,)


def test_build_qoi_energy_contract_and_jacobian():
    h, sigma_xi, n_qoi = build_qoi("energy")
    eta = jnp.arange(N, dtype=jnp.float32)

    assert n_qoi == 1
    assert sigma_xi.shape == (1, 1)
    assert h(eta).shape == (1,)
    assert jnp.allclose(h(eta), jnp.asarray([jnp.sum(eta**2)]))
    assert jnp.allclose(jax.jacfwd(h)(eta), 2 * eta[None, :])


def test_energy_conditional_sampler_shape_and_target_consistency():
    prior = GaussianPrior(
        prior=GaussianProcess(
            kernel=Gaussian(length_scale=1.0, sigma=1.0),
            mu=jnp.zeros(2),
        )
    )

    def h(eta):
        return jnp.asarray([jnp.sum(eta**2)])

    samples = sample_eta_given_theta(
        prior,
        jnp.asarray([1.0]),
        jax.random.key(0),
        h=h,
        Sigma_xi=0.2 * jnp.eye(1),
        n_samples=200,
        n_warmup=200,
        step_size=0.02,
        thinning=2,
    )

    energies = jnp.sum(samples**2, axis=1)
    assert samples.shape == (200, 2)
    assert bool(jnp.all(jnp.isfinite(samples)))
    assert 0.5 < float(jnp.mean(energies)) < 2.0


def test_linear_conditional_sampler_shape():
    prior = GaussianPrior(
        prior=GaussianProcess(
            kernel=Gaussian(length_scale=1.0, sigma=1.0),
            mu=jnp.zeros(2),
        )
    )
    samples = sample_eta_given_theta(
        prior,
        jnp.asarray([0.0]),
        jax.random.key(1),
        B=jnp.asarray([[1.0, 0.0]]),
        Sigma_xi=0.1 * jnp.eye(1),
        n_samples=5,
    )

    assert samples.shape == (5, 2)
