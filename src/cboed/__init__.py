"""Certified Expected Information Gain (EIG) bounds for Bayesian OED.

Enables 64-bit precision in JAX at import time: the certified bounds rely on
Cholesky factorizations and log-determinants that are numerically fragile in
the default 32-bit precision.
"""

import jax

jax.config.update("jax_enable_x64", True)
