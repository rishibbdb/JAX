import jax.numpy as jnp
import jax
import numpy as np

from jax.scipy.stats.gamma import pdf as gamma_pdf
from jax.scipy.stats.norm import pdf as norm_pdf

def c_multi_gamma_spe_prob(x, mix_probs, a, b, sigma=3.0):
    nmax = 10
    nint1 = 10
    nint2 = 15
    nint3 = 35
    eps = 1.e-6

    xmax = jnp.max(jnp.array([jnp.array(nmax * sigma), x + nmax * sigma]))
    diff = xmax-x
    xmin = jnp.max(jnp.array([jnp.array(0.0), x - diff]))
    x_m1 = xmin + 0.02*sigma
    x_m2 = x_m1 + 0.5*sigma

    # two combined the two integration regions
    xvals = jnp.concatenate([jnp.linspace(xmin, x_m1, nint1),
                             jnp.linspace(x_m1, x_m2, nint2),
                             jnp.linspace(x_m2, xmax, nint3)])

    dx = xvals[1:]-xvals[:-1]

    xvals = 0.5*(xvals[:-1]+xvals[1:])
    n_pdf = norm_pdf(xvals, loc=x, scale=sigma)

    a_e = jnp.expand_dims(a, axis=-1)
    b_e = jnp.expand_dims(b, axis=-1)
    mix_probs_e = jnp.expand_dims(mix_probs, axis=-1)

    xvals_e = jnp.expand_dims(xvals, axis=0)
    pdfs = jnp.sum(mix_probs_e * gamma_pdf(xvals_e, a_e, scale=1./b_e), axis=0)

    return jnp.sum(n_pdf * pdfs * dx)

c_multi_gamma_spe_prob_v = jax.vmap(c_multi_gamma_spe_prob, (0, 0, 0, 0, None), 0)


def c_multi_gamma_spe_prob_large_sigma(x, mix_probs, a, b, sigma=1000.):
    """
    ... for noise. tested for sigma of order 1000.
    """
    nmax = 6
    nint1 = 10
    nint2 = 15
    nint3 = 35
    eps = 1.e-6

    xmax = jnp.max(jnp.array([jnp.array(nmax * sigma), x + nmax * sigma]))
    diff = xmax-x
    xmin = jnp.max(jnp.array([jnp.array(0.0), x - diff]))
    x_m1 = xmin + 10
    x_m2 = x_m1 + 100

    # two combined the two integration regions
    xvals = jnp.concatenate([jnp.linspace(xmin, x_m1, nint1),
                             jnp.linspace(x_m1, x_m2, nint2),
                             jnp.linspace(x_m2, xmax, nint3)])

    dx = xvals[1:]-xvals[:-1]

    xvals = 0.5*(xvals[:-1]+xvals[1:])
    n_pdf = norm_pdf(xvals, loc=x, scale=sigma)

    a_e = jnp.expand_dims(a, axis=-1)
    b_e = jnp.expand_dims(b, axis=-1)
    mix_probs_e = jnp.expand_dims(mix_probs, axis=-1)

    xvals_e = jnp.expand_dims(xvals, axis=0)
    pdfs = jnp.sum(mix_probs_e * gamma_pdf(xvals_e, a_e, scale=1./b_e), axis=0)

    return jnp.sum(n_pdf * pdfs * dx)

c_multi_gamma_spe_prob_large_sigma_v = jax.vmap(c_multi_gamma_spe_prob_large_sigma, (0, 0, 0, 0, None), 0)


def c_multi_gamma_spe_prob_large_sigma_fine(x, mix_probs, a, b, sigma=1000.):
    """
    ... for noise. tested for sigma of order 1000.
    """
    nmax = 6
    nint1 = 20
    nint2 = 30
    nint3 = 70
    eps = 1.e-6

    xmax = jnp.max(jnp.array([jnp.array(nmax * sigma), x + nmax * sigma]))
    diff = xmax-x
    xmin = jnp.max(jnp.array([jnp.array(0.0), x - diff]))
    x_m1 = xmin + 10
    x_m2 = x_m1 + 100

    # two combined the two integration regions
    xvals = jnp.concatenate([jnp.linspace(xmin, x_m1, nint1),
                             jnp.linspace(x_m1, x_m2, nint2),
                             jnp.linspace(x_m2, xmax, nint3)])

    dx = xvals[1:]-xvals[:-1]

    xvals = 0.5*(xvals[:-1]+xvals[1:])
    n_pdf = norm_pdf(xvals, loc=x, scale=sigma)

    a_e = jnp.expand_dims(a, axis=-1)
    b_e = jnp.expand_dims(b, axis=-1)
    mix_probs_e = jnp.expand_dims(mix_probs, axis=-1)

    xvals_e = jnp.expand_dims(xvals, axis=0)
    pdfs = jnp.sum(mix_probs_e * gamma_pdf(xvals_e, a_e, scale=1./b_e), axis=0)

    return jnp.sum(n_pdf * pdfs * dx)

c_multi_gamma_spe_prob_large_sigma_fine_v = jax.vmap(c_multi_gamma_spe_prob_large_sigma_fine, (0, 0, 0, 0, None), 0)
