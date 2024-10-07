import jax.numpy as jnp
import jax
import numpy as np

from jax.scipy.stats.gamma import sf as gamma_sf
from jax.scipy.stats.gamma import pdf as gamma_pdf
from jax.scipy.stats.norm import pdf as norm_pdf
from jax.scipy.stats.norm import logpdf as norm_logpdf

from lib.gamma_sf_approx import gamma_sf_fast, c_coeffs, gamma_sf_fast_w_existing_coefficients

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from quadax import quadgk as quad

def c_multi_gamma_mpe_prob(x, logits, a, b, n, sigma):
    '''
    too slow due to gamma_sf / gammainc calls.
    '''
    g_pdf = tfd.MixtureSameFamily(
                  mixture_distribution=tfd.Categorical(
                      logits=logits
                      ),
                  components_distribution=tfd.Gamma(
                    concentration=a,
                    rate=b,
                    force_probs_to_zero_outside_support=True
                      )
                )

    gn = tfp.distributions.Normal(
                x,
                sigma,
                validate_args=False,
                allow_nan_stats=False,
                name='Normal'
            )

    nmax = 6
    nint = 11
    eps = 1.e-6

    xmax = jnp.max(jnp.array([jnp.array(nmax * sigma), x + nmax * sigma]))
    diff = xmax-x
    xmin = jnp.max(jnp.array([jnp.array(0.0)+eps, x - diff]))
    xvals = jnp.linspace(xmin, xmax, nint)

    n_pdf = gn.prob(0.5*(xvals[:-1]+xvals[1:]))
    sfs_power_n = jnp.power(g_pdf.survival_function(xvals), n)

    return jnp.sum( n_pdf * (sfs_power_n[:-1]-sfs_power_n[1:]) )

c_multi_gamma_mpe_prob_v = jax.vmap(c_multi_gamma_mpe_prob, (0, 0, 0, 0, 0, None), 0)


def c_multi_gamma_mpe_prob_pure_jax(x, mix_probs, a, b, n, sigma=3.0):
    '''
    too slow due to gamma_sf / gammainc calls.
    '''
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
    n_pdf = norm_pdf(0.5*(xvals[:-1]+xvals[1:]), loc=x, scale=sigma)

    a_e = jnp.expand_dims(a, axis=-1)
    b_e = jnp.expand_dims(b, axis=-1)
    mix_probs_e = jnp.expand_dims(mix_probs, axis=-1)

    xvals_e = jnp.expand_dims(xvals, axis=0)
    sfs = jnp.sum(mix_probs_e * gamma_sf(xvals_e, a_e, scale=1./b_e), axis=0)
    sfs_power_n = jnp.power(sfs, n)
    return jnp.sum(n_pdf * (sfs_power_n[:-1]-sfs_power_n[1:]))

c_multi_gamma_mpe_prob_pure_jax_v = jax.vmap(c_multi_gamma_mpe_prob_pure_jax, (0, 0, 0, 0, 0, None), 0)


def c_multi_gamma_mpe_prob_pure_jax_fast(x, mix_probs, a, b, n, sigma=3.0):
    """
    too heavy tail towards early times at large distances.
    """
    nmax = 15
    nint = 101
    eps = 1.e-6

    xmax = jnp.max(jnp.array([jnp.array(nmax * sigma), x + nmax * sigma]))
    diff = xmax-x
    xmin = jnp.max(jnp.array([jnp.array(0.0), x - diff]))

    xvals = jnp.linspace(xmin+eps, xmax, nint)
    n_pdf = norm_pdf(0.5*(xvals[:-1]+xvals[1:]), loc=x, scale=sigma)

    a_e = jnp.expand_dims(a, axis=-1)
    b_e = jnp.expand_dims(b, axis=-1)
    mix_probs_e = jnp.expand_dims(mix_probs, axis=-1)

    xvals_e = jnp.expand_dims(xvals, axis=0)
    sfs = jnp.sum(mix_probs_e * gamma_sf_fast(xvals_e, a_e, b_e), axis=0)
    sfs_power_n = jnp.power(sfs, n)
    return jnp.sum(n_pdf * (sfs_power_n[:-1]-sfs_power_n[1:]))

c_multi_gamma_mpe_prob_pure_jax_fast_v = jax.vmap(c_multi_gamma_mpe_prob_pure_jax_fast, (0, 0, 0, 0, 0, None), 0)


def integrand_fast(x, mix_probs, a, b, n_p, sigma, x0):
    c = c_coeffs(a)

    g_pdf = tfd.MixtureSameFamily(
                  mixture_distribution=tfd.Categorical(
                        probs=mix_probs
                      ),
                  components_distribution=tfd.Gamma(
                        concentration=a,
                        rate=b,
                        force_probs_to_zero_outside_support=True
                      )
                )

    tmp = jnp.sum(mix_probs * gamma_sf_fast_w_existing_coefficients(x, a, b, c), axis=-1)
    return norm_pdf(x, loc=x0, scale=sigma) * g_pdf.prob(x) * n_p * jnp.power(tmp, n_p-1.0)

def c_multi_gamma_mpe_prob_pure_jax_fast_qdx(x, mix_probs, a, b, n, sigma=3.0):
    delta = jnp.array(10.0)
    eps = 1.e-6
    xmax = jnp.max(jnp.array([delta*sigma, x + delta*sigma]))
    diff = xmax-x
    xmin = jnp.max(jnp.array([jnp.array(0.0)+eps, x - diff]))


    res = quad(integrand_fast,
                 jnp.array([xmin, xmax]),
                 args=(mix_probs, a, b, n, sigma, x),
                 epsabs=1.e-4,
                 epsrel=1.e-4,
                 order=31,
                 max_ninter=1
              )[0]

    return res

c_multi_gamma_mpe_prob_pure_jax_fast_qdx_v = jax.vmap(c_multi_gamma_mpe_prob_pure_jax_fast_qdx, (0, 0, 0, 0, 0, None), 0)


def c_multi_gamma_mpe_prob_midpoint(x, mix_probs, a, b, n, sigma=3.0):
    """
    Q < 30
    """
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
    sfs_power = jnp.power(jnp.sum(mix_probs_e * gamma_sf_fast(xvals_e, a_e, b_e), axis=0), n-1.0)
    pdfs = jnp.sum(mix_probs_e * gamma_pdf(xvals_e, a_e, scale=1./b_e), axis=0)

    return jnp.sum(n_pdf * n * pdfs * sfs_power * dx)

c_multi_gamma_mpe_prob_midpoint_v = jax.vmap(c_multi_gamma_mpe_prob_midpoint, (0, 0, 0, 0, 0, None), 0)


def integrand(x, mix_probs, a, b, n_p, sigma, x0):
    g_pdf = tfd.MixtureSameFamily(
                  mixture_distribution=tfd.Categorical(
                        probs=mix_probs
                      ),
                  components_distribution=tfd.Gamma(
                        concentration=a,
                        rate=b,
                        force_probs_to_zero_outside_support=True
                      )
                )

    return norm_pdf(x, loc=x0, scale=sigma) * g_pdf.prob(x) * n_p * jnp.power(g_pdf.survival_function(x), n_p-1.0)


def c_multi_gamma_mpe_prob_pure_jax_qdx(x, mix_probs, a, b, n, sigma=3.0):
    delta = jnp.array(8.0)
    eps = 1.e-6
    xmax = jnp.max(jnp.array([delta*sigma, x + delta*sigma]))
    diff = xmax-x
    #xmin = jnp.max(jnp.array([jnp.array(0.0)+eps, x - diff]))
    xmin = eps


    res = quad(integrand,
                 jnp.array([xmin, xmax]),
                 args=(mix_probs, a, b, n, sigma, x),
                 epsabs=1.e-4,
                 epsrel=1.e-4,
                 order=31,
                 max_ninter=1
              )[0]

    return res


def c_multi_gamma_mpe_prob_midpoint2(x, mix_probs, a, b, n, sigma=3.0):
    """
    Q < 30
    """
    nmax = 10
    nint1 = 10
    nint2 = 15
    nint3 = 35
    #eps = 1.e-12
    eps = 1.e-6

    #xmax = jnp.max(jnp.array([jnp.array(nmax * sigma), x + nmax * sigma]))
    #diff = xmax-x
    #xmin = jnp.max(jnp.array([jnp.array(0.0), x - diff]))
    #x_m1 = xmin + 0.02*sigma
    #x_m2 = x_m1 + 0.5*sigma

    ## two combined the two integration regions
    #xvals = jnp.concatenate([jnp.linspace(xmin, x_m1, nint1),
    #                         jnp.linspace(x_m1, x_m2, nint2),
    #                         jnp.linspace(x_m2, xmax, nint3)])

    x0 = eps
    x_m0 = 0.01
    xvals0 = jnp.linspace(x0, x_m0, 10)

    x_m1 = 0.05
    xvals1 = jnp.linspace(x_m0, x_m1, 10)

    x_m2 = 0.25
    xvals2 = jnp.linspace(x_m1, x_m2, 10)

    x_m25 = 0.75
    xvals25 = jnp.linspace(x_m2, x_m25, 10)

    x_m3 = 2.5
    xvals3 = jnp.linspace(x_m25, x_m3, 10)

    x_m4 = 8.0
    xvals4 = jnp.linspace(x_m3, x_m4, 20)

    #x_m5 = 6000.0
    #xvals5 = jnp.linspace(x_m4, x_m5, 20)

    #xmin = jnp.max(jnp.array([1.5 * eps, x - 4 * sigma]))
    #xmax = jnp.max(jnp.array([xmin+1.5*eps, x + 4 * sigma]))
    #xvals_x = jnp.linspace(xmin, xmax, 30)
    #xvals = jnp.sort(jnp.concatenate([xvals0, xvals1, xvals2, xvals25, xvals3, xvals4, xvals5, xvals_x]))
    xmin = jnp.max(jnp.array([1.5 * eps, x - 10 * sigma]))
    xmax = jnp.max(jnp.array([xmin+1.5*eps, x + 10 * sigma]))
    xvals_x = jnp.linspace(xmin, xmax, 101)
    xvals = jnp.sort(jnp.concatenate([xvals0, xvals1, xvals2, xvals25, xvals3, xvals4, xvals_x]))

    dx = xvals[1:]-xvals[:-1]

    xvals = 0.5*(xvals[:-1]+xvals[1:])
    n_pdf = norm_pdf(xvals, loc=x, scale=sigma)
    #n_log_pdf = norm_logpdf(xvals, loc=x, scale=sigma)

    a_e = jnp.expand_dims(a, axis=-1)
    b_e = jnp.expand_dims(b, axis=-1)
    mix_probs_e = jnp.expand_dims(mix_probs, axis=-1)

    xvals_e = jnp.expand_dims(xvals, axis=0)
    sfs = jnp.sum(mix_probs_e * gamma_sf_fast(xvals_e, a_e, b_e), axis=0)
    pdfs = jnp.sum(mix_probs_e * jnp.clip(gamma_pdf(xvals_e, a_e, scale=1./b_e), min=0, max=None), axis=0)

    return jnp.sum(n_pdf * n * pdfs * jnp.power(sfs, n-1.0) * dx)
    #return jnp.sum(jnp.exp(jnp.log(n_pdf) + jnp.log(n) + jnp.log(pdfs) + (n-1.0) * jnp.log(sfs) + jnp.log(dx)))
    #return jnp.sum(jnp.exp(n_log_pdf + jnp.log(n) + jnp.log(pdfs) + (n-1.0) * jnp.log(sfs) + jnp.log(dx)))

c_multi_gamma_mpe_prob_midpoint2_v = jax.vmap(c_multi_gamma_mpe_prob_midpoint2, (0, 0, 0, 0, 0, None), 0)


def mpe_pdf_no_conv(x, mix_probs, a, b, n):
    g_pdf = tfd.MixtureSameFamily(
                  mixture_distribution=tfd.Categorical(
                      probs=mix_probs
                      ),
                  components_distribution=tfd.Gamma(
                    concentration=a,
                    rate=b,
                    force_probs_to_zero_outside_support=True
                      )
    )
    return n * g_pdf.prob(x) * jnp.power(g_pdf.survival_function(x), n-1.0)


def combine(x, mix_probs, a, b, n, sigma):
    eps = jnp.array(1.e-12)
    crit = jnp.array(40.0)
    x_safe = jnp.where(x < eps, eps, x)
    probs_no_conv = mpe_pdf_no_conv(x_safe, mix_probs, a, b, n)
    probs_conv = c_multi_gamma_mpe_prob_midpoint2(x, mix_probs, a, b, n, sigma)
    return jnp.where(x < crit, probs_conv, probs_no_conv)

c_multi_gamma_mpe_prob_combined_v = jax.vmap(combine, (0, 0, 0, 0, 0, None), 0)
