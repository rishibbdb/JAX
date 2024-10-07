import jax.numpy as jnp
import jax
import numpy as np

from jax.scipy.special import gammaincc, erf
from jax.scipy.stats.norm import pdf as norm_pdf

_M_LN2 = jnp.log(2.0)
_M_SQRTPI = jnp.sqrt(np.pi)
_M_E = jnp.exp(1.0)
_M_SQRT2 = jnp.sqrt(2.0)


def c_multi_gamma_prob(x, mix_probs, a, b, sigma=3.0, delta=10.0):
    # todo: consider exploring logsumexp trick (potentially more stable)
    # e.g. https://github.com/tensorflow/probability/blob/65f265c62bb1e2d15ef3e25104afb245a6d52429/tensorflow_probability/python/distributions/mixture_same_family.py#L348
    # for now: implement naive mixture probs
    return jnp.sum(mix_probs * c_gamma_prob(x, a, b, sigma, delta), axis=-1)

c_multi_gamma_prob_v = jax.vmap(c_multi_gamma_prob, (0, 0, 0, 0, None, None), 0)


def c_gamma_prob(x, a, b, sigma=3.0, delta=10.0):
    # x < crit_x - delta => region 4
    # x > crit_x + delta => region 5
    # else: exact evaluation => region 1
    # makes a piecewise defined function
    # that ensures stable gradients.
    crit_x = b * sigma**2

    cond_region1 = jnp.logical_and(x < crit_x+delta, x > crit_x-delta)
    x_region1 = jnp.where(cond_region1, x, crit_x)

    cond_region3 = x >= crit_x+delta
    x_region3 = jnp.where(cond_region3, x, crit_x+delta)

    cond_region4 = x <= crit_x-delta
    x_region4 = jnp.where(cond_region4, x, crit_x-delta)


    yvals_region1 = _c_gamma_region1(x_region1, a, b, sigma=sigma)
    yvals_region3 = _c_gamma_region3(x_region3, a, b, sigma=sigma)
    yvals_region4 = _c_gamma_region4(x_region4, a, b, sigma=sigma)

    result1 = jnp.where(cond_region1, yvals_region1, 0.0)
    result3 = jnp.where(cond_region3, yvals_region3, 0.0)
    result4 = jnp.where(cond_region4, yvals_region4, 0.0)

    return result1 + result3 + result4

c_gamma_prob_v = jax.vmap(c_gamma_prob, (0, 0, 0, None, None), 0)


def _c_gamma_region1(x, a, b, sigma=3):
    """
    Implements convolution of gamma distribution with a normal distribution.
    Such distribution arises from adding gaussian noise to samples from a gamma distribution.
    See eq. 7 of arXiv:0704.1706 [astro-ph]. Used for region 1.
    This is the most "exact" calculation, relying on direct eval of hyp1f1
    """

    eta = b*sigma - x/sigma
    s_eta_sq = 0.5 * eta**2 # argument to hyp1f1 is always positive.

    fac1 = (b**a * sigma **(a-1) * jnp.exp(-0.5*(x/sigma)**2)) / 2**(0.5*(1+a))
    s1 = jax.scipy.special.hyp1f1(0.5*a, 0.5, s_eta_sq) / jax.scipy.special.gamma(0.5*(a+1))
    s2 = jax.scipy.special.hyp1f1(0.5*(a+1), 3./2., s_eta_sq) / jax.scipy.special.gamma(0.5*a)
    return fac1 * (s1 - np.sqrt(2)*eta*s2)


def _c_gamma_region3(x, a, b, sigma=3):
    """
    arXiv:0704.1706, eq. 12
    t >= rho sigma^2, a >= 1
    https://github.com/icecube/icetray/blob/7195b9ad8a76b22e0d7a1e9238147952b1645254/rpdf/private/rpdf/pandel.cxx#L207
    Note: a := ksi
    """

    rhosigma = b*sigma
    eta = rhosigma - x/sigma
    ksi21 = 2.*a - 1
    ksi212 = ksi21*ksi21
    ksi213 = ksi212*ksi21
    z = jnp.fabs(eta)/jnp.sqrt(2.*ksi21)
    sqrt1plusz2 = jnp.sqrt(1 + z*z)
    k = 0.5*(z*sqrt1plusz2 + jnp.log(z+sqrt1plusz2))
    beta=0.5*(z/sqrt1plusz2 - 1.)
    beta2 = beta*beta
    beta3 = beta2*beta
    beta4 = beta3*beta
    beta5 = beta4*beta
    beta6 = beta5*beta
    n1 = (20.*beta3 + 30.*beta2 + 9.*beta)/12.
    n2 = (6160.*beta6 + 18480.*beta5 + 19404.*beta4 + 8028.*beta3 + 945.*beta2)/288.
    n3 = (27227200.*beta6 + 122522400.*beta5 + 220540320.*beta4 + 200166120.*beta3 +\
          94064328.*beta2 + 20546550.*beta + 1403325.)*beta3/51840.

    sigma2 = sigma*sigma
    delay2 = x*x
    eta2 = eta*eta
    alpha=(-0.5*delay2/sigma2 + 0.25*eta2 - 0.5*a + 0.25 + k*ksi21 - 0.5*jnp.log(sqrt1plusz2) -\
           0.5*a*_M_LN2 + 0.5*(a-1.)*jnp.log(ksi21) + a*jnp.log(rhosigma))
    phi = 1. - n1/ksi21 + n2/ksi212 - n3/ksi213

    return jnp.exp(alpha)*phi/jax.scipy.special.gamma(a)/sigma


def _c_gamma_region4(x, a, b, sigma=3):
    """
    arXiv:0704.1706, eq. 13
    https://github.com/icecube/icetray/blob/e773449cfbb9e505dbcdeb3ae84242505fb7f253/rpdf/private/rpdf/pandel.cxx#L237
    t <= rho sigma^2, a >= 1
    Note: a := ksi
    """

    rhosigma = b*sigma
    eta = rhosigma - x/sigma
    ksi21 = 2.*a - 1
    ksi212 = ksi21*ksi21
    ksi213 = ksi212*ksi21
    z = jnp.fabs(eta)/jnp.sqrt(2.*ksi21)
    sqrt1plusz2 = jnp.sqrt(1 + z*z)
    k = 0.5*(z*sqrt1plusz2 + jnp.log(z+sqrt1plusz2))
    beta=0.5*(z/sqrt1plusz2 - 1.)
    beta2 = beta*beta
    beta3 = beta2*beta
    beta4 = beta3*beta
    beta5 = beta4*beta
    beta6 = beta5*beta
    n1 = (20.*beta3 + 30.*beta2 + 9.*beta)/12.
    n2 = (6160.*beta6 + 18480.*beta5 + 19404.*beta4 + 8028.*beta3 + 945.*beta2)/288.
    n3 = (27227200.*beta6 + 122522400.*beta5 + 220540320.*beta4 + 200166120.*beta3 +\
          94064328.*beta2 + 20546550.*beta + 1403325.)*beta3/51840.

    sigma2 = sigma*sigma
    delay2 = x*x
    eta2 = eta*eta

    u = jnp.power(2.*_M_E/ksi21, a/2.)*jnp.exp(-0.25)/_M_SQRT2
    psi = 1. + n1/ksi21 + n2/ksi212 + n3/ksi213
    cpandel= jnp.power(rhosigma, a)/sigma * jnp.exp(-0.5*delay2/sigma2+0.25*eta2) / (_M_SQRT2*_M_SQRTPI)

    return  cpandel * u * jnp.exp(-k*ksi21) * psi / jnp.sqrt(sqrt1plusz2)


def c_gamma_sf(x, a, b, sigma=3.0):
    """
    following arXiv:astro-ph/0506136
    """
    alpha = 2.5 # controls the split of the integral => precision.
    n_steps1 = 30 # controls the support points in midpoint integration
    n_steps2 = 10
    n_steps3 = 10
    eps = 1.e-15

    sqrt2sigma2 = jnp.sqrt(2.0*sigma**2)

    ymin = x - alpha*sqrt2sigma2 # start of numeric integration
    ymin = jnp.where(ymin >= 0.0, ymin, 0.0)

    ymax = x + alpha*sqrt2sigma2 # end of numeric integration
    ymax = jnp.where(ymax >= 0.0, ymax, 0.0)
    # todo: think about special case when ymin = ymax = 0.0
    # based on testing so far: no need to do anything.

    term1 = gammaincc(a, b*ymax) + gammaincc(a, b*ymin)
    term2 = jnp.power(b, a) # J in arXiv:astro-ph/0506136

    #x_int = jnp.linspace(ymin, ymax, n_steps+1, axis=-1)
    #dx = jnp.expand_dims(x_int[..., 1] - x_int[..., 0], axis=-1)

    mid_p1 = ymin + 0.01 * (ymax - ymin)
    mid_p2 = mid_p1 + 0.1 * (ymax - mid_p1)
    x_int = jnp.concatenate([
        jnp.linspace(ymin, mid_p1, n_steps2+1, axis=-1),
        jnp.linspace(mid_p1, mid_p2, n_steps3+1, axis=-1),
        jnp.linspace(mid_p2, ymax, n_steps1+1, axis=-1)
    ])

    dx = jnp.expand_dims(x_int[..., 1:] - x_int[..., :-1], axis=0)
    x_int = 0.5*(x_int[...,1:] + x_int[...,:-1])

    # add dimension to end for proper broadcasting during integration
    # and then integrate by brute force on even grid
    # todo: come up with something faster and more accurate?
    a_e = jnp.expand_dims(a, axis=-1)
    b_e = jnp.expand_dims(b, axis=-1)
    y_int = jnp.power(x_int, a_e-1) * jnp.exp(-b_e*x_int) * erf((x-x_int)/sqrt2sigma2)
    term2 *= jnp.sum(y_int * dx, axis=-1)

    sf = 0.5 * (term1 - term2/jax.scipy.special.gamma(a))
    return jnp.clip(sf, min=eps, max=1.0)


def c_multi_gamma_sf(x, mix_probs, a, b, sigma=3.0):
    probs = c_gamma_sf(x, a, b, sigma=sigma)
    return jnp.sum(mix_probs * probs, axis=-1)

c_multi_gamma_sf_v = jax.jit(jax.vmap(c_multi_gamma_sf, (0, 0, 0, 0, None), 0))


def c_multi_gamma_mpe_prob(x, mix_probs, a, b, n, sigma=3.0, delta=10.0):
    p = c_multi_gamma_prob(x, mix_probs, a, b, sigma=sigma, delta=delta)
    sf = c_multi_gamma_sf(x, mix_probs, a, b, sigma=sigma)
    return n * p * jnp.power(sf, n-1)

c_multi_gamma_mpe_prob_v1d_x = jax.jit(jax.vmap(c_multi_gamma_mpe_prob, (0, None, None, None, None, None, None), 0))


def c_multi_gamma_mpe_log_prob(x, mix_probs, a, b, n, sigma=3.0, delta=10.0):
    p = c_multi_gamma_prob(x, mix_probs, a, b, sigma=sigma, delta=delta)
    sf = c_multi_gamma_sf(x, mix_probs, a, b, sigma=sigma)
    return jnp.log(n) + jnp.log(p) + (n-1) * jnp.log(sf)


def postjitter_c_multi_gamma_mpe_prob(x, mix_probs, a, b, n, sigma=3.0, sigma_post=2.0):
    nmax = 5.0
    nmin = 10.0
    nint1 = 10
    nint2 = 10
    x0 = -14.0 # early integration start
    x1 = -1.0 # early integration end
    delta = 1.0


    xmin = jnp.max(jnp.array([x0, x - nmin * sigma_post]))
    xmax = jnp.max(jnp.array([x0 + nmax * sigma_post, x + nmax * sigma_post]))

    xvals = jnp.concatenate([jnp.linspace(x0, x1, nint1), x * jnp.ones(1), jnp.linspace(xmin, xmax, nint2)])
    xvals = jnp.sort(xvals)
    dx = xvals[1:] - xvals[:-1]
    xvals = 0.5*(xvals[:-1]+xvals[1:])
    return jnp.sum(norm_pdf(xvals, loc=x, scale=sigma_post) * c_multi_gamma_mpe_prob_v1d_x(xvals, mix_probs, a, b, n, sigma, delta) * dx)

postjitter_c_multi_gamma_mpe_prob_v = jax.jit(jax.vmap(postjitter_c_multi_gamma_mpe_prob, (0, 0, 0, 0, 0, None, None), 0))
