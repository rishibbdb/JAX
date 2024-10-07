#### experimental. do not use.

#### implement in a way that shares the gammaincc evaluations across branches.

#### add direct implementation of MPE formula, so that gammaincc evaluations can be shared across pdf and cdf evals.

import jax.numpy as jnp
import jax

from jax.scipy.special import gamma, gammaincc, gammainc
from jax.scipy.stats.norm import pdf as norm_pdf

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

__sigma_scale = 3.0

def c_multi_gamma_biweight_mpe_logprob(x, mix_probs, a, b, n_photons, sigma=3.0):
    s = __sigma_scale * sigma

    # share the two evaluations of the upper incomplete gamma function.
    # across all subsequent function calls to avoid
    # costly re-evaluations.
    g_a = gamma(a)
    gincc_a_bspx = gammaincc(a, b*(s+x)) * g_a

    x1 = jnp.where(x < s, s, x)
    gincc_a_bxms = gammaincc(a, b*(x1-s)) * g_a

    prob = _c_multi_gamma_biweight_prob(x, mix_probs, a, b, s, gincc_a_bspx, gincc_a_bxms)
    cdf = _c_multi_gamma_biweight_cdf(x, mix_probs, a, b, s, gincc_a_bspx, gincc_a_bxms)
    return jnp.log(n_photons) + jnp.log(prob) + (n_photons-1.0) * jnp.log(1.0-cdf)

c_multi_gamma_biweight_mpe_logprob_v = jax.jit(jax.vmap(c_multi_gamma_biweight_mpe_logprob,
                                                        (0, 0, 0, 0, 0, None),
                                                        0
                                                    )
                                                )

c_multi_gamma_biweight_mpe_logprob_v1d = jax.jit(jax.vmap(c_multi_gamma_biweight_mpe_logprob, (0, None, None, None, None, None), 0))


def postjitter_c_mpe_biweight(x, mix_probs, a, b, n, sigma=3.0, sigma_post=2.0):
    __sigma_scale = jnp.array(3.0)
    nmax = jnp.array(4.0)
    nmin = jnp.array(20.0)
    nint1 = 5
    nint2 = 10
    eps = jnp.array(1.e-6)
    x0 = -sigma * __sigma_scale # start of support of MPE convolved biweight

    xmax = jnp.max(jnp.array([x0 + nmax * sigma_post, x + nmax * sigma_post]))
    xmin = jnp.max(jnp.array([x0, x - nmin * sigma_post]))
    mid_p = xmin + 0.2 * (xmax-xmin)
    xvals = jnp.concatenate([jnp.linspace(xmin, mid_p, nint1), jnp.linspace(mid_p, xmax, nint2)])

    dx = xvals[1:] - xvals[:-1]
    xvals = 0.5*(xvals[:-1]+xvals[1:])
    return jnp.sum(norm_pdf(xvals, loc=x, scale=sigma_post) * jnp.exp(c_multi_gamma_biweight_mpe_logprob_v1d(xvals, mix_probs, a, b, n, sigma)) * dx)

postjitter_c_mpe_biweight_v = jax.jit(jax.vmap(postjitter_c_mpe_biweight, (0, 0, 0, 0, 0, None, None), 0))


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


def combine(x, mix_probs, a, b, n, sigma, sigma_post):
    crit = 40.0
    crit_cond = x < crit

    a_safe = jnp.where(crit_cond, jnp.ones((1, 3))+3.0, a)
    b_safe = jnp.where(crit_cond, jnp.ones((1, 3))*1.e-3, b)
    x_safe = jnp.where(crit_cond, 0.0, x)
    n_safe = jnp.where(crit_cond, 1.0, n)
    probs_no_conv = jnp.exp(c_multi_gamma_biweight_mpe_logprob(x_safe, mix_probs, a_safe, b_safe, n_safe, sigma))

    a_safe = jnp.where(crit_cond, a, jnp.ones((1, 3))+3.0)
    b_safe = jnp.where(crit_cond, b, jnp.ones((1, 3))*1.e-3)
    x_safe = jnp.where(crit_cond, x, 0.0)
    n_safe = jnp.where(crit_cond, n, 1.0)

    probs_conv = postjitter_c_mpe_biweight(x_safe, mix_probs, a_safe, b_safe, n_safe, sigma, sigma_post)
    return jnp.where(crit_cond, probs_conv, probs_no_conv)

postjitter_c_mpe_biweight_combined_v = jax.vmap(combine, (0, 0, 0, 0, 0, None, None), 0)


def _c_multi_gamma_biweight_prob(x, mix_probs, a, b, s, gincc_a_bspx, gincc_a_bxms):
    # todo: consider exploring logsumexp trick (potentially more stable)
    # e.g. https://github.com/tensorflow/probability/blob/65f265c62bb1e2d15ef3e25104afb245a6d52429/tensorflow_probability/python/distributions/mixture_same_family.py#L348
    # for now: implement naive mixture probs
    return jnp.sum(mix_probs * _c_gamma_biweight_prob(x, a, b, s, gincc_a_bspx, gincc_a_bxms), axis=-1)


def _branch0(x, a, b, s, gincc_a_bspx, gincc_a_bxms):
    # branch 0 (-s < x < +s)

    g_a = gamma(a)
    # use recurrence relation of gamma to avoid further gamma calls
    g_1pa = a * g_a
    g_2pa = (a+1) * g_1pa
    g_3pa = (a+2) * g_2pa
    g_4pa = (a+3) * g_3pa

    bspx = b*(s+x)
    bspx_pa = jnp.power(bspx, a)

    ginc_a = g_a - gincc_a_bspx
    #ginc_a = gammainc(a, bspx) * g_a
    # use recurrence relation of lower incomplete gamma function to avoid further gammainc calls
    exp_mbspx = jnp.exp(-bspx)
    ginc_1pa = a * ginc_a - bspx_pa * exp_mbspx
    ginc_2pa = (1+a) * ginc_1pa - bspx_pa*bspx * exp_mbspx
    ginc_3pa = (2+a) * ginc_2pa - bspx_pa*bspx*bspx * exp_mbspx
    ginc_4pa = (3+a) * ginc_3pa - bspx_pa*bspx*bspx*bspx * exp_mbspx

    fbx = 4*b*x
    t0 = b**4 * (s**4 - 2*s**2*x**2 + x**4)
    t1 = 4*b**3 * (s**2*x - x**3)
    t2 = b**2 * (6*x**2 - 2*s**2)

    tsum0 = (
                ginc_a * t0
                + ginc_1pa * t1
                + ginc_2pa * t2
                + ginc_4pa
                + (g_3pa - ginc_3pa) * fbx
                - g_2pa * (2*fbx + a*fbx)
    )

    pre_fac = 15.0/(16*b**4*s**5*g_a)
    return pre_fac * tsum0


def _branch1(x, a, b, s, gincc_a_bspx, gincc_a_bxms):
    # branch 1 (s > x)

    g_a = gamma(a)
    # use recurrence relation of gamma to avoid further gamma calls
    g_1pa = a * g_a
    g_2pa = (a+1) * g_1pa
    g_3pa = (a+2) * g_2pa
    g_4pa = (a+3) * g_3pa

    bspx = b*(s+x)
    bspx_pa = jnp.power(bspx, a)
    bxms = b*(x-s)
    bxms_pa = jnp.power(bxms, a)

    gincc_a = gincc_a_bspx
    # use recurrence relation of lower incomplete gamma function to avoid further gammainc calls
    exp_mbspx = jnp.exp(-bspx)
    gincc_1pa = a * gincc_a + bspx_pa * exp_mbspx
    gincc_2pa = (1+a) * gincc_1pa + bspx_pa*bspx * exp_mbspx
    gincc_3pa = (2+a) * gincc_2pa + bspx_pa*bspx*bspx * exp_mbspx
    gincc_4pa = (3+a) * gincc_3pa + bspx_pa*bspx*bspx*bspx * exp_mbspx

    gincc_a_m = gincc_a_bxms
    # use recurrence relation of lower incomplete gamma function to avoid further gammainc calls
    exp_mbxms = jnp.exp(-bxms)
    gincc_1pa_m = a * gincc_a_m + bxms_pa * exp_mbxms
    gincc_2pa_m = (1+a) * gincc_1pa_m + bxms_pa*bxms * exp_mbxms
    gincc_3pa_m = (2+a) * gincc_2pa_m + bxms_pa*bxms*bxms * exp_mbxms
    gincc_4pa_m = (3+a) * gincc_3pa_m + bxms_pa*bxms*bxms*bxms * exp_mbxms

    fbx = 4*b*x
    t0 = b**4 * (s**4 - 2*s**2*x**2 + x**4)
    t1 = 4*b**3 * (s**2*x - x**3)
    t2 = b**2 * (6*x**2 - 2*s**2)

    tsum1 = (
                (gincc_a_m - gincc_a) * t0
                + (gincc_1pa_m - gincc_1pa) * t1
                + (gincc_2pa_m - gincc_2pa) * t2
                + (gincc_3pa - gincc_3pa_m) * fbx
                + gincc_4pa_m - gincc_4pa
    )

    pre_fac = 15.0/(16*b**4*s**5*g_a)
    return pre_fac * tsum1


def _c_gamma_biweight_prob(x, a, b, s, gincc_a_bspx, gincc_a_bxms):
    x0 = jnp.where(x < s, x, s)
    b0 = _branch0(x0, a, b, s, gincc_a_bspx, gincc_a_bxms)

    x1 = jnp.where(x < s, s, x)
    b1 = _branch1(x1, a, b, s, gincc_a_bspx, gincc_a_bxms)
    return jnp.where(x < s, b0, b1)


def _branch0_cdf(x, a, b, s, gincc_a_bspx, gincc_a_bxms):
    g_a = gamma(a)
    bspx = b * (s+x)
    bx = b*x

    pre_factor = 1./(16.*b**5*s**5*(s+x)*g_a)

    c__11 = (
        3*(1 + a)*(2 + a)*(3 + a)*(4 + a)*x
        + 3*(2 + a)*(3 + a)*bspx*((4 + a)*s - (1 + 4*a)*x)
        - b**3*(s + x)**2*((8 + 7*a)*s**2 - 3*(3 + 7*a)*s*x + 3*(1 + 4*a)*x**2)
        - b**2*(s + x)*((a - 1)*(16 + 7*a)*s**2 + 3*(3 + a)*(2 + 3*a)*s*x - 6*(1 + 3*a*(2 + a))*x**2)
        + b**4*(8*s**5 + 15*s**4*x - 10*s**2*x**3)
    )

    c1 = jnp.exp(-bspx) * (
        3*b**(4 + a)*x**5*(s + x)**a
        + jnp.power(bspx ,a) * (3*(1 + a)*(2 + a)*(3 + a)*(4 + a)*s + c__11)
    )

    c__21 = (-3*a**5 + 15*a**4*(-2 + bx)
             + b**5*(s + x)**3*(8*s**2 - 9*s*x + 3*x**2)
             + 5*a**3*(-21 + 2*b*(9*x + b*(s**2 - 3*x**2)))
             - 15*a**2*(10 + b*(-11*x + 2*b*(x**2*(3 - bx) + s**2*(-1 + bx))))
             + a*(
                    -72 - 5*b*(3*b**3*s**4 - 18*x + 3*x*bx*(4 + bx*(-2 + bx))
                    + b*s**2*(-4 - 6*bx*(-1 + bx)))
             )
    )

    c2 = (s+x) * c__21 * (g_a - gincc_a_bspx)

    return pre_factor * (c1 + c2)


def _branch1_cdf(x, a, b, s, gincc_a_bspx, gincc_a_bxms):
    g_a = gamma(a)
    bspx = b * (s+x)
    bx = b*x

    pre_factor = 1./(16.*b**5*s**5*(s+x)*g_a)

    # branch 1 x >= s:
    c__11 = (
        3*a**4 + 72*(1 + b*s) + 3*a**3*(10 + b*(s - 4*x))
        + a**2*(105 + b*(s*(27 - 7*b*s) - 9*(7 + b*s)*x + 18*b*x**2))
        + a*(150 + b*(s*(78 - b*s*(9 + 7*b*s)) + (-87 + b*s*(-33 + 14*b*s))*x + 9*b*(4 + b*s)*x**2 - 12*b**2*x**3))
        + b*(-18*x + b*(8*s**2 - 9*s*x + 3*x**2)*(2 + bspx*(bspx-1)))
    )

    c__12 = (
        72 + 3*a**4 - 72*b*s - 3*a**3*(-10 + b*(s + 4*x))
        + b*(-18*x + b*(2 + b*(1 - (b * (x-s)))*(s - x))*(8*s**2 + 9*s*x + 3*x**2))
        + a**2*(105 + b*(-7*b*s**2 + 9*s*(-3 + bx) + 9*x*(-7 + 2*bx)))
        + a*(150 + b*(7*b**2*s**3 + b*s**2*(-9 + 14*bx) + s*(-78 + 3*bx*(11 - 3*bx)) - 3*x*(29 + 4*bx*(-3 + bx))))
    )

    c1 = jnp.exp(-bspx) * b**a * (jnp.power(s+x, a)*c__11 - jnp.exp(2*b*s)*jnp.power(x-s, a)*c__12)

    c__21 = (
        10*a*(1 + a)*(2 + a)*b*s**2 - 15*a*b**3*s**4 - 8*b**4*s**5
        + 15*(a*(1 + a)*(2 + a)*(3 + a) - 2*a*(1 + a)*b**2*s**2 + b**4*s**4)*x
        - 30*a*(2 + a*(3 + a) - b**2*s**2)*bx*x
        + 10*(3*a*(1 + a) - b**2*s**2)*bx**2*x
        - 15*a*bx**3*x
        + 3*bx**4*x
    )

    c__22 = (
        3*a**5 - 15*a**4*(-2 + bx)
        - b**5*(s + x)**3*(8*s**2 - 9*s*x + 3*x**2)
        + 5*a**3*(21 - 2*b*(9*x + b*(s**2 - 3*x**2)))
        + 15*a**2*(10 + b*(-11*x + 2*b*(x**2*(3 - bx) + s**2*(-1 + bx))))
        + a*(72 + 5*b*(3*b**3*s**4 - 18*x + 3*bx*x*(4 + bx*(-2 + bx)) + b*s**2*(-4 - 6*bx*(-1 + bx))))
    )

    c2 = (
        16*b**5*s**5*g_a - 3*a*(1 + a)*(2 + a)*(3 + a)*(4 + a)*gincc_a_bxms
        + b*gincc_a_bxms * c__21
        + gincc_a_bspx * c__22
    )

    return pre_factor * (c1 + c2) * (s+x)


def _c_gamma_biweight_cdf(x, a, b, s, gincc_a_bspx, gincc_a_bxms):
    x0 = jnp.where(x < s, x, s)
    b0 = _branch0_cdf(x0, a, b, s, gincc_a_bspx, gincc_a_bxms)

    x1 = jnp.where(x < s, s, x)
    b1 = _branch1_cdf(x1, a, b, s, gincc_a_bspx, gincc_a_bxms)
    return jnp.where(x < s, b0, b1)


def _c_multi_gamma_biweight_cdf(x, mix_probs, a, b, s, gincc_a_bspx, gincc_a_bxms):
    return jnp.sum(mix_probs * _c_gamma_biweight_cdf(x, a, b, s, gincc_a_bspx, gincc_a_bxms), axis=-1)
