import jax.numpy as jnp
import jax
import numpy as np

from jax.scipy.special import gamma, gammaincc

__sigma_scale = 3.0

def c_multi_gamma_biweight_prob(x, mix_probs, a, b, sigma=3.0):
    # todo: consider exploring logsumexp trick (potentially more stable)
    # e.g. https://github.com/tensorflow/probability/blob/65f265c62bb1e2d15ef3e25104afb245a6d52429/tensorflow_probability/python/distributions/mixture_same_family.py#L348
    # for now: implement naive mixture probs
    return jnp.sum(mix_probs * c_gamma_biweight_prob(x, a, b, sigma), axis=-1)

c_multi_gamma_biweight_prob_v = jax.vmap(c_multi_gamma_biweight_prob, (0, 0, 0, 0, None), 0)


def branch0(x, a, b, s):
    # branch 0 (-s < x < +s)
    g_a = gamma(a)
    g_1pa = gamma(1+a)
    g_2pa = gamma(2+a)
    g_3pa = gamma(3+a)
    g_4pa = gamma(4+a)

    bspx = b*(s+x)

    gincc_a = gammaincc(a, bspx) * g_a
    gincc_1pa = gammaincc(1+a, bspx)*g_1pa
    gincc_2pa = gammaincc(2+a, bspx)*g_2pa
    gincc_3pa = gammaincc(3+a, bspx)*g_3pa
    gincc_4pa = gammaincc(4+a, bspx)*g_4pa

    fbx = 4*b*x
    t0 = b**4 * (s**4 - 2*s**2*x**2 + x**4)
    t1 = 4*b**3 * (s**2*x - x**3)
    t2 = b**2 * (6*x**2 - 2*s**2)

    tsum0 = (
                (g_a - gincc_a) * t0
                + (g_1pa - gincc_1pa) * t1
                + (g_2pa - gincc_2pa) * t2
                + g_4pa - gincc_4pa
                + gincc_3pa * fbx
                - g_2pa * (2*fbx + a*fbx)
    )

    pre_fac = 15.0/(16*b**4*s**5*g_a)
    return pre_fac * tsum0


def branch1(x, a, b, s):
    # branch 1 (s > x)

    g_a = gamma(a)
    g_1pa = gamma(1+a)
    g_2pa = gamma(2+a)
    g_3pa = gamma(3+a)
    g_4pa = gamma(4+a)

    bspx = b*(s+x)
    bxms = b*(x-s)

    gincc_a = gammaincc(a, bspx) * g_a
    gincc_1pa = gammaincc(1+a, bspx)*g_1pa
    gincc_2pa = gammaincc(2+a, bspx)*g_2pa
    gincc_3pa = gammaincc(3+a, bspx)*g_3pa
    gincc_4pa = gammaincc(4+a, bspx)*g_4pa

    gincc_a_m = gammaincc(a, bxms) * g_a
    gincc_1pa_m = gammaincc(1+a, bxms)*g_1pa
    gincc_2pa_m = gammaincc(2+a, bxms)*g_2pa
    gincc_3pa_m = gammaincc(3+a, bxms)*g_3pa
    gincc_4pa_m = gammaincc(4+a, bxms)*g_4pa

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


def c_gamma_biweight_prob(x, a, b, sigma=3.0):
    s = __sigma_scale * sigma
    x0 = jnp.where(x < s, x, s)
    b0 = branch0(x0, a, b, s)

    x1 = jnp.where(x < s, s, x)
    b1 = branch1(x1, a, b, s)
    return jnp.where(x < s, b0, b1)

c_gamma_biweight_prob_v = jax.vmap(c_gamma_biweight_prob, (0, 0, 0, None), 0)


def branch0_cdf(x, a, b, s):
    g_a = gamma(a)
    bspx = b * (s+x)
    bspx_pa = jnp.power(bspx ,a)
    exp_mbspx = jnp.exp(-bspx)
    g_a_bspx = g_a * gammaincc(a, bspx)
    bx = b*x

    pre_factor = 1./(16.*b**5*s**5*(s+x)*g_a)

    c__11 = (
        3*(1 + a)*(2 + a)*(3 + a)*(4 + a)*x
        + 3*(2 + a)*(3 + a)*bspx*((4 + a)*s - (1 + 4*a)*x)
        - b**3*(s + x)**2*((8 + 7*a)*s**2 - 3*(3 + 7*a)*s*x + 3*(1 + 4*a)*x**2)
        - b**2*(s + x)*((a - 1)*(16 + 7*a)*s**2 + 3*(3 + a)*(2 + 3*a)*s*x - 6*(1 + 3*a*(2 + a))*x**2)
        + b**4*(8*s**5 + 15*s**4*x - 10*s**2*x**3)
    )

    c1 = exp_mbspx * (
        3*b**(4 + a)*x**5*(s + x)**a
        + bspx_pa * (3*(1 + a)*(2 + a)*(3 + a)*(4 + a)*s + c__11)
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

    c2 = (s+x) * c__21 * (g_a - g_a_bspx)

    return pre_factor * (c1 + c2)


def branch1_cdf(x, a, b, s):
    g_a = gamma(a)
    bspx = b * (s+x)
    bspx_pa = jnp.power(bspx ,a)
    exp_bspx = jnp.exp(bspx)
    exp_mbspx = jnp.exp(-bspx)
    g_a_bspx = g_a * gammaincc(a, bspx)
    bxms = b * (x-s)
    g_a_bxms = g_a * gammaincc(a, bxms)
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
        + b*(-18*x + b*(2 + b*(1 - bxms)*(s - x))*(8*s**2 + 9*s*x + 3*x**2))
        + a**2*(105 + b*(-7*b*s**2 + 9*s*(-3 + bx) + 9*x*(-7 + 2*bx)))
        + a*(150 + b*(7*b**2*s**3 + b*s**2*(-9 + 14*bx) + s*(-78 + 3*bx*(11 - 3*bx)) - 3*x*(29 + 4*bx*(-3 + bx))))
    )

    c1 = exp_mbspx * b**a * (jnp.power(s+x, a)*c__11 - jnp.exp(2*b*s)*jnp.power(x-s, a)*c__12)

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
        16*b**5*s**5*g_a - 3*a*(1 + a)*(2 + a)*(3 + a)*(4 + a)*g_a_bxms
        + b*g_a_bxms * c__21
        + g_a_bspx * c__22
    )

    return pre_factor * (c1 + c2) * (s+x)


def c_gamma_biweight_cdf(x, a, b, sigma=3.0):
    s = __sigma_scale * sigma
    x0 = jnp.where(x < s, x, s)
    b0 = branch0_cdf(x0, a, b, s)

    x1 = jnp.where(x < s, s, x)
    b1 = branch1_cdf(x1, a, b, s)
    return jnp.where(x < s, b0, b1)


def c_multi_gamma_biweight_cdf(x, mix_probs, a, b, sigma=3.0):
    return jnp.sum(mix_probs * c_gamma_biweight_cdf(x, a, b, sigma), axis=-1)

c_multi_gamma_biweight_cdf_v = jax.vmap(c_multi_gamma_biweight_cdf, (0, 0, 0, 0, None), 0)
