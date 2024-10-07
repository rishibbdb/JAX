import jax
import jax.numpy as jnp

"""
Fast approximation to lower incomplete gamma function.
Implementation follows the formulas given in

Geosci. Model Dev., 3, 329–336, 2010
www.geosci-model-dev.net/3/329/2010/
doi:10.5194/gmd-3-329-2010

by U. Blahak.
"""

# from Table 1.
p_coeffs = jnp.array([
    9.4368392235e-3,
    -1.0782666481e-4,
    -5.8969657295e-6,
    2.8939523781e-7,
    1.0043326298e-1,
    5.5637848465e-1
])

q_coeffs = jnp.array([
    1.1464706419e-1,
    2.6963429121,
    -2.9647038257,
    2.1080724954
])

r_coeffs = jnp.array([
    0.0,
    1.1428716184,
    -6.6981186438e-3,
    1.0480765092e-4
])

s_coeffs = jnp.array([
    1.0480765092,
    2.3423452308,
    -3.6174503174e-1,
    -3.1376557650,
    2.9092306039
])


def c_coeffs(a):
    """
    eq. 14-17
    """
    ap2 = jnp.power(a, 2)
    ap3 = jnp.power(a, 3)
    ap4 = jnp.power(a, 4)

    c1 = 1.0 + p_coeffs[0]*a + p_coeffs[1]*ap2 + p_coeffs[2]*ap3 + p_coeffs[3]*ap4 + p_coeffs[4]*(jnp.exp(-p_coeffs[5]*a)-1.0)
    c2 = q_coeffs[0] + q_coeffs[1]/a + q_coeffs[2]/ap2 + q_coeffs[3]/ap3
    c3 = r_coeffs[0] + r_coeffs[1]*a + r_coeffs[2]*ap2 + r_coeffs[3]*ap3
    c4 = s_coeffs[0] + s_coeffs[1]/a + s_coeffs[2]/ap2 + s_coeffs[3]/ap3 + s_coeffs[4]/ap4
    return jnp.array([c1, c2, c3, c4])


def tanh_approx(x):
    """
    eq. 23.
    an approximation to tanh.
    does not appear faster than the direct call.
    """
    ct = 9.37532
    crit = ct / 3
    y = (9*ct**2*x + 27*x**3) / (ct**3 + 27*ct*x**2)
    y = jnp.where(x <= -crit, -1.0, y)
    y = jnp.where(x >= crit, 1.0, y)
    return y


def regularized_lower_incomplete_gamma_approx(x, a):
    """
    note: the original paper treats the lower incomplete gamma function.
    The regularized lower incomplete gamma function differs by the
    normalization factor 1/Gamma[a].
    """

    c = c_coeffs(a)

    # eq. 13
    w = 0.5 + 0.5 * jnp.tanh(c[1]*(x-c[2]))

    # alternatively, use approximate tanh.
    #w = 0.5 + 0.5 * tanh_approx(c[1]*(x-c[2]))

    # eq. 12
    r1 = 1.0/jax.scipy.special.gamma(a) * jnp.exp(-x)*jnp.power(x, a)*(1.0/a + c[0]*x/(a*(a+1)) + (c[0]*x)**2/(a*(a+1)*(a+2)))*(1-w)
    r2 = w*(1.0-jnp.power(c[3], -x))
    return r1+r2


def regularized_lower_incomplete_gamma_approx_w_existing_coefficients(x, a, c):
    """
    note: the original paper treats the lower incomplete gamma function.
    The regularized lower incomplete gamma function differs by the
    normalization factor 1/Gamma[a].
    """

    # eq. 13
    w = 0.5 + 0.5 * jnp.tanh(c[1]*(x-c[2]))

    # alternatively, use approximate tanh.
    #w = 0.5 + 0.5 * tanh_approx(c[1]*(x-c[2]))

    # eq. 12
    r1 = 1.0/jax.scipy.special.gamma(a) * jnp.exp(-x)*jnp.power(x, a)*(1.0/a + c[0]*x/(a*(a+1)) + (c[0]*x)**2/(a*(a+1)*(a+2)))*(1-w)
    r2 = w*(1.0-jnp.power(c[3], -x))
    return r1+r2


def gamma_sf_fast(x, a, b):
    return jnp.clip(1.0-regularized_lower_incomplete_gamma_approx(x*b, a), min=0.0, max=1.0)


def gamma_sf_fast_w_existing_coefficients(x, a, b, c):
    return jnp.clip(1.0-regularized_lower_incomplete_gamma_approx_w_existing_coefficients(x*b, a, c), min=0.0, max=1.0)
