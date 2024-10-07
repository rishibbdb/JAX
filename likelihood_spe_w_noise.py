from lib.cgamma import c_multi_gamma_prob_v
from lib.c_spe_gamma import c_multi_gamma_spe_prob_large_sigma_v
import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import pdf as norm_pdf

def get_neg_c_triple_gamma_llh(eval_network_doms_and_track_fn):
    """
    here would be a smart docstring
    """

    @jax.jit
    def neg_c_triple_gamma_llh(track_direction,
                               track_vertex,
                               track_time,
                               event_data):

        # Constant parameters.
        sigma = jnp.array(3.0) # width of gaussian convolution
        sigma_noise = jnp.array(1000.0)
        end_of_physics = -jnp.array(100.0) # when to stop evaluating negative time residuals in units of sigma for physics pdf

        dom_pos = event_data[:, :3]
        first_hit_times = event_data[:, 3]
        charges = event_data[:, 4]
        n_photons = jnp.round(charges + 0.5)
        n_photons = jnp.clip(n_photons, min=1, max=200)

        logits, av, bv, geo_time = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)
        delay_time = first_hit_times - (geo_time + track_time)

        # Floor on negative time residuals.
        # Effectively a floor on the pdf.
        in_physics_range = delay_time > end_of_physics
        safe_delay_time = jnp.where(in_physics_range, delay_time, end_of_physics)

        mix_probs = jax.nn.softmax(logits)
        physics_probs = c_multi_gamma_prob_v(safe_delay_time,
                    mix_probs,
                    av,
                    bv,
                    sigma,
                    0.1)

        noise_probs = c_multi_gamma_spe_prob_large_sigma_v(delay_time,
                mix_probs,
                av,
                bv,
                sigma_noise)

        #noise_probs = norm_pdf(delay_time, (av[:, 1]-1)/bv[:, 1], scale=sigma_noise)
        #noise_probs = norm_pdf(delay_time, 0.0, scale=sigma_noise)

        noise_charge = jnp.array(0.005)
        floor_df = jnp.array(1./6000.)
        floor_weight = jnp.array(0.001)
        noise_weight = jnp.array(0.01)

        return -2.0 * jnp.sum(jnp.log(noise_weight*noise_probs + floor_weight*floor_df + (1.0-noise_weight-floor_weight)*physics_probs))



    return neg_c_triple_gamma_llh



def get_llh_and_grad_fs_for_iminuit_migrad(eval_network_doms_and_track):
    """
    """
    neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track)

    @jax.jit
    def neg_llh_5D(x, track_time, data):
        track_direction = x[:2]
        track_vertex = x[2:]
        return neg_llh(track_direction, track_vertex, track_time, data)

    grad_neg_llh_5D = jax.jit(jax.grad(neg_llh_5D, argnums=0))

    return neg_llh_5D, grad_neg_llh_5D


def get_llh_and_grad_fs_for_iminuit_migrad_profile(eval_network_doms_and_track):
    """
    """
    neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track)

	# this gradient is 3D (vertex) not 5D (vertex + direction)
    grad_neg_llh_3D = jax.jit(jax.grad(neg_llh, argnums=1))

    return neg_llh, grad_neg_llh_3D
