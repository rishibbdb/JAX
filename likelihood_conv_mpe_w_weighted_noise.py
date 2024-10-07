#from lib.c_mpe_gamma import c_multi_gamma_mpe_prob_midpoint2_v as c_multi_gamma_mpe_prob_midpoint_v
#from lib.c_mpe_gamma import c_multi_gamma_mpe_prob_pure_jax_fast_v as c_multi_gamma_mpe_prob_midpoint_v
from lib.c_mpe_gamma import c_multi_gamma_mpe_prob_combined_v as c_multi_gamma_mpe_prob_midpoint_v
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
        end_of_physics = jnp.array(30.0) # when to stop evaluating negative time residuals in units of sigma for physics pdf

        end_of_physics =  (-1.0) * end_of_physics * sigma

        dom_pos = event_data[:, :3]
        first_hit_times = event_data[:, 3]
        charges = event_data[:, 4]

        logits, av, bv, geo_time, predicted_charge = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)
        delay_time = first_hit_times - (geo_time + track_time)

        # Floor on negative time residuals.
        # Effectively a floor on the pdf.
        in_physics_range = delay_time > end_of_physics
        safe_delay_time = jnp.where(in_physics_range, delay_time, end_of_physics)

        n_photons = jnp.round(charges + 0.5)
        n_photons = jnp.clip(n_photons, min=1.0, max=200.0)
        mix_probs = jax.nn.softmax(logits)
        physics_probs = c_multi_gamma_mpe_prob_midpoint_v(safe_delay_time,
                    mix_probs,
                    av,
                    bv,
                    n_photons,
                    sigma)

        #noise_probs = c_multi_gamma_mpe_prob_midpoint_v(safe_delay_time,
        #            mix_probs,
        #            av,
        #            bv,
        #            n_photons,
        #            sigma_noise)

        noise_probs = c_multi_gamma_spe_prob_large_sigma_v(delay_time,
                mix_probs,
                av,
                bv,
                sigma_noise)

        #noise_probs = norm_pdf(delay_time, (av[:, 1]-1)/bv[:, 1], scale=sigma_noise)
        #noise_probs = norm_pdf(delay_time, 0.0, scale=sigma_noise)

        predicted_charge = jnp.clip(jnp.squeeze(jnp.sum(charges) / jnp.sum(predicted_charge) * predicted_charge), min=None, max=1.0)
        noise_charge = jnp.array(0.01)
        noise_weight = noise_charge / (noise_charge + predicted_charge)
        return -2.0 * jnp.sum(jnp.log(noise_weight*noise_probs + (1.0-noise_weight)*physics_probs))



    return neg_c_triple_gamma_llh
