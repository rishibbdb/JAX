#from lib.c_mpe_gamma import c_multi_gamma_mpe_prob_midpoint2_v as c_multi_gamma_mpe_prob_midpoint_v
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
        end_of_physics = jnp.array(20.0) # when to stop evaluating negative time residuals in units of sigma for physics pdf

        end_of_physics =  (-1.0) * end_of_physics * sigma

        dom_pos = event_data[:, :3]
        first_hit_times = event_data[:, 3]
        charges = event_data[:, 4]

        logits, av, bv, geo_time, predicted_charge = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)
        delay_time = first_hit_times - (geo_time + track_time)

        # take care of padding
        idx_padded = event_data[:, 0] != 0.0
        idx_padded_s = idx_padded.reshape((idx_padded.shape[0], 1))

        # replace padded values with some computable outputs that don't lead to nan.
        logits = jnp.where(idx_padded_s, logits, jnp.ones((1, 3)))
        av = jnp.where(idx_padded_s, av, jnp.ones((1, 3))+3.0)
        bv = jnp.where(idx_padded_s, bv, jnp.ones((1, 3))*1.e-3)
        delay_time = jnp.where(idx_padded, delay_time, 10.0)

        # Floor on negative time residuals.
        # Effectively a floor on the pdf.
        in_physics_range = delay_time > end_of_physics
        safe_delay_time = jnp.where(in_physics_range, delay_time, end_of_physics)

        mix_probs = jax.nn.softmax(logits)
        n_photons = jnp.round(charges + 0.5)
        n_photons = jnp.clip(n_photons, min=1.0, max=200.0)
        physics_probs = c_multi_gamma_mpe_prob_midpoint_v(safe_delay_time,
                    mix_probs,
                    av,
                    bv,
                    n_photons,
                    sigma)

        # The physics weight goes down, if predicted charge is less than ~1.
        # If no charge is expected, the hit is likely to be noise.
        # Scale predicted charge to Qtot (predicted  charge is for minimum ionizing muon)
        predicted_charge = jnp.where(idx_padded_s, predicted_charge, 0.0)
        predicted_charge = jnp.squeeze(jnp.sum(charges) / jnp.sum(predicted_charge) * predicted_charge)
        predicted_charge = jnp.clip(predicted_charge, min=None, max=1.0)

        #noise_probs = c_multi_gamma_spe_prob_large_sigma_v(delay_time,
        #        mix_probs,
        #        av,
        #        bv,
        #        sigma_noise)
        #noise_probs = norm_pdf(delay_time, (av[:, 1]-1)/bv[:, 1], scale=sigma_noise)
        noise_probs = norm_pdf(delay_time, 0.0, scale=sigma_noise)

        noise_charge = jnp.array(0.01)
        noise_weight = noise_charge / (noise_charge + predicted_charge)
        physics_weight = 1.0 - noise_weight

        log_probs = jnp.log(noise_weight*noise_probs + physics_weight*physics_probs)

        # remove contribution from padded hits to likelihood (mask to 0)
        log_probs = jnp.where(idx_padded, log_probs, jnp.array(0.0))
        return -2.0 * jnp.sum(log_probs)



    return neg_c_triple_gamma_llh
