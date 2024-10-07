from lib.cgamma import c_multi_gamma_prob_v
from lib.cgamma import c_multi_gamma_sf_v
from lib.c_spe_gamma import c_multi_gamma_spe_prob_large_sigma_v

import jax
import jax.numpy as jnp

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
        sigma = 3.0 # width of gaussian convolution
        X_safe = 10.0 # when to stop evaluating negative time residuals in units of sigma
        sigma_noise = jnp.array(1000.0)
        delta = 1.0 # how to combine the three regions that combine approximate and exact evaluation of hyp1f1 (required for convolutions). Small values are faster. Large values are more accurate.


        dom_pos = event_data[:, :3]
        first_hit_times = event_data[:, 3]
        charges = event_data[:, 4]

        logits, av, bv, geo_time, predicted_charge = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)

        mix_probs = jax.nn.softmax(logits)
        delay_time = first_hit_times - (geo_time + track_time)

        # Floor on negative time residuals.
        # Effectively a floor on the pdf.
        # Todo: think about noise.
        safe_delay_time = jnp.where(delay_time > -X_safe * sigma, delay_time, -X_safe * sigma)

        physics_probs = c_multi_gamma_prob_v(safe_delay_time,
                                     mix_probs,
                                     av,
                                     bv,
                                     sigma,
                                     delta)

        sfs = c_multi_gamma_sf_v(safe_delay_time, mix_probs, av, bv, sigma)

        n_photons = jnp.round(charges + 0.5)
        n_photons = jnp.clip(n_photons, min=1.0, max=200.0)
        physics_probs = n_photons * physics_probs * jnp.power(sfs, n_photons-1.0)

        noise_probs = c_multi_gamma_spe_prob_large_sigma_v(delay_time,
                mix_probs,
                av,
                bv,
                sigma_noise)

        predicted_charge = jnp.clip(jnp.squeeze(jnp.sum(charges) / jnp.sum(predicted_charge) * predicted_charge), min=None, max=1.0)
        noise_charge = jnp.array(0.01)
        noise_weight = noise_charge / (noise_charge + predicted_charge)

        return -2.0 * jnp.sum(jnp.log(noise_weight*noise_probs + (1.0-noise_weight)*physics_probs))

    return neg_c_triple_gamma_llh
