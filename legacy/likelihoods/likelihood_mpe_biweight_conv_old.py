from lib.cgamma_biweight_v2 import c_multi_gamma_biweight_prob_v
from lib.cgamma_biweight import c_multi_gamma_biweight_cdf_v
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
        X_safe = 2.9 # when to stop evaluating negative time residuals in units of sigma
        delta = 0.01 # how to combine the three regions that combine approximate and exact evaluation of hyp1f1 (required for convolutions). Small values are faster. Large values are more accurate.


        dom_pos = event_data[:, :3]
        first_hit_times = event_data[:, 3]
        charges = event_data[:, 4]
        n_photons = jnp.round(charges + 0.5)

        logits, av, bv, geo_time = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)

        mix_probs = jax.nn.softmax(logits)
        delay_time = first_hit_times - (geo_time + track_time)

        # Floor on negative time residuals.
        # Effectively a floor on the pdf.
        # Todo: think about noise.
        safe_delay_time = jnp.where(delay_time > -X_safe * sigma, delay_time, -X_safe * sigma)

        probs = c_multi_gamma_biweight_prob_v(safe_delay_time,
                                            mix_probs,
                                            av,
                                            bv,
                                            sigma)

        cdfs = c_multi_gamma_biweight_cdf_v(safe_delay_time, mix_probs, av, bv, sigma)

        mpe_log_probs = jnp.log(n_photons) + jnp.log(probs) + (n_photons-1.0) * jnp.log(1-cdfs)
        return -2.0 * jnp.sum(mpe_log_probs)


    return neg_c_triple_gamma_llh
