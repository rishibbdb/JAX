from lib.cgamma import postjitter_c_multi_gamma_mpe_prob_v
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
        sigma_post = 2.0
        X_safe = 20.0 # when to stop evaluating negative time residuals in units of sigma

        dom_pos = event_data[:, :3]
        first_hit_times = event_data[:, 3]
        charges = event_data[:, 4]
        n_photons = jnp.round(charges + 0.5)
        n_photons = jnp.clip(n_photons, min=1.0, max=1000.0)

        logits, av, bv, geo_time = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)

        mix_probs = jax.nn.softmax(logits)
        delay_time = first_hit_times - (geo_time + track_time)

        # Floor on negative time residuals.
        # Effectively a floor on the pdf.
        # Todo: think about noise.
        safe_delay_time = jnp.where(delay_time > -X_safe * sigma, delay_time, -X_safe * sigma)

        #probs = c_multi_gamma_prob_v(safe_delay_time,
        probs = postjitter_c_multi_gamma_mpe_prob_v(safe_delay_time,
                                     mix_probs,
                                     av,
                                     bv,
                                     n_photons,
                                     sigma,
                                     sigma_post)

        return -2.0 * jnp.sum(jnp.log(probs))

    return neg_c_triple_gamma_llh
