from lib.cgamma_biweight_log_mpe_prob import c_multi_gamma_biweight_mpe_logprob
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
        sigma = jnp.array(4.0) # width of gaussian convolution
        X_safe = jnp.array(2.9) # when to stop evaluating negative time residuals in units of sigma

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

        safe_delay_time = jnp.expand_dims(safe_delay_time, axis=-1)

        mpe_log_probs = c_multi_gamma_biweight_mpe_logprob(safe_delay_time, mix_probs, av, bv, n_photons, sigma)
        return -2.0 * jnp.sum(mpe_log_probs)


    return neg_c_triple_gamma_llh
