from lib.cgamma_biweight_log_mpe_prob import postjitter_c_mpe_biweight_v
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
        sigma = jnp.array(3.0) # width of gaussian convolution
        sigma_post = jnp.array(2.0)
        X_safe = jnp.array(10.0) # when to stop evaluating negative time residuals in units of sigma

        dom_pos = event_data[:, :3]
        first_hit_times = event_data[:, 3]
        charges = event_data[:, 4]
        n_photons = jnp.round(charges + 0.5)

        idx_padded = event_data[:, 0] != 0.0
        idx_padded_s = idx_padded.reshape((idx_padded.shape[0], 1))

        logits, av, bv, geo_time = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)
        logits = jnp.where(idx_padded_s, logits, jnp.ones((1, 3)))
        av = jnp.where(idx_padded_s, av, jnp.ones((1, 3))+3.0)
        bv = jnp.where(idx_padded_s, bv, jnp.ones((1, 3))*1.e-3)

        mix_probs = jax.nn.softmax(logits)
        delay_time = first_hit_times - (geo_time + track_time)

        # Floor on negative time residuals.
        # Effectively a floor on the pdf.
        # Todo: think about noise.
        safe_delay_time = jnp.where(delay_time > -X_safe * sigma, delay_time, -X_safe * sigma)
        #safe_delay_time = jnp.expand_dims(safe_delay_time, axis=-1)

        mpe_log_probs = jnp.log(postjitter_c_mpe_biweight_v(safe_delay_time, mix_probs, av, bv, n_photons, sigma, sigma_post))

        # mask padded region so it doesn't contribute to likelihood
        mpe_log_probs = jnp.where(idx_padded, mpe_log_probs, jnp.array(0.0))
        return -2.0 * jnp.sum(mpe_log_probs)


    return neg_c_triple_gamma_llh
