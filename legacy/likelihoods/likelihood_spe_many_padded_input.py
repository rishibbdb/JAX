from lib.cgamma import c_multi_gamma_prob_v
import jax
import jax.numpy as jnp


def get_neg_c_triple_gamma_llh(eval_network_doms_and_track_fn, n_pulses):
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
        X_safe = 20.0 # when to stop evaluating negative time residuals in units of sigma
        delta = 0.1 # how to combine the three regions that combine approximate and exact evaluation of hyp1f1 (required for convolutions). Small values are faster. Large values are more accurate.


        dom_pos = event_data[:, :3]
        hit_times = event_data[:, 3:3+n_pulses]
        # treat padded values in time dimension
        hit_charges = event_data[:, 3+n_pulses:]
        idx_padded_q = hit_charges != 0.0
        logits, av, bv, geo_time = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)

        # treat padded values for doms dimension
        idx_padded = event_data[:, 0] != 0.0
        idx_padded_s = idx_padded.reshape((idx_padded.shape[0], 1))
        # replace padded values with some computable outputs that don't lead to nan.
        logits = jnp.where(idx_padded_s, logits, jnp.ones((1, 3)))
        av = jnp.where(idx_padded_s, av, jnp.ones((1, 3))+3.0)
        bv = jnp.where(idx_padded_s, bv, jnp.ones((1, 3))*1.e-3)

        mix_probs = jax.nn.softmax(logits)

        # now prepare for broadcasting over time axis
        geo_time = jnp.expand_dims(geo_time, 1)
        delay_time = hit_times - (geo_time + track_time)

        # Floor on negative time residuals.
        # Effectively a floor on the pdf.
        # Todo: think about noise.
        safe_delay_time = jnp.where(delay_time > -X_safe * sigma, delay_time, -X_safe * sigma)

        # re-arrange so that dims are (n_doms, n_pulses, n_mixture_components)
        safe_delay_time = jnp.expand_dims(safe_delay_time, 2)

        mix_probs = jnp.expand_dims(mix_probs, 1)
        av = jnp.expand_dims(av, 1)
        bv = jnp.expand_dims(bv, 1)

        #y = jnp.where(idx_padded_q,
        #              jnp.log(c_multi_gamma_prob_v(safe_delay_time,
        #                                     mix_probs,
        #                                     av,
        #                                     bv,
        #                                     sigma,
        #                                     delta)),
        #              0.0)

        y = hit_charges * jnp.log(c_multi_gamma_prob_v(safe_delay_time,
                                             mix_probs,
                                             av,
                                             bv,
                                             sigma,
                                             delta))

        return -2.0 * jnp.sum(y)

    return neg_c_triple_gamma_llh
