from lib.cgamma import c_multi_gamma_prob_v
from lib.cgamma import c_multi_gamma_sf_v
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
        sigma = 2.0 # width of gaussian convolution
        X_safe = 20.0 # when to stop evaluating negative time residuals in units of sigma
        delta = 0.01 # how to combine the three regions that combine approximate and exact evaluation of hyp1f1 (required for convolutions). Small values are faster. Large values are more accurate.


        dom_pos = event_data[:, :3]
        first_hit_times = event_data[:, 3]
        charges = event_data[:, 4]

        #logits, av, bv, geo_time, dist = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)
        logits, av, bv, geo_time = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)

        idx_padded = event_data[:, 0] != 0.0
        idx_padded_s = idx_padded.reshape((idx_padded.shape[0], 1))

        # replace padded values with some computable outputs that don't lead to nan.
        logits = jnp.where(idx_padded_s, logits, jnp.ones((1, 3)))
        av = jnp.where(idx_padded_s, av, jnp.ones((1, 3))+3.0)
        bv = jnp.where(idx_padded_s, bv, jnp.ones((1, 3))*1.e-3)

        mix_probs = jax.nn.softmax(logits)
        delay_time = first_hit_times - (geo_time + track_time)

        # Floor on negative time residuals.
        # Effectively a floor on the pdf.
        # Todo: think about noise.
        safe_delay_time = jnp.where(delay_time > -X_safe * sigma, delay_time, -X_safe * sigma)

        probs = c_multi_gamma_prob_v(safe_delay_time,
                                     mix_probs,
                                     av,
                                     bv,
                                     sigma,
                                     delta)

        sfs = c_multi_gamma_sf_v(safe_delay_time, mix_probs, av, bv, sigma)

        #dist = jnp.clip(dist, max=20)
        #limit = 3.6*jnp.exp(0.23*dist)+1.0
        #limit = dist
        #n_photons = jnp.clip(charges, max=limit)+1
        #n_photons = charges
        n_photons = charges

        mpe_log_probs = jnp.where(idx_padded,
								  jnp.log(n_photons) + jnp.log(probs) + (n_photons-1.0) * jnp.log(sfs),
								  jnp.array(0.0))

        return -2.0 * jnp.sum(mpe_log_probs)

    return neg_c_triple_gamma_llh
