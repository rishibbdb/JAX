from lib.cgamma import c_multi_gamma_prob_v
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
        X_safe = 20.0 # when to stop evaluating negative time residuals in units of sigma
        delta = 0.1 # how to combine the three regions that combine approximate and exact evaluation of hyp1f1 (required for convolutions). Small values are faster. Large values are more accurate.


        dom_pos = event_data[:, :3]
        first_hit_times = event_data[:, 3]
        logits, av, bv, geo_time = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)

        mix_probs = jax.nn.softmax(logits)
        delay_time = first_hit_times - (geo_time + track_time)

        # Floor on negative time residuals.
        # Effectively a floor on the pdf.
        # Todo: think about noise.
        safe_delay_time = jnp.where(delay_time > -X_safe * sigma, delay_time, -X_safe * sigma)

        return -2.0 * jnp.sum(jnp.log(c_multi_gamma_prob_v(safe_delay_time,
                                                           mix_probs,
                                                           av,
                                                           bv,
                                                           sigma,
                                                           delta)))

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
