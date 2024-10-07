import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

### still under testing.
### this way of creating the likelihood would allow to vectorize over multiple events.
### i.e. directional reconstruction of multiple events in parallel on the gpu.
### see parallel_vectorized_over_events_minimizations.ipynb

def get_neg_mpe_llh_const_vertex_v2(eval_network_doms_and_track_fn, eps=jnp.float64(1.e-20), dtype=jnp.float64):

    @jax.jit
    def neg_mpe_llh_direction(track_direction, track_vertex, track_time, event_data):
        """
        track_direction: (zenith, azimuth) in radians
        track_vertex: (x, y, z)
        track_time: t (this time defines the fit vertex)
        event_data: 2D array (n_doms X 5) where columns are x,y,z of dom location, and t for first hit time, and estimated number of photon hits from Qtot.
        """

        dom_pos = event_data[:, :3]
        logits, av, bv, geo_time = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)

        gm = tfd.MixtureSameFamily(
                  mixture_distribution=tfd.Categorical(
                      logits=logits
                      ),
                  components_distribution=tfd.Gamma(
                    concentration=av,
                    rate=bv,
                    force_probs_to_zero_outside_support=True
                      )
                )

        first_hit_times = event_data[:, 3]
        n_photons = event_data[:, 4]

        delay_time = first_hit_times - (geo_time + track_time)
        llh = jnp.sum(jnp.log(n_photons * gm.prob(delay_time) * (1-gm.cdf(delay_time))**(n_photons-1) + eps))
        return -2*llh

    return neg_mpe_llh_direction


def get_neg_mpe_llh_const_vertex_v2_padded(eval_network_doms_and_track_fn, eps=jnp.float64(1.e-20), dtype=jnp.float64):

    @jax.jit
    def neg_mpe_llh_direction(track_direction, track_vertex, track_time, event_data):
        """
        track_direction: (zenith, azimuth) in radians
        track_vertex: (x, y, z)
        track_time: t (this time defines the fit vertex)
        event_data: 2D array (n_doms X 5) where columns are x,y,z of dom location, and t for first hit time, and estimated number of photon hits from Qtot.
        """

        dom_pos = event_data[:, :3]
        first_hit_times = event_data[:, 3]
        n_photons = event_data[:, 4]

        logits, av, bv, geo_time = eval_network_doms_and_track_fn(dom_pos, track_vertex, track_direction)

        delay_time = first_hit_times - (geo_time + track_time)
        delay_time = jnp.where(n_photons > 0, delay_time, 5.0)

        gm = tfd.MixtureSameFamily(
                  mixture_distribution=tfd.Categorical(
                      logits=logits
                      ),
                  components_distribution=tfd.Gamma(
                    concentration=av,
                    rate=bv,
                    force_probs_to_zero_outside_support=True
                      )
                )

        prob = jnp.where(n_photons > 0, n_photons * gm.prob(delay_time) * (1-gm.cdf(delay_time))**(n_photons-1) + eps, 1.)
        llh = jnp.sum(jnp.log(prob))

        return -2*llh

    return neg_mpe_llh_direction
