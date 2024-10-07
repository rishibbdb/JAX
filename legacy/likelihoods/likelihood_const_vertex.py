import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


def get_neg_mpe_llh_const_vertex(eval_network_doms_and_track_fn, event_data, track_vertex, track_time, eps=jnp.float64(1.e-20), dtype=jnp.float64):

    @jax.jit
    def neg_mpe_llh_direction(track_direction):
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
