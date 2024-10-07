import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
from typing import Any


def sample_times_clean(eval_network_doms_track_fn: Any,
                event_data: pd.DataFrame,
                track_pos: jnp.array,
                track_dir: jnp.array,
                track_time: jnp.float64,
                prng_key: jax.random.key) -> jnp.array:
    """
    track_dir: zenith, azimuth of src direction in radians
    """

    dom_positions = jnp.array(event_data[['x', 'y', 'z']].to_numpy())
    logits, av, bv, geo_time = eval_network_doms_track_fn(dom_positions,
                                                track_pos,
                                                track_dir)

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

    charges = np.round(event_data['charge'].to_numpy()+0.5).astype(int)
    n_tot = np.amax(charges)
    times = np.array(gm.sample(sample_shape=n_tot, seed=prng_key))

    n_doms = times.shape[1]
    first_times = np.zeros(n_doms)

    for i in range(n_doms):
        first_times[i] = np.amin(times[:charges[i], i])

    return jnp.array(first_times + geo_time + track_time, dtype=jnp.float64)


from lib.geo import cherenkov_cylinder_coordinates_v
from lib.geo import rho_dom_relative_to_track_v
from lib.trafos import transform_network_outputs_v, transform_network_inputs_v
from lib.geo import get_xyz_from_zenith_azimuth

def sample_times(event_data: pd.DataFrame,
                track_pos: jnp.array,
                track_dir: jnp.array,
                track_time: jnp.float64,
                network: Any,
                prng_key: jax.random.key) -> jnp.array:
    """
    track_dir: zenith, azimuth of src direction in radians
    """

    dom_positions = jnp.array(event_data[['x', 'y', 'z']].to_numpy())
    charges = np.round(event_data['charge'].to_numpy()+0.5).astype(int)
    track_dir_xyz = get_xyz_from_zenith_azimuth(track_dir)

    geo_time, closest_approach_dist, closest_approach_z = \
    cherenkov_cylinder_coordinates_v(dom_positions,
                                     track_pos,
                                     track_dir_xyz)

    closest_approach_rho = rho_dom_relative_to_track_v(dom_positions, track_pos, track_dir_xyz)

    track_zenith = track_dir[0]
    track_azimuth = track_dir[1]
    x = jnp.column_stack([closest_approach_dist,
                      closest_approach_rho,
                      closest_approach_z,
                      jnp.repeat(track_zenith, len(closest_approach_dist)),
                      jnp.repeat(track_azimuth, len(closest_approach_dist))])

    x_prime = transform_network_inputs_v(x)
    y_pred = network.eval_on_batch(x_prime)
    logits, av, bv = transform_network_outputs_v(y_pred)

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

    n_tot = np.amax(charges)
    times = np.array(gm.sample(sample_shape = n_tot, seed = prng_key))

    n_doms = times.shape[1]
    first_times = np.zeros(n_doms)

    for i in range(n_doms):
        first_times[i] = np.amin(times[:charges[i], i])

    return first_times + geo_time + track_time
