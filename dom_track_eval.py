import jax
import jax.numpy as jnp

from lib.geo import cherenkov_cylinder_coordinates_w_rho_v, cherenkov_cylinder_coordinates_w_rho2_v
from lib.geo import rho_dom_relative_to_track_v
from lib.geo import get_xyz_from_zenith_azimuth
from lib.trafos import transform_network_outputs_v, transform_network_inputs_v


def get_eval_network_doms_and_track(eval_network_v_fn, dtype=jnp.float64):
    """
    network eval function (vectorized across doms)
    """

    @jax.jit
    def eval_network_doms_and_track(dom_pos, track_vertex, track_dir):
        """
        track_direction: (zenith, azimuth) in radians
        track_vertex: (x, y, z)
        dom_pos: 2D array (n_doms X 3) where columns are x,y,z of dom location
        """
        track_dir_xyz = get_xyz_from_zenith_azimuth(track_dir)

        geo_time, closest_approach_dist, closest_approach_z, closest_approach_rho = \
            cherenkov_cylinder_coordinates_w_rho_v(dom_pos,
                                         track_vertex,
                                         track_dir_xyz)

        track_zenith = track_dir[0]
        track_azimuth = track_dir[1]
        x = jnp.column_stack([closest_approach_dist,
                          closest_approach_rho,
                          closest_approach_z,
                          jnp.repeat(track_zenith, len(closest_approach_dist)),
                          jnp.repeat(track_azimuth, len(closest_approach_dist))])

        # Cast to dtype. Enables network evaluation in fp32.
        # Which is significantly faster for consumer gpus
        # at essentially no loss of accuracy.
        x = jnp.array(x, dtype=dtype)

        x_prime = transform_network_inputs_v(x)
        y_pred = eval_network_v_fn(x_prime)
        logits, av, bv = transform_network_outputs_v(y_pred)

        # Cast to float64. Likelihoods need double precision.
        logits = jnp.array(logits, dtype=jnp.float64)
        av = jnp.array(av, dtype=jnp.float64)
        bv = jnp.array(bv, dtype=jnp.float64)
        geo_time = jnp.array(geo_time, dtype=jnp.float64)

        return logits, av, bv, geo_time

    return eval_network_doms_and_track


def get_eval_network_doms_and_track_w_charge(eval_network_v_fn, eval_charge_network_v_fn, dtype=jnp.float64):
    """
    network eval function (vectorized across doms)
    """

    @jax.jit
    def eval_network_doms_and_track(dom_pos, track_vertex, track_dir):
        """
        track_direction: (zenith, azimuth) in radians
        track_vertex: (x, y, z)
        dom_pos: 2D array (n_doms X 3) where columns are x,y,z of dom location
        """
        track_dir_xyz = get_xyz_from_zenith_azimuth(track_dir)

        geo_time, closest_approach_dist, closest_approach_z, closest_approach_rho = \
            cherenkov_cylinder_coordinates_w_rho_v(dom_pos,
                                         track_vertex,
                                         track_dir_xyz)

        track_zenith = track_dir[0]
        track_azimuth = track_dir[1]
        x = jnp.column_stack([closest_approach_dist,
                          closest_approach_rho,
                          closest_approach_z,
                          jnp.repeat(track_zenith, len(closest_approach_dist)),
                          jnp.repeat(track_azimuth, len(closest_approach_dist))])

        # Cast to dtype. Enables network evaluation in fp32.
        # Which is significantly faster for consumer gpus
        # at essentially no loss of accuracy.
        x = jnp.array(x, dtype=dtype)

        x_prime = transform_network_inputs_v(x)
        y_pred = eval_network_v_fn(x_prime)
        logits, av, bv = transform_network_outputs_v(y_pred)

        predicted_charge = eval_charge_network_v_fn(x_prime)

        # Cast to float64. Likelihoods need double precision.
        logits = jnp.array(logits, dtype=jnp.float64)
        av = jnp.array(av, dtype=jnp.float64)
        bv = jnp.array(bv, dtype=jnp.float64)
        geo_time = jnp.array(geo_time, dtype=jnp.float64)

        return logits, av, bv, geo_time, predicted_charge

    return eval_network_doms_and_track
