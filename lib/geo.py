import jax.numpy as jnp
import jax
import numpy as np
import pandas as pd

__n_ice_phase = 1.30799291638281
__n_ice_group = 1.32548384613875
__n_ice = __n_ice_group
__theta_cherenkov = np.arccos(1/__n_ice_phase)
__sin_theta_cherenkov = np.sin(__theta_cherenkov)
__tan_theta_cherenkov = np.tan(__theta_cherenkov)
__c = 0.299792458 # m / ns
__c_ice = __c/__n_ice_group

def center_track_pos_and_time_based_on_data(event_data: pd.DataFrame, track_pos, track_time, track_dir):
    track_dir_xyz = get_xyz_from_zenith_azimuth(track_dir)
    centered_track_time = np.sum(event_data['charge'] * event_data['time']) / np.sum(event_data['charge'])
    centered_track_pos = track_pos + (centered_track_time - track_time) * __c * track_dir_xyz
    return jnp.array(centered_track_pos), jnp.float64(centered_track_time)

def center_track_pos_and_time_based_on_data_batched(data, mctruth):
    track_dir = mctruth[2:4]
    track_time = mctruth[4]
    track_pos = mctruth[5:8]

    track_dir_xyz = get_xyz_from_zenith_azimuth(track_dir)
    charge = data[:, 4]
    time = data[:, 3]

    centered_track_time = np.sum(charge * time) / np.sum(charge)
    centered_track_pos = track_pos + (centered_track_time - track_time) * __c * track_dir_xyz
    return jnp.array(centered_track_pos), jnp.float64(centered_track_time)

center_track_pos_and_time_based_on_data_batched_v = jax.jit(jax.vmap(center_track_pos_and_time_based_on_data_batched, (0, 0), (0, 0)))


def geo_time(dom_pos, track_pos, track_dir):
    """
    roughly following https://github.com/icecube/icetray/blob/dde656a29dbd8330e5f54f9260550952f0269bc9/phys-services/private/phys-services/I3Calculator.cxx#L19

    dom_pos: 1D jax array with 3 components [x, y, z]
    track_pos: 1D jax array with 3 components [x, y, z]
    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
    """
    # vector from vertex to dom
    v_a = dom_pos - track_pos

    # distance muon travels from track vertex to point of closest approach.
    ds = jnp.dot(v_a, track_dir)

    # a vector parallel track with length ds
    ds_v = ds * track_dir

    # vector closest approach position to dom yields closest approach distance
    v_d = v_a - ds_v
    dc = jnp.linalg.norm(v_d)

    # distance that the photon travels
    dt = dc / __sin_theta_cherenkov

    # distance emission point to closest approach point
    dx = dc / __tan_theta_cherenkov

    return (ds - dx + dt * __n_ice_group) / __c

geo_time_v = jax.jit(jax.vmap(geo_time, (0, None, None), 0))


def cherenkov_cylinder_coordinates(dom_pos, track_pos, track_dir):
    """
    dom_pos: 1D jax array with 3 components [x, y, z]
    track_pos: 1D jax array with 3 components [x, y, z]
    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
    """
    # vector from vertex to dom
    v_a = dom_pos - track_pos

    # distance muon travels from track vertex to point of closest approach.
    ds = jnp.dot(v_a, track_dir)

    # a vector parallel track with length ds
    ds_v = ds * track_dir

    # vector closest approach position to dom yields closest approach distance
    v_d = v_a - ds_v
    dc = jnp.linalg.norm(v_d)

    # vector to closest approach position gives z-component
    v_c = track_pos + ds_v
    v_c_z = v_c[2]

    # distance that the photon travel
    dt = dc / __sin_theta_cherenkov

    # distance emission point to closest approach point
    dx = dc / __tan_theta_cherenkov

    ### missing: add last return value -> rho angle of track
    return (ds - dx + dt * __n_ice_group) / __c, dc, v_c_z


cherenkov_cylinder_coordinates_v = jax.jit(jax.vmap(cherenkov_cylinder_coordinates, (0, None, None), (0, 0, 0)))


def closest_distance_dom_track(dom_pos, track_pos, track_dir):
    """
    dom_pos: 1D jax array with 3 components [x, y, z]
    track_pos: 1D jax array with 3 components [x, y, z]
    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
    """

    # vector track vertex -> dom
    v_a = dom_pos - track_pos
    # vector: closest point on track -> dom
    v_d = v_a - jnp.dot(v_a, track_dir) * track_dir
    dist = jnp.linalg.norm(v_d)
    return dist

# Generalize to matrix input for dom_pos with shape (N_DOMs, 3).
# Output will be in form of (N_DOMs, 1)
closest_distance_dom_track_v = jax.jit(jax.vmap(closest_distance_dom_track, (0, None, None), 0))


def convert_spherical_to_cartesian_direction(x):
    """
    x = (theta, phi)
    """
    track_theta = x[0]
    track_phi = x[1]
    track_dir_x = jnp.sin(track_theta) * jnp.cos(track_phi)
    track_dir_y = jnp.sin(track_theta) * jnp.sin(track_phi)
    track_dir_z = jnp.cos(track_theta)
    direction = jnp.array([track_dir_x, track_dir_y, track_dir_z])
    return direction

# Generalize to matrix input for x with shape (N_DOMs, 2) for theta and phi angles.
# Output will be in form of (N_DOMs, 3) for dir_x, dir_y, dir_z
convert_spherical_to_cartesian_direction_v = jax.jit(jax.vmap(closest_distance_dom_track, 0, 0))


def get_xyz_from_zenith_azimuth(x):
    track_dir = convert_spherical_to_cartesian_direction(x)
    y = -1 * track_dir
    return y

get_xyz_from_zenith_azimuth_v = jax.jit(jax.vmap(get_xyz_from_zenith_azimuth, 0, 0))


def light_travel_time_i3calculator(dom_pos, track_pos, track_dir):
    """
    roughly following https://github.com/icecube/icetray/blob/dde656a29dbd8330e5f54f9260550952f0269bc9/phys-services/private/phys-services/I3Calculator.cxx#L19

    dom_pos: 1D jax array with 3 components [x, y, z]
    track_pos: 1D jax array with 3 components [x, y, z]
    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
    """
    dc = closest_distance_dom_track(dom_pos, track_pos, track_dir)

    # vector track support point -> dom
    v_a = dom_pos - track_pos

    # distance muon travels from support point to point of closest approach.
    ds = jnp.dot(v_a, track_dir)

    # distance that the photon travels
    dt = dc / __sin_theta_cherenkov

    # distance emission point to closest approach point
    dx = dc / __tan_theta_cherenkov

    return (ds - dx + dt * __n_ice_group) / __c

light_travel_time_i3calculator_v = jax.jit(jax.vmap(light_travel_time_i3calculator, (0, None, None), 0))


def closest_point_on_track(dom_pos, track_pos, track_dir):
    """
    dom_pos: 1D jax array with 3 components [x, y, z]
    track_pos: 1D jax array with 3 components [x, y, z]
    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
    """

    # vector track support point -> dom
    v_a = dom_pos - track_pos
    # vector: vector to closest point on track
    v_c = track_pos + jnp.dot(v_a, track_dir) * track_dir
    return v_c

closest_point_on_track_v = jax.jit(jax.vmap(closest_point_on_track, (0, None, None), 0))


def z_component_closest_point_on_track(dom_pos, track_pos, track_dir):
    """
    dom_pos: 1D jax array with 3 components [x, y, z]
    track_pos: 1D jax array with 3 components [x, y, z]
    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
    """

    # vector track support point -> dom
    v_a = dom_pos - track_pos
    # vector: vector to closest point on track
    v_c = track_pos + jnp.dot(v_a, track_dir) * track_dir
    return v_c[2]

z_component_closest_point_on_track_v = jax.jit(jax.vmap(z_component_closest_point_on_track, (0, None, None), 0))


def rho_dom_relative_to_track(dom_pos, track_pos, track_dir):
    """
    clean up and verify!

    dom_pos: 1D jax array with 3 components [x, y, z]
    track_pos: 1D jax array with 3 components [x, y, z]
    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
    """
    v1 = dom_pos - track_pos
    closestapproach = track_pos + jnp.dot(v1, track_dir)*track_dir
    v2 = dom_pos - closestapproach
    zdir = jnp.cross(track_dir, jnp.cross(jnp.array([0,0,1]), track_dir))
    positivedir = jnp.cross(track_dir, zdir)
    ypart = v2-zdir*jnp.dot(zdir, v2)
    zpart = v2-ypart
    z = jnp.dot(zpart, zdir)
    y = jnp.dot(ypart, positivedir)
    return jnp.arctan2(y,z)

rho_dom_relative_to_track_v = jax.jit(jax.vmap(rho_dom_relative_to_track, (0, None, None), 0))


def get_perpendicular_dir(track_dir):
    '''
    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
    '''
    dirx = track_dir[0]
    diry = track_dir[1]
    dirz = track_dir[2]

    perpz = jnp.where(jnp.logical_or(dirz == 1.0, dirz == -1.0), 0.0, jnp.sqrt(dirx**2 + diry**2))
    perpx = -dirx * dirz/perpz
    perpy = -diry * dirz/perpz
    return jnp.array([perpx, perpy, perpz])


def cherenkov_cylinder_coordinates_w_rho(dom_pos, track_pos, track_dir):
    """
    dom_pos: 1D jax array with 3 components [x, y, z]
    track_pos: 1D jax array with 3 components [x, y, z]
    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
    """
    # vector from vertex to dom
    v_a = dom_pos - track_pos

    # distance muon travels from track vertex to point of closest approach.
    ds = jnp.dot(v_a, track_dir)

    # a vector parallel track with length ds
    ds_v = ds * track_dir

    # vector closest approach position to dom yields closest approach distance
    v_d = v_a - ds_v
    dc = jnp.linalg.norm(v_d)

    # vector to closest approach position gives z-component
    v_c = track_pos + ds_v
    v_c_z = v_c[2]

    # distance that the photon travel
    dt = dc / __sin_theta_cherenkov

    # distance emission point to closest approach point
    dx = dc / __tan_theta_cherenkov

    # compute rho angle, following I3PhotonicsService
	# https://github.com/icecube/icetray/blob/e117b063b1340dc565a7459e31d8307bcf0b05b5/photonics-service/private/photonics-service/I3PhotonicsService.cxx#L152
    perp_dir = get_perpendicular_dir(track_dir)
    cos_rho = jnp.dot(v_d, perp_dir) / dc
    cos_rho = jnp.where(dc > 0.0, cos_rho, 0.0)
    rho = jnp.arccos(cos_rho)

    rhosign = jnp.dot(jnp.cross(v_d, perp_dir), track_dir)
    rho = jnp.where(rhosign <= 0.0, rho, 2.0*jnp.pi-rho)
    rho = jnp.where(rho <= jnp.pi, rho, rho-2.0*jnp.pi)

    return (ds - dx + dt * __n_ice_group) / __c, dc, v_c_z, rho

cherenkov_cylinder_coordinates_w_rho_v = jax.jit(jax.vmap(cherenkov_cylinder_coordinates_w_rho, (0, None, None), (0, 0, 0, 0)))


def cherenkov_cylinder_coordinates_w_rho2(dom_pos, track_pos, track_dir):
    """
    dom_pos: 1D jax array with 3 components [x, y, z]
    track_pos: 1D jax array with 3 components [x, y, z]
    track_dir: 1D jax array with 3 components [dir_x, dir_y, dir_z]
    """
    # vector from vertex to dom
    v_a = dom_pos - track_pos

    # distance muon travels from track vertex to point of closest approach.
    ds = jnp.dot(v_a, track_dir)

    # a vector parallel track with length ds
    ds_v = ds * track_dir

    # vector closest approach position to dom yields closest approach distance
    v_d = v_a - ds_v
    dc = jnp.linalg.norm(v_d)

    # vector to closest approach position gives z-component
    v_c = track_pos + ds_v
    v_c_z = v_c[2]

    # distance that the photon travel
    dt = dc / __sin_theta_cherenkov

    # distance emission point to closest approach point
    dx = dc / __tan_theta_cherenkov

    # compute rho angle, following I3PhotonicsService
    zdir = jnp.cross(track_dir, jnp.cross(jnp.array([0,0,1]), track_dir))
    positivedir = jnp.cross(track_dir, zdir)
    z = jnp.dot(zdir, v_d)
    ypart = v_d-zdir*z
    y = jnp.dot(positivedir, ypart)
    rho = jnp.arctan2(y,z)
    rho = jnp.where(dc > 0.0, rho, 0.0)
    return (ds - dx + dt * __n_ice_group) / __c, dc, v_c_z, rho

cherenkov_cylinder_coordinates_w_rho2_v = jax.jit(jax.vmap(cherenkov_cylinder_coordinates_w_rho2, (0, None, None), (0, 0, 0, 0)))
