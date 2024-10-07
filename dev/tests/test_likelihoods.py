import sys
sys.path.insert(0, "/home/storage/hans/jax_reco")

import jax.numpy as jnp
from jax.scipy import optimize
import jax
jax.config.update("jax_enable_x64", True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys, os

from lib.sim_data_i3 import I3SimHandlerFtr
from lib.plotting_tools import plot_event, adjust_plot_1d
from lib.geo import center_track_pos_and_time_based_on_data

from lib.network import get_network_eval_v_fn

from dom_track_eval import get_eval_network_doms_and_track
from time_sampler import sample_times_clean
from likelihood_const_vertex import get_neg_mpe_llh_const_vertex

# Event Index.
event_index = 20

# Random number seed.
key = jax.random.PRNGKey(2)

# Get network and eval logic.
eval_network_v = get_network_eval_v_fn(bpath='/home/storage/hans/jax_reco/data/network')
eval_network_doms_and_track = get_eval_network_doms_and_track(eval_network_v)

# Get an IceCube event.
bp = '/home/storage2/hans/i3files/21217'
sim_handler = I3SimHandlerFtr(os.path.join(bp, 'meta_ds_21217_from_35000_to_53530.ftr'),
                              os.path.join(bp, 'pulses_ds_21217_from_35000_to_53530.ftr'),
                              '/home/storage/hans/jax_reco/data/icecube/detector_geometry.csv')

meta, pulses = sim_handler.get_event_data(event_index)
print(f"muon energy: {meta['muon_energy_at_detector']/1.e3:.1f} TeV")

# Get dom locations, first hit times, and total charges (for each dom).
event_data = sim_handler.get_per_dom_summary_from_sim_data(meta, pulses)

print("n_doms", len(event_data))

# Produce and save an event view.
plot_event(event_data, geo=sim_handler.geo, outfile=f"event_view_{event_index}.png")

# Let's generate some new first hit times following our triple pandel model.
# (avoid problems with time smearing for now -> to be implemented: gaussian convoluted triple pandel.)
track_pos = jnp.array([meta['muon_pos_x'], meta['muon_pos_y'], meta['muon_pos_z']])
track_time = meta['muon_time']
track_zenith = meta['muon_zenith']
track_azimuth = meta['muon_azimuth']
track_src = jnp.array([track_zenith, track_azimuth])

print("old track vertex:", track_pos)
centered_track_pos, centered_track_time = center_track_pos_and_time_based_on_data(event_data, track_pos, track_time, track_src)
print("new track vertex:", centered_track_pos)

key, subkey = jax.random.split(key)
first_times = sample_times_clean(eval_network_doms_and_track, event_data, track_pos, track_src, track_time, subkey)

# Compare to original first hit times by plotting.
fig, ax = plt.subplots()
ax.scatter(event_data['time'], first_times, label='first hit')
plt.plot([0, 100000], [0, 100000], "r--")

plot_args = {'xlim':[9500, 13500],
                 'ylim':[9500, 13500],
                 'xlabel':'original time [ns]',
                 'ylabel':'resampled time [ns]'}

adjust_plot_1d(fig, ax, plot_args=plot_args)
plt.tight_layout()
plt.savefig(f"resampled_times_ev_{event_index}.png", dpi=300)

# Use these random times as fake data.
# Create some n_photons from qtot (by rounding up).
n_photons = np.round(event_data['charge'].to_numpy()+0.5)

# Combine into single data tensor for fitting.
fake_event_data = jnp.column_stack([
                                        jnp.array(event_data[['x', 'y', 'z']].to_numpy()),
                                        jnp.array(first_times),
                                        jnp.array(n_photons)
                                   ])

# Send to GPU.
fake_event_data.devices()
centered_track_pos.devices()
centered_track_time.devices()
track_src.devices()

del event_data

# Get LLH function.
neg_llh = get_neg_mpe_llh_const_vertex(eval_network_doms_and_track, fake_event_data, centered_track_pos, centered_track_time)
neg_llh_v = jax.jit(jax.vmap(neg_llh, 0, 0))

# And do a minimization.
@jax.jit
def minimize(x0):
    return optimize.minimize(neg_llh, x0, method="BFGS")

result = minimize(track_src)
print(result)

# And plot likelihood space.
n_eval = 100
zenith = np.linspace(track_src[0]-0.05, track_src[0]+0.05, n_eval)
azimuth = np.linspace(track_src[1]-0.05, track_src[1]+0.05, n_eval)
X, Y = np.meshgrid(zenith, azimuth)

init_dirs = np.column_stack([X.flatten(), Y.flatten()])
logls = neg_llh_v(init_dirs)

logls = logls.reshape(X.shape)

fig, ax = plt.subplots()
min_logl = np.amin(logls)
delta_logl = logls - np.amin(logls)
pc = ax.pcolormesh(np.rad2deg(X), np.rad2deg(Y), delta_logl, vmin=0, vmax=np.min([1.2*np.amax(delta_logl)]), shading='auto')
cbar = fig.colorbar(pc)
cbar.ax.tick_params(labelsize=16)
cbar.ax.get_yaxis().labelpad = 5
cbar.set_label("-2$\\Delta$log $L_{MPE}$", fontsize=20)
cbar.outline.set_linewidth(1.5)

ax.set_xlabel("zenith [deg]", fontsize=16)
ax.set_ylabel("azimuth [deg]", fontsize=16)
ax.set_xlim(np.rad2deg([track_src[0]-0.05, track_src[0]+0.05]))
ax.set_ylim(np.rad2deg([track_src[1]-0.05, track_src[1]+0.05]))
ax.tick_params(axis='both', which='both', width=1.5, colors='0.0', labelsize=16)

ax.scatter(np.rad2deg(track_src[0]), np.rad2deg(track_src[1]), marker="*", color='red', label="True")

b_zenith = result.x[0]
b_azimuth = result.x[1]
ax.scatter(np.rad2deg([b_zenith]), np.rad2deg([b_azimuth]), marker="x", color='cyan', label='2D-fit')
plt.legend()
plt.tight_layout()
plt.savefig(f"likelihood_space_ev_{event_index}.png", dpi=300)

