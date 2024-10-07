#!/usr/bin/env python

from iminuit import Minuit
import sys, os
sys.path.insert(0, "/home/storage/hans/jax_reco")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from jax.scipy import optimize

import pandas as pd
import numpy as np
from lib.simdata_i3 import I3SimHandlerFtr
from lib.geo import center_track_pos_and_time_based_on_data
from lib.network import get_network_eval_v_fn
from dom_track_eval import get_eval_network_doms_and_track
from likelihood_spe import get_neg_c_triple_gamma_llh

# Event Index.
event_index = 1

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

# Make MCTruth seed.
track_pos = jnp.array([meta['muon_pos_x'], meta['muon_pos_y'], meta['muon_pos_z']])
track_time = meta['muon_time']
track_zenith = meta['muon_zenith']
track_azimuth = meta['muon_azimuth']
track_src = jnp.array([track_zenith, track_azimuth])

print("original seed vertex:", track_pos)
centered_track_pos, centered_track_time = center_track_pos_and_time_based_on_data(event_data, track_pos, track_time, track_src)
print("shifted seed vertex:", centered_track_pos)

# Create some n_photons from qtot (by rounding up).
n_photons = np.round(event_data['charge'].to_numpy()+0.5)

# Combine into single data tensor for fitting.
fitting_event_data = jnp.array(event_data[['x', 'y', 'z', 'time']].to_numpy())

# Setup likelihood
neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track)

scale = 10.0
@jax.jit
def neg_llh_5D(x, track_time, data):
    return neg_llh(x[:2], x[2:]*scale, track_time, data)

@jax.jit
def minimize_bfgs(x0, track_time, data):
    result = optimize.minimize(neg_llh_5D,
                                x0,
                                args=(track_time, data),
                                method="BFGS",
                                options={'maxiter':50, 'gtol':1.e-5})
    return result.fun, result.x

x0 = jnp.concatenate([track_src, centered_track_pos/scale])
best_logl, best_x = minimize_bfgs(x0, centered_track_time, fitting_event_data)

print("... solution found.")
print(f"-2*logl={best_logl:.3f}")
print(f"zenith={best_x[0]:.3f}rad")
print(f"azimuth={best_x[1]:.3f}rad")
print(f"x={best_x[2]*scale:.3f}m")
print(f"y={best_x[3]*scale:.3f}m")
print(f"z={best_x[4]*scale:.3f}m")
print(f"at fix time t={centered_track_time:.3f}ns")


