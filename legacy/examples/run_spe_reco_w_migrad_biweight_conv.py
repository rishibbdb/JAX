#!/usr/bin/env python

from iminuit import Minuit
import sys, os
sys.path.insert(0, "/home/storage/hans/jax_reco_new")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

import pandas as pd
import numpy as np
from lib.simdata_i3 import I3SimHandler
from lib.geo import center_track_pos_and_time_based_on_data
from lib.network import get_network_eval_v_fn
from dom_track_eval import get_eval_network_doms_and_track
from likelihood_spe_biweight_conv import get_neg_c_triple_gamma_llh, get_llh_for_iminuit_migrad
from lib.experimental_methods import remove_early_pulses

# Event Index.
event_index = 0

# Get network and eval logic.
eval_network_v = get_network_eval_v_fn(bpath='/home/storage/hans/jax_reco/data/network')
eval_network_doms_and_track = get_eval_network_doms_and_track(eval_network_v)

# Get an IceCube event.
bp = '/home/storage2/hans/i3files/21217'
sim_handler = I3SimHandler(os.path.join(bp, 'meta_ds_21217_from_35000_to_53530.ftr'),
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
fitting_event_data_unclean = jnp.array(event_data[['x', 'y', 'z', 'time']].to_numpy())
print(fitting_event_data_unclean.shape)
fitting_event_data = remove_early_pulses(eval_network_doms_and_track,
                                        fitting_event_data_unclean,
                                        centered_track_pos,
                                        track_src,
                                        centered_track_time)
print(fitting_event_data.shape)

obj_fn = get_llh_for_iminuit_migrad(eval_network_doms_and_track)

# put the thing below into a for loop if you want to reconstruct many events (without jit-recompiling everything)
f_prime = lambda x: obj_fn(x, centered_track_time, fitting_event_data)

x0 = jnp.concatenate([track_src, centered_track_pos])
print(f_prime(x0))

m = Minuit(f_prime, x0)
m.errordef = Minuit.LIKELIHOOD
m.limits = ((0.0, np.pi), (0.0, 2.0 * np.pi), (-500.0, 500.0),  (-500.0, 500.0),  (-500.0, 500.0))
m.strategy = 0
m.migrad()

print("... solution found.")
print(f"-2*logl={m.fval:.3f}")
print(f"zenith={m.values[0]:.3f}rad")
print(f"azimuth={m.values[1]:.3f}rad")
print(f"x={m.values[2]:.3f}m")
print(f"y={m.values[3]:.3f}m")
print(f"z={m.values[4]:.3f}m")
print(f"at fix time t={centered_track_time:.3f}ns")


