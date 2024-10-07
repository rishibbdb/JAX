import sys, os
sys.path.insert(0, "/home/storage/hans/jax_reco_new")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import jax.numpy as jnp
from jax.scipy import optimize
import jax
jax.config.update("jax_enable_x64", True)
import optimistix as optx

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lib.simdata_i3 import I3SimBatchHandlerTFRecord
from lib.geo import center_track_pos_and_time_based_on_data_batched_v
from lib.experimental_methods import get_clean_pulses_fn_v
from lib.network import get_network_eval_v_fn

from likelihood_spe_padded_input import get_neg_c_triple_gamma_llh
from lib.geo import get_xyz_from_zenith_azimuth, __c
from dom_track_eval import get_eval_network_doms_and_track_altrho as get_eval_network_doms_and_track
import time

dtype = jnp.float32
eval_network_v = get_network_eval_v_fn(bpath='/home/storage/hans/jax_reco/data/network',
                                       dtype=dtype)
eval_network_doms_and_track = get_eval_network_doms_and_track(eval_network_v, dtype=dtype)

# Create padded batches (with different seq length).
tfrecord = "/home/storage2/hans/i3files/21220/data_ds_21220_from_0_to_10000_10_to_100TeV.tfrecord"
batch_maker = I3SimBatchHandlerTFRecord(tfrecord, batch_size=4096)
batch_iter = batch_maker.get_batch_iterator()

# Until LLH has a noise-term, we need to remove crazy early noise pulses
clean_pulses_fn_v = get_clean_pulses_fn_v(eval_network_doms_and_track)

# And set up likelihood for batched processing
neg_llh = get_neg_c_triple_gamma_llh(eval_network_doms_and_track)

scale = 20.0
@jax.jit
def neg_llh_5D(x, args):
        centered_track_time = args[0]
        fitting_event_data = args[1]

        # project back if outside of [0, pi] x [0, 2*pi]
        zenith = x[0] / scale
        azimuth = x[1] / scale
        zenith = jnp.fmod(zenith, 2.0*jnp.pi)
        zenith = jnp.where(zenith < 0, zenith+2.0*jnp.pi, zenith)
        cond = zenith > jnp.pi
        zenith = jnp.where(cond, -1.0*zenith+2.0*jnp.pi, zenith)
        azimuth = jnp.where(cond, azimuth-jnp.pi, azimuth)

        azimuth = jnp.fmod(azimuth, 2.0*jnp.pi)
        azimuth = jnp.where(azimuth < 0, azimuth+2.0*jnp.pi, azimuth)

        projected_dir = jnp.array([zenith, azimuth])
        return neg_llh(projected_dir, x[2:]*scale, centered_track_time, fitting_event_data)

# optimization logic for one event.
def optimize_one_event(data, track_src, centered_track_time, centered_track_pos):
    args=[centered_track_time, data]
    solver = optx.BFGS(rtol=1e-8, atol=1e-4, use_inverse=True)
    x0 = jnp.concatenate([track_src*scale, centered_track_pos/scale])
    best_x = optx.minimise(neg_llh_5D, solver, x0, args=args, throw=False).value
    best_logl = neg_llh_5D(best_x, args=args)
    return best_logl, best_x

# make it work on a batch.
optimize_one_batch = jax.jit(jax.vmap(optimize_one_event, (0, 0, 0, 0), (0, 0)))

def reconstruct_one_batch(data, mctruth):
    data_clean_padded = clean_pulses_fn_v(data, mctruth)
    centered_track_positions, centered_track_times = \
            center_track_pos_and_time_based_on_data_batched_v(data_clean_padded, mctruth)
    track_src_v = mctruth[:, 2:4]
    result_logl, result_x = optimize_one_batch(data_clean_padded,
                                           track_src_v,
                                           centered_track_times,
                                           centered_track_positions)
    return result_x

# main loop over batches
n_batches = 10

results = []
for i in range(n_batches):
    try:
        data, mctruth = batch_iter.next() # [Nev, Ndom, Nobs], [Nev, Naux]
        print(f"processing batch with shape ({data.shape[0]}, {data.shape[1]}, {data.shape[2]})")
        data = jnp.array(data.numpy())
        mctruth = jnp.array(mctruth.numpy())
        tic = time.time()
        result_x = reconstruct_one_batch(data, mctruth)
        toc = time.time()
        y = jnp.column_stack([mctruth, result_x])
        results.append(y)
        print(f"took {toc-tic:.1f}s.")

    except Exception as e:
        print(e)
        print("Reached end of batch iterator early. Stopping reconstruction here.")
        break

# store results.
results = jnp.concatenate(results)
np.save("reco_result_tfrecord_altrho.npy", results)

