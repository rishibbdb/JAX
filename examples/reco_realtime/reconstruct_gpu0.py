import sys, os
sys.path.insert(0, "/home/storage/hans/jax_reco_new")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

from likelihood_mpe_padded_input_tests import get_neg_c_triple_gamma_llh
from lib.geo import get_xyz_from_zenith_azimuth, __c
from dom_track_eval import get_eval_network_doms_and_track as get_eval_network_doms_and_track
import time

dtype = jnp.float32
eval_network_v = get_network_eval_v_fn(bpath='/home/storage/hans/jax_reco/data/network',
                                       dtype=dtype)
eval_network_doms_and_track = get_eval_network_doms_and_track(eval_network_v, dtype=dtype)


event_ids = ['1022', '10393', '10644', '10738', '11086', '11232', '13011',
       '13945', '14017', '14230', '15243', '16416', '16443', '1663',
       '1722', '17475', '18846', '19455', '20027', '21113', '21663',
       '22232', '22510', '22617', '23574', '23638', '23862', '24530',
       '24726', '25181', '25596', '25632', '27063', '27188', '27285',
       '28188', '28400', '29040', '29707', '3062', '31920', '31989',
       '32781', '32839', '33119', '33656', '34506', '35349', '37086',
       '37263', '37448', '37786', '37811', '39166', '39962', '40023',
       '41381', '41586', '42566', '42568', '42677', '43153', '43483',
       '4397', '44081', '48309', '48448', '48632', '49067', '50832',
       '51687', '51956', '54374', '55301', '55526', '55533', '56041',
       '5620', '56741', '56774', '57174', '57394', '57723', '59010',
       '59029', '59089', '59099', '59228', '62274', '62512', '63373',
       '65472', '6586', '8', '8604', '8674', '8840', '9410', '9419',
       '9505']


for event_id in event_ids[:50]:
    try:
        bp = '/home/storage2/hans/i3files/alerts/bfrv2'
        tfrecord = os.path.join(bp, f'event_{event_id}_N100_merged_w_energy_loss_preds_from_0_to_10_1st_pulse_sigma_0.7.tfrecord')
        batch_maker = I3SimBatchHandlerTFRecord(tfrecord, batch_size=100)
        batch_iter = batch_maker.get_batch_iterator()
        data, mctruth = batch_iter.next()
        data, mctruth = (jnp.array(data), jnp.array(mctruth))

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

        # make logl calculation work on a batch.
        neg_llh_one_batch = jax.jit(jax.vmap(neg_llh, (0, 0, 0, 0), 0))

        def reconstruct_one_batch(data, mctruth):
            data_clean_padded = clean_pulses_fn_v(data, mctruth)
            centered_track_positions, centered_track_times = \
                    center_track_pos_and_time_based_on_data_batched_v(data_clean_padded, mctruth)

            track_src_v = mctruth[:, 2:4]

            true_logl = neg_llh_one_batch(track_src_v,
                                          centered_track_positions,
                                          centered_track_times,
                                          data_clean_padded)

            result_logl, result_x = optimize_one_batch(data_clean_padded,
                                                   track_src_v,
                                                   centered_track_times,
                                                   centered_track_positions)

            return true_logl - result_logl, result_x


        print(f"processing batch with shape ({data.shape[0]}, {data.shape[1]}, {data.shape[2]})")
        charge = jnp.round(data[..., 4]+0.5)
        data = jnp.concatenate([data[..., :4], jnp.expand_dims(charge, axis=-1)], axis=-1)
        tic = time.time()
        delta_logl, result_x = reconstruct_one_batch(data, mctruth)
        toc = time.time()
        y = jnp.column_stack([mctruth, result_x, delta_logl])
        print(f"took {toc-tic:.1f}s.")

        # store results.
        np.save(f"/home/storage/hans/jax_reco_new/examples/reco_realtime/results/reco_result_{event_id}_tfrecord.npy", y)
    except:
        continue





