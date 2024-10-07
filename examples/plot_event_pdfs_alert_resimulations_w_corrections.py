import sys, os
sys.path.insert(0, "/home/storage/hans/jax_reco_new")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from collections import defaultdict

import jax.numpy as jnp
from jax.scipy import optimize
import jax
jax.config.update("jax_enable_x64", True)
import optimistix as optx

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from lib.simdata_i3 import I3SimBatchHandlerTFRecord
from lib.simdata_i3 import I3SimHandler

from lib.experimental_methods import get_clean_pulses_fn_v
from lib.network import get_network_eval_v_fn
from lib.geo import cherenkov_cylinder_coordinates_w_rho_v as cherenkov_cylinder_coordinates_w_rho_v
from lib.geo import get_xyz_from_zenith_azimuth
from lib.cgamma import c_multi_gamma_prob, c_multi_gamma_sf
from lib.plotting import adjust_plot_1d

from dom_track_eval import get_eval_network_doms_and_track2 as get_eval_network_doms_and_track

import time

from collections import defaultdict

dtype = jnp.float32
eval_network_v = get_network_eval_v_fn(bpath='/home/storage/hans/jax_reco_new/data/network',
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

def make_event_plot(event_id, sigma):
    global eval_network_doms_and_track, geo
    bp = '/home/storage2/hans/i3files/alerts/bfrv2/'
    sim_handler = I3SimHandler(os.path.join(bp, f'meta_ds_event_{event_id}_N100_from_0_to_100_1st_pulse_charge_correction.ftr'),
                              os.path.join(bp, f'pulses_ds_event_{event_id}_N100_from_0_to_100_1st_pulse_charge_correction.ftr'),
                              '/home/storage/hans/jax_reco/data/icecube/detector_geometry.csv')

    pdf = PdfPages(f"pdfs_w_corr_{event_id}_sigma_{sigma:.1f}_dist_r0.5.pdf")

    hit_x = []
    hit_y = []
    hit_z = []
    hit_t = []
    hit_q = []
    hit_q_corr = []

    for i in range(100):
        try:
            meta, pulses = sim_handler.get_event_data(i)
        except:
            continue

        #print(f"muon energy: {meta['muon_energy_at_detector']/1.e3:.1f} TeV")

        track_time = meta['muon_time']
        track_pos = [meta['muon_pos_x'], meta['muon_pos_y'], meta['muon_pos_z']]
        track_dir = [meta['muon_zenith'], meta['muon_azimuth']]

        # Get dom locations, first hit times, and total charges (for each dom).
        event_data = sim_handler.get_per_dom_summary_from_sim_data(meta, pulses)
        event_data_corr = sim_handler.get_per_dom_summary_from_sim_data(meta, pulses,
                                                                        charge_key = f'corrected_charge_{sigma:.1f}')

        if i == 0:

            # Plot event.
            fig = plt.figure(figsize=(8,6))
            ax = plt.subplot(projection='3d')
            ax.set_xlabel('pos.x [m]', fontsize=16, labelpad=-25)
            ax.set_ylabel('pos.y [m]', fontsize=16, labelpad=-25)
            ax.set_zlabel('pos.z [m]', fontsize=16, labelpad=-25)
            ax.set_title(f'event {event_id}')

            df = event_data
            idx = df['charge'] > 0
            geo = sim_handler.geo

            try:
                im = ax.scatter(geo['x'], geo['y'], geo['z'], s=0.5, c='0.7', alpha=0.4)
            except:
                pass

            im = ax.scatter(df[idx]['x'], df[idx]['y'], df[idx]['z'], s=np.sqrt(df[idx]['charge']*100), c=df[idx]['time'],
                            cmap='rainbow_r',  edgecolors='k', zorder=1000)
            ax.tick_params(axis='both', which='both', width=1.5, colors='0.0', labelsize=16)
            cb = plt.colorbar(im, orientation="vertical", pad=0.1)
            cb.set_label(label='time [ns]', size='x-large')
            cb.ax.tick_params(labelsize='x-large')
            pdf.savefig(fig)

        hit_t.append(event_data['time'].values - track_time)
        hit_x.append(event_data['x'].values)
        hit_y.append(event_data['y'].values)
        hit_z.append(event_data['z'].values)
        hit_q.append(event_data['charge'].values)
        hit_q_corr.append(event_data_corr['charge'].values)

    dom_data = defaultdict(lambda: defaultdict(list))

    for i in range(len(hit_x)):
        for j in range(len(hit_x[i])):
            q = hit_q[i][j]
            q_corr = hit_q_corr[i][j]

            x = hit_x[i][j]
            y = hit_y[i][j]
            z = hit_z[i][j]

            t = hit_t[i][j]
            dom_data[(x,y,z)]['first_hit_time'].append(t)
            dom_data[(x,y,z)]['q_tot'].append(q)
            dom_data[(x,y,z)]['q_tot_corr'].append(q_corr)

    for key in dom_data.keys():
        qs = dom_data[key]['q_tot']
        dom_data[key]['mean_q_tot'] = np.median(qs)

        qs = dom_data[key]['q_tot_corr']
        dom_data[key]['mean_q_tot_corr'] = np.median(qs)

    sorting = 'charge'

    dom_pos = []
    for key in dom_data.keys():
        dom_pos.append(jnp.array(key).reshape((1,3)))

    dom_pos = jnp.concatenate(dom_pos, axis=0)
    track_pos = jnp.array(track_pos)
    track_dir = jnp.array(track_dir)

    # evaluate network for these doms
    # notice that all true track vertex and directions are the same. So we use the first one.
    logits, av, bv, geo_time = eval_network_doms_and_track(dom_pos, track_pos, track_dir)
    mix_probs = jax.nn.softmax(logits)

    # get also the other DOM info
    track_dir_xyz = get_xyz_from_zenith_azimuth(track_dir)

    geo_time, closest_approach_dist, closest_approach_z, closest_approach_rho = \
                cherenkov_cylinder_coordinates_w_rho_v(dom_pos,
                                             track_pos,
                                             track_dir_xyz)

    # convert first_hit_times to delay_times by subtracting geo_times
    # and add more dom data to dom_data dict
    for i in range(len(dom_pos)):
        pos = tuple(np.array(dom_pos[i]))
        gt = geo_time[i]
        for j in range(len(dom_data[pos]['first_hit_time'])):
            dom_data[pos]['first_hit_time'][j] -= float(gt)

        dom_data[pos]['closest_approach_dist'] = closest_approach_dist[i]
        dom_data[pos]['closest_approach_z'] = closest_approach_rho[i]
        dom_data[pos]['closest_approach_rho'] = closest_approach_z[i]
        dom_data[pos]['mix_probs'] = np.array(mix_probs[i])
        dom_data[pos]['a'] = np.array(av[i])
        dom_data[pos]['b'] = np.array(bv[i])

    # charge sorted mapping
    dom_positions = list(dom_data.keys())

    if sorting == 'charge':
        dom_positions.sort(key=lambda x: dom_data[x]['mean_q_tot'], reverse=True)
    else:
        dom_positions.sort(key=lambda x: dom_data[x]['closest_approach_dist'] ,reverse=False)

    n_plots = len(dom_positions)
    n_doms_per_page = 3
    xvals = np.linspace(-20, 3000, 30000)
    sigma = 3.0
    delta = 0.01

    c_multi_gamma_prob_vx = jax.vmap(c_multi_gamma_prob, (0, None, None, None, None, None), 0)
    c_multi_gamma_sf_vx = jax.vmap(c_multi_gamma_sf, (0, None, None, None, None), 0)

    #for i in range(0, int(np.min([n_plots, 60])), n_doms_per_page):
    for i in range(0, 100, n_doms_per_page):
            print(i)
            fig, ax = plt.subplots(n_doms_per_page, 3)
            for j in range(n_doms_per_page):
                pos = tuple(dom_positions[i])
                g_mix_p = dom_data[pos]['mix_probs']
                g_a = dom_data[pos]['a']
                g_b = dom_data[pos]['b']
                mode = (g_a[1]-1)/g_b[1]

                dist = dom_data[pos]['closest_approach_dist']
                z = dom_data[pos]['closest_approach_z']
                rho = dom_data[pos]['closest_approach_rho']
                for k in range(3):
                    tax = ax[j, k]

                    if k == 0:
                        tax.set_title(f"event {event_id} (dist={dist:.1f}m, z ={z:.0f}m, rho={rho:.0f}deg)", fontsize=6)
                        yval = c_multi_gamma_prob_vx(xvals, g_mix_p, g_a, g_b, sigma, delta)
                        tax.plot(xvals, yval, label='SPE PDF', color='black')
                        tax.set_ylim([0.0, 1.2 * np.amax(yval)])

                    elif k == 1:
                        n_p_orig = dom_data[pos]['mean_q_tot']
                        n_p_orig = np.round(n_p_orig+0.5)
                        n_p = np.floor(np.min([3.6*np.exp(0.23*dist)+1, n_p_orig]))
                        n_p = np.clip(n_p, None, 30)

                        n_p_corr = dom_data[pos]['mean_q_tot_corr']
                        n_p_corr = np.round(n_p_corr+0.5)
                        n_p_corr = np.floor(np.min([3.6*np.exp(0.23*dist)+1, n_p_corr]))

                        probs = c_multi_gamma_prob_vx(xvals, g_mix_p, g_a, g_b, 3.0, 0.1)
                        sfs = c_multi_gamma_sf_vx(xvals, g_mix_p, g_a, g_b, 3.0)

                        yval1 = n_p_orig * probs * jnp.power(sfs, n_p_orig-1)
                        tax.plot(xvals, yval1, label=f'MPE PDF (q={n_p_orig})', color='tab:blue', lw=1)

                        yval2 = n_p * probs * jnp.power(sfs, n_p-1)
                        tax.plot(xvals, yval2, label=f'MPE PDF (q={n_p})', color='tab:orange', lw=1)

                        yval3 = n_p_corr * probs * jnp.power(sfs, n_p_corr-1)
                        tax.plot(xvals, yval3, label=f'MPE PDF (q={n_p_corr})', color='tab:red', lw=2)
                        tax.set_ylim([0.0, 1.2*np.amax([np.amax(yval1), np.amax(yval2), np.amax(yval3)])])


                    if k == 0 or k == 1:
                        tax.set_xlabel('delay time [ns]')
                        tax.set_ylabel('pdf')
                        for tx in dom_data[pos]['first_hit_time']:
                            if tx > -20:
                                tax.axvline(tx, alpha=0.1, color='black', lw=0.5)

                        idx = np.array(dom_data[pos]['first_hit_time']) > -20
                        xmax = np.max([20,  1.2 * np.amax(dom_data[pos]['first_hit_time'])])
                        tax.hist(np.array(dom_data[pos]['first_hit_time'])[idx], density=True, alpha=0.5,
                                     label='first hit (from MC)', color='tab:green')
                        tax.legend(fontsize=5)
                        tax.set_xlim([-20 , np.max([20, 3 * mode])])

                    if k == 2:
                        tax.hist(dom_data[pos]['q_tot'], color='tab:blue', histtype='step', lw=2)
                        tax.hist(dom_data[pos]['q_tot'], color='tab:blue', alpha=0.5)
                        tax.axvline(dom_data[pos]['mean_q_tot'], alpha=0.5, color='black', lw=1)
                        tax.set_xlabel('total charge [p.e.]')
                        tax.set_ylabel('counts')


                i+=1

            plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=1.0)

            pdf.savefig(fig)
            plt.close()

    pdf.close()



for e_id in event_ids:
    print('working on', e_id)
    sigma = 0.7
    make_event_plot(e_id, sigma)
