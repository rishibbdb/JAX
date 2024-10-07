#!/usr/bin/env python

from icecube import dataio, dataclasses, photonics_service, icetray
from icecube.icetray import I3Tray

import feather
import pandas as pd
import numpy as np
import glob
import os, sys

sys.path.insert(0, "/home/storage/hans/jax_reco_new")

from _lib.pulse_extraction_from_i3 import get_pulse_info_w_qtot_correction
from _lib.muon_energy import add_muon_energy
from _lib.unfold_per_loss import Unfold

from argparse import ArgumentParser
from collections import defaultdict

parser = ArgumentParser()

parser.add_argument("-id", "--indir", type=str,
                  default="/home/storage2/hans/i3files/alerts/bfrv2/",
                  dest="INDIR",
                  help="directory containing the .i3 files")

parser.add_argument("-ib", "--infile_base", type=str,
                  default="event_1722_N100",
                  dest="INFILE_BASE",
                  help="part of filename that is common to all .i3 files")

parser.add_argument("-is", "--infile_suffix", type=str,
                  default=".i3.zst",
                  dest="INFILE_SUFFIX",
                  help="suffix of .i3 files. Typically .i3.zst")

parser.add_argument("-did", "--dataset_id", type=int,
                  default=0,
                  dest="DATASET_ID",
                  help="ID of IceCube dataset")

parser.add_argument("-s", "--file_index_start", type=int,
                  default=0,
                  dest="FILE_INDEX_START",
                  help="start index of range of files to be converted (included)")

parser.add_argument("-e", "--file_index_end", type=int,
                  default=100,
                  dest="FILE_INDEX_END",
                  help="end index of range of files to be converted (excluded)")

parser.add_argument("-o", "--outdir", type=str,
                  default="/home/storage2/hans/i3files/alerts/bfrv2/",
                  dest="OUTDIR",
                  help="directory where to write output feather files")

parser.add_argument('--recompute_true_muon_energy', action='store_true',
                  dest="RECOMPUTE_MU_E")

args = parser.parse_args()

dataset_id = args.DATASET_ID
indir = args.INDIR
infile_base = args.INFILE_BASE
infile_suffix = args.INFILE_SUFFIX
file_index_start = args.FILE_INDEX_START
file_index_end = args.FILE_INDEX_END
outdir = args.OUTDIR

merged_outfile = os.path.join(indir, f'{infile_base}_merged_w_energy_loss_preds.i3.zst')

if args.RECOMPUTE_MU_E:
    from _lib.muon_energy import add_muon_energy

n_events_per_file = int(1.e5) # unique event ids.

# Select variable names from frame
meta_keys = dict()
meta_keys['pulses'] = 'TWSRTHVInIcePulsesIC'
meta_keys['mc_primary_neutrino'] = 'MCPrimary1'
meta_keys['mc_most_energetic_muon'] = 'MCMostEnergeticTrack'
meta_keys['spline_mpe'] = 'SplineMPEIC'
meta_keys['mc_muon_energy_at_interaction'] = 'TrueMuonEnergyAtInteraction'
meta_keys['mc_muon_energy_at_detector_entry']  = 'TrueMuoneEnergyAtDetectorEntry'
meta_keys['mc_muon_energy_at_detector_leave'] = 'TrueMuoneEnergyAtDetectorLeave'
min_muon_energy_at_detector = 0 # GeV
max_muon_energy_at_detector = 1000000 # GeV

# in old datsets, the background I3MCTree is kept separately
# from the I3MCTree. Hence checking for coincident events depends
# on the dataset id

if dataset_id in [21002, 21047, 21124]:
    meta_keys['bkg_mc_tree'] = 'BackgroundI3MCTree_preMuonProp'

elif dataset_id in [21217]:
    meta_keys['bkg_mc_tree'] = 'BackgroundI3MCTree'

else:
    # assume new datasets by default
    meta_keys['bkg_mc_tree'] = 'I3MCTree'


# collect all existing files
infiles = []
for file_index in range(file_index_start, file_index_end):
    infile = os.path.join(indir, f"{infile_base}_Part0{file_index:.0f}{infile_suffix}")
    if os.path.exists(infile):
        infiles.append(infile)
    else:
        infile = os.path.join(indir, f"{infile_base}_Part0{file_index:.0f}{infile_suffix}")
        if os.path.exists(infile):
            infiles.append(infile)
print(infiles)

print(f"processing {len(infiles)} .i3 files.")


# first process millipede unfolding
# then proceed with remaining code

Pulses = 'InIcePulses'
Pulses_RW = 'UncleanedInIcePulsesTimeRange'

icemodel = 'ftp-v1'
spline_table_dir = '/home/storage2/hans/photon-tables/'
tilttabledir = os.path.expandvars(
            f'$I3_BUILD/ice-models/resources/models/ICEMODEL/spice_{icemodel}/')
eff_distance = os.path.join(
            spline_table_dir,
            f'cascade_effectivedistance_spice_{icemodel}_z20.eff.fits')
eff_distance_prob = os.path.join(
            spline_table_dir,
            f'cascade_effectivedistance_spice_{icemodel}_z20.prob.fits')
eff_distance_tmod = os.path.join(
            spline_table_dir,
            f'cascade_effectivedistance_spice_{icemodel}_z20.tmod.fits')

bulk_fmt = defaultdict(
    lambda: f'cascade_single_spice_{icemodel}_flat_z20_a5')
bulk_fmt['1'] = f'ems_spice{icemodel}_z20_a10'
bulk_fmt['mie'] = f'ems_{icemodel}_z20_a10'
bulk_fmt['lea'] = f'cascade_single_spice_{icemodel}_flat_z20_a10'

bulk_prob_ver = defaultdict(lambda: '')
bulk_prob_ver['ftp-v1'] = '.v2'

cs_args = dict(amplitudetable=os.path.join(spline_table_dir, f'{bulk_fmt[icemodel]}.abs.fits'),
                   timingtable=os.path.join(
                       spline_table_dir, f'{bulk_fmt[icemodel]}.prob{bulk_prob_ver[icemodel]}.fits'),
                   effectivedistancetable=eff_distance,
                   effectivedistancetableprob=eff_distance_prob,
                   effectivedistancetabletmod=eff_distance_tmod,
                   tiltTableDir=tilttabledir)
cascade_service = photonics_service.I3PhotoSplineService(**cs_args)

gcd_file = '/home/storage2/hans/i3files/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz'
infiles = [gcd_file] + infiles

tray = I3Tray()
tray.AddModule("I3Reader", "reader", FilenameList = infiles)
tray.Add(add_muon_energy)

excludedDOMs = []

tray.Add(
            Unfold,
            Loss_Vector_Name='I3MCTree',
            FitName='MCMostEnergeticTrack',
            PhotonsPerBin=0,
            BinSigma=0,
            CascadePhotonicsService = cascade_service,
            ExcludedDOMs = excludedDOMs,
            Pulses = Pulses,
            ReadoutWindow= Pulses_RW,
         )

tray.AddModule("I3Writer","writer",FileName = merged_outfile,
                            Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics],
                DropOrphanStreams = [icetray.I3Frame.DAQ])

tray.AddModule("TrashCan","trash")
tray.Execute()
tray.Finish()


# outer loop over all infiles
event_first_pulse_idx = 0 # inclusive
event_last_pulse_idx = 0 # inclusive

meta_frames = []
pulse_frames = []

event_count = 0

g = dataio.I3File(gcd_file)
geo_frame = g.pop_frame()
g.close()

f = dataio.I3File(merged_outfile)
pulse_data = {'event_id': [], 'sensor_id': [], 'time': [], 'corrected_charge':[], 'charge': [], 'is_HLC':[]}

meta_data = {'event_id': [], 'idx_start': [], 'idx_end': [], 'n_channel_HLC': []}
meta_data.update({'neutrino_energy': [], 'muon_energy': [], 'muon_energy_at_detector': []})
meta_data.update({'muon_energy_lost': [], 'q_tot': [], 'n_channel': []})
meta_data.update({'muon_zenith': [], 'muon_azimuth': [], 'muon_time': []})
meta_data.update({'muon_pos_x': [], 'muon_pos_y': [], 'muon_pos_z': []})
meta_data.update({'spline_mpe_zenith': [], 'spline_mpe_azimuth': [], 'spline_mpe_time': []})
meta_data.update({'spline_mpe_pos_x': [], 'spline_mpe_pos_y': [], 'spline_mpe_pos_z': []})

while f.more():
    try:
        frame = f.pop_physics()

    except:
        print("Cant load frame. Skip!")
        continue


    # Try to read all keys. Skip if something is missing.
    try:
        event_header = frame['I3EventHeader']
        #interaction_type = frame['I3MCWeightDict']['InteractionType']
        interaction_type = 1.0
        primary_neutrino = frame[meta_keys['mc_primary_neutrino']] # I3Particle
        spline_mpe = frame[meta_keys['spline_mpe']]
    except:
        print("Missing a key in block 1. Skip!")
        continue

    try:
        most_energetic_track = frame[meta_keys['mc_most_energetic_muon']] # I3Particle
        muon_energy_at_interaction = frame[meta_keys['mc_muon_energy_at_interaction']].value # I3Double
        muon_energy_at_det =  frame[meta_keys['mc_muon_energy_at_detector_entry']].value # I3Double
        muon_energy_leaving = frame[meta_keys['mc_muon_energy_at_detector_leave']].value # I3Double
    except:
        print("Missing a key in block 2. Skip!")
        continue


    # Keep only muon neutrino CC events.
    is_CC_interaction = interaction_type < 1.5

    # Check if the muon enters the detector with some energy selection
    #pass_muon_energy = np.isfinite(muon_energy_at_det) and muon_energy_at_det > min_muon_energy_at_detector and muon_energy_at_det < max_muon_energy_at_detector
    pass_muon_energy = True


    # Track sanity check. is MCMostEnergeticTrack energy similar to MuonEnergy at interaction point?
    energy_ratio = muon_energy_at_interaction  / most_energetic_track.energy
    found_correct_muon = energy_ratio < 0.9 or energy_ratio > 0.9

    has_sensible_muon = np.logical_and(pass_muon_energy, energy_ratio)

    # There are no coincident events in the frame
    if meta_keys['bkg_mc_tree'] == 'I3MCTree':
        has_no_coinc = len(frame['I3MCTree'].get_primaries()) == 1

    else:
        has_no_coinc = len(frame[meta_keys['bkg_mc_tree']]) == 0

    has_sensible_muon = np.logical_and(has_sensible_muon, has_no_coinc)
    has_Sensible_muon = True

    if np.logical_and(is_CC_interaction, has_sensible_muon):
        # Retain event.
        event_count += 1

        # Define unique event identifier.
        event_id = event_header.run_id * n_events_per_file + event_header.event_id

        # Get all pulses.
        event_pulse_data, summary = get_pulse_info_w_qtot_correction(frame, geo_frame, event_id, pulses_key=meta_keys['pulses'])
        # Store.
        for key in pulse_data.keys():
            pulse_data[key] += event_pulse_data[key]

        # Get meta_data.
        event_last_pulse_idx = event_first_pulse_idx + summary['n_pulses'] - 1
        meta_data['event_id'].append(event_id)
        meta_data['idx_start'].append(event_first_pulse_idx)
        meta_data['idx_end'].append(event_last_pulse_idx)

        meta_data['neutrino_energy'].append(primary_neutrino.energy)
        meta_data['muon_energy'].append(muon_energy_at_interaction)
        meta_data['muon_energy_at_detector'].append(muon_energy_at_det)

        if np.isfinite(muon_energy_leaving):
            meta_data['muon_energy_lost'].append(muon_energy_at_det - muon_energy_leaving)
        else:
            # lost all energy inside the detector
            meta_data['muon_energy_lost'].append(muon_energy_at_det)

        meta_data['q_tot'].append(summary['q_tot'])
        meta_data['n_channel'].append(summary['n_channel'])
        meta_data['n_channel_HLC'].append(summary['n_channel_HLC'])
        meta_data['muon_zenith'].append(most_energetic_track.dir.zenith)
        meta_data['muon_azimuth'].append(most_energetic_track.dir.azimuth)
        meta_data['muon_time'].append(most_energetic_track.time)
        meta_data['muon_pos_x'].append(most_energetic_track.pos.x)
        meta_data['muon_pos_y'].append(most_energetic_track.pos.y)
        meta_data['muon_pos_z'].append(most_energetic_track.pos.z)
        meta_data['spline_mpe_zenith'].append(spline_mpe.dir.zenith)
        meta_data['spline_mpe_azimuth'].append(spline_mpe.dir.azimuth)
        meta_data['spline_mpe_time'].append(spline_mpe.time)
        meta_data['spline_mpe_pos_x'].append(spline_mpe.pos.x)
        meta_data['spline_mpe_pos_y'].append(spline_mpe.pos.y)
        meta_data['spline_mpe_pos_z'].append(spline_mpe.pos.z)


        # Last action:
        # update first pulse idx for next event.
        event_first_pulse_idx = event_last_pulse_idx + 1


# Convert to data frames.
df_pulses = pd.DataFrame.from_dict(pulse_data)
df_meta = pd.DataFrame.from_dict(meta_data)

meta_frames.append(df_meta)
pulse_frames.append(df_pulses)

f.close()

# Combine all files into one.
# Write to disk using feather format.
df_pulses = pd.concat(pulse_frames)
df_meta = pd.concat(meta_frames)

ofile = os.path.join(outdir, f"pulses_ds_{infile_base}_from_{file_index_start}_to_{file_index_end}_1st_pulse_charge_correction.ftr")
df_pulses.reset_index(drop=True).to_feather(ofile, compression='zstd')
ofile = os.path.join(outdir, f"meta_ds_{infile_base}_from_{file_index_start}_to_{file_index_end}_1st_pulse_charge_correction.ftr")
df_meta.reset_index(drop=True).to_feather(ofile, compression='zstd')

print(f"stored {event_count} events in outfile {ofile}")


