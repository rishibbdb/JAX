from icecube.icetray import I3Units
from icecube import icetray, dataio, dataclasses, millipede, DomTools
from icecube import WaveCalibrator, wavedeform, photonics_service
from icecube import gulliver, gulliver_modules, phys_services
import numpy
import library
from icecube.millipede import MonopodFit, MuMillipedeFit, TaupedeFit, HighEnergyExclusions
from icecube.icetray import I3Tray
import sys, os
from collections import defaultdict
from unfold_per_loss import Unfold

from _lib.muon_energy import add_muon_energy

#Pulses = 'MillipedeHVSplitPulses'
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

gcd = '/home/storage2/hans/i3files/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz'
#i3 = '/home/storage2/hans/i3files/alerts/bfrv2/event_10644_N100_Part00.i3.zst'
i3 = '/home/storage2/hans/i3files/alerts/bfrv2/event_1722_N100_Part00.i3.zst'
#i3 = '/home/storage2/hans/i3files/alerts/bfrv2/event_1022_N100_Part00.i3.zst'
infiles = [gcd, i3]

tray = I3Tray()
tray.AddModule("I3Reader", "reader", FilenameList = infiles)
tray.Add(add_muon_energy)
tray.Add('Delete', keys=['BrightDOMs'])

#excludedDOMs = tray.Add(HighEnergyExclusions,
#                        Pulses=Pulses,
#                        BrightDOMThreshold=10,
#                        BadDomsList='BadDomsList',
#                        CalibrationErrata='CalibrationErrata',
#                        SaturationWindows='SaturationWindows')
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

tray.AddModule("I3Writer","writer",FileName = "test.i3.zst",
                            Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics],
                DropOrphanStreams = [icetray.I3Frame.DAQ])

tray.AddModule("TrashCan","trash")
tray.Execute(10)
tray.Finish()
