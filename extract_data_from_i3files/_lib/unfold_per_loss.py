from icecube.icetray import I3Units
from icecube import icetray, dataio, dataclasses, millipede, DomTools
from icecube import WaveCalibrator, wavedeform, photonics_service
from icecube import gulliver, gulliver_modules, phys_services
import numpy
import library
from collections import defaultdict

class Unfold(icetray.I3Module):
    """ A class originally gifted by K. Jero (from JvS?)
    """

    def __init__(self, ctx):
        super(Unfold, self).__init__(ctx)
        self.AddParameter('Loss_Vector_Name',
                          'Name of the loss vector or I3MCTree',
                          'I3MCTree')
        self.AddParameter('Pulses', 'Name of the Pulses', 'InIcePulses')
        self.AddParameter('FitName',
                          'Name of the fit to use for closest approach distance calculations',
                          'SeedTrack')
        self.AddParameter('CascadePhotonicsService',
                          'Photonics service for cascades',
                          None)
        self.AddParameter('ExcludedDOMs',
                          'DOMs to exclude',
                          [])
        self.AddParameter('PhotonsPerBin',
                          'Number of photoelectrons to include in each timeslice',
                          0)
        self.AddParameter('BinSigma',
                          'Bayesian blocking sigma',
                          0)

        self.AddParameter('ReadoutWindow',
                          'UncleanedInIcePulsesTimeRange',
                          0)

        self.AddOutBox('OutBox')


    def Configure(self):
        self.input_loss_vect_name = self.GetParameter('Loss_Vector_Name')
        self.pulses = self.GetParameter('Pulses')
        self.fitname = self.GetParameter('FitName')
        self.cscd_service = self.GetParameter('CascadePhotonicsService')
        self.exclude_doms = self.GetParameter('ExcludedDOMs')
        self.ppb = self.GetParameter('PhotonsPerBin')
        self.bs = self.GetParameter('BinSigma')
        self.readout_window = self.GetParameter('ReadoutWindow')


    def Physics(self, frame):
        if not frame.Has(self.input_loss_vect_name):
            #print('no unfolding for {}'.format(self.input_loss_vect_name))
            self.PushFrame(frame)
            return True

        self.millipede = millipede.PyPyMillipede(self.context)
        #print('Lets unfold the true losses')
        ExQdict = defaultdict(list)

        if self.input_loss_vect_name == 'I3MCTree':
            I3MCTree = frame['I3MCTree']
            # if loss.energy>1 and (loss.pos.x**2+loss.pos.y**2)**.5<600 and
            # loss.pos.z<600 and loss.pos.z>-600]
            sources = []
            for p in I3MCTree.get_daughters(I3MCTree[0]):
                for loss in I3MCTree.get_daughters(p):
                    if not 'Mu' in str(loss.type):
                        sources.append(loss)

            #print(len(sources))

                #loss for loss in I3MCTree.get_daughters(p) for p in
                #    I3MCTree.get_daughters(
                #        I3MCTree[0])]

        elif isinstance(frame[self.input_loss_vect_name], dataclasses.I3Particle) :
            sources = [frame[self.input_loss_vect_name]]
        else:
            sources = frame[self.input_loss_vect_name]
        #for s in sources:
        #    print('time:', s.time, 'energy:', s.energy)

        # This line needs to call get_photonics not the service itself
        self.millipede.SetParameter('CascadePhotonicsService', self.cscd_service)
        self.millipede.SetParameter('ExcludedDOMs', self.exclude_doms)
        self.millipede.SetParameter('Pulses', self.pulses)
        self.millipede.SetParameter('PhotonsPerBin', self.ppb)
        self.millipede.SetParameter('BinSigma', self.bs)
        self.millipede.SetParameter('ReadoutWindow', self.readout_window)
        self.millipede.DatamapFromFrame(frame)
        response = self.millipede.GetResponseMatrix(sources)
        #print('Fit Statistics For Losses:', self.millipede.FitStatistics(sources, response, params=None))
        edeps = [p.energy for p in sources]
        responsemat = response.to_I3Matrix()

        all_expectations = numpy.asarray(responsemat) * numpy.asarray(edeps).reshape((1, len(edeps)))

        thisreco = frame[self.fitname]
        I3OMGeo = frame['I3Geometry'].omgeo
        I3EH = frame['I3EventHeader']
        try:
            ps = frame[self.pulses].apply(frame)
        except:
            ps = frame[self.pulses]

        for i in range(len(edeps)):
            #print(f"loss {i}, {edeps[i]}")
            expectations = all_expectations[:, i]
            itera = -1
            ExcludedDOMlist = library.excluded_doms(frame, self.exclude_doms)
            for k, dc in self.millipede.domCache.items():
                valid = dc.valid
                if k in ExcludedDOMlist:
                    for v in valid:
                        if v:
                            itera += 1
                    continue

                thisexdomq = 0
                for v in valid:
                    if v:
                        itera += 1
                        try:
                            thisexdomq += expectations[itera]
                        except:
                            print('Problem extracting expected Q')
                            pass

                ExQdict[k].append(thisexdomq)
        frame[self.fitname + '_' + self.input_loss_vect_name +
                  '_ExQ'] = dataclasses.I3MapKeyVectorDouble(ExQdict)
        self.PushFrame(frame)
        #print('Unfold done\n')
