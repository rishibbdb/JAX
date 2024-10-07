#!/usr/bin/env python

from icecube import icetray, dataclasses, dataio, simclasses
from icecube.icetray import I3Units
import numpy as np

"""
sourced from
https://github.com/icecube/icetray/blob/main/finallevel_filter_diffusenumu/python/post_level5/segment_MuonEnergy.py
https://github.com/icecube/icetray/commit/ddd018cad01f0a23e3367d1378646eab85836a56
"""

def get_interacting_neutrino(fr):
    """ Starting with the MCPrimary1 we go through the tree until we find the interacting neutrino """
    if not fr.Has("MCPrimary1"):
        raise KeyError("Missing MCPrimary1")
    initial_neutrino = fr["MCPrimary1"]

    if not fr.Has("I3MCTree"):
        raise KeyError("Missing I3MCTree")
    tree = fr["I3MCTree"]

    # check for LeptonInjector simulation which does not mark the neutrino as in_ice
    if fr.Has("EventProperties"):
        return initial_neutrino

    # go through the tree until we find the interacting neutrino (neutrino && inice)
    next_particle = [initial_neutrino]
    while len(next_particle) > 0:
        d = next_particle.pop()
        if d.is_neutrino and d.location_type_string == "InIce":
            return d
        next_particle.extend(tree.get_daughters(d))
    raise RuntimeError("Did not find an interacting neutrino!")


def get_interacting_neutrino_and_daughters(fr):
    """
    Looks for the MCprimary and then sorts its daughters

    Returns:
        PrimaryNeutrino
        Lepton if existing, else None
        List of Hadrons, else empty list
        Outgoing Neutrino, else None
    """
    tree = fr["I3MCTree"]
    neutrino = get_interacting_neutrino(fr)
    children = tree.get_daughters(neutrino)
    if len(children) != 2:
        print("Expected only two childs from interaction but found %d"%len(children))
        # raise RuntimeWarning()
        return None, None, [None], None
    lepton = None
    outgoing_neutrino = None
    hadrons = []
    for ch in children:
        if ch.type == dataclasses.I3Particle.Hadrons:
            hadrons.append(ch)
        elif int(ch.type) in [12,14,16,-12,-14,-16]:
            outgoing_neutrino = ch
        else:
            lepton = ch

    if len(hadrons)==1 and lepton is not None:
        return neutrino, lepton, hadrons, outgoing_neutrino
    elif lepton is None and outgoing_neutrino is None:
        print("Found an event with no outgoing lepton. This is fine if you're" \
              "simulating NuE, e.g. in the hadronic decay of Glashow events. ")
        return neutrino, lepton, hadrons, outgoing_neutrino
    elif len(hadrons)==0 and outgoing_neutrino is not None:
        print("Found an event without hadrons. This is fine if you're" \
              "simulating NuE, e.g. in the leptonic decay of Glashow events.")
        return neutrino, lepton, hadrons, outgoing_neutrino
    elif len(hadrons)==1 and outgoing_neutrino is not None:
        print("Found an event with outgoing neutrino and hadrons. E.g. NC-event")
        return neutrino, lepton, hadrons, outgoing_neutrino
    else:
        return None, None, [None], None
        #print("ANOTHER CASE", fr["I3MCTree"])


def intersect_cylinder(pos, dir, center=dataclasses.I3Position(31.25, 19.64, 0), radius=800*I3Units.m, length=1600*I3Units.m):
    # following https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/projects/phys-services/trunk/public/phys-services/surfaces/detail/CylinderBase.h
    h = (None, None)
    r = (None, None)
    offset = pos-center

    b = offset.x * np.cos(dir.azimuth) + offset.y * np.sin(dir.azimuth)

    d = b**2 + radius**2 - offset.x**2 - offset.y**2

    sinth = np.sin(dir.zenith)
    costh = np.cos(dir.zenith)

    if d > 0:
        d = np.sqrt(d)

        if costh != 0:
            h = (offset.z-length/2) / costh, (offset.z+length/2) / costh
        if sinth != 0:
            r = (b - d) / sinth, (b + d) / sinth
        if costh == 0:
            if np.abs(offset.z) < lenfth/2:
                h = r
            else:
                h = (None, None)
        elif sinth == 0:
            if np.hypot(offset.x, offset.y) >= radius:
                h = (None, None)
        else:
            if np.min(h) >= np.max(r) or np.max(h) <= np.min(r):
                h = (None, None)
            else:
                h = (np.max([np.min(r), np.min(h)]), np.min([np.max(r), np.max(h)]))

    return h


def timeShift(p, MMCTrack):
    # following https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/projects/MuonGun/releases/V00-02-03/private/MuonGun/Track.cxx#L116
    shifted = simclasses.I3MMCTrack(MMCTrack)
    mmcPos = dataclasses.I3Position(MMCTrack.GetXi(), MMCTrack.GetYi(), MMCTrack.GetZi())
    d = (p.pos - mmcPos).magnitude
    dt = p.time + d/p.speed - MMCTrack.ti
    shifted.ti = MMCTrack.ti + dt
    shifted.xi = MMCTrack.xi
    shifted.yi = MMCTrack.yi
    shifted.zi = MMCTrack.zi
    shifted.Ei = MMCTrack.Ei
    shifted.tc = MMCTrack.tc + dt
    shifted.xc = MMCTrack.xc
    shifted.yc = MMCTrack.yc
    shifted.zc = MMCTrack.zc
    shifted.Ec = MMCTrack.Ec
    shifted.tf = MMCTrack.tf + dt
    shifted.xf = MMCTrack.xf
    shifted.yf = MMCTrack.yf
    shifted.zf = MMCTrack.zf
    shifted.Ef = MMCTrack.Ef
    shifted.particle = p

    return shifted


class Checkpoint(object):
    # following https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/projects/MuonGun/releases/V00-02-03/public/MuonGun/Track.h
    def __init__(self, length, energy, offset):
        self.length = length
        self.energy = energy
        self.offset = offset

    def __lt__(self, other):
        return self.length < other.length

    def __repr__(self):
        return "CP(l=%f, e=%f, o=%d)"%(self.length, self.energy, self.offset)


class LossSum(object):
    # following https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/projects/MuonGun/releases/V00-02-03/public/MuonGun/Track.h
    def __init__(self, length, energy):
        self.length = length
        self.energy = energy

    def __repr__(self):
        return "LS(l=%f, e=%f)"%(self.length, self.energy)

    def __lt__(self, other):
        return self.length < other.length


def get_checkpoints_and_losses(MMCTrack, daughters):
    # following https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/projects/MuonGun/releases/V00-02-03/private/MuonGun/Track.cxx#L16
    checkpoints = [Checkpoint(0, MMCTrack.particle.energy, 0)] # length, abs energy, time diff
    losses = [LossSum(checkpoints[-1].length, 0)] # length, energy loss since last checkpoint

    # energy loss before entering the volume
    if MMCTrack.Ei > 0:
        d = (dataclasses.I3Position(MMCTrack.xi, MMCTrack.yi, MMCTrack.zi) - MMCTrack.particle.pos).magnitude
        checkpoints.append( Checkpoint(d, MMCTrack.Ei, len(losses)) )

    # energy loss within the volume
    elost = 0
    for p in daughters:
        elost += p.energy
        d = (p.pos - MMCTrack.particle.pos).magnitude
        losses.append( LossSum(d, elost) )

    # energy loss after the volume
    if MMCTrack.Ef > 0:
        d = (dataclasses.I3Position(MMCTrack.xf, MMCTrack.yf, MMCTrack.zf) - MMCTrack.particle.pos).magnitude
        checkpoints.append( Checkpoint(d, MMCTrack.Ef, len(losses)) )
        losses.append( LossSum(checkpoints[-1].length, 0) )

    checkpoints.append( Checkpoint(MMCTrack.particle.length, 0, len(losses)) )
    losses.append( LossSum(MMCTrack.particle.length , 0) )

    return checkpoints, losses


def getEnergy(length, MMCTrack, checkpoints, losses):
    # following https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/projects/MuonGun/releases/V00-02-03/private/MuonGun/Track.cxx#L62
    if (not np.isfinite(length)) or length >= MMCTrack.particle.length:
        return 0
    if length <= 0:
        return MMCTrack.particle.energy

    # checkpoints before and after 'length'
    cp_before = None
    cp_after = None
    for i in range(len(checkpoints)):
        if not checkpoints[i].length < length:
            cp_before = checkpoints[np.max([i-1,0])]
            cp_after = checkpoints[np.max([i-1,0])+1]
            break

    # first and last loss between checkpoints
    l1 = 0 + cp_before.offset-1 if cp_before.offset > 0 else 0
    l2 = 0 + cp_after.offset

    # last loss before 'length'
    ls = losses[l1]
    for i in range(l1, l2+1):
        if not losses[i].length < length:
            ls = losses[np.max([i-1,l1])]
            break

    # continues loss rate not taken into account by losses
    conti_rate = (cp_before.energy - cp_after.energy - losses[l2-1].energy)/(cp_after.length-cp_before.length)

    assert ls.energy <= cp_before.energy, "sum of losses is smaller than energy at least checkpoint"

    # energy = energy at checkpoint - discrete losses - continues losses
    return cp_before.energy - ls.energy - conti_rate*(length-cp_before.length)

def EnergyAtEgdeNoMuonGun(fr):
    MMCList = fr["MMCTrackList"]
    tree = fr["I3MCTree"]

    for MMCTrack in MMCList:
        if not tree.parent(MMCTrack.particle).type in [dataclasses.I3Particle.NuMu, dataclasses.I3Particle.NuMuBar]:
            continue

        if not tree.has(MMCTrack.particle):
            raise
        p = tree.get_particle(MMCTrack.particle)
        daughters = tree.children(p)
        if len(daughters) != 0:
            shifted = timeShift(p, MMCTrack)
            checkpoints, losses = get_checkpoints_and_losses(shifted, daughters)
            intersections = intersect_cylinder(shifted.particle.pos, shifted.particle.dir)
            return getEnergy(intersections[0], shifted, checkpoints, losses), getEnergy(intersections[1], shifted, checkpoints, losses)
    return None, None


def add_muon_energy(frame):
    try:
        e_first, e_last = EnergyAtEgdeNoMuonGun(frame)
        frame.Put("TrueMuoneEnergyAtDetectorEntry", dataclasses.I3Double(e_first))
        frame.Put("TrueMuoneEnergyAtDetectorLeave", dataclasses.I3Double(e_last))
    except:
        frame.Put("TrueMuoneEnergyAtDetectorEntry", dataclasses.I3Double(np.nan))
        frame.Put("TrueMuoneEnergyAtDetectorLeave", dataclasses.I3Double(np.nan))
    nu, lepton, hadrons, nu_out = get_interacting_neutrino_and_daughters(frame)
    mu_e = np.nan if lepton is None else lepton.energy
    frame.Put("TrueMuonEnergyAtInteraction", dataclasses.I3Double(mu_e))
    if not frame.Has('MCMostEnergeticTrack'):
        frame.Put("MCMostEnergeticTrack", lepton)
