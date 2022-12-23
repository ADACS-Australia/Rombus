import numpy as np
import lalsimulation
import lal
from typing import Dict, Protocol
from dataclasses import dataclass, field

class IsDataclass(Protocol):
    # Checking for this attribute is currently the most reliable way to 
    # ascertain that something is a dataclass
    __dataclass_fields__: Dict

class model(object):
    model_dtype = 'complex'
    
    def init_cache(self):
        @dataclass
        class ModelCache(object):
            fmin: float
            fmax: float
            deltaF: float
            nf: int
            fseries: np.ndarray
            fmin_index: float
            WFdict: Dict
        fmin = 20
        fmax = 1024
        deltaF = 1.0 / 4.0
        nf = int((fmax-fmin)/deltaF)+1
        fseries = np.linspace(fmin, fmax, nf)
        fmin_index = int(fmin / deltaF)
        WFdict = lal.CreateDict()
        return ModelCache(
            fmin = fmin,
            fmax = fmax,
            deltaF = deltaF,
            nf = nf,
            fseries = fseries,
            fmin_index = fmin_index,
            WFdict = WFdict
            )
    
    def init_domain(self,cache):
        return cache.fseries
    
    def compute_model(self,params: np.array, domain, cache: IsDataclass) -> np.array:
        m1 = params[0]
        m2 = params[1]
        chi1L = params[2]
        chi2L = params[3]
        chip = params[4]
        thetaJ = params[5]
        alpha = params[6]
        l1 =0
        l2 =0
    
        m1 *= lal.lal.MSUN_SI
        m2 *= lal.lal.MSUN_SI
    
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(cache.WFdict, l1)
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(cache.WFdict, l2)
    
        if not np.array_equiv(domain,cache.fseries):
            h = lalsimulation.SimIMRPhenomPFrequencySequence(
                domain,
                chi1L,
                chi2L,
                chip,
                thetaJ,
                m1,
                m2,
                1e6 * lal.lal.PC_SI * 100,
                alpha,
                0,
                40,
                lalsimulation.IMRPhenomPv2NRTidal_V,
                lalsimulation.NRTidalv2_V,
                cache.WFdict,
            )
            h = h[0].data.data
        else:
            h = lalsimulation.SimIMRPhenomP(
                chi1L,
                chi2L,
                chip,
                thetaJ,
                m1,
                m2,
                1e6 * lal.lal.PC_SI * 100,
                alpha,
                0,
                cache.deltaF,
                cache.fmin,
                cache.fmax,
                40,
                lalsimulation.IMRPhenomPv2NRTidal_V,
                lalsimulation.NRTidalv2_V,
                cache.WFdict,
            )
            h = h[0].data.data[cache.fmin_index: len(h[0].data.data)]
            if len(h) < cache.nf:
                h = np.append(h, np.zeros(cache.nf - len(h), dtype=complex))
    
        return h
