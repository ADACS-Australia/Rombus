import numpy as np
import lalsimulation
import lal
from rombus.core import RombusModel

class model(RombusModel):

    model_dtype = 'complex'

    def init(self):
        self.fmin = 20
        self.fmax = 1024
        self.deltaF = 1.0 / 4.0
        self.nf = int((self.fmax-self.fmin)/self.deltaF)+1
        self.fseries = np.linspace(self.fmin, self.fmax, self.nf)
        self.fmin_index = int(self.fmin / self.deltaF)
        self.WFdict = lal.CreateDict()

    def init_domain(self):
        return self.fseries

    def compute(self, params: np.array, domain) -> np.array:
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

        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(self.WFdict, l1)
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(self.WFdict, l2)

        if not np.array_equiv(domain,self.fseries):
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
                self.WFdict,
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
                self.deltaF,
                self.fmin,
                self.fmax,
                40,
                lalsimulation.IMRPhenomPv2NRTidal_V,
                lalsimulation.NRTidalv2_V,
                self.WFdict,
            )
            h = h[0].data.data[self.fmin_index: len(h[0].data.data)]
            if len(h) < self.nf:
                h = np.append(h, np.zeros(self.nf - len(h), dtype=complex))

        return h
