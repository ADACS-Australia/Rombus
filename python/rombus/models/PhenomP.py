from typing import NamedTuple

import lal
import lalsimulation
import numpy as np

from rombus.model import RombusModel


class model(RombusModel):

    model_dtype = complex

    params.add("m1", 30, 35)  # noqa F821
    params.add("m2", 30, 35)  # noqa F821
    params.add("chi1L", 0, 0.1)  # noqa F821
    params.add("chi2L", 0, 0.1)  # noqa F821
    params.add("chip", 0, 0.1)  # noqa F821
    params.add("thetaJ", -np.pi / 2, np.pi / 2)  # noqa F821
    params.add("alpha", 0, np.pi / 2)  # noqa F821
    params.set_validation(lambda p: p.m1 >= p.m2)  # noqa F821

    def cache(self):
        self.l1 = 0
        self.l2 = 0
        self.fmin = 20
        self.fmax = 1024
        self.deltaF = 1.0 / 4.0
        self.nf = int((self.fmax - self.fmin) / self.deltaF) + 1
        self.fseries = np.linspace(self.fmin, self.fmax, self.nf)
        self.fmin_index = int(self.fmin / self.deltaF)
        self.WFdict = lal.CreateDict()

    def set_domain(self):
        return self.fseries

    def compute(self, params: NamedTuple, domain) -> np.array:

        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(self.WFdict, self.l1)
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(self.WFdict, self.l2)

        if not np.array_equiv(domain, self.fseries):
            h = lalsimulation.SimIMRPhenomPFrequencySequence(
                domain,
                params.chi1L,
                params.chi2L,
                params.chip,
                params.thetaJ,
                params.m1 * lal.lal.MSUN_SI,
                params.m2 * lal.lal.MSUN_SI,
                1e6 * lal.lal.PC_SI * 100,
                params.alpha,
                0,
                40,
                lalsimulation.IMRPhenomPv2NRTidal_V,
                lalsimulation.NRTidalv2_V,
                self.WFdict,
            )
            h = h[0].data.data
        else:
            h = lalsimulation.SimIMRPhenomP(
                params.chi1L,
                params.chi2L,
                params.chip,
                params.thetaJ,
                params.m1 * lal.lal.MSUN_SI,
                params.m2 * lal.lal.MSUN_SI,
                1e6 * lal.lal.PC_SI * 100,
                params.alpha,
                0,
                self.deltaF,
                self.fmin,
                self.fmax,
                40,
                lalsimulation.IMRPhenomPv2NRTidal_V,
                lalsimulation.NRTidalv2_V,
                self.WFdict,
            )
            h = h[0].data.data[self.fmin_index : len(h[0].data.data)]
            if len(h) < self.nf:
                h = np.append(h, np.zeros(self.nf - len(h), dtype=complex))

        return h
