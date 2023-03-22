from typing import NamedTuple

import lal  # type: ignore
import lalsimulation  # type: ignore
import numpy as np

from rombus.model import RombusModel


class model(RombusModel):

    model_dtype = complex

    # N.B.: mypy struggles with NamedTuples, so typing is turned off for the following
    params.add("m1", 30, 35)  # type: ignore # noqa F821
    params.add("m2", 30, 35)  # type: ignore # noqa F821
    params.add("chi1L", 0, 0.1)  # type: ignore # noqa F821
    params.add("chi2L", 0, 0.1)  # type: ignore # noqa F821
    params.add("chip", 0, 0.1)  # type: ignore # noqa F821
    params.add("thetaJ", -np.pi / 2, np.pi / 2)  # type: ignore # noqa F821
    params.add("alpha", 0, np.pi / 2)  # type: ignore # noqa F821
    params.set_validation(lambda p: p.m1 >= p.m2)  # type: ignore # noqa F821

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

    def set_domain(self) -> np.ndarray:
        return self.fseries

    def compute(self, params: NamedTuple, domain: np.ndarray) -> np.ndarray:

        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(self.WFdict, self.l1)
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(self.WFdict, self.l2)

        if not np.array_equiv(domain, self.fseries):
            h = lalsimulation.SimIMRPhenomPFrequencySequence(
                domain,
                params.chi1L,  # type: ignore
                params.chi2L,  # type: ignore
                params.chip,  # type: ignore
                params.thetaJ,  # type: ignore
                params.m1 * lal.lal.MSUN_SI,  # type: ignore
                params.m2 * lal.lal.MSUN_SI,  # type: ignore
                1e6 * lal.lal.PC_SI * 100,
                params.alpha,  # type: ignore
                0,
                40,
                lalsimulation.IMRPhenomPv2NRTidal_V,
                lalsimulation.NRTidalv2_V,
                self.WFdict,
            )
            h = h[0].data.data
        else:
            h = lalsimulation.SimIMRPhenomP(
                params.chi1L,  # type: ignore
                params.chi2L,  # type: ignore
                params.chip,  # type: ignore
                params.thetaJ,  # type: ignore
                params.m1 * lal.lal.MSUN_SI,  # type: ignore
                params.m2 * lal.lal.MSUN_SI,  # type: ignore
                1e6 * lal.lal.PC_SI * 100,
                params.alpha,  # type: ignore
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
