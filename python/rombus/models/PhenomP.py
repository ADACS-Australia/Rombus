from typing import NamedTuple

import lal  # type: ignore
import lalsimulation  # type: ignore
import numpy as np

from rombus.model import RombusModel


class Model(RombusModel):

    # Set some constants
    l_1 = 0
    l_2 = 0
    f_min = 20.0
    f_max = 1024.0
    delta_F = 1.0 / 4.0
    n_f = int((f_max - f_min) / delta_F) + 1
    f_min_index = int(f_min / delta_F)
    WFdict = lal.CreateDict()

    # Set the domain over-and-on which the ROM will be defined
    coordinate.set("f", f_min, f_max, n_f, label="$f$")  # type: ignore # noqa F821

    # Set the ordinate the model will map the domain to
    ordinate.set("h", dtype=np.dtype("float64"), label="$h$")  # type: ignore # noqa F821

    # N.B.: mypy struggles with NamedTuples, so typing is turned off for the following
    params.add("m1", 30, 35)  # type: ignore # noqa F821
    params.add("m2", 30, 35)  # type: ignore # noqa F821
    params.add("chi1L", 0, 0.1)  # type: ignore # noqa F821
    params.add("chi2L", 0, 0.1)  # type: ignore # noqa F821
    params.add("chip", 0, 0.1)  # type: ignore # noqa F821
    params.add("thetaJ", -np.pi / 2, np.pi / 2)  # type: ignore # noqa F821
    params.add("alpha", 0, np.pi / 2)  # type: ignore # noqa F821
    params.set_validation(lambda p: p.m1 >= p.m2)  # type: ignore # noqa F821

    def compute(self, params: NamedTuple, domain: np.ndarray) -> np.ndarray:

        from rombus._core.log import log

        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(self.WFdict, self.l_1)
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(self.WFdict, self.l_2)
        if not np.array_equiv(domain, self.domain):
            d_i_last = 0.0
            for d_i in domain:
                if d_i <= d_i_last:
                    log.comment(f"XXXX: {d_i} {d_i-d_i_last}")
                d_i_last = d_i
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
                self.delta_F,
                self.f_min,
                self.f_max,
                40,
                lalsimulation.IMRPhenomPv2NRTidal_V,
                lalsimulation.NRTidalv2_V,
                self.WFdict,
            )
            h = h[0].data.data[self.f_min_index : len(h[0].data.data)]
            if len(h) < self.n_domain:
                h = np.append(h, np.zeros(self.n_domain - len(h), dtype=complex))

        return h
