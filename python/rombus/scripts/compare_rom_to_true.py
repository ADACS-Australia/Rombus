import lalsimulation 
import numpy as np
import lal
import sys
import pylab as plt

basis = np.load("B_matrix.npy")
fnodes = np.load("fnodes.npy")


def signal_at_nodes(fnodes, m1, m2, chi1L, chi2L, chip, thetaJ, alpha):

        l1 = 0  # params[7]
        l2 = 0  # params[8]

        m1 *= lal.lal.MSUN_SI

        m2 *= lal.lal.MSUN_SI

        WFdict = lal.CreateDict()

        h = lalsimulation.SimIMRPhenomPFrequencySequence(
            fnodes,
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
            WFdict,
        )

        return h[0].data.data

def ROM(fnodes, basis, m1, m2, chi1L, chi2L, chip, thetaJ, alpha):
	_signal_at_nodes = signal_at_nodes(fnodes, m1, m2, chi1L, chi2L, chip, thetaJ, alpha)
	return np.dot(_signal_at_nodes, basis)

def full_model(fmin, fmax, deltaF, m1, m2, chi1L, chi2L, chip, thetaJ, alpha):

	WFdict = lal.CreateDict()

	m1 *= lal.lal.MSUN_SI
	m2 *= lal.lal.MSUN_SI

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
            deltaF,
            fmin,
            fmax,
            40,
            lalsimulation.IMRPhenomPv2NRTidal_V,
            lalsimulation.NRTidalv2_V,
            WFdict,
        )
	hplus = h[0].data.data[int(fmin/deltaF):int(fmax/deltaF)+1]
	return hplus 

def main(GREEDY_POINTS):

    greedypoints = np.load(GREEDY_POINTS)
    
    m_min = 20 
    m_max = 30
    
    
    m1 = np.random.uniform(low=m_min, high=m_max)
    m2  = np.random.uniform(low=m_min, high=m_max)
    chi1L = np.random.uniform(low=0, high=0.8)
    chi2L  = np.random.uniform(low=0, high=0.8)
    chip  = np.random.uniform(low=0, high=0.8)
    thetaJ = np.random.uniform(low=0, high=np.pi)
    alpha = np.random.uniform(low=0, high=np.pi)
    
    fmin = 20
    fmax = 1024 
    deltaF = 1./4.
    fseries = np.linspace(fmin, fmax, int((fmax-fmin)/deltaF)+1)
    
    h_rom = ROM(fnodes, basis, m1, m2, chi1L, chi2L, chip, thetaJ, alpha)
    
    h_full = full_model(fmin, fmax, deltaF, m1, m2, chi1L, chi2L, chip, thetaJ, alpha)
    
    plt.semilogx(fseries, h_rom, label='ROM', alpha=0.5, linestyle='--')
    plt.semilogx(fseries, h_full, label='Full model', alpha=0.5)
    plt.scatter(fnodes, signal_at_nodes(fnodes, m1, m2, chi1L, chi2L, chip, thetaJ, alpha), s=1)
    plt.legend()
    plt.savefig("comparison.pdf", bbox_inches='tight')

if __name__ == "__main__":
    GREEDY_POINTS = sys.argv[1]
    main(GREEDY_POINTS)
