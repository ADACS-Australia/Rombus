import numpy as np
#from pycbc import types, fft, waveform, lalsimulation
import lalsimulation
import os.path
from numpy.linalg import qr
from misc import *
import lal

def MGS(RB, next_vec, iter):
    
    dim_RB = iter 
    for i in range(dim_RB):
    
        ## --- ortho_basis = ortho_basis - L2_proj*basis; --- ##
        L2 = np.vdot(RB[i], next_vec)
        next_vec -= RB[i]*L2
	
   
    norm = np.sqrt(np.vdot(next_vec, next_vec))	
    next_vec /= norm

    return next_vec, norm

def IMGS(RB, next_vec, iter):

	ortho_condition = .5
	norm_prev = np.sqrt(np.vdot(next_vec, next_vec))
	flag = False

	while not flag: 
    
		next_vec, norm_current = MGS(RB, next_vec, iter)
 
		next_vec *= norm_current 
 
		if norm_current/norm_prev <= ortho_condition:
         
			norm_prev = norm_current
 
         
		else:
			flag = True
 
		norm_current  = np.sqrt(np.vdot(next_vec, next_vec)) 
		next_vec /= norm_current 
 
	RB[iter] = next_vec#np.vstack((RB, next_vec)) 
	return RB[iter]
	

fmin = 20
fmax = 4096 
deltaF = 1./4.
fseries = np.linspace(fmin, fmax, int((fmax-fmin)/deltaF)+1)
fmin_index = int(fmin/deltaF)


greedypoints = np.loadtxt("GreedyPoints.txt")#[0:35**2]
TS = np.zeros([len(greedypoints), len(fseries)], dtype='complex')

i = 0
#_L1 = np.linspace(0, 5000, 35)
#_L2 = np.linspace(0, 5000, 35)

#a,b = np.meshgrid(_L1,_L2)
#positions = np.vstack([a.ravel(), b.ravel()])

#L1 = positions[0]
#L2 = positions[1]

for ii, params in enumerate(greedypoints[:]):

	print( "iteration %d/%d"%(i,len(greedypoints)) )
	
	m1 = params[0]
	m2 = params[1]
	chi1L = params[2]
	chi2L = params[3]
	chip = params[4]
	thetaJ = params[5]
	alpha = params[6]
	l1 = params[7]
	l2 = params[8]
	
	m1 *= lal.lal.MSUN_SI
	
	m2 *= lal.lal.MSUN_SI
	
	WFdict = lal.CreateDict()

	lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(WFdict, l1)
	lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(WFdict, l2)
	
	#hp = lalsimulation.SimIMRPhenomDNRTidal(0, deltaF, fmin, fmax, 40, 1e6*lal.lal.PC_SI*100, m1, m2, chi1L, chi2L, l1, l2, None)
	h = lalsimulation.SimIMRPhenomP(chi1L, chi2L, chip, thetaJ,
	m1, m2, 1e6*lal.lal.PC_SI*100, alpha, 0, deltaF, fmin, fmax, 40, lalsimulation.IMRPhenomPv2NRTidal_V, lalsimulation.NRTidalv2_V, WFdict)
	
	h = h[0].data.data[fmin_index:len(h[0].data.data)]	
		
	
	if len(h) < len(fseries):
	
		h = np.append(h, np.zeros(len(fseries)-len(h), dtype = complex))
	h /= np.sqrt(np.vdot(h,h)) 
	
	TS[i] = h
	
	i += 1

error = 1
iter = 1
RB_matrix = np.zeros_like(TS) 
RB_matrix[0] = TS[0]
pc = np.zeros(len(TS)*len(TS), complex).reshape(len(TS), len(TS))
indices = [0]

while error > 1e-16:


	pc = project_onto_basis(1., RB_matrix, TS, pc, iter-1) 
	
	#residual = TS - projections
	# Find projection errors
	projection_errors = [1 - dot_product(1., pc[jj][:iter], pc[jj][:iter]) for jj in range(len(pc))]
	index = np.argmax(projection_errors) # Find Training-space index of waveform with largest proj. error 
	error = projection_errors[index] 
	print (error, iter, index)
	indices.append(index) 
	#Gram-Schmidt to get the next basis and normalize
	
	RB_matrix[iter] = IMGS(RB_matrix, TS[index], iter)
	
	if iter % 100 == 0:
	       np.save("RB_matrix", RB_matrix[0:iter]/np.sqrt(deltaF))
	       print ("saving on iter %d"%(iter))
	iter += 1
	
	if iter == len(RB_matrix) - 1:
		break
RB_matrix = RB_matrix[~np.all(RB_matrix == 0, axis=1)]

#np.save("RB_matrix",RB_matrix/np.sqrt(deltaF))
np.save("RB_matrix",RB_matrix)
np.save("GreedyPoints", greedypoints[indices])
