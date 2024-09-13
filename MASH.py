import numpy as np
from numpy import random as rn
from copy import deepcopy as dc
import random
from time import time as timee
from numba import jit, objmode, typeof, complex128
from scipy.optimize import root
from scipy import interpolate
from scipy.integrate import solve_ivp
# import model_GHTC as m
# from model_GHTC import H, dH, dH0, initR, initc, get_xj, LBasisChange, LBasisChangeRevert, getLRates, getLStates, ForceCustom, HopDirCustom # import the functions from the model
import model as m
from model import H, dH, dH0, initR, initc, get_xj, ForceCustom, HopDirCustom # import the functions from the model
    
@jit(nopython=True)
def expE(sd, dt):
    sd.c[:] = np.exp(-1j*dt*sd.E)*sd.c
    
@jit(nopython=True)
def createEU(sd):
    sd.E[:], sd.U[:,:] = np.linalg.eigh(H(sd.R, sd.xj))
    overlap_mat = np.conj(sd.U).T
    for n in range(m.NStates):
        f = np.exp(1j*np.angle(overlap_mat[n,n]))
        sd.U[:,n] = f * sd.U[:,n] # phase correction
    
@jit(nopython=True)
def createEU_GTC(sd):
    sd.E_GTC[:], sd.U_GTC[:,:] = np.linalg.eigh(H(np.zeros(m.NR), sd.xj))
    overlap_mat = np.conj(sd.U).T
    for n in range(m.NStates):
        f = np.exp(1j*np.angle(overlap_mat[n,n]))
        sd.U_GTC[:,n] = f * sd.U_GTC[:,n] # phase correction
    
@jit(nopython=True)
def updateEU(sd0, sd1):
    sd1.E[:], sd1.U[:,:] = np.linalg.eigh(H(sd1.R, sd1.xj))
    overlap_mat = np.conj(sd1.U).T @ sd0.U
    for n in range(m.NStates):
        f = np.exp(1j*np.angle(overlap_mat[n,n]))
        sd1.U[:,n] = f * sd1.U[:,n] # phase correction
        
@jit(nopython=True)
def DtoA_2D(matD, sd): # convert operator from diabatic to adiabatic basis
    return np.conj(sd.U).T @ matD @ sd.U

@jit(nopython=True)
def DtoA_3D(matD, sd): # convert operator from diabatic to adiabatic basis
    matA = np.zeros(np.shape(matD), dtype = np.complex128)
    for i in range(len(matD[:,0,0])):
        matA[i] = np.conj(sd.U).T @ matD[i] @ sd.U
    return matA
        
@jit(nopython=True)
def updateForce(sd): # this calculates the classical force on each nuclear DOF using only the active state
    if(m.useForceCustom):
        ForceCustom(sd)
    else:
        sd.F[:] = -dH0(sd) # state-independent force
        sd.F[:] += np.real(-DtoA_3D(dH(sd), sd)[:,sd.acst,sd.acst]) # state-dependent force for diabatic dH
    
@jit(nopython=True)
def PropH(sd0, sd1, dt):
    expE(sd1, dt/2)
    sd1.P[:] += 0.5 * sd0.F * dt
    sd1.R[:] += sd1.P / sd1.M * dt
    updateEU(sd0, sd1)
    updateForce(sd1)
    sd1.P[:] += 0.5 * sd1.F * dt
    sd1.c[:] = np.conj(sd1.U).T @ sd0.U @ sd1.c # NECESSARY FOR DYNAMICS: rotate c to (new) adiabatic basis
    expE(sd1, dt/2)
        
@jit(nopython=True)
def getHopDir(sd, hd):   
    if(m.useHopDirCustom):
        return HopDirCustom(sd, hd)
    hop_dir = np.zeros(m.NR)
    dc = np.zeros((m.NR, m.NStates, m.NStates), dtype=np.complex128)
    dE = np.expand_dims(sd.E, axis=1) - sd.E
    for i in range(m.NStates):
        dE[i,i] = 1.0
    for i in range(m.NR):
        dc[i,:,:] = DtoA_2D(dH(sd)[i], sd) / dE
        for j in range(m.NStates):
            dc[i,j,j] = 0.0
    for k in range(m.NR):
        for n in range(m.NStates): # general rescaling direction, slow
            hop_dir[k] += np.real(dc[k,n,sd.acst]*sd.c[sd.acst]*np.conj(sd.c[n])-dc[k,n,hd.acst_att]*sd.c[hd.acst_att]*np.conj(sd.c[n])) / np.sqrt(sd.M[k]) 
    return hop_dir
        
@jit(nopython=True)
def hop(sd, hd): # attempt a hop
    hd.rescale_mag = 1.0
    sd.P[:] = sd.P/np.sqrt(sd.M) # mass-scaled momentum
    potdiff = np.real(sd.E[hd.acst_att] - sd.E[sd.acst])
    hop_dir = getHopDir(sd, hd)
    P_proj = np.dot(sd.P,hop_dir) * hop_dir / np.dot(hop_dir,hop_dir) # projected P along hop_dir
    P_proj_norm = np.sqrt(np.dot(P_proj,P_proj))
    P_orth = sd.P - P_proj # orthogonal P
    if(P_proj_norm**2 < 2*potdiff): # rejected hop
        P_proj = -P_proj # reverse projected momentum
        sd.P[:] = P_orth + P_proj
        hd.accepted = False
        sd.P[:] = sd.P*np.sqrt(sd.M)
    else: # accepted hop
        hd.rescale_mag = np.sqrt(P_proj_norm**2 - 2*potdiff)/P_proj_norm
        P_proj = np.sqrt(P_proj_norm**2 - 2*potdiff)/P_proj_norm * P_proj # scale projected momentum
        sd.P[:] = P_orth + P_proj
        hd.accepted = True
        sd.P[:] = sd.P*np.sqrt(sd.M)
        
@jit(nopython=True)
def pop_diff(sd): # return population difference between current active state and other highest populated state
    pop = np.abs(sd.c)**2
    pop[sd.acst] = 0.0
    diff = np.abs(sd.c[sd.acst])**2 - np.max(pop)
    if(np.abs(diff)>10**(-15)):
        return diff
    else:
        return 0.0 # return 0 if difference is too small (avoids floating point issues)
    
@jit(nopython=True)
def getACST_att(sd, hd): # return population difference between current active state and other highest populated state
    pop = np.abs(sd.c)**2
    pop[sd.acst] = 0.0
    hd.acst_att = np.argmax(pop)
    
@jit(nopython=True)
def rho(c): # returns the density matrix estimator (populations and coherences)
    return m.alpha * np.outer(c,np.conj(c)) - m.beta * np.identity(m.NStates) # works in any basis

@jit(nopython=True)
def getACST(sd): # return state with largest population
    sd.acst = np.argmax(np.abs(sd.c))
        
@jit(nopython=True)
def dPeq0(sd0, sd1, hd): # dP = 0.0
    hd.t0 += hd.dt
    hd.dt = 0.0
    hd.dt_r = 0.0
    sd0.copy(sd1)

@jit(nopython=True)
def dPlt0(sd0, sd1, hd): # dP < 0.0
    hd.dt_r = 1.0*hd.dt

@jit(nopython=True)
def dPgt0(sd0, sd1, hd): # dP > 0.0
    hd.t0 += hd.dt
    hd.dt_r -= hd.dt
    hd.dt = 1.0*hd.dt_r
    sd0.copy(sd1)
    PropH(sd0, sd1, hd.dt)
    sd1.dP = pop_diff(sd1)
    if(sd1.dP>0.0):
        if(m.dtF - (hd.t_st + hd.t0) - hd.dt < 10**-12): # full dtF finished without needing hop
            sd0.copy(sd1)
            hd.attempt = False
            return
        hd.t0 += hd.dt
        hd.dt_r = m.dtF - (hd.t_st + hd.t0)
        hd.dt = 1.0*hd.dt_r
        sd0.copy(sd1)
        PropH(sd0, sd1, hd.dt)
        sd1.dP = pop_diff(sd1)
        if(sd1.dP>0.0): # full dtF finished without needing hop
            sd0.copy(sd1)
            hd.attempt = False
            return
        elif(sd1.dP==0.0):
            dPeq0(sd0, sd1, hd)
        elif(sd1.dP<0.0):
            dPlt0(sd0, sd1, hd)
    elif(sd1.dP==0.0):
        dPeq0(sd0, sd1, hd)
    elif(sd1.dP<0.0):
        dPlt0(sd0, sd1, hd)
    
@jit(nopython=True)
def Interpolation(sd0, sd1, hd):
    hd.dt = hd.dt * pop_diff(sd0) / (pop_diff(sd0) - pop_diff(sd1))
    if(hd.dt < m.dtLB):
        hd.dt = 1.0*m.dtLB
    elif(hd.dt_r/2 - m.dtLB < hd.dt < hd.dt_r/2):
        hd.dt = hd.dt_r/2 - m.dtLB
    elif(hd.dt_r/2 <= hd.dt < hd.dt_r/2 + m.dtLB):
        hd.dt = hd.dt_r/2 + m.dtLB
    elif(hd.dt_r - 2*m.dtLB < hd.dt):
        hd.dt = hd.dt_r - 2*m.dtLB

@jit(nopython=True)
def Interpolate_Propagate(sd0, sd1, hd): # interpolate and propagate to find precise hopping time t0
    Interpolation(sd0, sd1, hd)
    sd1.copy(sd0)
    PropH(sd0, sd1, hd.dt)
    sd1.dP = pop_diff(sd1)
    if(hd.dt < hd.dt_r/2): # interpolation in first half of dt_r
        if(sd1.dP>0.0):
            hd.t0 += hd.dt
            hd.dt = hd.dt_r/2 - hd.dt
            hd.dt_r = hd.dt_r/2 + hd.dt
            sd0.copy(sd1)
            PropH(sd0, sd1, hd.dt)
            sd1.dP = pop_diff(sd1)
            if(sd1.dP>0.0):
                dPgt0(sd0, sd1, hd)
            elif(sd1.dP==0.0):
                dPeq0(sd0, sd1, hd)
            elif(sd1.dP<0.0):
                dPlt0(sd0, sd1, hd)
        elif(sd1.dP==0.0):
            dPeq0(sd0, sd1, hd)
        elif(sd1.dP<0.0):
            dPlt0(sd0, sd1, hd)
    elif(hd.dt >= hd.dt_r/2): # interpolation in second half of dt_r
        if(sd1.dP>0.0):
            dPgt0(sd0, sd1, hd)
        elif(sd1.dP==0.0):
            dPeq0(sd0, sd1, hd)
        elif(sd1.dP<0.0):
            hd.dt -= hd.dt_r/2
            hd.dt_r = hd.dt_r/2 + hd.dt
            hd.dt = hd.dt_r - hd.dt
            sd1.copy(sd0)
            PropH(sd0, sd1, hd.dt)
            sd1.dP = pop_diff(sd1)
            if(sd1.dP>0.0):
                dPgt0(sd0, sd1, hd)
            elif(sd1.dP==0.0):
                dPeq0(sd0, sd1, hd)
            elif(sd1.dP<0.0):
                dPlt0(sd0, sd1, hd)
         
@jit(nopython=True)   
def hop_setup(sd0, sd1, hd): # determine rescaling conditions and perform hop attempt
    print("Hop attempt!")
    getACST_att(sd1, hd)
    hop(sd1, hd)
    if(hd.accepted):
        hd.t_st += hd.dt
        sd1.acst = 1*hd.acst_att
        updateForce(sd1)
    if(hd.N_int == m.max_int or hd.N_hop_att == m.max_hop_att):
        hd.attempt = False
    sd0.copy(sd1)
    
@jit(nopython=True)   
def hop_maxed(sd0, sd1, hd): # if max interpolations or hop attempts reached, complete full timestep
    hd.dt = m.dtF - (hd.t_st + hd.t0)
    sd1.copy(sd0)
    PropH(sd0, sd1, hd.dt)
    sd1.dP = pop_diff(sd1)
    if(sd1.dP>0):
        sd0.copy(sd1)
        hd.attempt = False
    
@jit(nopython=True) # do a full step of propagation
def fullStep(sd0, sd1, hd):
    sd1.copy(sd0)
    hd.dt = 1.0*m.dtF
    if(m.hop_type=='express'):
        PropH(sd0, sd1, hd.dt)
        sd1.dP = pop_diff(sd1)
        if(sd1.dP<=0.0):
            hop_setup(sd0, sd1, hd)
    elif(m.hop_type=='full'):
        hd.t_st = 0.0
        hd.t0 = 0.0
        hd.dt_r = 1.0*hd.dt
        hd.N_hop_att = 0
        hd.attempt = True
        while(hd.N_hop_att < m.max_hop_att and hd.attempt == True): # do hops until no hop needed or max_hop_att reached
            print("N_hop_att: ", hd.N_hop_att)
            hd.N_int = 0
            PropH(sd0, sd1, hd.dt)
            sd1.dP = pop_diff(sd1)
            if(sd1.dP>0.0): # no more hops needed
                sd0.copy(sd1)
                hd.attempt = False
            elif(sd1.dP==0.0): # attempt hop now
                sd0.copy(sd1)
            elif(sd1.dP<0.0):
                while(hd.dt > m.dtUB and hd.attempt == True and hd.N_int < m.max_int):
                    Interpolate_Propagate(sd0, sd1, hd)
                    hd.N_int += 1
            if((hd.N_int == m.max_int or hd.N_hop_att == m.max_hop_att) and hd.dt > m.dtLB/2 and hd.attempt==True):
                print("!!! N_int / N_hop_att too big !!!", hd.N_int, hd.N_hop_att)
                hop_maxed(sd0, sd1, hd)
            if(hd.attempt==False): # don't attempt a hop
                break
            hd.N_hop_att += 1
            hd.t_st += hd.t0
            hop_setup(sd0, sd1, hd)
            hd.t0 = 0.0
            hd.dt = m.dtF - hd.t_st
            hd.dt_r = 1.0*hd.dt
            if(hd.dt < m.dtLB/2):
                hd.attempt = False
                break

def runTraj():
    
    # Create output arrays
    
    '''popACST = np.zeros((m.NStates,m.NStepsPrint+1))
    popADI = np.zeros((m.NStates,m.NStepsPrint+1))
    popTC = np.zeros((m.NStates,m.NStepsPrint+1))
    popDIA = np.zeros((m.NStates,m.NStepsPrint+1))'''
    
    Hinputs_index = np.arange(1,1+m.NMol)
    
    psi = np.zeros((m.NTraj,m.NStates,m.NStepsPrint+1), dtype=np.complex128) # stores wavefunction in desired basis for each (printed) timestep and for each trajectory
    Hinputs = np.zeros((m.NTraj,m.NMol,m.NStepsPrint+1)) # stores average nuclear position for each (printed) timestep and for each trajectory
    active_state = np.zeros((m.NTraj,m.NStepsPrint+1)) # stores active state for each (printed) timestep and for each trajectory
    xjs = np.zeros((m.NTraj,m.NMol)) # stores wavefunction in desired basis for each (printed) timestep and for each trajectory
    rho_ensemble = np.zeros((m.NStates,m.NStates,m.NStepsPrint+1), dtype=np.complex128)
    
    for itraj in range(m.NTraj): # repeat simulation for each trajectory
        
        # Initialize state and hop class objects
        
        sd0 = m.state_data() # state data before timestep propagation
        sd1 = m.state_data() # state data after timestep propagation
        hd = m.hop_data()
        
        # Initialize trajectory information
        
        initR(sd0) # initialize nuclear positions and momenta
        get_xj(sd0)
        createEU(sd0)
        createEU_GTC(sd0)
        initc(sd0)
        getACST(sd0)
        updateForce(sd0) # initial force
        sd0.M[:] = m.M
        
        iskip = 0 # counting variable to determine when to store the current timestep data
        for t in range(m.NSteps): # single trajectory
            print("t: ", t)
            
            # Save output variables
            
            if(t % m.NSkip == 0):
                Hinputs[itraj,:,iskip] = np.real(H(sd0.R, sd0.xj)[Hinputs_index,Hinputs_index]) # store nuclear positions
                active_state[itraj, iskip] = sd0.acst # store active state
                if(m.outputBasis==0): # adiabatic basis
                    psi[itraj,:,iskip] = sd0.c # store wavefunction
                    rho_ensemble[:,:,iskip] += rho(sd0.c)
                elif(m.outputBasis==1): # no-nuclei eigenbasis
                    psi[itraj,:,iskip] = np.conj(sd0.U_GTC).T @ sd0.U @ sd0.c # store wavefunction
                    rho_ensemble[:,:,iskip] += rho(np.conj(sd0.U_GTC).T @ sd0.U @ sd0.c)
                elif(m.outputBasis==2): # diabatic basis
                    psi[itraj,:,iskip] = sd0.U @ sd0.c # store wavefunction
                    rho_ensemble[:,:,iskip] += rho(sd0.U @ sd0.c)
                iskip += 1
                
            # Do full timestep
            
            fullStep(sd0, sd1, hd)
            
        # Save output variables for last time
        
        Hinputs[itraj,:,iskip] = np.real(H(sd0.R, sd0.xj)[Hinputs_index,Hinputs_index]) # store nuclear positions
        active_state[itraj, iskip] = sd0.acst # store active state
        if(m.outputBasis==0): # adiabatic basis
            psi[itraj,:,iskip] = sd0.c # store wavefunction
            rho_ensemble[:,:,iskip] += rho(sd0.c)
        elif(m.outputBasis==1): # no-nuclei eigenbasis
            psi[itraj,:,iskip] = np.conj(sd0.U_GTC).T @ sd0.U @ sd0.c # store wavefunction
            rho_ensemble[:,:,iskip] += rho(np.conj(sd0.U_GTC).T @ sd0.U @ sd0.c)
        elif(m.outputBasis==2): # diabatic basis
            psi[itraj,:,iskip] = sd0.U @ sd0.c # store wavefunction
            rho_ensemble[:,:,iskip] += rho(sd0.U @ sd0.c)
        
    # Return output variables
    
    return psi, Hinputs, active_state, xjs, rho_ensemble