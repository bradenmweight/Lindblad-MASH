import numpy as np
from numba import jit, objmode
import model as m
from numpy import random as rn
    
from model import H, dH, dH0, initR # import the functions from the model



def initc(sd): # initialization of the coefficients
    sd.c = np.sqrt(sd.beta/sd.alpha) * np.ones((m.NStates), dtype=np.complex128)
    sd.c[m.initState] = np.sqrt((1+sd.beta)/sd.alpha)
    for n in range(m.NStates):
        sd.c[n] = sd.c[n] * np.exp( 1j* 2 * np.pi * rn.random() )
    #sd.c[:] = np.conj(U).T @ U_GTC @ c
    return sd

def expE(sd, dt):
    sd.c[:] = np.exp(-1j*dt*sd.E)*sd.c
    
def get_adiabatic_E_U(sd):
    sd.E[:], sd.U[:,:] = np.linalg.eigh( H(sd.R) )
    overlap_mat = np.conj(sd.U).T
    for n in range(m.NStates):
        f = np.exp(1j*np.angle(overlap_mat[n,n]))
        sd.U[:,n] = f * sd.U[:,n] # phase correction
    
def updateEU(sd0, sd1):
    sd1.E[:], sd1.U[:,:] = np.linalg.eigh( H(sd1.R) )
    overlap_mat = np.conj(sd1.U).T @ sd0.U
    for n in range(m.NStates):
        f = np.exp(1j*np.angle(overlap_mat[n,n]))
        sd1.U[:,n] = f * sd1.U[:,n] # phase correction
        
def DtoA_2D(matD, sd): # convert operator from diabatic to adiabatic basis
    return np.conj(sd.U).T @ matD @ sd.U

def DtoA_3D(matD, sd): # convert operator from diabatic to adiabatic basis
    matA = np.zeros(np.shape(matD), dtype = np.complex128)
    for i in range(len(matD[:,0,0])):
        matA[i] = np.conj(sd.U).T @ matD[i] @ sd.U
    return matA
        
def updateForce(sd): # this calculates the classical force on each nuclear DOF using only the active state
    sd.F[:] = -dH0(sd.R) # state-independent force
    dH_dia  = dH(sd.R) # diabatic dH
    dH_ad   = DtoA_3D(dH(sd.R), sd)[:,sd.acst,sd.acst] # adiabatic dH, only active state
    sd.F[:] += np.real(-dH_ad) # state-dependent force for diabatic dH
    
def PropH(sd0, sd1, dt):
    expE(sd1, dt/2)
    sd1.P[:] += 0.5 * sd0.F * dt
    sd1.R[:] += sd1.P / sd1.M * dt
    updateEU(sd0, sd1)
    updateForce(sd1)
    sd1.P[:] += 0.5 * sd1.F * dt
    sd1.c[:] = np.conj(sd1.U).T @ sd0.U @ sd1.c # NECESSARY FOR DYNAMICS: rotate c to (new) adiabatic basis
    expE(sd1, dt/2)
        
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
        
def pop_diff(sd): # return population difference between current active state and other highest populated state
    pop = np.abs(sd.c)**2
    pop[sd.acst] = 0.0
    diff = np.abs(sd.c[sd.acst])**2 - np.max(pop)
    if(np.abs(diff)>10**(-15)):
        return diff
    else:
        return 0.0 # return 0 if difference is too small (avoids floating point issues)
    
def getACST_att(sd, hd): # return population difference between current active state and other highest populated state
    pop = np.abs(sd.c)**2
    pop[sd.acst] = 0.0
    hd.acst_att = np.argmax(pop)
    
def rho(sd, rotation=None): # returns the density matrix estimator (populations and coherences)
    if ( rotation is not None ):
        c_dia = rotation @ sd.c
        return sd.alpha * np.outer(c_dia,np.conj(c_dia)) - sd.beta * np.identity(m.NStates)
    else:
        return sd.alpha * np.outer(sd.c,np.conj(sd.c)) - sd.beta * np.identity(m.NStates) # works in any basis

def getACST(sd): # return state with largest population
    sd.acst = np.argmax(np.abs(sd.c))
         
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
    # sd0.copy(sd1)
    sd1 = sd0
    
def fullStep(sd0, sd1, hd):
    # sd0.copy(sd1)
    sd1 = sd0
    hd.dt = 1.0*m.dtF
    PropH(sd0, sd1, hd.dt)
    sd1.dP = pop_diff(sd1)
    if(sd1.dP<=0.0):
        hop_setup(sd0, sd1, hd)


def runTraj():
    
    # Create output arrays
    
    active_state     = np.zeros((m.NTraj,m.NStepsPrint+1)) # stores active state for each (printed) timestep and for each trajectory
    psi_ad           = np.zeros((m.NTraj,m.NStates,m.NStepsPrint+1),   dtype=np.complex128) # stores wavefunction in desired basis for each (printed) timestep and for each trajectory
    rho_ensemble_ad  = np.zeros((m.NStates,m.NStates,m.NStepsPrint+1), dtype=np.complex128)
    psi_dia          = np.zeros((m.NTraj,m.NStates,m.NStepsPrint+1),   dtype=np.complex128) # stores wavefunction in desired basis for each (printed) timestep and for each trajectory
    rho_ensemble_dia = np.zeros((m.NStates,m.NStates,m.NStepsPrint+1), dtype=np.complex128)
    
    for itraj in range(m.NTraj): # repeat simulation for each trajectory
        print( itraj )
        # Initialize state and hop class objects
        sd0 = state_data() # state data before timestep propagation
        sd1 = state_data() # state data after timestep propagation
        hd  = hop_data()
        
        # Initialize trajectory information
        sd0.R, sd0.P = initR() # initialize nuclear positions and momenta
        initc(sd0) # initialize nuclear positions and momenta
        get_adiabatic_E_U(sd0)
        getACST(sd0)
        updateForce(sd0) # initial force
        sd0.M[:] = m.M
        
        iskip = 0 # counting variable to determine when to store the current timestep data
        for t in range(m.NSteps): # single trajectory
            #print("t: ", t)
            
            # Save output variables
            if(t % m.NSkip == 0):
                active_state[itraj, iskip]   = sd0.acst
                psi_ad[itraj,:,iskip]        = sd0.c
                psi_dia[itraj,:,iskip]       = sd0.U @ sd0.c
                rho_ensemble_ad[:,:,iskip]  += rho(sd0)
                rho_ensemble_dia[:,:,iskip] += rho(sd0, rotation=sd0.U)
                iskip += 1
                
            # Do full timestep
            fullStep(sd0, sd1, hd)
            
        # Save output variables for last time
        active_state[itraj, iskip]   = sd0.acst
        psi_ad[itraj,:,iskip]        = sd0.c
        psi_dia[itraj,:,iskip]       = sd0.U @ sd0.c
        rho_ensemble_ad[:,:,iskip]  += rho(sd0)
        rho_ensemble_dia[:,:,iskip] += rho(sd0, rotation=sd0.U)
        
    # Return output variables
    time = np.linspace( 0.0, m.NSteps * m.dtF, m.NStepsPrint+1 )
    return time, active_state, psi_ad, rho_ensemble_ad/m.NTraj, psi_dia, rho_ensemble_dia/m.NTraj


















### Data classes for storing trajectory information ###

class state_data(object): # storage object for state variables (wavefunction, nuclei, etc.)
    def __init__(self):
        
        # Necessary MASH variables
        self.c = np.zeros(m.NStates, dtype=np.complex128) # state wavefunction
        self.acst = 0 # active state
        self.R = np.zeros(m.NR)
        self.P = np.zeros(m.NR)
        self.F = np.zeros(m.NR)
        self.M = np.zeros(m.NR)
        self.E = np.zeros(m.NStates) # H eigenenergies
        self.U = np.zeros((m.NStates,m.NStates), dtype=np.complex128) # H eigenvectors
        self.dP = 0.0
        self.sumN = np.sum(np.array([1/n for n in range(1,m.NStates+1)]))
        self.alpha = (m.NStates - 1)/(self.sumN - 1)
        self.beta = (self.alpha - 1)/m.NStates
        self.ZPE = self.beta/self.alpha


    # # Copy values from other state_data object
    # def copy(self, sd_new):
    #     with objmode():
    #         for i in range(len(state_types)):
    #             setattr(self,state_types[i][0],copy.deepcopy(getattr(sd_new,state_types[i][0])))
        
class hop_data(object): # storage object for hopping variables (timesteps, hop conditions, etc.)
    def __init__(self):
        
        # Time variables
        
        self.dt = 0.0 # current timestep used to propagate
        self.t_st = 0.0 # starting time of hop attempt relative to full timestep
        self.t0 = 0.0 # time relative to t_st
        self.dt_r = 0.0 # smallest timestep of propagation from t0 that changes most populated state
        
        # Hopping conditions
        
        self.acst_att = 0 # state to attempt hop towards
        self.N_hop_att = 0 # number of hop attempts performed during current full timestep
        self.N_int = 0 # number of interpolations performed during current hop attempt
        self.d_dt_H = 0.0 # acst-acst_att population change time derivative due to Hamiltonian
        self.d_dt_L = 0.0 # acst-acst_att population change time derivative due to Lindblad
        self.attempt = True # keep attempting current hop
        self.accepted = True # current hop attempt is energetically acceptable
        self.accepted_test = True # test hop attempt is energetically acceptable
        self.rescale = False # rescale momenta after hop attempt
        self.reverse = False # reverse momenta after hop attempt
        self.rescale_mag = 0.0 # rescale factor of momenta along projected direction
