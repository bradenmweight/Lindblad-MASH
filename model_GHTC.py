import numpy as np
from numba import int32, float64, complex128, boolean
from numba.experimental import jitclass
from numba import jit, objmode
from math import gamma as GammaF
from numpy import random as rn
import copy

### Conversion constants ###

eV_au = 0.0367493036 # eV to au
meV_au = 0.0000367493036 # meV to au
ps_au = 41341.374575751 # picosecond to au
fs_au = 41.341374575751 # femtosecond to au
c = 137.03599 # speed of light in au

### Functions for defining parameters ###

@jit(nopython=True)
def OhmicFamily(ω, ωm, λ, s, cutoff): # this defines the bath using the Ohmic spectral density
    if(cutoff=='exp'):
        fm = np.exp(-ω/ωm)
        C = λ / (ωm**s * GammaF(s))
    elif(cutoff=='gaus'):
        fm = np.exp(-(ω/(np.sqrt(2/s)*ωm))**2)
        C = 2*λ / ((np.sqrt(2/s)*ωm)**s * GammaF(s/2))
    elif(cutoff=='dl'):
        fm = ωm**2 / (ω**2 + ωm**2)
        C = 2*np.sin(s*np.pi/2)*λ / (np.pi * ωm**s)
    return C * ω**s * fm

@jit(nopython=True)
def bathParam(N, ωm, λ, s, cutoff): # this calculates the parameters of the bath based on the spectral density function
    ω = np.linspace(np.log(10**(-4)*ωm),np.log(100*ωm),20000)
    ω = np.power(np.exp(1), ω).astype(ω.dtype)
    dω = ω[1:] - ω[:-1]
    Jω = OhmicFamily(ω, ωm, λ, s, cutoff)
    Fω = np.zeros((len(dω)))
    for iω in range(len(dω)):
        Fω[iω] = np.sum( Jω[:iω]/ω[:iω] * dω[:iω] ) 
    λs = Fω[-1] 
    ωk = np.zeros((N))
    ck = np.zeros((N))
    for i in range(N):
        j = i+1
        ωk[i] = ω[np.argmin(np.abs(Fω - ((j-0.5)/N) *  λs))] 
        ck[i] = 2.0 * ωk[i] * (λs/(2.0 * N))**0.5 
    dk = 0.5 * ck**2.0 / ωk**3
    Rk = np.sqrt(2.0 * dk/ ωk)
    return Rk, ωk

### Parameters ###

# Trajectory/Timing parameters

NSteps = 2000 # number of full steps in the simulation
NStepsPrint = 500 # number of full steps that are stored/printed/outputed
NSkip = int(NSteps/NStepsPrint) # used to enforce NStepsPrint
totalTime = 1.000*ps_au # total amount of simulation time
dtF = totalTime/NSteps # full timestep 
NTraj = 1

# System parameters

NMol = 20 # number of molecules (one donor state and one highly excited state per molecule)
NMod = 20 # number of photon modes (only positive angles included)
NStates = 1 + NMol + NMod # total number of states
NBath = 50 # number of nuclear DOFs for one molecule
NR = NMol*NBath # total number of nuclear DOFs
M = np.ones(NR) # mass of each nuclear DOF (for bath DOFs, this is arbitrary)
ωm = 0.004*eV_au
λHH = 0.005*eV_au
s = 3
cutoff = 'gaus'
RkHH, ωkHH = bathParam(NBath, ωm, λHH, s, cutoff)
dHCustom = -ωkHH**2 * RkHH # H derivative array
β = 1053 # 1/kT of bath in au (1053 is 300K)
ΔEHH = 2.422*eV_au - λHH # energy of donor in eV
ωc0 = 2.421*eV_au # lowest energy photon mode in eV
nr = 1.5 # refractive index of cavity
ωc_max = ωc0 + 0.150*eV_au # highest energy photon mode in eV
kz = ωc0 * (nr/c) # z-component of photon mode momentum
kx_max = np.sqrt((ωc_max * (nr/c))**2-kz**2) # largest x-component of photon mode momentum
Lx = 2 * np.pi * (NMod-1)/kx_max # length of cavity in x direction (parallel to plates)
kx = 2 * np.pi * np.arange(0,NMod) / Lx # array of x-component of photon mode momentum
ωc = (c/nr) * (kx**2.0 + kz**2.0)**0.5 # array of photon energies
gHH0 = 0.0155*eV_au/np.sqrt(NMol) # 0-indidence light-matter coupling (number is total collective coupling in eV)
θ = np.arctan(kx/kz) # angles of photon modes 
gHH = gHH0 * (ωc/ωc0)**0.5 #* np.cos(θ) # light-matter coupling for each mode
useForceCustom = True
useHopDirCustom = True

# Lindblad parameters

LBasis = 2 # 0 = adiabatic, 1 = TC eigen (no nuclei), 2 = diabatic-Fock, -1 = no loss
Q = 300
ΓG1toG0 = ωc0 / Q # cavity loss rate
ΓG0toHH = 0.000*eV_au/NMol # incoherent Lindblad decay rate from highly excited state to donor state
ΓHHtoG0 = 1 / (30*ps_au) # matter loss rate
Γpulse = 1 / (100*fs_au)
NJump = NMol + NMod # number of jump operators
dontRescaleStates = np.array([0]) # adiabatic states that should not cause rescaling or frustrated hops (e.g. collective ground state G0)

# Initialization/Output parameters

initBasis = 0 # 0 = adiabatic, 1 = TC eigen (no nuclei), 2 = diabatic-Fock
initState = NStates//2
outputBasis = 0 # 0 = adiabatic, 1 = TC eigen (no nuclei), 2 = diabatic-Fock
sample_type = 'focused' # 'focused' or 'gaussian', probably not important

# MASH hop parameters

hop_type = 'full' # 'express' is faster, 'full' uses max_hop and num_bisect
max_hop_att = 50 # max number of attempted hops during a full timestep
max_int = 100 # max number of interpolations during a hop attempt
tolUB = 10**(-4) # upper bound tolerance for timestep size during interpolations
tolLB = 10**(-6) # lower bound tolerance for timestep size during interpolations (prevents floating point errors)
dtUB = dtF * tolUB
dtLB = dtF * tolLB

# MASH constants (ignore)

sumN = np.sum(np.array([1/n for n in range(1,NStates+1)]))
alpha = (NStates - 1)/(sumN - 1)
beta = (alpha - 1)/NStates
ZPE = beta/alpha

### Functions ###

@jit(nopython=True)
def get_xj(sd): # create molecule positions for light-matter coupling in GHTC Hamiltonian
    sd.xj[:] = np.arange(0,NMol) * (Lx/(NMol-1)) # positions of molecules (evenly spaced)
    #sd.xj[:] = np.sort(rn.rand(NMol)*Lx) # positions of molecules (random)

@jit(nopython=True)
def H(R, xj): # GHTC Hamiltonian (collective ground state energy has been set to 0)
    H = np.zeros((NStates,NStates), dtype = np.complex128)
    H[0,0] = 0 # collective ground state energy
    for j in range(NMol):
        H[1+j, 1+j] = ΔEHH # HH energy shift
        H[1+j, 1+j] += np.sum(0.5 * ωkHH**2 * (R[j*NBath:(j+1)*NBath]-RkHH)**2) # HH bath energy for jth molecule
        H[1+j, 1+j] -= np.sum(0.5 * ωkHH**2 * (R[j*NBath:(j+1)*NBath])**2) # subtract ground state energy for jth molecule
    for k in range(NMod):
        H[1+NMol+k, 1+NMol+k] = ωc[k] #  # kth photon mode energy
    for j in range(NMol):
        for k in range(NMod):
            fx = np.exp(1j * kx[k] * xj[j]) # light-matter coupling phase of kth mode and jth molecule (no phase when NMod = 1)
            H[1+NMol+k, 1+j] = gHH[k] * fx # light-matter coupling between kth mode and jth molecule
            H[1+j, 1+NMol+k] = np.conj(gHH[k] * fx) # light-matter coupling between kth mode and jth molecule (complex conjugate)
    return H

@jit(nopython=True)
def dH(sd): # derivative of state-dependent potential
    dH = np.zeros((NR,NStates,NStates), dtype = np.complex128)
    for j in range(NMol):
        dH[j * NBath : (j + 1) * NBath,1+j,1+j] = -ωkHH**2 * RkHH
    return dH

@jit(nopython=True)
def dH0(sd): # derivative of state-independent potential which is the collective ground state (needs to be included even though H had ground state energy removed)
    dH0 = np.zeros(NR) # state independent, so only need to do NR derivatives once
    for j in range(NMol):
        dH0[j * NBath : (j + 1) * NBath] = ωkHH**2 * sd.R[j * NBath : (j + 1) * NBath] # derivative of unshifted harmonic oscillator
    return dH0

@jit(nopython=True)
def initR(sd): # initialize R and P
    R0 = 0.0 # average initial nuclear position (could be array specific to each nuclear DOF)
    P0 = 0.0 # average initial nuclear momentum (could be array specific to each nuclear DOF)
    #σPHH = np.sqrt(ωkHH/(2.0 * np.tanh(0.5 * β * ωkHH))) # Wigner standard dev of nuclear momentum (mass = 1)
    #σRHH = σPHH/(ωkHH) # Wigner standard dev of nuclear position (mass = 1)
    σPHH = np.sqrt(1/β)*np.ones(NBath) # classical distribution
    σRHH = 1/np.sqrt(β*ωkHH**2) # classical distribution
    for Ri in range(NR):
        imode = (Ri%NBath) # current bath mode
        sd.R[Ri] = rn.normal() * σRHH[imode] + R0 # gaussian random variable centered around R0
        sd.P[Ri] = rn.normal() * σPHH[imode] + P0 # gaussian random variable centered around P0

@jit(nopython=True)
def initc(sd): # initialization of the coefficients
    if(sample_type=='focused'): # focused sampling
        sd.c[:] = np.sqrt(beta/alpha) * np.ones((NStates), dtype = np.complex128)
        sd.c[initState] = np.sqrt((1+beta)/alpha)
        for n in range(NStates):
            uni = rn.random()
            sd.c[n] = sd.c[n] * np.exp(1j*2*np.pi*uni)
    elif(sample_type=='gaussian'): # sampling with gaussian random components
        while(True):
            rand_pairs = np.zeros((NStates,2))
            for n in range(NStates):
                rand_pairs[n,0] = rn.normal()
                rand_pairs[n,1] = rn.normal()
            c_mag = rand_pairs[:,0]**2+rand_pairs[:,1]**2
            if(c_mag[initState] == np.max(c_mag)):
                break
        norm = np.sqrt(np.sum(c_mag))
        sd.c[:] = (rand_pairs[:,0]+1j*rand_pairs[:,1])/norm
    if(initBasis==1): # GTC eigenbasis
        sd.c[:] = np.conj(sd.U).T @ sd.U_GTC @ sd.c
    if(initBasis==2): # diabatic basis
        sd.c[:] = np.conj(sd.U).T @ sd.c
        
@jit(nopython=True)
def LBasisChange(sd): # initialization of the coefficients
    if(LBasis == 0): # GHTC eigenbasis
        pass
    elif(LBasis == 1): # GTC eigenbasis
        sd.c[:] = np.conj(sd.U_GTC.T) @ sd.U @ sd.c
    elif(LBasis == 2): # diabatic-Fock basis
        sd.c[:] = sd.U @ sd.c
        
@jit(nopython=True)
def LBasisChangeRevert(sd): # initialization of the coefficients
    if(LBasis == 0): # GHTC eigenbasis
        pass
    elif(LBasis == 1): # GTC eigenbasis
        sd.c[:] = np.conj(sd.U.T) @ sd.U_GTC @ sd.c
    elif(LBasis == 2): # diabatic-Fock basis
        sd.c[:] = np.conj(sd.U).T @ sd.c
        
@jit(nopython=True)
def getLRates(sd): # initialization of the coefficients
    LRates = np.zeros(NJump)
    if(LBasis == 0): # GHTC eigenbasis
        polPC = np.array([np.sum(np.abs(sd.U[NMol+1:,ii])**2) for ii in range(NStates)])
        LRates[:NStates-1] = ΓG0toHH*(1-polPC[1:]) # incoherent driving
        LRates[NStates-1:] = ΓG1toG0*polPC[1:] # cavity loss
    elif(LBasis == 1): # GTC eigenbasis
        polPC = np.array([np.sum(np.abs(sd.U_GTC[NMol+1:,ii])**2) for ii in range(NStates)])
        LRates[:NStates-1] = ΓG0toHH*(1-polPC[1:]) # incoherent driving
        LRates[NStates-1:] = ΓG1toG0*polPC[1:] # cavity loss
    elif(LBasis == 2): # diabatic-Fock basis
        LRates[:NMol] = ΓG0toHH*np.ones(NMol) # incoherent driving
        LRates[NMol:] = ΓG1toG0*np.ones(NMod) # cavity loss
    return LRates

@jit(nopython=True)
def getLStates(sd): # initialization of the coefficients
    LStates = np.zeros((NJump,2), dtype=np.int32)
    if(LBasis == 0): # GHTC eigenbasis
        for n in range(1,NStates):
            LStates[n-1] = np.array([0,n]) # incoherent driving
            LStates[NStates+n-2] = np.array([n,0]) # cavity loss
    elif(LBasis == 1): # GTC eigenbasis
        for n in range(1,NStates):
            LStates[n-1] = np.array([0,n]) # incoherent driving
            LStates[NStates+n-2] = np.array([n,0]) # cavity loss
    elif(LBasis == 2): # diabatic-Fock basis
        for j in range(NMol):
            LStates[j] = np.array([0,1+j]) # incoherent driving
        for k in range(NMod):
            LStates[NMol+k] = np.array([1+NMol+k,0]) # cavity loss
    return LStates

@jit(nopython=True)
def createEU_GTC(sd):
    sd.E_GTC[:], sd.U_GTC[:,:] = np.linalg.eigh(H(np.zeros(NR), sd.xj))
    overlap_mat = np.conj(sd.U).T
    for n in range(NStates):
        f = np.exp(1j*np.angle(overlap_mat[n,n]))
        sd.U_GTC[:,n] = f * sd.U_GTC[:,n] # phase correction
        
@jit(nopython=True)
def ForceCustom(sd): # this calculates the classical force on each nuclear DOF using only the active state
    sd.F[:] = -dH0(sd) # state-independent force
    for j in range(NMol):
        site_index = 1+j # state label of jth site/molecule
        sd.F[j*NBath:(j+1)*NBath] -= np.abs(sd.U[site_index,sd.acst])**2*dHCustom # optimized for Holstein-like bath

@jit(nopython=True)
def HopDirCustom(sd, hd):
    hop_dir = np.zeros(NR)
    for j in range(NMol):
        site_index = 1+j # state label of jth site/molecule
        for n in range(NStates): # optimized rescaling direction for Holstein-like bath
            if(sd.acst!=n):
                hop_dir[j*NBath:(j+1)*NBath] += np.real(np.conj(sd.U[site_index,n]) * sd.U[site_index,sd.acst] / (sd.E[sd.acst] - sd.E[n]) * dHCustom * sd.c[sd.acst]*np.conj(sd.c[n])) #
            if(hd.acst_att!=n):
                hop_dir[j*NBath:(j+1)*NBath] += -np.real(np.conj(sd.U[site_index,n]) * sd.U[site_index,hd.acst_att] / (sd.E[hd.acst_att] - sd.E[n]) * dHCustom * sd.c[hd.acst_att]*np.conj(sd.c[n])) 
    return hop_dir

@jit(nopython=True)
def Hinput(Hinputs, xj): # reconstruct H based on saved diagonal energies
    Hinput = H(np.zeros(NR), xj)
    for j in range(NMol):
        Hinput[1+j,1+j] = Hinputs[j]
    return Hinput

### Numba data types ###

state_types = [
    
    # Necessary MASH variables
    
    ('c', complex128[::1]), # state wavefunction
    ('acst', int32), # active state
    ('R', float64[::1]),
    ('P', float64[::1]),
    ('F', float64[::1]),
    ('M', float64[::1]),
    ('E', float64[::1]), # H eigenenergies
    ('U', complex128[:,::1]), # H eigenvectors
    ('dP', float64),
    
    # GHTC specific variables
    
    ('xj', float64[::1]),
    ('E_GTC', float64[::1]),
    ('U_GTC', complex128[:,::1]),
    
    # Lindblad variables
    
    ('dr1T', float64[::1]),
    ('Z1T', float64[::1]),
    ('randLB', float64[:,::1]),
    ('sample', boolean)
]

hop_types = [
    
    # Time variables
    
    ('dt', float64), # current timestep used to propagate
    ('t_st', float64), # starting time of hop attempt relative to full timestep
    ('t0', float64), # time relative to t_st
    ('dt_r', float64), # smallest timestep of propagation from t0 that changes most populated state
    
    # Hopping conditions
    
    ('acst_att', int32), # state to attempt hop towards
    ('N_hop_att', int32), # number of hop attempts performed during current full timestep
    ('N_int', int32), # number of interpolations performed during current hop attempt
    ('d_dt_H', float64), # acst-acst_att population change time derivative due to Hamiltonian
    ('d_dt_L', float64), # acst-acst_att population change time derivative due to Lindblad
    ('attempt', boolean), # keep attempting current hop
    ('accepted', boolean), # current hop attempt is energetically acceptable
    ('accepted_test', boolean), # test hop attempt is energetically acceptable
    ('rescale', boolean), # rescale momenta after hop attempt
    ('reverse', boolean), # reverse momenta after hop attempt
    ('rescale_mag', float64) # rescale factor of momenta along projected direction
]

### Data classes for storing trajectory information ###

@jitclass(state_types)
class state_data(object): # storage object for state variables (wavefunction, nuclei, etc.)
    def __init__(self):
        
        # Necessary MASH variables
        
        self.c = np.zeros(NStates, dtype=np.complex128) # state wavefunction
        self.acst = 0 # active state
        self.R = np.zeros(NR)
        self.P = np.zeros(NR)
        self.F = np.zeros(NR)
        self.M = np.zeros(NR)
        self.E = np.zeros(NStates) # H eigenenergies
        self.U = np.zeros((NStates,NStates), dtype=np.complex128) # H eigenvectors
        self.dP = 0.0
        
        # GHTC specific variables
        
        self.xj = np.zeros(NMol)
        self.E_GTC = np.zeros(NStates)
        self.U_GTC = np.zeros((NStates,NStates), dtype=np.complex128)
        
        # Lindblad variables
        
        self.dr1T = np.zeros(NStates)
        self.Z1T = np.zeros(NStates)
        self.randLB = np.zeros((NJump,2))
        self.sample = True
        
    # Copy values from other state_data object
        
    def copy(self, sd_new):
        with objmode():
            for i in range(len(state_types)):
                setattr(self,state_types[i][0],copy.deepcopy(getattr(sd_new,state_types[i][0])))
        
@jitclass(hop_types)
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
