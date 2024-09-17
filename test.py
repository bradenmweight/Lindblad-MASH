import numpy as np
from matplotlib import pyplot as plt

from MASH import runTraj
from model import exact_results

exact_data = exact_results()

time, active_state, psi_ad, rho_ensemble_ad, psi_dia, rho_ensemble_dia = runTraj()

# Diabatic Results
plt.plot( exact_data[:,0], exact_data[:,1].real, c='black', lw=6, alpha=0.5, label="Exact D0" )
plt.plot( exact_data[:,0], exact_data[:,2].real, c='black', lw=6, alpha=0.5, label="Exact D1" )
plt.plot( time, rho_ensemble_dia[0,0,:].real, c='red', lw=2, label="D0" )
plt.plot( time, rho_ensemble_dia[1,1,:].real, c='red', lw=2, label="D1" )

# Adiabatic Results
# plt.plot( exact_data[:,0], exact_data[:,5].real, c='black', lw=6, alpha=0.5, label="Exact A0" )
# plt.plot( exact_data[:,0], exact_data[:,6].real, c='black', lw=6, alpha=0.5, label="Exact A1" )
# plt.plot( time, rho_ensemble_ad[0,0,:].real, c='red', lw=2, label="A0" )
# plt.plot( time, rho_ensemble_ad[1,1,:].real, c='red', lw=2, label="A1" )

plt.legend()
plt.xlabel("Time (a.u.)", fontsize=15)
plt.ylabel("Population", fontsize=15)
plt.xlim(time[0], time[-1])
plt.savefig("test.jpg", dpi=300)