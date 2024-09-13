#!/scratch/ekoessle/myenvs/PyEnv1/bin/python
#SBATCH -p action
#SBATCH -x bhd0005,bhc0024,bhd0020
#SBATCH --job-name=sbatcharray
#SBATCH --nodes=1 --ntasks=1
#SBATCH --mem-per-cpu=2GB
#SBATCH -t 1:00:00
#SBATCH --output=out_%A_%a.out
#SBATCH --error=err_%A_%a.err

import sys, os
sys.path.append(os.popen("pwd").read().split("/tmpdir")[0]) # include parent directory which has method and model files
#-------------------------
# import LMASH as method
# import model_GHTC as model
import method
import model
import time
import numpy as np

JOBID = str(os.environ["SLURM_ARRAY_JOB_ID"]) # get ID of this job
TASKID = str(os.environ["SLURM_ARRAY_TASK_ID"]) # get ID of this task within the array

if(int(TASKID)==1):
    open('ARRAYJOBID_' + JOBID,'a').close()

t0 = time.time()

result = method.runTraj() # run the method

t1 = time.time()
print("Job time: ",t1-t0)

# Write individual job data to files

NStates = model.NStates
NSkip = model.NSkip
NStepsPrint = model.NStepsPrint
NTraj = model.NTraj
NR = model.NR
NSteps = model.NSteps
NMol = model.NMol
N_Hinputs = NMol

psi = np.zeros((NStates,NTraj*(NStepsPrint+1)), dtype = result[0].dtype)
Hinputs = np.zeros((N_Hinputs,NTraj*(NStepsPrint+1)), dtype = result[1].dtype)
active_state = np.zeros((NTraj*(NStepsPrint+1)), dtype = result[2].dtype)
acst_sum = np.zeros((NStates,(NStepsPrint+1)))
xjs = np.zeros((NMol,NTraj), dtype = result[3].dtype)
pop = np.zeros((NStates,(NStepsPrint+1)), dtype = result[4].dtype)

for ii in range(NTraj):
    psi[:,ii*(NStepsPrint+1) : (ii+1)*(NStepsPrint+1)] = result[0][ii,:,:]
    Hinputs[:,ii*(NStepsPrint+1) : (ii+1)*(NStepsPrint+1)] = result[1][ii,:,:]
    active_state[ii*(NStepsPrint+1) : (ii+1)*(NStepsPrint+1)] = result[2][ii,:]
    xjs[:,ii] = result[3][ii,:]
for t in range(result[0].shape[-1]):
    for ii in range(NStates):
        acst_sum[ii,t] += np.sum([int(active_state[i*(NStepsPrint+1)+t]==ii) for i in range(NTraj)])
        pop[ii,t] += result[4][ii,ii,t]

psiFile = np.savetxt(f"./psi_{JOBID}_{TASKID}.txt",psi) 
HinputsFile = np.savetxt(f"./Hinputs_{JOBID}_{TASKID}.txt",Hinputs)
active_statefile = np.savetxt(f"./ACST_{JOBID}_{TASKID}.txt",active_state)
xjsFile = np.savetxt(f"./xjs_{JOBID}_{TASKID}.txt",xjs) 
acst_popFile = open(f"./ACST_pop_{JOBID}_{TASKID}.txt","w")
popFile = open(f"./pop_{JOBID}_{TASKID}.txt","w")
for t in range(result[0].shape[-1]):
    acst_popFile.write(f"{t * model.NSkip * model.dtF} \t")
    popFile.write(f"{t * model.NSkip * model.dtF} \t")
    for i in range(model.NStates):
        acst_popFile.write(str(acst_sum[i,t].real / NTraj) + "\t")
        popFile.write(str(pop[i,t].real / NTraj) + "\t")
    acst_popFile.write("\n")
    popFile.write("\n")
acst_popFile.close()
popFile.close()

t1 = time.time()
print("Total time: ", t1-t0)