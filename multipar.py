#!/scratch/ekoessle/myenvs/PyEnv1/bin/python
#SBATCH -p action
#SBATCH -x bhd0005,bhc0024,bhd0020
#SBATCH -o output_multipar.log
#SBATCH --mem-per-cpu=10GB
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

NARRAY = str(1) # number of jobs
filename = "GTHC"

import os, sys
import subprocess
import time
import numpy as np
from pathlib import Path

manual = 0
JOBIDnum = 22951491
ARRAYJOBIDnum = 22951498

if(manual==1):
    JOBID = str(JOBIDnum)
    ARRAYJOBID = str(ARRAYJOBIDnum)
    os.chdir("data")
else:
    JOBID = str(os.environ["SLURM_JOB_ID"]) # get ID of this job

    Path("data/tmpdir_" + JOBID).mkdir(parents=True, exist_ok=True) # make temporary directory for individual job files
    os.chdir("data/tmpdir_" + JOBID) # change to temporary directory
    command = str("sbatch -W --array [1-" + NARRAY + "] ../../sbatcharray.py") # command to submit job array

    open(filename,'a').close()

    t0 = time.time()

    ARRAYJOBID = str(subprocess.check_output(command, shell=True)).replace("b'Submitted batch job ","").replace("\\n'","") # runs job array and saves job ID of the array

    t1 = time.time()
    print("Job ID: " + JOBID)
    print("Array time: ",t1-t0)

    os.chdir("..") # go back to original directory

# Gather data 

psi = np.loadtxt("tmpdir_" + JOBID + "/psi_" + ARRAYJOBID + "_1.txt", dtype = np.complex128) # load first job to get parameters
Hinputs = np.loadtxt("tmpdir_" + JOBID + "/Hinputs_" + ARRAYJOBID + "_1.txt") # load first job to get parameters
active_state = np.loadtxt("tmpdir_" + JOBID + "/ACST_" + ARRAYJOBID + "_1.txt") # load first job to get parameters
xjs = np.loadtxt("tmpdir_" + JOBID + "/xjs_" + ARRAYJOBID + "_1.txt") # load first job to get parameters
ntrajs = int(len(xjs[0,:]))
steps = int(len(active_state)) # number of printed steps (NSteps//NSkip) times number of single-run trajectories
psi = np.zeros((len(psi[:,0]),steps*int(NARRAY)), dtype = psi.dtype) # initialize zeros matrix using parameters
Hinputs = np.zeros((len(Hinputs[:,0]),steps*int(NARRAY)), dtype = Hinputs.dtype) # initialize zeros matrix using parameters
active_state = np.zeros((steps*int(NARRAY)), dtype = active_state.dtype) # initialize zeros matrix using parameters
xjs = np.zeros((len(xjs[:,0]),ntrajs*int(NARRAY)), dtype = xjs.dtype) # initialize zeros matrix using parameters

for i in range(int(NARRAY)):
    if(i==0):
        acst_pop = np.loadtxt("tmpdir_" + JOBID + "/ACST_pop_" + ARRAYJOBID + "_1.txt") # load first job
        pop = np.loadtxt("tmpdir_" + JOBID + "/pop_" + ARRAYJOBID + "_1.txt") # load first job
    else:
        acst_pop += np.loadtxt("tmpdir_" + JOBID + "/ACST_pop_" + ARRAYJOBID + "_" + str(i+1) + ".txt") # load first job
        pop += np.loadtxt("tmpdir_" + JOBID + "/pop_" + ARRAYJOBID + "_" + str(i+1) + ".txt") # load first job
    psi[:,i*steps : (i+1)*steps] = np.loadtxt("tmpdir_" + JOBID + "/psi_" + ARRAYJOBID + "_" + str(i+1) + ".txt", dtype = np.complex128) # append each line with next trajectory(s)
    Hinputs[:,i*steps : (i+1)*steps] = np.loadtxt("tmpdir_" + JOBID + "/Hinputs_" + ARRAYJOBID + "_" + str(i+1) + ".txt") # append each line with next trajectory(s)
    active_state[i*steps : (i+1)*steps] = np.loadtxt("tmpdir_" + JOBID + "/ACST_" + ARRAYJOBID + "_" + str(i+1) + ".txt") # append each line with next trajectory(s)
    xjs[:,i*ntrajs : (i+1)*ntrajs] = np.loadtxt("tmpdir_" + JOBID + "/xjs_" + ARRAYJOBID + "_" + str(i+1) + ".txt") # append each line with next trajectory(s)

psiFile = np.savetxt(f"./psi_{filename}.txt",psi)
HinputsFile = np.savetxt(f"./Hinputs_{filename}.txt",Hinputs)
active_stateFile = np.savetxt(f"./ACST_{filename}.txt",active_state)
xjsFile = np.savetxt(f"./xjs_{filename}.txt",xjs)

acst_pop = acst_pop / int(NARRAY) # totalVALID # divide to go from sum to average
pop = pop / int(NARRAY) # totalVALID # divide to go from sum to average
acst_popFile = np.savetxt(f"./ACST_pop_{filename}.txt",acst_pop)
popFile = np.savetxt(f"./pop_{filename}.txt",pop)

# os.system("rm -rf tmpdir_" + JOBID) # delete temporary folder (risky since this removes original job file data)