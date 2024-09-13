filename_top = "Run_GHTC" # choose a name for this simulation run

method_filename = "LMASH"
model_filename = "model_GHTC"
LID_filename = "LossIntDataFull"

import os, sys
import shutil
import subprocess
from pathlib import Path
sys.path.append(os.getcwd()) 

dirname = filename_top
while(os.path.exists(Path(dirname))):
    if(dirname==filename_top):
        dirname = filename_top + "_2"
    else:
        dirind = str(int(dirname.split("_")[-1]) + 1)
        dirname = filename_top + "_" + dirind
Path(dirname).mkdir(parents=True, exist_ok=True) # make directory for all files to be copied in

shutil.copyfile("Run.py", dirname + "/Run.py")
shutil.copyfile(method_filename + ".py", dirname + "/" + method_filename + ".py")
shutil.copyfile(model_filename + ".py", dirname + "/" + model_filename + ".py")
shutil.copyfile(LID_filename + ".txt", dirname + "/" + LID_filename + ".txt")

os.chdir(dirname)

Path("data").mkdir(parents=True, exist_ok=True)
shutil.copyfile(model_filename + ".py", "data/model.py")
shutil.copyfile(method_filename + ".py", "data/method.py")
shutil.copyfile(LID_filename + ".txt", "data/LossIntDataFull.txt")
    
os.chdir("data")
sys.path.append(os.getcwd()) 

import method
import model
import numpy as np

result = method.runTraj()

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

psiFile = np.savetxt(f"./psi_{filename_top}.txt",psi) 
HinputsFile = np.savetxt(f"./Hinputs_{filename_top}.txt",Hinputs)
active_statefile = np.savetxt(f"./ACST_{filename_top}.txt",active_state)
xjsFile = np.savetxt(f"./xjs_{filename_top}.txt",xjs) 
acst_popFile = open(f"./ACST_pop_{filename_top}.txt","w")
popFile = open(f"./pop_{filename_top}.txt","w")
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
