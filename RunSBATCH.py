#!/scratch/ekoessle/myenvs/PyEnv1/bin/python
#SBATCH -p debug
#SBATCH -x bhd0005,bhc0024,bhd0020
#SBATCH --output=run_sbatch.out
#SBATCH --error=run_sbatch.err
#SBATCH --mem-per-cpu=4GB
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

filename_top = "RunSBATCH" # choose a name for this simulation run

multipar_filename = "multipar"
sbatcharray_filename = "sbatcharray"
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

shutil.copyfile("RunSBATCH.py", dirname + "/RunSBATCH.py")
shutil.copyfile(multipar_filename + ".py", dirname + "/" + multipar_filename + ".py")
shutil.copyfile(sbatcharray_filename + ".py", dirname + "/" + sbatcharray_filename + ".py")
shutil.copyfile(method_filename + ".py", dirname + "/" + method_filename + ".py")
shutil.copyfile(model_filename + ".py", dirname + "/" + model_filename + ".py")
shutil.copyfile(LID_filename + ".txt", dirname + "/" + LID_filename + ".txt")

os.chdir(dirname)

Path("data").mkdir(parents=True, exist_ok=True)
shutil.copyfile(model_filename + ".py", "data/model.py")
shutil.copyfile(method_filename + ".py", "data/method.py")
shutil.copyfile(LID_filename + ".txt", "data/LossIntDataFull.txt")
    
command = str("sbatch " + multipar_filename + ".py") # command to submit multipar file
subprocess.Popen(command, shell=True)
