#!/bin/bash
#SBATCH --account=IscrC_DbStM       # account name
#SBATCH --partition=boost_usr_prod # partition name
#SBATCH --nodes=1          # number of nodes
#SBATCH --ntasks-per-node=1 # number of tasks per node
#SBATCH --cpus-per-task=2
#SBATCH --gpus=3
# # SBATCH --mem=<memory>GB             # total memory per node requested in GB (optional)
#SBATCH --error=ale.err            # standard error file
#SBATCH --output=ale.out           # standard output file
#SBATCH --qos=boost_qos_lprod             # quality of service (optional)
#SBATCH --time=05:00:00              # time limits: here 1 hour

srun ./run.sh
# module load profile/deeplrn cineca-ai/3.0.1
# srun --account IscrC_DbStM --partition boost_usr_prod --time 01:00:00 --nodes 1 --ntasks-per-node=1 --cpus-per-task=1 --gpus=1 --pty /bin/bash