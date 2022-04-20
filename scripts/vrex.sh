#!/usr/bin/bash
#SBATCH --chdir /home/kothari/l5kit
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 40G
#SBATCH --reservation=courses
#SBATCH --account=civil-459
#SBATCH --gres gpu:1
#SBATCH --time 10:00:00


python examples/dro/dro_train.py
