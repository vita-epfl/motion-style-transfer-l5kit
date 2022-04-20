#!/usr/bin/bash
#SBATCH --chdir /home/kothari/l5kit
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --mem 300G
#SBATCH --reservation=courses
#SBATCH --account=civil-459
#SBATCH --gres gpu:2
#SBATCH --time 12:00:00
#SBATCH --output slurm_outs/vit_small_w12.out

python examples/dro/train.py