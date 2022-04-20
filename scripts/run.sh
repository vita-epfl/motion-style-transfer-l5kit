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
#SBATCH --output slurm_outs/eval_time.out

python examples/dro/train.py