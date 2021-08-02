#!/bin/bash

#SBATCH --partition=p_ps848          # Partition (job queue)
#SBATCH --job-name=BERTTesting       # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                   # Total # of tasks across all nodes
#SBATCH --nodelist=volta002
#SBATCH --cpus-per-task=12           # Cores per task (>1 if multithread tasks)
#SBATCH --gres=gpu:2
#SBATCH --mem=180000                 # Real memory (RAM) required (MB)
#SBATCH --time=10:00:00               # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.%N.%j.out     # STDOUT output file
#SBATCH --error=slurm.%N.%j.err      # STDERR output file (optional)
echo "Resource allocated"


source /home/joshcoop/.bashrc
# activate my own environment
source /home/joshcoop/miniconda3/bin/activate BERTEnv37

echo "Directory changed"

echo "Running Confusion Matrix Creation"
python prep_filtering_functions.py

echo "Confusion Matrix Creation Completed"