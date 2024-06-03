#!/bin/bash
#SBATCH --time=00-00:25:00      # Job time allocation
#SBATCH --gres=gpu:1            # Request GPUs
#SBATCH --mem=16G               # Memory
#SBATCH -c 1                    # Number of cores
#SBATCH -J test_experiments_JOBSUFFIX        # Job name
#SBATCH -o log_fit.out          # Output file

# Load environment
#module load anaconda
module load mamba
module load gcc
source activate ml-spm
export OMP_NUM_THREADS=1

# Print job info
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME
echo "Running on nodes: "$SLURM_JOB_NODELIST

# Run fit script
rm -r ~/.cache # Sometimes it gets stuck if there are existing builds of cuda extensions
python predict_experiments.py
python plot_predictions.py
