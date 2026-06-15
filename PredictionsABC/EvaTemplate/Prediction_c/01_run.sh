#!/bin/bash
#SBATCH --time=00-00:25:00      # Job time allocation
#SBATCH --gres=gpu:1            # Request GPUs
#SBATCH --mem=16G               # Memory
#SBATCH -c 1                    # Number of cores
#SBATCH -J JOBNAME_Prediction_c        # Job name
#SBATCH -o log_fit.out          # Output file

# Load environment
#module load anaconda
module load triton/2024.1-gcc
module load mamba
module load gcc
source activate ml-spm
export OMP_NUM_THREADS=1

# Print job info
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME
echo "Running on nodes: "$SLURM_JOB_NODELIST

# Run fit script
#rm -rf ~/.cache # Sometimes it gets stuck if there are existing builds of cuda extensions
python predict_experiments.py
