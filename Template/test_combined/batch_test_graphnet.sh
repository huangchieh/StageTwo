#!/bin/bash
#SBATCH --time=00-05:00:00      # Job time allocation
#SBATCH --gres=gpu:1            # Request GPUs
#SBATCH --constraint=ampere|volta # Request specific nodes
#SBATCH --mem=64G               # Memory
#SBATCH -c 4                    # Number of cores
#SBATCH -J test_combined_JOBSUFFIX        # Job name
#SBATCH -o log_fit.out          # Output file

# Load environment
# module load anaconda
module load mamba gcc
source activate ml-spm
export OMP_NUM_THREADS=1

# Print job info
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME
echo "Running on nodes: "$SLURM_JOB_NODELIST

# Run fit script
rm -r ~/.cache # Sometimes it gets stuck if there are existing builds of cuda extensions
# Prepare the data_dir in config.yaml
sed -i "s/Water-bilayer_FB_L30/CASE/g" config.yaml
python test_graphnet.py
