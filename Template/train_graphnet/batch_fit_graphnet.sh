#!/bin/bash
#SBATCH --time=00-12:00:00      # Job time allocation
#SBATCH --gres=gpu:1            # Request GPUs
#SBATCH --constraint=a100|volta # Request specific nodes
#SBATCH --mem=64G               # Memory
#SBATCH -c 4                    # Number of cores
#SBATCH -J train_graphnet_JOBSUFFIX  # Job name
#SBATCH -o log_fit.out          # Output file

# Load environment
module load anaconda
source activate ml-spm
export OMP_NUM_THREADS=1

# Print job info
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME
echo "Running on nodes: "$SLURM_JOB_NODELIST

# Run fit script
rm -r ~/.cache # Sometimes it gets stuck if there are existing builds of cuda extensions
python fit_graphnet.py \
    --run_dir ./ \
    --data_dir  DATA_DIR \
    --urls_train "URLS_TRAIN" \
    --urls_val "URLS_VAL" \
    --urls_test "URLS_TEST" \
    --random_seed 0 \
    --train True \
    --test True \
    --predict True \
    --epochs 400 \
    --num_workers 8 \
    --batch_size 16 \
    --avg_best_epochs 10 \
    --pred_batches 3 \
    --lr 1e-3 \
    --zmin -2.5 \
    --z_lims -2.9 0.5 \
    --peak_std 0.20 \
    --box_res 0.125 0.125 0.100 \
    --classes 1 8 \
    --class_colors 'w' 'r' \
    --edge_cutoff 3.0 \
    --afm_cutoff 1.125 \
    --loss_weights 1.0 1.0 \
    --loss_labels "NLL (Node)" "NLL (Edge)" \
    --timings
