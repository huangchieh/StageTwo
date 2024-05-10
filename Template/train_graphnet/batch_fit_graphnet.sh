#!/bin/bash
#SBATCH --time=00-12:00:00      # Job time allocation
#SBATCH --gres=gpu:1            # Request GPUs
#SBATCH --constraint=a100|volta # Request specific nodes
#SBATCH --mem=64G               # Memory
#SBATCH -c 4                    # Number of cores
#SBATCH -J test_train_graphnet  # Job name
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
    --data_dir /scratch/phys/project/sin/AFM_Hartree_DB/AFM_sims/striped/Water-Au111-FB/Water-Au111_FB_L50 \
    --urls_train "Water-K-{1..10}_train_{0..31}.tar" \
    --urls_val "Water-K-{1..5}_val_{0..7}.tar" \
    --urls_test "Water-K-{1..10}_test_{0..7}.tar" \
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
