#!/bin/bash
#SBATCH --time=05-00:00:00      # Job time allocation
#SBATCH --gres=gpu:4            # Request GPUs
#SBATCH --constraint=a100       # Request specific nodes
#SBATCH --mem=64G               # Memory
#SBATCH --nodes=1               # Total number of nodes 
#SBATCH --ntasks-per-node=1     # 1 MPI task per node, torchrun starts the tasks for each GPU
#SBATCH -c 16                   # Number of cores per task/node
#SBATCH -J test_train_posnet    # Job name
#SBATCH -o log_fit.out          # Output file

# Load environment
module load mamba # On new triton
source activate ml-spm
export OMP_NUM_THREADS=1

# Print job info
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME
echo "Running on nodes: "$SLURM_JOB_NODELIST

# Divide number of cores by number of GPUs to get number of workers per GPU
num_gpus=$(echo "$SLURM_JOB_GPUS" | sed -e $'s/,/\\\n/g' | wc -l)
num_workers=$((SLURM_CPUS_PER_TASK/num_gpus))
echo "GPUs: $num_gpus, num_workers: $num_workers"

torchrun \
    --standalone \
    --nnodes $SLURM_NNODES \
    --nproc_per_node $num_gpus \
    --max_restarts 0 \
    fit_posnet.py \
        --run_dir ./ \
        --data_dir /scratch/phys/project/sin/AFM_Hartree_DB/AFM_sims/striped/Water-Au111-FB \
        --urls_train "Water-Au111/Water-K-{1..10}_train_{0..31}.tar::Water-Au111_FB_L50/Water-K-{1..10}_train_{0..31}.tar" \
        --urls_val "Water-Au111/Water-K-{1..5}_val_{0..7}.tar::Water-Au111_FB_L50/Water-K-{1..5}_val_{0..7}.tar" \
        --urls_test "Water-Au111/Water-K-{1..10}_test_{0..7}.tar::Water-Au111_FB_L50/Water-K-{1..10}_test_{0..7}.tar" \
        --random_seed 0 \
        --train True \
        --test True \
        --predict True \
        --epochs 4000 \
        --num_workers $num_workers \
        --batch_size 8 \
        --avg_best_epochs 5 \
        --pred_batches 10 \
        --lr 1e-4 \
        --zmin -10.0 \
        --z_lims -2.9 0.5 \
        --peak_std 0.20 \
        --box_res 0.125 0.125 0.100 \
        --loss_labels "MSE (pos.)" \
        --timings
