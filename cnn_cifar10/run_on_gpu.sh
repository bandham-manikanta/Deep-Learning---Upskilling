#!/bin/bash
#SBATCH --job-name="hyper-parameter tuning"
# #SBATCH --cpus-per-task=96  # Utilize all available CPU cores
#SBATCH --tasks-per-node=4  # One task per GPU
#SBATCH --gres=gpu:4 # Adjust based on your GPU requirement
#SBATCH --nodes=1
#SBATCH --partition=a100-long  # Use the long partition for longer runtime
#SBATCH --time=8:00:00  # Set the maximum time limit

# Print the current directory for debugging
# echo "Current working directory: $(pwd)"

# Generate dynamic file names
OUTPUT_FILE="output_hyp_tuning_${SLURM_JOB_ID}_$(date +%Y%m%d-%H%M%S).txt"

# Redirect stdout and stderr to the files
exec > $OUTPUT_FILE 2>&1

# Activate your virtual environment
# module load python/3.12.0  
source /gpfs/projects/MaffeiGroup/dl_venv/bin/activate

# Run your Python script
python train_efficientnet.py
# python train_vit_transfer.py