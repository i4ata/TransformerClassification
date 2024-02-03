#!/bin/bash
#SBATCH --time=00:59:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=custom_transformer
#SBATCH --mem=8000

module purge
module load Python/3.9.6-GCCcore-11.2.0

source ~/env/bin/activate

srun python train.py --use_custom False --name pretrained_vit --batch_size 32
srun python train.py --use_custom True --name my_vit