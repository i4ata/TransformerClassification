#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=training_vit
#SBATCH --mem=2000

source ~/env/bin/activate .

python -m src.train --use_custom True --name my_vit

deactivate