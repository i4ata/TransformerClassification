#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=training_vit
#SBATCH --mem=2000

source ~/env/bin/activate .

python -m src.train --model custom --name my_vit
python -m src.train --model pretrained --name pretrained_vit

deactivate
