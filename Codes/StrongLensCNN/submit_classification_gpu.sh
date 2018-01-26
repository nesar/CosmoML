#!/bin/sh
#SBATCH -p cp100

# ONLY for batch submissions. Otherwise use the interactive session from salloc_classification_gpu.sh

echo [$SECONDS] setting up environment

export PATH="/cosmo/software/anaconda3/bin:$PATH"
export PYTHONPATH=/cosmo/software/anaconda3/lib/python3.6/site-packages/:$HOME/.conda/envs/nes_keras/lib/python3.6/site-packages/

mkdir slurm-$SLURM_JOBID
cd slurm-$SLURM_JOBID
mkdir ModelOutClassification


cp ../train.py ./
cp -r ../model_architectures ./
cp ../load_train_data.py ./
cp ../__init__.py ./
cp -r ../trained_models ./
 
srun -p cp100 python train.py

echo [$SECONDS] job completed

