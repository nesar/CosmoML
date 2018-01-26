#!/bin/sh

echo [$SECONDS] setting up environment

export PATH="/cosmo/software/anaconda3/bin:$PATH"
export PYTHONPATH=/cosmo/software/anaconda3/lib/python3.6/site-packages/:$HOME/.conda/envs/nes_keras/lib/python3.6/site-packages/

echo [$SLURM_JOBID]

#salloc -N 1
salloc -p cp100
