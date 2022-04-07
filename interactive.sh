#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=2:00:00

# Load modules
module load python/3.8.10

# Create environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install dependencies
python -m pip install --no-index --no-deps $HOME/python_wheels/en_core_web_sm-3.1.0-py3-none-any.whl
python -m pip install --no-index 'chardet<4.0,>=2.0' 'click<7.2.0,>=7.1.1'

# Install nlproar
# Copy the module files to localscratch to avoid conflicts when building the .egg-link
cd $HOME/workspace/nlproar
python -m pip install --no-index -e .

# Enable offline model
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_GCE_CHECK=true
