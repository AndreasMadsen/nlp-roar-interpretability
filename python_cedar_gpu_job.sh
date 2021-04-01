#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=24G
#SBATCH --time=2:00:00

# Load modules
module load python/3.8.2

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip3 install --no-index --no-deps $HOME/python_wheels/en_core_web_sm-2.2.0.tar.gz
pip3 install --no-index 'chardet<4.0,>=2.0'

# Install comp550
# Copy the module files to localscratch to avoid conflicts when building the .egg-link
mkdir $SLURM_TMPDIR/comp550
cp -r -t $SLURM_TMPDIR/comp550 $HOME/workspace/comp550/setup.py $HOME/workspace/comp550/comp550
cd $SLURM_TMPDIR/comp550
pip3 install --no-index --find-links $HOME/python_wheels -e .

# Run code
cd $SLURM_TMPDIR
python3 -u -X faulthandler "$HOME/workspace/comp550/$1" "${@:2}" --use-gpu True --num-workers 4 --persistent-dir $SCRATCH/comp550
