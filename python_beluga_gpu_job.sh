#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=2:00:00

# Load modules
module load python/3.8.2

# Create environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install dependencies
python -m pip install --no-index --no-deps $HOME/python_wheels/en_core_web_sm-3.0.0-py3-none-any.whl
python -m pip install --no-index 'chardet<4.0,>=2.0' 'click<7.2.0,>=7.1.1'

# Install comp550
# Copy the module files to localscratch to avoid conflicts when building the .egg-link
mkdir $SLURM_TMPDIR/comp550
cp -r -t $SLURM_TMPDIR/comp550 $HOME/workspace/comp550/setup.py $HOME/workspace/comp550/comp550
cd $SLURM_TMPDIR/comp550
python -m pip install --no-index -e .

# Run code
cd $SLURM_TMPDIR
for seed in $(echo $RUN_SEEDS)
do
    echo Running $seed
    python -u -X faulthandler "$HOME/workspace/comp550/$1" "${@:2}"  --seed "$seed" --use-gpu True --num-workers 4 --persistent-dir $SCRATCH/comp550
done
