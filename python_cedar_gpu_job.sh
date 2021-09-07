#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100l:1
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
mkdir $SLURM_TMPDIR/nlproar
cp -r -t $SLURM_TMPDIR/nlproar $HOME/workspace/nlproar/setup.py $HOME/workspace/nlproar/nlproar
cd $SLURM_TMPDIR/nlproar
python -m pip install --no-index -e .

# Run code
cd $SLURM_TMPDIR

if [ -z "${RUN_SEEDS}" ]; then
    python -u -X faulthandler "$HOME/workspace/nlproar/$1" "${@:2}" --use-gpu True --num-workers 4 --persistent-dir $SCRATCH/nlproar
else
    for seed in $(echo $RUN_SEEDS)
    do
        echo Running $seed
        python -u -X faulthandler "$HOME/workspace/nlproar/$1" "${@:2}"  --seed "$seed" --use-gpu True --num-workers 4 --persistent-dir $SCRATCH/nlproar
    done
fi
