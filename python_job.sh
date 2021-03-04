#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=2:00:00

# Load modules
module load python/3.6
module load StdEnv/2020
module load scipy-stack/2020b

virtualenv --system-site-packages --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip3 install --no-index --find-links $HOME/python_wheels \
    'numpy>=1.19.0' 'tqdm>=4.53.0' 'torch>=1.7.0' 'pytorch-lightning>=1.2.0' \
    'spacy>=2.2.0' $HOME/python_wheels/en_core_web_sm-2.2.0.tar.gz 'torchtext>=0.6.0' \
    'scikit-learn>=0.23.0' 'nltk>=3.5' 'gensim>=3.8.0' 'pandas>=1.1.0'

# Install comp550
cd $HOME/workspace/comp550
pip3 install --no-index --no-deps -e .
python3 -u -X faulthandler "$@" --use-gpu True --num-workers 4 --persistent-dir $SCRATCH/comp550
