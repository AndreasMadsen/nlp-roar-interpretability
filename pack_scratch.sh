#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH -J pack-scratch
#SBATCH -o /home/anmadc/logs/%x.%j.out
#SBATCH -e /home/anmadc/logs/%x.%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=32:00:00

module load python/3.8.2

# Create environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index tqdm

# Suggestion from https://docs.computecanada.ca/wiki/Handling_large_collections_of_files
tar -vc --use-compress-program="pigz -p 4" -f $SCRATCH/nlproar.tar.gz $SCRATCH/nlproar/ | tqdm --total $(find $SCRATCH/nlproar/ | wc -l) > /dev/null
