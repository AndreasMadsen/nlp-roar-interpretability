#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH -J unpack-scratch
#SBATCH -o /home/anmadc/logs/%x.%j.out
#SBATCH -e /home/anmadc/logs/%x.%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=16:00:00

tar -xvf --touch $SCRATCH/comp550.tar.gz -C /
