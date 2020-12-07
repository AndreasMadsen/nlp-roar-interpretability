#!/bin/bash
for seed in {0..4}
do
    sbatch --time=1:00:0 -J sst_s-${seed} -o $SCRATCH"/logs/slurm-%j.err" -e $SCRATCH"/logs/slurm-%j.err" ./python_job.sh \
        experiments/stanford_sentiment.py \
        --seed ${seed}
done
