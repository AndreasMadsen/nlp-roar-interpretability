#!/bin/bash
for seed in {0..4}
do
    sbatch --time=0:15:0 --mem=12G -J sst_s-${seed} -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err"  ./python_job.sh \
        experiments/stanford_sentiment.py \
        --seed ${seed}
done
