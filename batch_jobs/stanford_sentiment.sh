#!/bin/bash
for seed in {0..4}
do
    if [ ! -f $SCRATCH"/comp550/results/sst_s-${seed}.json" ]; then
        echo sst_s-${seed}
        sbatch --time=0:15:0 --mem=12G -J sst_s-${seed} -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err"  ./python_job.sh \
            experiments/stanford_sentiment.py \
            --seed ${seed}
    fi
done
