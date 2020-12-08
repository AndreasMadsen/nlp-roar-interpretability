#!/bin/bash
for seed in {0..4}
do
    sbatch --time=1:30:0 --mem=24G -J nli_s-${seed} -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" ./python_job.sh \
        experiments/stanford_nli.py \
        --seed ${seed}
done
