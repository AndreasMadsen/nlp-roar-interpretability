#!/bin/bash
for seed in {0..4}
do
    if [ ! -f $SCRATCH"/comp550/results/snli_s-${seed}_k-0_m-a_r-0.json" ]; then
        echo snli_s-${seed}
        sbatch --time=1:30:0 --mem=24G -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
            -J snli_s-${seed}_k-0_m-a_r-0 ./python_job.sh \
            experiments/stanford_nli.py \
            --seed ${seed}
    fi
done
