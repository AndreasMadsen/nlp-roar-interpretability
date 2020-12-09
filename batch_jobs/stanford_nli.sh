#!/bin/bash
for seed in {0..4}
do
    if [ ! -f $SCRATCH"/comp550/results/snli_s-${seed}.json" ]; then
        echo snli_s-${seed}
        sbatch --time=1:30:0 --mem=24G -J snli_s-${seed} -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" ./python_job.sh \
            experiments/stanford_nli.py \
            --seed ${seed}
    fi
done
