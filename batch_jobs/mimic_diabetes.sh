#!/bin/bash
for seed in {0..0}
do
    if [ ! -f $SCRATCH"/comp550/results/mimic-t-d_s-${seed}.json" ]; then
        echo mimic_t-d_s-${seed}
        sbatch --time=2:50:0 --mem=24G -J mimic_t-d_s-${seed} -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" ./python_job.sh \
            experiments/mimic.py \
            --seed ${seed} \
            --subset diabetes
    fi
done
