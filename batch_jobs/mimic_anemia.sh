#!/bin/bash
for seed in {0..4}
do
    if [ ! -f $SCRATCH"/comp550/results/mimic-t-a_s-${seed}.json" ]; then
        echo mimic_t-a_s-${seed}
        sbatch --time=0:30:0 --mem=16G -J mimic_t-a_s-${seed} -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" ./python_job.sh \
            experiments/mimic.py \
            --seed ${seed} \
            --subset anemia
    fi
done
