#!/bin/bash
declare -A time=( ["anemia"]="0:20:0" ["diabetes"]="0:40:0")

for seed in {0..4}
do
    for subset in 'anemia' 'diabetes'
    do
        if [ ! -f $SCRATCH"/comp550/results/mimic-${subset::1}_s-${seed}.json" ]; then
            echo mimic-${subset::1}_s-${seed}
            sbatch --time=${time[$subset]} --mem=16G \
                -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                -J mimic-${subset::1}_s-${seed} ./python_job.sh \
                experiments/mimic.py \
                --seed ${seed} \
                --subset ${subset}
        fi
    done
done
