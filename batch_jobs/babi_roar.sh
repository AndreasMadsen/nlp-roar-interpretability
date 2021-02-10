#!/bin/bash
for seed in {0..4}
do
    for type in 1 2 3
    do
        for importance_measure in 'random' 'attention' 'gradient'
        do
            for k in {1..10}
            do
                if [ ! -f $SCRATCH"/comp550/results/babi-${type}_s-${seed}_k-${k}_m-${importance_measure::1}_r-0.json" ]; then
                    echo babi-${type}_s-${seed}_k-${k}_m-${importance_measure::1}_r-0
                    sbatch --time=0:50:0 --mem=24G \
                        -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                        -J babi-${type}_s-${seed}_k-${k}_m-${importance_measure::1}_r-0 ./python_job.sh \
                        experiments/babi.py \
                        --seed ${seed} --k ${k} --importance-measure ${importance_measure} \
                        --task ${type}
                fi
            done
        done
    done
done
