#!/bin/bash
declare -A time=( ["random"]="1:20:0" ["attention"]="1:20:0" ["gradient"]="9:00:0")

for seed in {0..4}
do
    for subset in 'anemia' 'diabetes'
    do
        for importance_measure in 'random' 'attention' 'gradient'
        do
            for k in {1..10}
            do
                if [ ! -f $SCRATCH"/comp550/results/mimic-${subset::1}_s-${seed}_k-${k}_m-${importance_measure::1}_r-0.json" ]; then
                    echo mimic-${subset::1}_s-${seed}_k-${k}_m-${importance_measure::1}_r-0
                    sbatch --time=${time[$importance_measure]} --mem=16G \
                        -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                        -J mimic-${subset::1}_s-${seed}_k-${k}_m-${importance_measure::1}_r-0 ./python_job.sh \
                        experiments/mimic.py \
                        --seed ${seed} --k ${k} --importance-measure ${importance_measure} \
                        --subset ${subset}
                fi
            done
        done
    done
done
