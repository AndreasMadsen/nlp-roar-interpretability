#!/bin/bash
for seed in {0..4}
do
    for subset in 'anemia' 'diabetes'
    do
        for importance_measure in 'random' 'attention'
        do
            for k in {1..10}
            do
                if [ ! -f $SCRATCH"/comp550/results/mimic_t-${subset::1}_s-${seed}_k-${k}_m-${importance_measure::1}_r-0.json" ]; then
                    echo mimic_t-${subset::1}_s-${seed}_k-${k}_m-${importance_measure::1}_r-0
                    sbatch --time=0:40:0 --mem=16G \
                        -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                        -J mimic-t-${subset::1}_s-${seed}_k-${k}_m-${importance_measure::1}_r-0 ./python_job.sh \
                        experiments/mimic.py \
                        --seed ${seed} --k ${k} --importance-measure ${importance_measure} \
                        --subset ${subset}
                fi
            done
        done
    done
done
