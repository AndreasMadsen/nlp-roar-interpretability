#!/bin/bash
# jobs: 5 * 2 * 4 * (10 + 9) = 760

declare -A time=( ["anemia random"]="0:25:0"   ["anemia attention"]="0:25:0"   ["anemia gradient"]="0:30:0" ["anemia integrated-gradient"]="0:45:0"
                  ["diabetes random"]="0:40:0" ["diabetes attention"]="0:40:0" ["diabetes gradient"]="0:50:0" ["diabetes integrated-gradient"]="1:10:0" )

for seed in {0..4}
do
    for subset in 'anemia' 'diabetes'
    do
    for importance_measure in 'random' 'attention' 'gradient' 'integrated-gradient'
        do
            for k in {1..10}
            do
                if [ ! -f $SCRATCH"/comp550/results/mimic-${subset::1}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0.json" ]; then
                    echo mimic-${subset::1}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0
                    sbatch --time=${time[$subset $importance_measure]} --mem=32G \
                        -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                        -J mimic-${subset::1}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0 ./python_job.sh \
                        experiments/mimic.py \
                        --seed ${seed} --k ${k} --recursive-step-size 1 \
                        --roar-strategy count --importance-measure ${importance_measure} \
                        --subset ${subset}
                fi
            done

            for k in {10..90..10}
            do
                if [ ! -f $SCRATCH"/comp550/results/mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0.json" ]; then
                    echo mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0
                    sbatch --time=${time[$subset $importance_measure]} --mem=32G \
                        -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                        -J mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0 ./python_job.sh \
                        experiments/mimic.py \
                        --seed ${seed} --k ${k} --recursive-step-size 10 \
                        --roar-strategy quantile --importance-measure ${importance_measure} \
                        --subset ${subset}
                fi
            done
        done
    done
done
