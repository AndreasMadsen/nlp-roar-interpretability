#!/bin/bash
declare -A time=( ["anemia random"]="0:20:0" ["anemia attention"]="0:20:0" ["anemia gradient"]="0:50:0"
                  ["diabetes random"]="0:40:0" ["diabetes attention"]="0:40:0" ["diabetes gradient"]="2:30:0")

for seed in {0..4}
do
    for subset in 'anemia' 'diabetes'
    do
    for importance_measure in 'random' 'attention' 'gradient'
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

            for k in {5..95..5}
            do
                if [ ! -f $SCRATCH"/comp550/results/mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0.json" ]; then
                    echo mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0
                    sbatch --time=${time[$subset $importance_measure]} --mem=32G \
                        -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                        -J mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-0 ./python_job.sh \
                        experiments/mimic.py \
                        --seed ${seed} --k ${k} --recursive-step-size 5 \
                        --roar-strategy quantile --importance-measure ${importance_measure} \
                        --subset ${subset}
                fi
            done
        done
    done
done
