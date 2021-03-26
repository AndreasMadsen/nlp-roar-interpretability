#!/bin/bash
# jobs: 5 * 2 * 3 * (10 + 9) = 570
source "batch_jobs/_job_script.sh"

# Actual time:    ["anemia random"]="0:09:0"   ["anemia attention"]="0:09:0"   ["anemia gradient"]="0:11:0" ["anemia integrated-gradient"]="0:24:0"
#                 ["diabetes random"]="0:17:0" ["diabetes attention"]="0:17:0" ["diabetes gradient"]="0:23:0" ["diabetes integrated-gradient"]="0:49:0"
declare -A time=( ["anemia random"]="0:25:0"   ["anemia attention"]="0:25:0"   ["anemia gradient"]="0:30:0" ["anemia integrated-gradient"]="0:45:0"
                  ["diabetes random"]="0:40:0" ["diabetes attention"]="0:40:0" ["diabetes gradient"]="0:50:0" ["diabetes integrated-gradient"]="1:10:0" )

for seed in {0..4}
do
    for subset in 'anemia' 'diabetes'
    do
        for importance_measure in 'attention' 'gradient' 'integrated-gradient'
        do
            dependency=''

            for k in {1..10}
            do
                if [ ! -f $SCRATCH"/comp550/results/mimic-${subset::1}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-1.json" ]; then
                    echo mimic-${subset::1}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-1
                    if last_jobid=$(
                        sbatch --time=${time[$subset $importance_measure]} --mem=8G --parsable ${dependency} \
                            -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                            -J mimic-${subset::1}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-1 $(job_script gpu) \
                            experiments/mimic.py --recursive \
                            --seed ${seed} --k ${k} --recursive-step-size 1 \
                            --roar-strategy count --importance-measure ${importance_measure} \
                            --subset ${subset}
                    ); then
                        echo "Submitted batch job ${last_jobid}"
                        dependency="--dependency=afterok:${last_jobid}"
                    else
                        echo "Could not submit batch job, skipping"
                        break
                    fi
                fi
            done

            dependency=''

            for k in {10..90..10}
            do
                if [ ! -f $SCRATCH"/comp550/results/mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-1.json" ]; then
                    echo mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-1
                    if last_jobid=$(
                        sbatch --time=${time[$subset $importance_measure]} --mem=8G --parsable ${dependency} \
                            -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                            -J mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-1 $(job_script gpu) \
                            experiments/mimic.py --recursive \
                            --seed ${seed} --k ${k} --recursive-step-size 10 \
                            --roar-strategy quantile --importance-measure ${importance_measure} \
                            --subset ${subset}
                    ); then
                        echo "Submitted batch job ${last_jobid}"
                        dependency="--dependency=afterok:${last_jobid}"
                    else
                        echo "Could not submit batch job, skipping"
                        break
                    fi
                fi
            done
        done
    done
done
