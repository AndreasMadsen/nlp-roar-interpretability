#!/bin/bash
# jobs: 5 * 2 * 3 * (10 + 9) = 570
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual time:    ["anemia random"]="0:09:0"   ["anemia mutual-information"]="0:25:0"   ["anemia attention"]="0:09:0"   ["anemia gradient"]="0:11:0" ["anemia integrated-gradient"]="0:44:0"
#                 ["diabetes random"]="0:17:0" ["diabetes mutual-information"]="0:40:0" ["diabetes attention"]="0:17:0" ["diabetes gradient"]="0:23:0" ["diabetes integrated-gradient"]="1:46:0"
declare -A time=( ["anemia random"]="0:25:0"   ["anemia mutual-information"]="0:25:0"   ["anemia attention"]="0:25:0"   ["anemia gradient"]="0:30:0" ["anemia integrated-gradient"]="1:05:0"
                  ["diabetes random"]="0:40:0" ["diabetes mutual-information"]="0:40:0" ["diabetes attention"]="0:40:0" ["diabetes gradient"]="0:50:0" ["diabetes integrated-gradient"]="2:05:0" )

for seed in $(echo "$seeds")
do
    for subset in 'anemia' 'diabetes'
    do
        for importance_measure in 'mutual-information' 'attention' 'gradient' 'integrated-gradient'
        do
            riemann_samples=$([ "$importance_measure" == integrated-gradient ] && echo 50 || echo 0)
            dependency=''

            for k in {1..10}
            do
                if last_jobid=$(
                    submit_seeds "${time[$subset $importance_measure]}" "$seed" "roar/mimic-${subset::1}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-1_rs-${riemann_samples}.json" \
                        --mem=8G --parsable $dependency \
                        $(job_script gpu) \
                        experiments/mimic.py --recursive \
                        --k "$k" --recursive-step-size 1 \
                        --roar-strategy count --importance-measure "$importance_measure" \
                        --subset "$subset"
                ); then
                    if [ ! "$last_jobid" == "skipping" ]; then
                        echo "Submitted batch job $last_jobid"
                        dependency="--dependency=afterok:$last_jobid"
                    fi
                else
                    echo "Could not submit batch job, skipping"
                    break
                fi
            done

            dependency=''

            for k in {10..90..10}
            do
                if last_jobid=$(
                    submit_seeds "${time[$subset $importance_measure]}" "$seed" "roar/mimic-${subset::1}_s-${seed}_k-${k}_y-q_m-${importance_measure::1}_r-1_rs-${riemann_samples}.json" \
                        --mem=8G --parsable $dependency \
                        $(job_script gpu) \
                        experiments/mimic.py --recursive \
                        --k "$k" --recursive-step-size 10 \
                        --roar-strategy quantile --importance-measure "$importance_measure" \
                        --subset "$subset"
                ); then
                    if [ ! "$last_jobid" == "skipping" ]; then
                        echo "Submitted batch job $last_jobid"
                        dependency="--dependency=afterok:$last_jobid"
                    fi
                else
                    echo "Could not submit batch job, skipping"
                    break
                fi
            done
        done
    done
done
