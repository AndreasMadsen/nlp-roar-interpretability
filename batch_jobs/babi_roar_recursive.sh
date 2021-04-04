#!/bin/bash
# jobs: 5 * 3 * 3 * (10 + 9) = 855
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual time:    ["1 random"]="0:08:0" ["1 attention"]="0:09:0" ["1 gradient"]="0:08:0" ["1 integrated-gradient"]="0:10:0"
#                 ["2 random"]="0:11:0" ["2 attention"]="0:12:0" ["2 gradient"]="0:12:0" ["2 integrated-gradient"]="0:15:0"
#                 ["3 random"]="0:25:0" ["3 attention"]="0:25:0" ["3 gradient"]="0:25:0" ["3 integrated-gradient"]="0:32:0" )
declare -A time=( ["1 random"]="0:20:0" ["1 attention"]="0:20:0" ["1 gradient"]="0:20:0" ["1 integrated-gradient"]="1:20:0"
                  ["2 random"]="0:25:0" ["2 attention"]="0:25:0" ["2 gradient"]="0:25:0" ["2 integrated-gradient"]="1:25:0"
                  ["3 random"]="0:35:0" ["3 attention"]="0:35:0" ["3 gradient"]="0:35:0" ["3 integrated-gradient"]="1:45:0" )

for type in 1 2 3
do
    for importance_measure in 'attention' 'gradient' 'integrated-gradient'
    do
        riemann_samples=$(( $importance_measure == integrated-gradient ? 50 : 0 ))
        dependency=''

        for k in {1..10}
        do
            if last_jobid=$(
                submit_seeds "${time[$type $importance_measure]}" "$seeds" "roar/babi-${type}_s-%s_k-${k}_y-c_m-${importance_measure::1}_r-1_rs-${riemann_samples}.json" \
                    --mem=6G --parsable $dependency \
                    $(job_script gpu) \
                    experiments/babi.py --recursive \
                    --k "$k" --recursive-step-size 1 \
                    --roar-strategy count --importance-measure "$importance_measure" \
                    --task "$type"
            ); then
                echo "Submitted batch job $last_jobid"
                dependency="--dependency=afterok:$last_jobid"
            else
                echo "Could not submit batch job, skipping"
                break
            fi
        done

        dependency=''

        for k in {10..90..10}
        do
            if last_jobid=$(
                submit_seeds "${time[$type $importance_measure]}" "$seeds" "roar/babi-${type}_s-%s_k-${k}_y-q_m-${importance_measure::1}_r-1_rs-${riemann_samples}.json" \
                    --mem=6G --parsable $dependency \
                    $(job_script gpu) \
                    experiments/babi.py --recursive \
                    --k "$k" --recursive-step-size 10 \
                    --roar-strategy quantile --importance-measure "$importance_measure" \
                    --task "$type"
            ); then
                echo "Submitted batch job $last_jobid"
                dependency="--dependency=afterok:$last_jobid"
            else
                echo "Could not submit batch job, skipping"
                break
            fi
        done
    done
done
