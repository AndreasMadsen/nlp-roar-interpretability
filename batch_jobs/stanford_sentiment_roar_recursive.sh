#!/bin/bash
# jobs: 5 * 1 * 3 * (10 + 9) = 285
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual time:    ["random"]="0:02:0" ["mutual-information"]="0:??:0" ["attention"]="0:02:0" ["gradient"]="0:02:0" ["integrated-gradient"]="0:03:0"
declare -A time=( ["random"]="0:15:0" ["mutual-information"]="0:15:0" ["attention"]="0:15:0" ["gradient"]="0:15:0" ["integrated-gradient"]="0:15:0" )

for importance_measure in 'mutual-information' 'attention' 'gradient' 'integrated-gradient'
do
    riemann_samples=$([ "$importance_measure" == integrated-gradient ] && echo 50 || echo 0)
    dependency=''

    for k in {1..10}
    do
        if last_jobid=$(
            submit_seeds "${time[$importance_measure]}" "$seeds" "roar/sst_s-%s_k-${k}_y-c_m-${importance_measure::1}_r-1_rs-${riemann_samples}.json" \
            --mem=6G --parsable $dependency \
            $(job_script gpu) \
            experiments/stanford_sentiment.py --recursive \
            --k "$k" --recursive-step-size 1 \
            --roar-strategy count --importance-measure "$importance_measure"
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
            submit_seeds "${time[$importance_measure]}" "$seeds" "roar/sst_s-%s_k-${k}_y-q_m-${importance_measure::1}_r-1_rs-${riemann_samples}.json" \
            --mem=6G --parsable $dependency \
            $(job_script gpu) \
            experiments/stanford_sentiment.py --recursive \
            --k "$k" --recursive-step-size 10 \
            --roar-strategy quantile --importance-measure "$importance_measure"
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
