#!/bin/bash
# jobs: 5 * 3 * 4 * (10 + 9) = 1140
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual   pre_time=( ["1 random"]="0:01:0" ["1 mutual-information"]="0:04:0" ["1 attention"]="0:02:0" ["1 gradient"]="0:02:0" ["1 integrated-gradient"]="0:02:0"
#                     ["2 random"]="0:02:0" ["2 mutual-information"]="0:04:0" ["2 attention"]="0:02:0" ["2 gradient"]="0:02:0" ["2 integrated-gradient"]="0:03:0"
#                     ["3 random"]="0:02:0" ["3 mutual-information"]="0:05:0" ["3 attention"]="0:02:0" ["3 gradient"]="0:02:0" ["3 integrated-gradient"]="0:08:0" )
declare -A pre_time=( ["1 random"]="0:15:0" ["1 mutual-information"]="0:15:0" ["1 attention"]="0:15:0" ["1 gradient"]="0:15:0" ["1 integrated-gradient"]="0:15:0"
                      ["2 random"]="0:15:0" ["2 mutual-information"]="0:15:0" ["2 attention"]="0:15:0" ["2 gradient"]="0:15:0" ["2 integrated-gradient"]="0:15:0"
                      ["3 random"]="0:15:0" ["3 mutual-information"]="0:15:0" ["3 attention"]="0:15:0" ["3 gradient"]="0:15:0" ["3 integrated-gradient"]="0:20:0" )

# Actual   roar_time=( ["1"]="0:08:0" ["2"]="0:12:0" ["3"]="0:24:0" )
declare -A roar_time=( ["1"]="0:20:0" ["2"]="0:25:0" ["3"]="0:35:0" )

for type in 1 2 3
do
    for importance_measure in 'random' 'mutual-information' 'attention' 'gradient' 'integrated-gradient'
    do
        riemann_samples=$([ "$importance_measure" == integrated-gradient ] && echo 50 || echo 0)
        dependency=''

        if precompute_jobid=$(
            submit_seeds ${pre_time[$type $importance_measure]} "$seeds" "importance_measure/babi-${type}-pre_s-%s_m-${importance_measure::1}_rs-${riemann_samples}.csv.gz" \
                --mem=6G --parsable \
                $(job_script gpu) \
                experiments/compute_importance_measure.py \
                --dataset "babi-${type}" \
                --importance-measure "$importance_measure" \
                --importance-caching build
        ); then
            if [ ! "$precompute_jobid" == "skipping" ]; then
                echo "Submitted precompute batch job $precompute_jobid"
                dependency="--dependency=afterok:$precompute_jobid"
            fi
        else
            echo "Could not submit precompute batch job, skipping"
            break
        fi

        for k in {1..10}
        do
            submit_seeds ${roar_time[$type]} "$seeds" "roar/babi-${type}_s-%s_k-${k}_y-c_m-${importance_measure::1}_r-0_rs-${riemann_samples}.json" \
                --mem=6G $dependency \
                $(job_script gpu) \
                experiments/babi.py \
                --k "$k" --recursive-step-size 1 \
                --roar-strategy count --importance-measure "$importance_measure" \
                --importance-caching use \
                --task "$type"
        done

        for k in {10..100..10}
        do
            if [ "$k" -le 90 ] || [ "$importance_measure" = "random" ]; then
                submit_seeds ${roar_time[$type]} "$seeds" "roar/babi-${type}_s-%s_k-${k}_y-q_m-${importance_measure::1}_r-0_rs-${riemann_samples}.json" \
                    --mem=6G $dependency \
                    $(job_script gpu) \
                    experiments/babi.py \
                    --k "$k" --recursive-step-size 10 \
                    --roar-strategy quantile --importance-measure "$importance_measure" \
                    --importance-caching use \
                    --task "$type"
            fi
        done
    done
done
