#!/bin/bash
# jobs: 5 * 2 * 4 * (10 + 9) = 760
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual   pre_time=( ["anemia random"]="0:04:0"   ["anemia attention"]="0:05:0"   ["anemia gradient"]="0:07:0"   ["anemia integrated-gradient"]="0:36:0"
#                     ["diabetes random"]="0:05:0" ["diabetes attention"]="0:06:0" ["diabetes gradient"]="0:12:0" ["diabetes integrated-gradient"]="1:29:0" )
declare -A pre_time=( ["anemia random"]="0:15:0"   ["anemia attention"]="0:15:0"   ["anemia gradient"]="0:20:0"   ["anemia integrated-gradient"]="0:55:0"
                      ["diabetes random"]="0:20:0" ["diabetes attention"]="0:20:0" ["diabetes gradient"]="0:30:0" ["diabetes integrated-gradient"]="1:50:0" )

# Actual   roar_time=( ["anemia"]="0:09:0"   ["diabetes"]="0:17:0" )
declare -A roar_time=( ["anemia"]="0:20:0"   ["diabetes"]="0:30:0" )


for seed in $(echo "$seeds")
do
    for subset in 'anemia' 'diabetes'
    do
    for importance_measure in 'random' 'attention' 'gradient' 'integrated-gradient'
        do
            riemann_samples=$(( $importance_measure == integrated-gradient ? 50 : 0 ))

            if precompute_jobid=$(
                submit_seeds ${pre_time[$subset $importance_measure]} "$seed" "importance-measure/mimic-${subset::1}-pre_s-${seed}_m-${importance_measure::1}_rs-${riemann_samples}.csv.gz" \
                    --mem=8G --parsable \
                    $(job_script gpu) \
                    experiments/compute_importance_measure.py \
                    --dataset "mimic-${subset::1}" \
                    --importance-measure "$importance_measure" \
                    --importance-caching build
            );  then
                echo "Submitted precompute batch job $precompute_jobid"
            else
                echo "Could not submit precompute batch job, skipping"
                break
            fi

            for k in {1..10}
            do
                submit_seeds ${roar_time[$subset]} "$seed" "roar/mimic-${subset::1}_s-%s_k-${k}_y-c_m-${importance_measure::1}_r-0_rs-${riemann_samples}.json"\
                        --mem=8G --dependency=afterok:"$precompute_jobid" \
                    -J mimic-${subset::1}_s-${seed}_k-${k}_y-c_m-${importance_measure::1}_r-0_rs-${riemann_samples} $(job_script gpu) \
                    experiments/mimic.py \
                    --k "$k" --recursive-step-size 1 \
                    --roar-strategy count --importance-measure "$importance_measure" \
                    --importance-caching use \
                    --subset "$subset"
            done

            for k in {10..90..10}
            do
                submit_seeds ${roar_time[$subset]} "$seed"  "roar/mimic-${subset::1}_s-%s_k-${k}_y-q_m-${importance_measure::1}_r-0_rs-${riemann_samples}.json" \
                    --mem=8G --dependency=afterok:"$precompute_jobid" \
                    $(job_script gpu) \
                    experiments/mimic.py \
                    --k "$k" --recursive-step-size 10 \
                    --roar-strategy quantile --importance-measure "$importance_measure" \
                    --importance-caching use \
                    --subset "$subset"
            done
        done
    done
done
