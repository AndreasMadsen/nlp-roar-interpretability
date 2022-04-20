#!/bin/bash
# jobs: 5 * 2 * 4 * (10 + 9) = 760
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual   pre_time=( ["rnn anemia random"]="0:04:0"       ["rnn anemia mutual-information"]="0:05:0"       ["rnn anemia attention"]="0:05:0"       ["rnn anemia gradient"]="0:07:0"       ["rnn anemia integrated-gradient"]="0:36:0"       ["rnn times-input-gradient"]="0:??:0"
#                     ["rnn diabetes random"]="0:05:0"     ["rnn diabetes mutual-information"]="0:07:0"     ["rnn diabetes attention"]="0:06:0"     ["rnn diabetes gradient"]="0:12:0"     ["rnn diabetes integrated-gradient"]="1:29:0"     ["rnn times-input-gradient"]="0:??:0"
#                     ["roberta anemia random"]="0:03:0"   ["roberta anemia mutual-information"]="0:??:0"                                           ["roberta anemia gradient"]="0:??:0"   ["roberta anemia integrated-gradient"]="0:??:0"   ["roberta anemia times-input-gradient"]="0:??:0"
#                     ["roberta diabetes random"]="0:03:0" ["roberta diabetes mutual-information"]="0:??:0"                                         ["roberta diabetes gradient"]="0:??:0" ["roberta diabetes integrated-gradient"]="0:??:0" ["roberta diabetes times-input-gradient"]="0:??:0"
declare -A pre_time=( ["rnn anemia random"]="0:15:0"       ["rnn anemia mutual-information"]="0:15:0"       ["rnn anemia attention"]="0:15:0"       ["rnn anemia gradient"]="0:20:0"       ["rnn anemia integrated-gradient"]="0:55:0"       ["rnn anemia times-input-gradient"]="0:20:0"
                      ["rnn diabetes random"]="0:20:0"     ["rnn diabetes mutual-information"]="0:20:0"     ["rnn diabetes attention"]="0:20:0"     ["rnn diabetes gradient"]="0:30:0"     ["rnn diabetes integrated-gradient"]="1:50:0"     ["rnn diabetes times-input-gradient"]="0:30:0"
                      ["roberta anemia random"]="0:20:0"   ["roberta anemia mutual-information"]="0:??:0"                                           ["roberta anemia gradient"]="0:??:0"   ["roberta anemia integrated-gradient"]="0:??:0"   ["roberta anemia times-input-gradient"]="0:??:0"
                      ["roberta diabetes random"]="0:20:0" ["roberta diabetes mutual-information"]="0:??:0"                                         ["roberta diabetes gradient"]="0:??:0" ["roberta diabetes integrated-gradient"]="0:??:0" ["roberta diabetes times-input-gradient"]="0:??:0" )

# Actual   roar_time=( ["rnn anemia"]="0:09:0"     ["rnn diabetes"]="0:17:0"
#                      ["roberta anemia"]="0:03:0" ["roberta diabetes"]="0:07:0" )
declare -A roar_time=( ["rnn anemia"]="0:20:0"     ["rnn diabetes"]="0:30:0"
                       ["roberta anemia"]="0:20:0" ["roberta diabetes"]="0:30:0" )

for seed in $(echo "$seeds")
do
    for model_type in 'rnn' 'roberta'
    do
        for subset in 'anemia' 'diabetes'
        do
            for importance_measure in 'random' 'attention' 'gradient' 'integrated-gradient' 'times-input-gradient'
            do
                if [ "$model_type" == "roberta" ] && [ "$importance_measure" != 'random' ]; then
                    continue
                fi
                riemann_samples=$([ "$importance_measure" == integrated-gradient ] && echo 50 || echo 0)

                dependency=''

                if precompute_jobid=$(
                    submit_seeds ${pre_time[$model_type $subset $importance_measure]} "$seed" "importance_measure/mimic-${subset::1}_${model_type}-pre_s-${seed}_m-${importance_measure::1}_rs-${riemann_samples}.csv.gz" \
                        --parsable \
                        $(job_script gpu) \
                        experiments/compute_importance_measure.py \
                        --dataset "mimic-${subset::1}" \
                        --model-type "$model_type" \
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
                    if [ "$model_type" == "roberta" ]; then
                        continue
                    fi

                    submit_seeds ${roar_time[$model_type $subset]} "$seed" "roar/mimic-${subset::1}_${model_type}_s-%s_k-${k}_y-c_m-${importance_measure::1}_r-0_rs-${riemann_samples}.json"\
                        $dependency \
                        $(job_script gpu) \
                        experiments/mimic.py \
                        --model-type "$model_type" \
                        --k "$k" --recursive-step-size 1 \
                        --roar-strategy count --importance-measure "$importance_measure" \
                        --importance-caching use \
                        --subset "$subset"
                done

                for k in {10..100..10}
                do
                    if [ "$k" -le 90 ] || [ "$importance_measure" = "random" ]; then
                        submit_seeds ${roar_time[$model_type $subset]} "$seed"  "roar/mimic-${subset::1}_${model_type}_s-%s_k-${k}_y-q_m-${importance_measure::1}_r-0_rs-${riemann_samples}.json" \
                            $dependency \
                            $(job_script gpu) \
                            experiments/mimic.py \
                            --model-type "$model_type" \
                            --k "$k" --recursive-step-size 10 \
                            --roar-strategy quantile --importance-measure "$importance_measure" \
                            --importance-caching use \
                            --subset "$subset"
                    fi
                done
            done
        done
    done
done
