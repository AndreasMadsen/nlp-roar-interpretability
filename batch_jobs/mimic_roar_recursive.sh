#!/bin/bash
# jobs: 5 * 2 * 3 * (10 + 9) = 570
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual time:    ["rnn anemia random"]="0:09:0"       ["rnn anemia mutual-information"]="0:10:0"       ["rnn anemia attention"]="0:09:0"       ["rnn anemia gradient"]="0:11:0"       ["rnn anemia integrated-gradient"]="0:44:0"       ["rnn anemia times-input-gradient"]="0:??:0"
#                 ["rnn diabetes random"]="0:17:0"     ["rnn diabetes mutual-information"]="0:18:0"     ["rnn diabetes attention"]="0:17:0"     ["rnn diabetes gradient"]="0:23:0"     ["rnn diabetes integrated-gradient"]="1:46:0"     ["rnn diabetes times-input-gradient"]="0:??:0"
#                 ["roberta anemia random"]="0:??:0"   ["roberta anemia mutual-information"]="0:??:0"                                           ["roberta anemia gradient"]="0:04:0"   ["roberta anemia integrated-gradient"]="0:27:0"   ["roberta anemia times-input-gradient"]="0:04:0"
#                 ["roberta diabetes random"]="0:??:0" ["roberta diabetes mutual-information"]="0:??:0"                                         ["roberta diabetes gradient"]="0:07:0" ["roberta diabetes integrated-gradient"]="1:09:0" ["roberta diabetes times-input-gradient"]="0:07:0" )
declare -A time=( ["rnn anemia random"]="0:25:0"       ["rnn anemia mutual-information"]="0:35:0"       ["rnn anemia attention"]="0:25:0"       ["rnn anemia gradient"]="0:30:0"       ["rnn anemia integrated-gradient"]="1:05:0"       ["rnn anemia times-input-gradient"]="0:30:0"
                  ["rnn diabetes random"]="0:40:0"     ["rnn diabetes mutual-information"]="0:40:0"     ["rnn diabetes attention"]="0:40:0"     ["rnn diabetes gradient"]="0:50:0"     ["rnn diabetes integrated-gradient"]="2:05:0"     ["rnn diabetes times-input-gradient"]="0:50:0"
                  ["roberta anemia random"]="0:??:0"   ["roberta anemia mutual-information"]="0:??:0"                                           ["roberta anemia gradient"]="0:30:0"   ["roberta anemia integrated-gradient"]="1:00:0"   ["roberta anemia times-input-gradient"]="0:30:0"
                  ["roberta diabetes random"]="0:??:0" ["roberta diabetes mutual-information"]="0:??:0"                                         ["roberta diabetes gradient"]="0:30:0" ["roberta diabetes integrated-gradient"]="1:40:0" ["roberta diabetes times-input-gradient"]="0:30:0" )

for seed in $(echo "$seeds")
do
    for model_type in 'rnn' 'roberta'
    do
        for subset in 'anemia' 'diabetes'
        do
            for importance_measure in 'attention' 'gradient' 'integrated-gradient' 'times-input-gradient'
            do
                if [ "$model_type" == "roberta" ] && [ "$importance_measure" == 'attention' ]; then
                    continue
                fi

                riemann_samples=$([ "$importance_measure" == integrated-gradient ] && echo 50 || echo 0)

                dependency=''

                for k in {1..10}
                do
                    if last_jobid=$(
                        submit_seeds "${time[$model_type $subset $importance_measure]}" "$seed" "roar/mimic-${subset::1}_${model_type}_s-%s_k-${k}_y-c_m-${importance_measure::1}_r-1_rs-${riemann_samples}.json" \
                            --parsable $dependency \
                            $(job_script gpu) \
                            experiments/mimic.py --recursive \
                            --model-type "$model_type" \
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
                        submit_seeds "${time[$model_type $subset $importance_measure]}" "$seed" "roar/mimic-${subset::1}_${model_type}_s-%s_k-${k}_y-q_m-${importance_measure::1}_r-1_rs-${riemann_samples}.json" \
                            --parsable $dependency \
                            $(job_script gpu) \
                            experiments/mimic.py --recursive \
                            --model-type "$model_type" \
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
done
