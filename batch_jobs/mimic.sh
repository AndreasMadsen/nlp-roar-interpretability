#!/bin/bash
# jobs: 5*2=10
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

#        Actual time: ["rnn anemia"]="0:07:0" ["rnn diabetes"]="0:15:0"
#        Actual time: ["roberta anemia"]="0:03:0" ["roberta diabetes"]="0:06:0"
declare -A time=( ["rnn anemia"]="0:20:0" ["rnn diabetes"]="0:40:0"
                  ["roberta anemia"]="0:20:0" ["roberta diabetes"]="0:30:0" )

for seed in $(echo "$seeds")
do
    for model_type in 'rnn' 'roberta'
    do
        for subset in 'anemia' 'diabetes'
        do
            submit_seeds "${time[$model_type $subset]}" "$seed" "roar/mimic-${subset::1}_${model_type}_s-%s.json" \
                $(job_script gpu) \
                experiments/mimic.py \
                --subset "$subset" \
                --model-type "$model_type"
        done
    done
done
