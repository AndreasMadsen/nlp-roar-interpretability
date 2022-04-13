#!/bin/bash
# jobs: 5*2=10
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

#        Actual time: ["anemia"]="0:07:0" ["diabetes"]="0:15:0"
declare -A time_rnn=( ["anemia"]="0:20:0" ["diabetes"]="0:40:0")
#            Actual time: ["anemia"]="0:09:0" ["diabetes"]="0:10:0"
declare -A time_roberta=( ["anemia"]="0:20:0" ["diabetes"]="0:40:0")

for seed in $(echo "$seeds")
do
    for subset in 'anemia' 'diabetes'
    do
        submit_seeds "${time_rnn[$subset]}" "$seed" "roar/mimic-${subset::1}_rnn_s-%s.json" \
            --mem=8G \
            $(job_script gpu) \
            experiments/mimic.py \
            --subset "$subset"

        submit_seeds "${time_roberta[$subset]}" "$seed" "roar/mimic-${subset::1}_roberta_s-%s.json" \
            --mem=64G \
            $(job_script gpu) \
            experiments/mimic.py \
            --subset "$subset" \
            --model-type roberta --batch-size 8 --max-epochs 3
    done
done
