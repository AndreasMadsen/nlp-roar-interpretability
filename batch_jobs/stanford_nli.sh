#!/bin/bash
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

for seed in $(echo "$seeds")
do
    # Actual time: "0:44:0"
    submit_seeds "1:10:0" "$seed" "roar/snli_rnn_s-%s.json" \
        --mem=24G \
        $(job_script gpu) \
        experiments/stanford_nli.py

    # Actual time: "0:??:0"
    submit_seeds "3:00:0" "$seed" "roar/snli_roberta_s-%s.json" \
        --mem=64G \
        $(job_script gpu) \
        experiments/stanford_nli.py \
        --model-type roberta --batch-size 8 --max-epochs 3
done
