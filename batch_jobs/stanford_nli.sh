#!/bin/bash
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual time: "0:44:0"
for seed in $(echo "$seeds")
do
    submit_seeds "1:10:0" "$seed" "roar/snli_s-%s.json" \
        --mem=24G \
        $(job_script gpu) \
        experiments/stanford_nli.py
done
