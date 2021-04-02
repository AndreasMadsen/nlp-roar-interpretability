#!/bin/bash
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual time: "0:04:0"

submit_seeds 0:15:0 "$seeds" "roar/imdb_s-%s.json" \
    --mem=6G \
    $(job_script gpu) \
    experiments/imdb.py
