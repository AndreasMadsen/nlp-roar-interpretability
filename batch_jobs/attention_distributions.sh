#!/bin/bash

sbatch --time=0:30:0 --mem=20G \
    -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
    -J "attention_distributions" ./python_job.sh \
    experiments/attention_distributions.py \
    --num-seeds 5
