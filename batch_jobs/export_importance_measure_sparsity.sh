#!/bin/bash
source "batch_jobs/_job_script.sh"

sbatch --time=2:00:00 -J "importance_measure_sparsity" \
    -o "$SCRATCH"/comp550/logs/%x.%j.out -e "$SCRATCH"/comp550/logs/%x.%j.err \
    --mem=12G \
    $(job_script cpu) \
    export/importance_measure_sparsity.py
