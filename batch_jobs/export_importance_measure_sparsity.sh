#!/bin/bash
source "batch_jobs/_job_script.sh"

sbatch --time=1:00:00 -J "importance_measure_sparsity" \
    -o "$SCRATCH"/nlproar/logs/%x.%j.out -e "$SCRATCH"/nlproar/logs/%x.%j.err \
    $(job_script cpu) \
    export/importance_measure_sparsity.py --stage preprocess
