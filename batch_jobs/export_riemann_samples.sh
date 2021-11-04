#!/bin/bash
source "batch_jobs/_job_script.sh"

sbatch --time=4:00:00 -J "export_riemann_samples" \
    -o "$SCRATCH"/nlproar/logs/%x.%j.out -e "$SCRATCH"/nlproar/logs/%x.%j.err \
    --mem=8G \
    $(job_script cpu) \
    export/riemann_samples.py
