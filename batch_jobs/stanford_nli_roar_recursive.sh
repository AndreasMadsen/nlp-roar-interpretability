#!/bin/bash
for seed in {0..4}
do
    for importance_measure in 'attention'
    do
        dependency=''

        for k in {1..10}
        do
            if [ ! -f $SCRATCH"/comp550/results/snli_roar_s-${seed}_k-${k}_m-${importance_measure::1}_r-1.json" ]; then
                echo snli_roar_s-${seed}_k-${k}_m-${importance_measure::1}_r-1
                if last_jobid=$(
                    sbatch --time=1:30:0 --mem=32G --parsable ${dependency} \
                    -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                    -J snli_roar_s-${seed}_k-${k}_m-${importance_measure::1}_r-1 ./python_job.sh \
                    experiments/stanford_nli.py --recursive \
                    --seed ${seed} --k ${k} --importance-measure ${importance_measure}
                ); then
                    echo "Submitted batch job ${last_jobid}"
                    dependency="--dependency=afterok:${last_jobid}"
                else
                    echo "Could not submit batch job, skipping"
                    break
                fi
            fi
        done
    done
done
