#!/bin/bash
for seed in {0..4}
do
    for subset in 'anemia' 'diabetes'
    do
        for importance_measure in 'attention' 'gradient'
        do
            dependency=''

            for k in {1..10}
            do
                if [ ! -f $SCRATCH"/comp550/results/mimic-${subset::1}_s-${seed}_k-${k}_m-${importance_measure::1}_r-1.json" ]; then
                    echo mimic-${subset::1}_s-${seed}_k-${k}_m-${importance_measure::1}_r-1
                    if last_jobid=$(
                        sbatch --time=0:40:0 --mem=16G --parsable ${dependency} \
                            -o $SCRATCH"/comp550/logs/%x.%j.out" -e $SCRATCH"/comp550/logs/%x.%j.err" \
                            -J mimic-${subset::1}_s-${seed}_k-${k}_m-${importance_measure::1}_r-1 ./python_job.sh \
                            experiments/mimic.py --recursive \
                            --seed ${seed} --k ${k} --importance-measure ${importance_measure} \
                            --subset ${subset}
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
done
