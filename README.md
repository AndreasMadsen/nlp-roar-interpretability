# COMP550 Interpretability

**Measuring if attention is explanation with ROAR**

## Install

```bash
git clone https://github.com/AndreasMadsen/python-comp550-interpretability.git comp550-interpretability
cd comp550-interpretability
python -m pip install -e .
```

## Experiments

To run the baseline model (no masking) there is a script for each dataset.
The scripts will download the datasets themself, the only exception is MIMIC
which is free by require authorization to access.

* SST: `python experiments/stanford_sentiment.py`
* SNLI: `python experiments/stanford_nli.py`
* IMDB: `python experiments/imdb.py`
* MIMIC (Diabetes): `python experiments/mimic.py --subset diabetes`
* MIMIC (Anemia): `python experiments/mimic.py --subset anemia`
* bABI-1: `python experiments/mimic.py --task 1`
* bABI-2: `python experiments/mimic.py --task 2`
* bABI-3: `python experiments/mimic.py --task 3`

There are similar scripts for ROAR experiments. These experiments assume that
the model checkpoints from baseline scripts already exists.
* SST: `python experiments/stanford_sentiment_roar.py`
* SNLI: `python experiments/stanford_nli_roar.py`
* IMDB: `python experiments/imdb_roar.py`
* MIMIC (Diabetes): `python experiments/mimic_roar.py --subset diabetes`
* MIMIC (Anemia): `python experiments/mimic_roar.py --subset anemia`
* bABI-1: `python experiments/mimic_roar.py --task 1`
* bABI-2: `python experiments/mimic_roar.py --task 2`
* bABI-3: `python experiments/mimic_roar.py --task 3`

## Running on a HPC setup

For downloading dataset dependencies and packages not avaliable
on Beluga (compute-canada) we provide a `download.sh` script.

Additionally, we provide script for submitting all jobs to a Slurm
queue, in `batch_jobs/`. Note again, that the ROAR script assume
there are checkpoints for the baseline models.

The jobs automatically `$SCRATCH/comp550` as the presistent dir.

## MIMIC

See https://mimic.physionet.org/gettingstarted/access/ for how to access MIMIC.
You will need to download `DIAGNOSES_ICD.csv.gz` and `NOTEEVENTS.csv.gz` and
place them in `mimic/` relative to your presistent dir.
