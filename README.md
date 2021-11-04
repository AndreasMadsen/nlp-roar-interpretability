# NLP ROAR Interpretability

**Official code for: [Evaluating the Faithfulness of Importance Measures in NLP by Recursively Masking Allegedly Important Tokens and Retraining](https://arxiv.org/abs/2110.08412)**

## Install

```bash
git clone https://github.com/AndreasMadsen/nlp-roar-interpretability.git
cd nlp-roar-interpretability
python -m pip install -e .
```

## Experiments

### Tasks

There are scripts for each dataset. Note that some tasks share a dataset.
Use this list to identify how to train a model for each task.
* SST: `python experiments/stanford_sentiment.py`
* SNLI: `python experiments/stanford_nli.py`
* IMDB: `python experiments/imdb.py`
* MIMIC (Diabetes): `python experiments/mimic.py --subset diabetes`
* MIMIC (Anemia): `python experiments/mimic.py --subset anemia`
* bABI-1: `python experiments/babi.py --task 1`
* bABI-2: `python experiments/babi.py --task 2`
* bABI-3: `python experiments/babi.py --task 3`

### Parameters

Each of the above scripts `stanford_sentiment`, `stanford_nli`, `imdb`,
`mimic`, and `babi` take the same set of CLI arguments. You can learn
about each argument with `--help`. The most important arguments which
will allow you to run the experiments presented in the paper are:

* `--importance-measure`: this specifies which importance measure is used. It can be either `random`, `mutual-information`, `attention` , `gradient`, or `integrated-gradient`.
* `--seed`: specifies the seed used to initialize the model.
* `--roar-strategy`: should ROAR masking be done absoloute (`count`) or relative (`quantile`),
* `--k`: the proportion of tokens in % to mask if `--roar-strategy quantile` is used. The number of tokens if `--roar-strategy count` is used.
* `--recursive`: indicates that model to use for computing the importance measure has `--k` set to `--k` - `--recursive-step-size` instead of `0` as used in classic ROAR.

Note, for `--k` > 0, the reference model must already be trained. For example, in the non-recursive case, this means that a model trained with `--k 0` must already available.

## Running on a HPC setup

For downloading dataset dependencies we provide a `download.sh` script.

Additionally, we provide script for submitting all jobs to a Slurm
queue, in `batch_jobs/`. Note again, that the ROAR script assume
there are checkpoints for the baseline `--k 0` models.

The jobs automatically use `$SCRATCH/nlproar` as the presistent dir.

## MIMIC

See https://mimic.physionet.org/gettingstarted/access/ for how to access MIMIC.
You will need to download `DIAGNOSES_ICD.csv.gz` and `NOTEEVENTS.csv.gz` and
place them in `mimic/` relative to your presistent dir.
