import argparse
from functools import partial
import os.path as path
import os
import csv
import json
import gzip
import shutil

import torch
from tqdm import tqdm

from comp550.dataset import SNLIDataset, SSTDataset, IMDBDataset, BabiDataset, MimicDataset
from comp550.model import SingleSequenceToClass, MultipleSequenceToClass
from comp550.util import generate_experiment_id
from comp550.explain import ImportanceMeasure

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Quantify attention sparsity")
parser.add_argument("--persistent-dir",
                    action="store",
                    default=os.path.realpath(os.path.join(thisdir, "..")),
                    type=str,
                    help="Directory where all persistent data will be stored")
parser.add_argument("--seed",
                    action="store",
                    default=0,
                    type=int,
                    help="The seed for each model to measure")
parser.add_argument("--importance-measure",
                    action="store",
                    default='attention',
                    type=str,
                    choices=['random', 'attention', 'gradient', 'integrated-gradient'],
                    help="Use 'random', 'attention', 'gradient', or 'integrated-gradient' as the importance measure.")
parser.add_argument("--use-gpu",
                    action="store",
                    default=torch.cuda.is_available(),
                    type=bool,
                    help=f"Should GPUs be used (detected automatically as {torch.cuda.is_available()})")
parser.add_argument("--num-workers",
                    action="store",
                    default=4,
                    type=int,
                    help="The number of workers to use in data loading")

if __name__ == "__main__":
    args = parser.parse_args()

    print("Computing attention distributions:")
    print(f" - seed: {args.seed}")
    print(f' - importance_measure: {args.importance_measure}')

    experiments = [
        (SSTDataset, SingleSequenceToClass.load_from_checkpoint),
        (SNLIDataset, MultipleSequenceToClass.load_from_checkpoint),
        (IMDBDataset, SingleSequenceToClass.load_from_checkpoint),
        (partial(BabiDataset, task=1), partial(MultipleSequenceToClass.load_from_checkpoint, hidden_size=32)),
        (partial(BabiDataset, task=2), partial(MultipleSequenceToClass.load_from_checkpoint, hidden_size=32)),
        (partial(BabiDataset, task=3), partial(MultipleSequenceToClass.load_from_checkpoint, hidden_size=32))
    ]

    if path.exists(f'{args.persistent_dir}/mimic'):
        experiments += [
            (partial(MimicDataset, subset='diabetes', mimicdir=f'{args.persistent_dir}/mimic', batch_size=8), SingleSequenceToClass.load_from_checkpoint),
            (partial(MimicDataset, subset='anemia', mimicdir=f'{args.persistent_dir}/mimic', batch_size=8), SingleSequenceToClass.load_from_checkpoint)
        ]

    os.makedirs(f'/tmp/results/attention', exist_ok=True)
    os.makedirs(f'{args.persistent_dir}/results/attention', exist_ok=True)
    for dataset_cls, model_cls in tqdm(experiments, desc="Analyzing models"):
        dataset = dataset_cls(cachedir=f"{args.persistent_dir}/cache",
                                num_workers=args.num_workers)
        dataset.prepare_data()

        experiment_id = generate_experiment_id(dataset.name, args.seed)
        csv_name = f"{dataset.name}_s-{args.seed}_m-{args.importance_measure[0]}"

        model = model_cls(checkpoint_path=f'{args.persistent_dir}/checkpoints/{experiment_id}/checkpoint.ckpt',
                        embedding=dataset.embedding(),
                        num_of_classes=len(dataset.label_names))

        importance_measure = ImportanceMeasure(model, dataset, args.importance_measure,
                                               use_gpu=args.use_gpu,
                                               num_workers=args.num_workers,
                                               seed=args.seed)

        # Write to /tmp to avoid high IO on a HPC system
        with gzip.open(f'/tmp/results/attention/{csv_name}.csv.gz', 'wt', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=['split', 'observation', 'index', 'importance'])
            writer.writeheader()

            for split in ['train', 'val', 'test']:
                # Compute attention distribution for each dataset, seed, and split
                for observation_i, (observation, importance) in enumerate(tqdm(
                    importance_measure.evaluate(split),
                    desc='Explaining observations',
                    leave=False
                )):
                    writer.writerows([{
                        'split': split,
                        'observation': observation_i,
                        'index': val_i,
                        'importance': val
                    } for val_i, val in enumerate(importance)])

        shutil.move(f'/tmp/results/attention/{csv_name}.csv.gz',
                    f'{args.persistent_dir}/results/attention/{csv_name}.csv.gz')
