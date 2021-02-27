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

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Quantify attention sparsity")
parser.add_argument(
    "--persistent-dir",
    action="store",
    default=os.path.realpath(os.path.join(thisdir, "..")),
    type=str,
    help="Directory where all persistent data will be stored",
)
parser.add_argument(
    "--num-seeds",
    action="store",
    default=5,
    type=int,
    help="The number of seeds used for each model",
)
parser.add_argument(
    "--use-gpu",
    action="store",
    default=torch.cuda.is_available(),
    type=bool,
    help=f"Should GPUs be used (detected automatically as {torch.cuda.is_available()})",
)
parser.add_argument(
    "--num-workers",
    action="store",
    default=4,
    type=int,
    help="The number of workers to use in data loading",
)


def _get_alphas(dataloader_fn, model, device):
    for batch in dataloader_fn(shuffle=False):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        with torch.no_grad():
            _, batch_alpha = model(batch)
            yield from (alpha[:length].tolist() for alpha, length in zip(batch_alpha, batch['length']))


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if args.use_gpu else "cpu")

    print("Computing attention distributions:")
    print(f" - num_seeds: {args.num_seeds}")

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
            (partial(MimicDataset, subset='diabetes', mimicdir=f'{args.persistent_dir}/mimic'), SingleSequenceToClass.load_from_checkpoint),
            (partial(MimicDataset, subset='anemia', mimicdir=f'{args.persistent_dir}/mimic'), SingleSequenceToClass.load_from_checkpoint)
        ]

    os.makedirs(f'/tmp/results/attention', exist_ok=True)
    os.makedirs(f'{args.persistent_dir}/results/attention', exist_ok=True)
    for dataset_cls, model_cls in tqdm(experiments, desc="Computing attention distributions:"):
        dataset = dataset_cls(cachedir=f"{args.persistent_dir}/cache",
                                num_workers=args.num_workers)
        dataset.prepare_data()
        dataset.setup("fit")
        dataset.setup("test")

        for seed in range(args.num_seeds):
            experiment_id = generate_experiment_id(dataset.name, seed)

            model = model_cls(checkpoint_path=f'{args.persistent_dir}/checkpoints/{experiment_id}/checkpoint.ckpt',
                            embedding=dataset.embedding(),
                            num_of_classes=len(dataset.label_names))
            model = model.to(device)

            # Write to /tmp to avoid high IO on a HPC system
            with gzip.open(f'/tmp/results/attention/{experiment_id}.csv.gz', 'wt', newline='') as fp:
                writer = csv.DictWriter(fp, fieldnames=['split', 'observation', 'index', 'alpha'])
                writer.writeheader()

                splits = [
                    ('train', dataset.train_dataloader),
                    ('val', dataset.val_dataloader),
                    ('test', dataset.test_dataloader)
                ]
                for split_name, split_dataloader in splits:
                    # Compute attention distribution for each dataset, seed, and split
                    for observation_i, observation_alphas in enumerate(_get_alphas(split_dataloader, model, device)):
                        writer.writerows([{
                            'split': split_name,
                            'observation': observation_i,
                            'index': alpha_i,
                            'alpha': alpha
                        } for alpha_i, alpha in enumerate(observation_alphas)])

            shutil.move(f'/tmp/results/attention/{experiment_id}.csv.gz',
                        f'{args.persistent_dir}/results/attention/{experiment_id}.csv.gz')
