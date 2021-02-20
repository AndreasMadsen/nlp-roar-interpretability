import argparse
from functools import partial
import os
import pickle

import torch
from tqdm import tqdm

from comp550.dataset import (
    SNLIDataModule,
    StanfordSentimentDataset,
    IMDBDataModule,
    BabiDataModule,
    MimicDataset,
)
from comp550.model import SingleSequenceToClass, MultipleSequenceToClass

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


def _get_alphas(dataset, model, device):
    model = model.to(device)

    alphas = []
    for batch in dataset.train_dataloader():
        # Potentially move tensors to GPU
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        with torch.no_grad():
            h3, alpha = model(batch)
        alphas.extend(alpha.tolist())

    return alphas


def _load_model(dataset, checkpoint_path):
    # Load either a SingleSequenceToClass or MultipleSequenceToClass model
    # depending on the dataset with the correct parameters.
    if dataset.name in ["babi-1", "babi-2", "babi-3"]:
        model = MultipleSequenceToClass.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            embedding=dataset.embedding(),
            hidden_size=32,
            num_of_classes=len(dataset.label_names),
        )
    elif dataset.name == "snli":
        model = MultipleSequenceToClass.load_from_checkpoint(
            checkpoint_path=checkpoint_path, embedding=dataset.embedding()
        )
    else:
        model = SingleSequenceToClass.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            embedding=dataset.embedding(),
        )
    return model


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if args.use_gpu else "cpu")

    print("Computing attention distributions:")
    print(f" - num_seeds: {args.num_seeds}")

    datasets = [
        StanfordSentimentDataset,
        SNLIDataModule,
        IMDBDataModule,
        partial(BabiDataModule, task=1),
        partial(BabiDataModule, task=2),
        partial(BabiDataModule, task=3),
    ]

    results = {}
    for dataset_cls in tqdm(datasets, desc="Computing attention distributions:"):
        dataset = dataset_cls(
            cachedir=f"{args.persistent_dir}/cache", num_workers=args.num_workers
        )
        dataset.prepare_data()
        dataset.setup("fit")

        for seed in range(args.num_seeds):
            # Assumes experiment IDs are of the form "{dataset.name}_s-{seed}"
            experiment_id = f"{dataset.name}_s-{seed}"

            checkpoint_path = (
                f"{args.persistent_dir}/checkpoints/{experiment_id}/checkpoint.ckpt"
            )
            model = _load_model(dataset, checkpoint_path)

            # Compute attention distributions
            alphas = _get_alphas(dataset, model, device)
            results[experiment_id] = {
                "dataset_name": dataset.name,
                "seed": seed,
                "alphas": alphas,
            }

    os.makedirs(f"{args.persistent_dir}/cache/encoded", exist_ok=True)
    with open(f"{args.persistent_dir}/cache/encoded/alphas.pkl", "wb") as f:
        pickle.dump(results, f)
