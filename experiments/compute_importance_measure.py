import argparse
from functools import partial
import os.path as path
import os
import csv
import json
import gzip
import shutil
import time

import torch
from tqdm import tqdm

from nlproar.dataset import SNLIDataset, SSTDataset, IMDBDataset, BabiDataset, MimicDataset
from nlproar.model import select_multiple_sequence_to_class, select_single_sequence_to_class
from nlproar.util import generate_experiment_id, optimal_roar_batch_size
from nlproar.explain import ImportanceMeasure

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Quantify attention sparsity")
parser.add_argument("--persistent-dir",
                    action="store",
                    default=os.path.realpath(os.path.join(thisdir, "..")),
                    type=str,
                    help="Directory where all persistent data will be stored")
parser.add_argument("--model-type",
                    action="store",
                    default='rnn',
                    type=str,
                    choices=['roberta', 'xlnet', 'rnn'],
                    help="The model to use either rnn, roberta, or xlnet.")
parser.add_argument("--seed",
                    action="store",
                    default=0,
                    type=int,
                    help="The seed for each model to measure")
parser.add_argument("--importance-measure",
                    action="store",
                    default='attention',
                    type=str,
                    choices=['random', 'mutual-information', 'attention', 'gradient', 'integrated-gradient', 'times-input-gradient'],
                    help="Use 'random', 'mutual-information', 'attention', 'gradient', or 'integrated-gradient' as the importance measure.")
parser.add_argument("--dataset",
                    action="store",
                    default='sst',
                    type=str,
                    choices=['sst', 'snli', 'imdb', 'babi-1', 'babi-2', 'babi-3', 'mimic-a', 'mimic-d'],
                    help="Specify which dataset to compute.")
parser.add_argument("--riemann-samples",
                    action="store",
                    default=50,
                    type=int,
                    help="The number of samples used in the integrated-gradient method")
parser.add_argument("--importance-caching",
                    action="store",
                    default=None,
                    type=str,
                    choices=['use', 'build'],
                    help="How should the cache be used for the importance measure, default is no cache involvement.")
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
    print(f' - model_type: {args.model_type}')
    print(f" - seed: {args.seed}")
    print(f' - dataset: {args.dataset}')
    print(f' - importance_measure: {args.importance_measure}')

    # Get dataset class and model class
    SingleSequenceToClass = select_single_sequence_to_class(args.model_type)
    MultipleSequenceToClass = select_multiple_sequence_to_class(args.model_type)

    dataset_cls, model_cls = ({
        'sst': (SSTDataset, SingleSequenceToClass.load_from_checkpoint),
        'snli': (SNLIDataset, MultipleSequenceToClass.load_from_checkpoint),
        'imdb': (IMDBDataset, SingleSequenceToClass.load_from_checkpoint),
        'babi-1': (partial(BabiDataset, task=1), partial(MultipleSequenceToClass.load_from_checkpoint, hidden_size=32)),
        'babi-2': (partial(BabiDataset, task=2), partial(MultipleSequenceToClass.load_from_checkpoint, hidden_size=32)),
        'babi-3': (partial(BabiDataset, task=3), partial(MultipleSequenceToClass.load_from_checkpoint, hidden_size=32)),
        'mimic-d': (partial(MimicDataset, subset='diabetes', mimicdir=f'{args.persistent_dir}/mimic'), SingleSequenceToClass.load_from_checkpoint),
        'mimic-a': (partial(MimicDataset, subset='anemia', mimicdir=f'{args.persistent_dir}/mimic'), SingleSequenceToClass.load_from_checkpoint)
    })[args.dataset]

    # Load dataset
    dataset = dataset_cls(cachedir=f"{args.persistent_dir}/cache", model_type=args.model_type,
                          num_workers=args.num_workers)
    dataset.prepare_data()

    # Load model
    experiment_id = generate_experiment_id(f'{dataset.name}_{dataset.model_type}', args.seed, k=0)
    model = model_cls(checkpoint_path=f'{args.persistent_dir}/checkpoints/{experiment_id}/checkpoint.ckpt',
                      cachedir=f'{args.persistent_dir}/cache',
                      embedding=dataset.embedding(),
                      num_of_classes=len(dataset.label_names))

    # Load importance measure
    importance_measure = ImportanceMeasure(
        model, dataset, args.importance_measure,
        riemann_samples=args.riemann_samples,
        use_gpu=args.use_gpu,
        num_workers=args.num_workers,
        batch_size=optimal_roar_batch_size(dataset.name, args.model_type, args.importance_measure, args.use_gpu),
        seed=args.seed,
        caching=args.importance_caching,
        cachedir=f'{args.persistent_dir}/cache'
    )

    # Write to /tmp to avoid high IO on a HPC system
    os.makedirs(f'/tmp/results/importance_measure', exist_ok=True)
    os.makedirs(f'{args.persistent_dir}/results/importance_measure', exist_ok=True)
    csv_name = generate_experiment_id(f'{dataset.name}_{dataset.model_type}-pre', args.seed,
                                      importance_measure=args.importance_measure,
                                      riemann_samples=args.riemann_samples)
    with gzip.open(f'/tmp/results/importance_measure/{csv_name}.csv.gz', 'wt', newline='') as fp:
        writer = csv.DictWriter(fp, extrasaction='ignore', fieldnames=['split', 'observation', 'index', 'token', 'importance', 'walltime'])
        writer.writeheader()

        for split in ['train', 'val', 'test']:
            start_time = time.time()

            # Compute attention distribution for each dataset, seed, and split
            for observation, importance in tqdm(
                importance_measure.evaluate(split),
                desc=f'Explaining {split} observations',
                leave=False
            ):
                # Get walltime for each batch of computation
                walltime = time.time() - start_time

                writer.writerows([{
                    'split': split,
                    'observation': observation['index'].tolist(),
                    'index': index,
                    'token': token_val,
                    'importance': importance_val,
                    'walltime': walltime
                } for index, (token_val, importance_val)
                    in enumerate(zip(observation['sentence'].tolist(), importance.tolist()))])

                start_time = time.time()

    shutil.move(f'/tmp/results/importance_measure/{csv_name}.csv.gz',
                f'{args.persistent_dir}/results/importance_measure/{csv_name}.csv.gz')
