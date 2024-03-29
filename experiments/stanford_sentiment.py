import argparse
import os.path as path
import shutil
import json
import os
import tempfile

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from nlproar.dataset import SSTDataset, ROARDataset
from nlproar.model import select_single_sequence_to_class
from nlproar.util import generate_experiment_id, optimal_roar_batch_size

# On compute canada the ulimit -n is reached, unless this strategy is used.
torch.multiprocessing.set_sharing_strategy('file_system')

thisdir = path.dirname(path.realpath(__file__))
parser = argparse.ArgumentParser(description="Run ROAR benchmark for SST")
parser.add_argument('--persistent-dir',
                    action='store',
                    default=path.realpath(path.join(thisdir, '..')),
                    type=str,
                    help='Directory where all persistent data will be stored')
parser.add_argument("--model-type",
                    action="store",
                    default='rnn',
                    type=str,
                    choices=['roberta', 'longformer', 'xlnet', 'rnn'],
                    help="The model to use either rnn, roberta, or xlnet.")
parser.add_argument("--k",
                    action="store",
                    default=0,
                    type=int,
                    help="The proportion of tokens to mask.")
parser.add_argument("--roar-strategy",
                    action="store",
                    default='count',
                    type=str,
                    choices=['count', 'quantile'],
                    help="The meaning of k in terms of how to mask tokens.")
parser.add_argument("--recursive",
                    action='store_true',
                    default=False,
                    help="Should ROAR masking be applied recursively.")
parser.add_argument("--recursive-step-size",
                    action="store",
                    default=1,
                    type=int,
                    help="The proportion of tokens to mask.")
parser.add_argument("--importance-measure",
                    action="store",
                    default='attention',
                    type=str,
                    choices=['random', 'mutual-information', 'attention', 'gradient', 'integrated-gradient', 'times-input-gradient'],
                    help="Use 'random', 'mutual-information', 'attention', 'gradient', or 'integrated-gradient' as the importance measure.")
parser.add_argument("--riemann-samples",
                    action="store",
                    default=50,
                    type=int,
                    help="The number of samples used in the integrated-gradient method")
parser.add_argument("--seed", action="store", default=0, type=int, help="Random seed")
parser.add_argument("--num-workers",
                    action="store",
                    default=4,
                    type=int,
                    help="The number of workers to use in data loading")
# epochs = 8 (https://github.com/successar/AttentionExplanation/blob/master/ExperimentsBC.py#L11)
parser.add_argument('--batch-size',
                    action='store',
                    default=None, #32,
                    type=int,
                    help='The batch size to use')
parser.add_argument("--max-epochs",
                    action="store",
                    default=None, #8,
                    type=int,
                    help="The max number of epochs to use")
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

if __name__ == "__main__":
    args = parser.parse_args()
    if args.max_epochs is None:
        args.max_epochs = ({ 'rnn': 8, 'roberta': 3, 'longformer': 3, 'xlnet': 3 })[args.model_type]
    if args.batch_size is None:
        args.batch_size = ({ 'rnn': 32, 'roberta': 16, 'longformer': 8, 'xlnet': 8 })[args.model_type]

    torch.set_num_threads(max(1, args.num_workers))
    pl.seed_everything(args.seed, workers=True)

    SingleSequenceToClass = select_single_sequence_to_class(args.model_type)
    experiment_id = generate_experiment_id(f'sst_{args.model_type}', args.seed,
                                           k=args.k,
                                           strategy=args.roar_strategy,
                                           importance_measure=args.importance_measure,
                                           recursive=args.recursive,
                                           riemann_samples=args.riemann_samples)

    print('Running SST-ROAR experiment:')
    print(f' - model_type: {args.model_type}')
    print(f' - k: {args.k}')
    print(f' - seed: {args.seed}')
    print(f' - strategy: {args.roar_strategy}')
    print(f' - recursive: {args.recursive}')
    print(f' - recursive_step_size: {args.recursive_step_size}')
    print(f' - importance_measure: {args.importance_measure}')

    # Load base dataset
    base_dataset = SSTDataset(
        cachedir=f'{args.persistent_dir}/cache',
        model_type=args.model_type,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )
    base_dataset.prepare_data()

    # Create main dataset
    if args.k == 0:
        main_dataset = base_dataset
    else:
        base_experiment_id = generate_experiment_id(f'sst_{args.model_type}', args.seed,
                                                    k=args.k-args.recursive_step_size if args.recursive else 0,
                                                    strategy=args.roar_strategy,
                                                    importance_measure=args.importance_measure,
                                                    recursive=args.recursive,
                                                    riemann_samples=args.riemann_samples)
        main_dataset = ROARDataset(
            cachedir=f'{args.persistent_dir}/cache',
            model=SingleSequenceToClass.load_from_checkpoint(
                checkpoint_path=f'{args.persistent_dir}/checkpoints/{base_experiment_id}/checkpoint.ckpt',
                cachedir=f'{args.persistent_dir}/cache',
                embedding=base_dataset.embedding()
            ),
            base_dataset=base_dataset,
            k=args.k,
            strategy=args.roar_strategy,
            recursive=args.recursive,
            recursive_step_size=args.recursive_step_size,
            importance_measure=args.importance_measure,
            riemann_samples=args.riemann_samples,
            use_gpu=args.use_gpu,
            build_batch_size=optimal_roar_batch_size(
                base_dataset.name, base_dataset.model_type,
                args.importance_measure, args.use_gpu),
            importance_caching=args.importance_caching,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        main_dataset.prepare_data()

    logger = TensorBoardLogger(f'{args.persistent_dir}/tensorboard', name=experiment_id)
    model = SingleSequenceToClass(f'{args.persistent_dir}/cache', main_dataset.embedding())

    # Source uses the best model, measured with AUC metric, and evaluates every epoch.
    #  https://github.com/successar/AttentionExplanation/blob/master/Trainers/TrainerBC.py#L28
    checkpoint_callback = ModelCheckpoint(
        monitor="auroc_val",
        dirpath=tempfile.mkdtemp(),
        filename="checkpoint-{epoch:02d}-{auroc_val:.2f}",
        mode="max",
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
        deterministic=args.model_type != 'longformer', # TODO: debug
        logger=logger,
        gpus=int(args.use_gpu),
    )
    trainer.fit(model, main_dataset)
    main_dataset.clean('fit')

    os.makedirs(f'{args.persistent_dir}/checkpoints/{experiment_id}', exist_ok=True)
    shutil.copyfile(
        checkpoint_callback.best_model_path,
        f'{args.persistent_dir}/checkpoints/{experiment_id}/checkpoint.ckpt',
    )

    print("best checkpoint:", checkpoint_callback.best_model_path)
    results = trainer.test(
        datamodule=main_dataset,
        verbose=False,
        ckpt_path=checkpoint_callback.best_model_path,
    )[0]
    print(results)

    os.makedirs(f'{args.persistent_dir}/results/roar', exist_ok=True)
    with open(f'{args.persistent_dir}/results/roar/{experiment_id}.json', "w") as f:
        json.dump({"seed": args.seed,
                   "dataset": main_dataset.name, "model_type": main_dataset.model_type,
                   "strategy": args.roar_strategy, "k": args.k, "recursive": args.recursive,
                   "importance_measure": args.importance_measure,
                   **results}, f)
