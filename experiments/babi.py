import argparse
import json
import os
import os.path as path
import shutil
import tempfile

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from nlproar.dataset import BabiDataset, ROARDataset
from nlproar.model import select_multiple_sequence_to_class
from nlproar.util import generate_experiment_id, optimal_roar_batch_size

# On compute canada the ulimit -n is reached, unless this strategy is used.
torch.multiprocessing.set_sharing_strategy('file_system')

thisdir = path.dirname(path.realpath(__file__))
parser = argparse.ArgumentParser(description='Run babi task')
parser.add_argument('--persistent-dir',
                    action='store',
                    default=path.realpath(path.join(thisdir, '..')),
                    type=str,
                    help='Directory where all persistent data will be stored')
parser.add_argument("--model-type",
                    action="store",
                    default='rnn',
                    type=str,
                    choices=['roberta', 'rnn', 'xlnet'],
                    help="The model to use either rnn or roberta.")
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
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Random seed')
parser.add_argument('--num-workers',
                    action='store',
                    default=4,
                    type=int,
                    help='The number of workers to use in data loading')
# epochs = 100 (https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/Trainers/DatasetQA.py#L106)
parser.add_argument('--max-epochs',
                    action='store',
                    default=None,
                    type=int,
                    help='The max number of epochs to use')
parser.add_argument('--batch-size',
                    action='store',
                    default=None,
                    type=int,
                    help='The batch size to use')
parser.add_argument("--importance-caching",
                    action="store",
                    default=None,
                    type=str,
                    choices=['use', 'build'],
                    help="How should the cache be used for the importance measure, default is no cache involvement.")
parser.add_argument('--use-gpu',
                    action='store',
                    default=torch.cuda.is_available(),
                    type=bool,
                    help=f'Should GPUs be used (detected automatically as {torch.cuda.is_available()})')
parser.add_argument('--task',
                    action='store',
                    default=1,
                    type=int,
                    help='Babi task - to choose between [1,2,3]')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.max_epochs is None:
        args.max_epochs = ({ 'rnn': 100, 'roberta': 8, 'longformer': 8, 'xlnet': 8 })[args.model_type]
    if args.batch_size is None:
        args.batch_size = ({ 'rnn': 50, 'roberta': 16, 'longformer': 16, 'xlnet': 16 })[args.model_type]

    torch.set_num_threads(max(1, args.num_workers))
    pl.seed_everything(args.seed, workers=True)

    MultipleSequenceToClass = select_multiple_sequence_to_class(args.model_type)
    experiment_id = generate_experiment_id(f'babi-{args.task}_{args.model_type}', args.seed,
                                           k=args.k,
                                           strategy=args.roar_strategy,
                                           importance_measure=args.importance_measure,
                                           recursive=args.recursive,
                                           riemann_samples=args.riemann_samples)

    print(f'Running babi-{args.task}-ROAR experiment:')
    print(f' - model_type: {args.model_type}')
    print(f' - k: {args.k}')
    print(f' - seed: {args.seed}')
    print(f' - strategy: {args.roar_strategy}')
    print(f' - recursive: {args.recursive}')
    print(f' - recursive_step_size: {args.recursive_step_size}')
    print(f' - importance_measure: {args.importance_measure}')

    base_dataset = BabiDataset(
        cachedir=f'{args.persistent_dir}/cache',
        model_type=args.model_type,
        num_workers=args.num_workers,
        task=args.task,
        batch_size=args.batch_size
    )
    base_dataset.prepare_data()

    # Create main dataset
    if args.k == 0:
        main_dataset = base_dataset
    else:
        base_experiment_id = generate_experiment_id(f'babi-{args.task}_{args.model_type}', args.seed,
                                                    k=args.k-args.recursive_step_size if args.recursive else 0,
                                                    strategy=args.roar_strategy,
                                                    importance_measure=args.importance_measure,
                                                    recursive=args.recursive,
                                                    riemann_samples=args.riemann_samples)
        main_dataset = ROARDataset(
            cachedir=f'{args.persistent_dir}/cache',
            model=MultipleSequenceToClass.load_from_checkpoint(
                checkpoint_path=f'{args.persistent_dir}/checkpoints/{base_experiment_id}/checkpoint.ckpt',
                cachedir=f'{args.persistent_dir}/cache',
                embedding=base_dataset.embedding(),
                hidden_size=32,
                num_of_classes=len(base_dataset.label_names)
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
    model = MultipleSequenceToClass(f'{args.persistent_dir}/cache', main_dataset.embedding(),
                                    hidden_size=32, num_of_classes=len(main_dataset.label_names))


    checkpoint_callback = ModelCheckpoint(
        monitor="acc_val",
        dirpath=tempfile.mkdtemp(),
        filename="checkpoint-{epoch:02d}-{acc_val:.2f}",
        mode="max",
    )

    trainer = Trainer(max_epochs=args.max_epochs,
                      check_val_every_n_epoch=1,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      gpus=int(args.use_gpu))
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
