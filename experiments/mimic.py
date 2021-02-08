import argparse
import os.path as path
import shutil
import json
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from comp550.dataset import MimicDataset, ROARDataset
from comp550.model import SingleSequenceToClass

thisdir = path.dirname(path.realpath(__file__))
parser = argparse.ArgumentParser(description="Run ROAR benchmark for SST")
parser.add_argument('--persistent-dir',
                    action='store',
                    default=path.realpath(path.join(thisdir, '..')),
                    type=str,
                    help='Directory where all persistent data will be stored')
parser.add_argument('--subset',
                    action='store',
                    default='diabetes',
                    type=str,
                    help='Should be either "diabetes" or "anemia"')
parser.add_argument("--k",
                    action="store",
                    default=0,
                    type=int,
                    help="The proportion of tokens to mask.")
parser.add_argument("--recursive",
                    action='store_true',
                    default=False,
                    help="Should ROAR masking be applied recursively.")
parser.add_argument("--importance-measure",
                    action="store",
                    default='attention',
                    type=str,
                    help="Use 'random' or 'attention' as the importance measure.")
parser.add_argument("--seed", action="store", default=0, type=int, help="Random seed")
parser.add_argument("--num-workers",
                    action="store",
                    default=4,
                    type=int,
                    help="The number of workers to use in data loading")
# epochs = 8 (https://github.com/successar/AttentionExplanation/blob/master/ExperimentsBC.py#L11)
parser.add_argument("--max-epochs",
                    action="store",
                    default=8,
                    type=int,
                    help="The max number of epochs to use")
parser.add_argument("--use-gpu",
                    action="store",
                    default=torch.cuda.is_available(),
                    type=bool,
                    help=f"Should GPUs be used (detected automatically as {torch.cuda.is_available()})")

if __name__ == "__main__":
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.num_workers))
    pl.seed_everything(args.seed)
    experiment_id = f"mimic_t-{args.subset[0]}_s-{args.seed}_k-{args.k}_m-{args.importance_measure[0]}_r-{int(args.recursive)}"

    print('Running MIMIC experiment:')
    print(f' - k: {args.k}')
    print(f' - recursive: {args.recursive}')
    print(f' - importance_measure: {args.importance_measure}')
    print(f' - seed: {args.seed}')
    print(f' - subset: {args.subset}')

    # Create ROAR dataset
    base_dataset = MimicDataset(cachedir=f'{args.persistent_dir}/cache',
                                mimicdir=f'{args.persistent_dir}/mimic',
                                subset=args.subset,
                                seed=args.seed, num_workers=args.num_workers)
    base_dataset.prepare_data()

    # Create main dataset
    if args.k == 0:
        main_dataset = base_dataset
    else:
        base_model = SingleSequenceToClass.load_from_checkpoint(
            checkpoint_path=f'{args.persistent_dir}/checkpoints/sst_s-{args.seed}/checkpoint.ckpt',
            embedding=base_dataset.embedding()
        )
        main_dataset = ROARDataset(
            cachedir=f'{args.persistent_dir}/cache',
            model=base_model,
            base_dataset=base_dataset,
            k=args.k,
            recursive=args.recursive,
            importance_measure=args.importance_measure,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        main_dataset.prepare_data()

    logger = TensorBoardLogger(f'{args.persistent_dir}/tensorboard', name=experiment_id)
    model = SingleSequenceToClass(base_dataset.embedding())

    # Source uses the best model, measured with AUC metric, and evaluates every epoch.
    #  https://github.com/successar/AttentionExplanation/blob/master/Trainers/TrainerBC.py#L28
    checkpoint_callback = ModelCheckpoint(
        monitor="auc_val",
        dirpath=f'{args.persistent_dir}/checkpoints/{experiment_id}',
        filename="checkpoint-{epoch:02d}-{auc_val:.2f}",
        mode="max",
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
        deterministic=True,
        logger=logger,
        gpus=int(args.use_gpu),
    )
    trainer.fit(model, main_dataset)

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

    os.makedirs(f'{args.persistent_dir}/results', exist_ok=True)
    with open(f'{args.persistent_dir}/results/{experiment_id}.json', "w") as f:
        json.dump({"seed": args.seed, "dataset": f"mimic-{args.subset}",
                   "k": args.k, "recursive": args.recursive, "importance_measure": args.importance_measure,
                   **results}, f)
