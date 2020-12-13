import argparse
import json
import os
import os.path as path
import shutil

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from comp550.dataset import BabiDataModule, ROARDataset
from comp550.model import MultipleSequenceToClass

torch.multiprocessing.set_sharing_strategy('file_system')

thisdir = path.dirname(path.realpath(__file__))
parser = argparse.ArgumentParser(description='Run babi task')
parser.add_argument('--persistent-dir',
                    action='store',
                    default=path.realpath(path.join(thisdir, '..')),
                    type=str,
                    help='Directory where all persistent data will be stored')
parser.add_argument("--k",
                    action="store",
                    default=1,
                    type=int,
                    help="The proportion of tokens to mask.")
parser.add_argument("--masking",
                    action="store",
                    default='top-k',
                    type=str,
                    help="Use 'random' or 'top-k' masking.")
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
                    default=100,
                    type=int,
                    help='The max number of epochs to use')
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
    torch.set_num_threads(max(1, args.num_workers))
    pl.seed_everything(args.seed)
    experiment_id = f"babi_t-{args.task}_roar_s-{args.seed}_k-{args.k}_m-{args.masking[0]}"

    print(f'Running babi-{args.task}-ROAR experiment:')
    print(f' - k: {args.k}')
    print(f' - seed: {args.seed}')
    print(f' - masking: {args.masking}')

    base_dataset = BabiDataModule(
        cachedir=f'{args.persistent_dir}/cache',
        num_workers=args.num_workers,
        task_idx=args.task
    )
    base_dataset.prepare_data()
    base_model = MultipleSequenceToClass.load_from_checkpoint(
        checkpoint_path=f"{args.persistent_dir}/checkpoints/babi_t-{args.task}_s-{args.seed}/checkpoint.ckpt",
        embedding=base_dataset.embedding(),
        hidden_size=32,
        num_of_classes=len(base_dataset.label_names)
    )
    roar_dataset = ROARDataset(
        model=base_model,
        base_dataset=base_dataset,
        k=args.k,
        masking=args.masking,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=32,
    )

    logger = TensorBoardLogger(f'{args.persistent_dir}/tensorboard', name=experiment_id)
    model = MultipleSequenceToClass(
        base_dataset.embedding(), hidden_size=32, num_of_classes=len(base_dataset.label_names))

    checkpoint_callback = ModelCheckpoint(
        monitor="acc_val",
        dirpath=f'{args.persistent_dir}/checkpoints/{experiment_id}',
        filename="checkpoint-{epoch:02d}-{acc_val:.2f}",
        mode="max",
    )

    trainer = Trainer(max_epochs=args.max_epochs,
                      check_val_every_n_epoch=1,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      gpus=int(args.use_gpu))
    trainer.fit(model, roar_dataset)

    shutil.copyfile(
        checkpoint_callback.best_model_path,
        f'{args.persistent_dir}/checkpoints/{experiment_id}/checkpoint.ckpt',
    )

    print("best checkpoint:", checkpoint_callback.best_model_path)
    results = trainer.test(
        datamodule=roar_dataset,
        verbose=False,
        ckpt_path=checkpoint_callback.best_model_path,
    )[0]
    print(results)

    os.makedirs(f'{args.persistent_dir}/results', exist_ok=True)
    with open(f'{args.persistent_dir}/results/{experiment_id}.json', "w") as f:
        json.dump({"seed": args.seed, "dataset": f"babi_t-{args.task}", "roar": True,
                   "k": args.k, "masking": args.masking,
                   **results}, f)
