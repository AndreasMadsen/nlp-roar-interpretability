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

from comp550.dataset import BabiDataModule
from comp550.model import MultipleSequenceToClass

thisdir = path.dirname(path.realpath(__file__))
parser = argparse.ArgumentParser(description='Run babi task')
parser.add_argument('--persistent-dir',
                    action='store',
                    default=path.realpath(path.join(thisdir, '..')),
                    type=str,
                    help='Directory where all persistent data will be stored')
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
    experiment_id = f"babi_t-{args.task}_s-{args.seed}"

    print(f'Running babi-{args.task} experiment:')
    print(f' - seed: {args.seed}')

    dataset = BabiDataModule(
        cachedir=f'{args.persistent_dir}/cache',
        num_workers=args.num_workers,
        task_idx=args.task
    )
    dataset.prepare_data()

    logger = TensorBoardLogger(f'{args.persistent_dir}/tensorboard', name=experiment_id)
    model = MultipleSequenceToClass(
        dataset.embedding(), hidden_size=32, num_of_classes=len(dataset.label_names))

    # Original implementation chooses the best checkpoint on the basis of accuracy
    # https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/Trainers/TrainerQA.py#L12
    checkpoint_callback = ModelCheckpoint(
        monitor='acc_val',
        dirpath=f'{args.persistent_dir}/checkpoints/{experiment_id}',
        filename='checkpoint-{epoch:02d}-{acc_val:.2f}',
        mode='max')

    trainer = Trainer(max_epochs=args.max_epochs,
                      check_val_every_n_epoch=1,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      gpus=int(args.use_gpu))
    trainer.fit(model, dataset)

    shutil.copyfile(
        checkpoint_callback.best_model_path,
        f'{args.persistent_dir}/checkpoints/{experiment_id}/checkpoint.ckpt')
    print('best checkpoint:', checkpoint_callback.best_model_path)

    results = trainer.test(
        datamodule=dataset,
        verbose=False,
        ckpt_path=checkpoint_callback.best_model_path,
    )[0]
    print(results)

    os.makedirs(f'{args.persistent_dir}/results', exist_ok=True)
    with open(f'{args.persistent_dir}/results/{experiment_id}.json', "w") as f:
        json.dump({"seed": args.seed, "dataset": f"babi_t-{args.task}", "roar": False,
                   **results}, f)
