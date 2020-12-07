import argparse
import os.path as path
import shutil
import json
import os

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from comp550.dataset import SNLIDataModule
from comp550.model import MultipleSequenceToClass

thisdir = path.dirname(path.realpath(__file__))
parser = argparse.ArgumentParser()
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
# epochs = 25 (https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/ExperimentsQA.py#L10)
parser.add_argument('--max-epochs',
                    action='store',
                    default=25,
                    type=int,
                    help='The max number of epochs to use')
parser.add_argument('--use-gpu',
                    action='store',
                    default=torch.cuda.is_available(),
                    type=bool,
                    help=f'Should GPUs be used (detected automatically as {torch.cuda.is_available()})')

if __name__ == "__main__":
    args = parser.parse_args()
    torch.set_num_threads(args.num_workers)
    seed_everything(args.seed)
    experiment_id = f"snli_s-{args.seed}"

    print('Running SNLI experiment:')
    print(f' - k: {args.k}')
    print(f' - seed: {args.seed}')
    print(f' - random-masking: {args.random_masking}')

    dataset = SNLIDataModule(
        cachedir=f'{args.persistent_dir}/cache', num_workers=args.num_workers)
    dataset.prepare_data()

    logger = TensorBoardLogger(
        f'{args.persistent_dir}/tensorboard', name=experiment_id)
    model = MultipleSequenceToClass(dataset.embedding())

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
        json.dump({"seed": args.seed, "dataset": "snli", "roar": False,
                   **results}, f)