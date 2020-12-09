import argparse
import os.path as path
import shutil

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from comp550.dataset import IMDBDataModule
from comp550.model import SingleSequenceToClass

parser = argparse.ArgumentParser(description='Run example task')
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
# epochs = 8 (https://github.com/successar/AttentionExplanation/blob/master/ExperimentsBC.py#L11)
parser.add_argument('--max-epochs',
                    action='store',
                    default=8,
                    type=int,
                    help='The max number of epochs to use')
parser.add_argument('--use-gpu',
                    action='store',
                    default=torch.cuda.is_available(),
                    type=bool,
                    help=f'Should GPUs be used (detected automatically as {torch.cuda.is_available()})')

if __name__ == '__main__':
    args = parser.parse_args()

    thisdir = path.dirname(path.realpath(__file__))

    dataset = IMDBDataModule(
        cachedir=thisdir + '/../cache', num_workers=args.num_workers)
    dataset.prepare_data()

    logger = TensorBoardLogger(
        thisdir + '/../tensorboard', name='imdb_sentiment')
    model = SingleSequenceToClass(dataset.embedding())

    # Source uses the best model, measured with AUC metric, and evaluates every epoch.
    #  https://github.com/successar/AttentionExplanation/blob/master/Trainers/TrainerBC.py#L28
    checkpoint_callback = ModelCheckpoint(
        monitor='auc_val',
        dirpath=thisdir + '/../checkpoints/imdb_sentiment',
        filename='checkpoint-{epoch:02d}-{auc_val:.2f}',
        mode='max')
    pl.seed_everything(args.seed)
    trainer = pl.Trainer(max_epochs=args.max_epochs, check_val_every_n_epoch=1,
                         callbacks=[checkpoint_callback], deterministic=True,
                         logger=logger, gpus=int(args.use_gpu))
    trainer.fit(model, dataset)

    shutil.copyfile(
        checkpoint_callback.best_model_path,
        thisdir + '/../checkpoints/imdb_sentiment/checkpoint.ckpt')

    print('best checkpoint:', checkpoint_callback.best_model_path)
    print(trainer.test(datamodule=dataset, verbose=False,
                       ckpt_path=checkpoint_callback.best_model_path)[0])
