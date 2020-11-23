import argparse
import os.path as path
import shutil

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from comp550.dataset import StanfordSentimentDataset
from comp550.model import SingleSequenceToClass

parser = argparse.ArgumentParser(description='Run example task')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Random seed')
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

    # NOTE: UserWarning: The given NumPy array is not writeable, is related to
    # https://github.com/huggingface/datasets/issues/616
    # Maybe use `python3 -W ignore standford_sentiment.py` for now.
    dataset = StanfordSentimentDataset(seed=args.seed, cachedir=thisdir + '/../cache')
    dataset.prepare_data()
    dataset.setup()

    logger = TensorBoardLogger(thisdir + '/../tensorboard', name='standford_sentiment')
    model = SingleSequenceToClass(dataset.embedding())

    # Source uses the best model, measured with AUC metric, and evaluates every epoch.
    #  https://github.com/successar/AttentionExplanation/blob/master/Trainers/TrainerBC.py#L28
    checkpoint_callback = ModelCheckpoint(
        monitor='auc_val',
        dirpath=thisdir + '/../checkpoints/standford_sentiment',
        filename='checkpoint-{epoch:02d}-{auc_val:.2f}',
        mode='max')
    trainer = pl.Trainer(max_epochs=args.max_epochs, check_val_every_n_epoch=1,
                         callbacks=[checkpoint_callback],
                         logger=logger, gpus=int(args.use_gpu))
    trainer.fit(model, dataset)

    shutil.copyfile(
        checkpoint_callback.best_model_path,
        thisdir + '/../checkpoints/standford_sentiment/checkpoint.ckpt')

    # TODO: SST does not have test-labels defined, so this doesn't work
    # trainer.test(datamodule=dataset)
