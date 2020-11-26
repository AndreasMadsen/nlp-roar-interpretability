import argparse
import os.path as path
import shutil

from comp550.dataset import SNLIDataModule
from comp550.model import MultipleSequenceToClass
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size',
                    required=False, type=int, default=128)
'''
Max epochs is 25
https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/ExperimentsQA.py#L10
'''

parser = Trainer.add_argparse_args(parser)

if __name__ == "__main__":

    args = parser.parse_args()
    seed_everything(args.seed)

    thisdir = path.dirname(path.realpath(__file__))

    dataset = SNLIDataModule(
        cachedir=thisdir + '/../cache', batch_size=args.batch_size)
    dataset.prepare_data()
    dataset.setup(stage="fit")

    logger = TensorBoardLogger(
        thisdir + '/../tensorboard', name='standford_snli')
    model = MultipleSequenceToClass(dataset.embedding())

    checkpoint_callback = ModelCheckpoint(
        monitor='auc_val',
        dirpath=thisdir + '/../checkpoints/standford_snli',
        filename='checkpoint-{epoch:02d}-{auc_val:.2f}',
        mode='max')

    trainer = Trainer.from_argparse_args(args,
                                         check_val_every_n_epoch=1,
                                         callbacks=[checkpoint_callback],
                                         logger=logger,
                                         profiler=True,
                                         num_sanity_val_steps=0)
    trainer.fit(model, dataset)

    shutil.copyfile(
        checkpoint_callback.best_model_path,
        thisdir + '/../checkpoints/standford_snli/checkpoint.ckpt')
