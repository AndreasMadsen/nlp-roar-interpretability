import argparse
import os.path as path
import shutil
import torch

from comp550.dataset import SNLIDataModule
from comp550.model import MultipleSequenceToClass
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

parser = argparse.ArgumentParser()

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
    seed_everything(args.seed)

    thisdir = path.dirname(path.realpath(__file__))

    dataset = SNLIDataModule(
        cachedir=thisdir + '/../cache', num_workers=args.num_workers)
    dataset.prepare_data()

    logger = TensorBoardLogger(
        thisdir + '/../tensorboard', name='standford_snli')
    model = MultipleSequenceToClass(dataset.embedding())

    '''
    Original implementation chooses the best checkpoint on the basis of accuracy
    https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/Trainers/TrainerQA.py#L12
    '''
    checkpoint_callback = ModelCheckpoint(
        monitor='acc_val',
        dirpath=thisdir + '/../checkpoints/standford_snli',
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
        thisdir + '/../checkpoints/standford_snli/checkpoint.ckpt')
    print('best checkpoint:', checkpoint_callback.best_model_path)
    print(trainer.test(datamodule=dataset, verbose=False,
                       ckpt_path=checkpoint_callback.best_model_path)[0])
