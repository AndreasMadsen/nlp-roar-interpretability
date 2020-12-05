import argparse
from os import path
import shutil

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from comp550.model import SingleSequenceToClass, MultipleSequenceToClass
from comp550.dataset import ROARDataset, StanfordSentimentDataset, SNLIDataModule

torch.multiprocessing.set_sharing_strategy("file_system")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    action="store",
    default="sst",
    choices=["sst", "snli"],
    type=str,
    help="The dataset to use.",
)
parser.add_argument("--seed", action="store", default=0, type=int, help="Random seed.")
parser.add_argument(
    "--num-workers",
    action="store",
    default=4,
    type=int,
    help="The number of workers to use in data loading.",
)
parser.add_argument(
    "--max-epochs",
    action="store",
    default=8,
    type=int,
    help="The max number of epochs to use.",
)
parser.add_argument(
    "--use-gpu",
    action="store",
    default=torch.cuda.is_available(),
    type=bool,
    help="Should GPUs be used (detected automatically as %s)"
    % torch.cuda.is_available(),
)

if __name__ == "__main__":
    args = parser.parse_args()

    thisdir = path.dirname(path.realpath(__file__))

    # Load the original dataset
    if args.dataset == "sst":
        base_dataset = StanfordSentimentDataset(
            cachedir=thisdir + "/../cache", seed=args.seed, num_workers=args.num_workers
        )
    else:
        pass
    base_dataset.prepare_data()

    base_model = SingleSequenceToClass.load_from_checkpoint(
        checkpoint_path=thisdir + "/../checkpoints/standford_sentiment/checkpoint.ckpt",
        embedding=base_dataset.embedding(),
    )

    # Load the ROAR dataset
    roar_dataset = ROARDataset(
        cachedir=thisdir + "/../cache",
        model=base_model,
        base_dataset=base_dataset,
        k=5,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    roar_dataset.prepare_data()

    model = SingleSequenceToClass(embedding=base_dataset.embedding())

    logger = TensorBoardLogger(thisdir + "/../tensorboard", name="standford_sentiment")

    # Source uses the best model, measured with AUC metric, and evaluates every epoch.
    #  https://github.com/successar/AttentionExplanation/blob/master/Trainers/TrainerBC.py#L28
    checkpoint_callback = ModelCheckpoint(
        monitor="auc_val",
        dirpath=thisdir + "/../checkpoints/standford_sentiment",
        filename="checkpoint-{epoch:02d}-{auc_val:.2f}",
        mode="max",
    )
    pl.seed_everything(args.seed)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
        deterministic=True,
        logger=logger,
        gpus=int(args.use_gpu),
    )
    trainer.fit(model, roar_dataset)

    shutil.copyfile(
        checkpoint_callback.best_model_path,
        thisdir + "/../checkpoints/standford_sentiment/checkpoint.ckpt",
    )

    print("best checkpoint:", checkpoint_callback.best_model_path)
    print(
        trainer.test(
            datamodule=roar_dataset,
            verbose=False,
            ckpt_path=checkpoint_callback.best_model_path,
        )[0]
    )
