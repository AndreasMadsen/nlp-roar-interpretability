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
    "model_checkpoint_path",
    action="store",
    type=str,
    help="Path to the model checkpoint used for masking tokens for ROAR."
)
parser.add_argument(
    "--dataset",
    action="store",
    default="sst",
    choices=["sst", "snli"],
    type=str,
    help="The dataset to use.",
)
parser.add_argument(
    "--k",
    action="store",
    default=0.1,
    type=float,
    help="The proportion of tokens to mask for each instance."
)
parser.add_argument(
    "--do-random-masking",
    action="store_true",
    help="Whether to mask random tokens or not."
)
parser.add_argument(
    "--seed", 
    action="store", 
    default=0, 
    type=int, 
    help="Random seed."
)
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

    print("Loading dataset %s." % args.dataset)
    if args.dataset == "sst":
        base_dataset = StanfordSentimentDataset(
            cachedir=thisdir + "/../cache", seed=args.seed, num_workers=args.num_workers
        )
        base_dataset.prepare_data()
        base_model = SingleSequenceToClass.load_from_checkpoint(
            checkpoint_path=args.model_checkpoint_path,
            embedding=base_dataset.embedding()
        )
    else:
        base_dataset = SNLIDataModule(
            cachedir=thisdir + "/../cache", num_workers=args.num_workers
        )
        base_dataset.prepare_data()
        base_model = MultipleSequenceToClass.load_from_checkpoint(
            checkpoint_path=args.model_checkpoint_path,
            embedding=base_dataset.embedding()
        )

    print("Masking %s dataset." % args.dataset)
    roar_dataset = ROARDataset(
        model=base_model,
        base_dataset=base_dataset,
        k=args.k,
        do_random_masking=args.do_random_masking,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    roar_dataset.prepare_data()

    if args.dataset == "sst":
        model = SingleSequenceToClass(embedding=base_dataset.embedding())
    else:
        model = MultipleSequenceToClass(embedding=base_dataset.embedding())

    logger = TensorBoardLogger(thisdir + "/../tensorboard", name="roar_%s" % args.dataset)

    # Source uses the best model, measured with AUC metric, and evaluates every epoch.
    #  https://github.com/successar/AttentionExplanation/blob/master/Trainers/TrainerBC.py#L28
    checkpoint_callback = ModelCheckpoint(
        monitor="auc_val",
        dirpath=thisdir + "/../checkpoints/roar_%s" % args.dataset,
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
        thisdir + "/../checkpoints/roar_%s/checkpoint.ckpt" % args.dataset,
    )

    print("best checkpoint:", checkpoint_callback.best_model_path)
    print(
        trainer.test(
            datamodule=roar_dataset,
            verbose=False,
            ckpt_path=checkpoint_callback.best_model_path,
        )[0]
    )
