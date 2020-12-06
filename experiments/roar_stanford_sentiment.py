import argparse
import json
import os
import os.path as path
import shutil

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from comp550.dataset import StanfordSentimentDataset, ROARDataset
from comp550.model import SingleSequenceToClass

torch.multiprocessing.set_sharing_strategy("file_system")

parser = argparse.ArgumentParser(description="Run ROAR benchmark for SST")
parser.add_argument(
    "model_checkpoint_path",
    type=str,
    help="Path to the model checkpoint to use for ROAR masking.",
)
parser.add_argument(
    "--k", action="store", default=1, type=int, help="The proportion of tokens to mask."
)
parser.add_argument(
    "--do-random-masking",
    action="store_true",
    default=False,
    help="Whether to mask random tokens or not.",
)
parser.add_argument("--seed", action="store", default=0, type=int, help="Random seed")
parser.add_argument(
    "--num-workers",
    action="store",
    default=4,
    type=int,
    help="The number of workers to use in data loading",
)
# epochs = 8 (https://github.com/successar/AttentionExplanation/blob/master/ExperimentsBC.py#L11)
parser.add_argument(
    "--max-epochs",
    action="store",
    default=8,
    type=int,
    help="The max number of epochs to use",
)
parser.add_argument(
    "--use-gpu",
    action="store",
    default=torch.cuda.is_available(),
    type=bool,
    help=f"Should GPUs be used (detected automatically as {torch.cuda.is_available()})",
)

if __name__ == "__main__":
    args = parser.parse_args()

    thisdir = path.dirname(path.realpath(__file__))
    experiment_id = f"sst_{args.seed}_{args.k}_{args.do_random_masking}"

    if not path.exists(thisdir + "/../results"):
        os.makedirs(thisdir + "/../results")

    base_dataset = StanfordSentimentDataset(
        cachedir=thisdir + "/../cache", seed=args.seed, num_workers=args.num_workers
    )
    base_dataset.prepare_data()

    base_model = SingleSequenceToClass.load_from_checkpoint(
        checkpoint_path=args.model_checkpoint_path, embedding=base_dataset.embedding()
    )

    roar_dataset = ROARDataset(
        model=base_model,
        base_dataset=base_dataset,
        k=args.k,
        do_random_masking=args.do_random_masking,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    logger = TensorBoardLogger(thisdir + "/../tensorboard", name="standford_sentiment")
    model = SingleSequenceToClass(base_dataset.embedding())

    # Source uses the best model, measured with AUC metric, and evaluates every epoch.
    #  https://github.com/successar/AttentionExplanation/blob/master/Trainers/TrainerBC.py#L28
    checkpoint_callback = ModelCheckpoint(
        monitor="auc_val",
        dirpath=thisdir + f"/../checkpoints/{experiment_id}",
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
        thisdir + f"/../checkpoints/{experiment_id}/checkpoint.ckpt",
    )

    print("best checkpoint:", checkpoint_callback.best_model_path)
    results = trainer.test(
        datamodule=roar_dataset,
        verbose=False,
        ckpt_path=checkpoint_callback.best_model_path,
    )[0]
    print(results)

    with open(thisdir + f"/../results/{experiment_id}.json", "w") as f:
        json.dump({"seed": args.seed, "dataset": "sst", "k": args.k, **results}, f)
