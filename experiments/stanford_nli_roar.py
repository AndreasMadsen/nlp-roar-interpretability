import argparse
import os.path as path
import shutil
import json
import os

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from comp550.dataset import SNLIDataModule, ROARDataset
from comp550.model import MultipleSequenceToClass

parser = argparse.ArgumentParser(description="Run ROAR benchmark for SNLI.")
parser.add_argument("--k",
                    action="store",
                    default=1,
                    type=int,
                    help="The proportion of tokens to mask.")
parser.add_argument("--random-masking",
                    action="store_true",
                    default=False,
                    help="Whether to mask random tokens or not.",)
parser.add_argument("--seed", action="store", default=0, type=int, help="Random seed")
parser.add_argument("--num-workers",
                    action="store",
                    default=4,
                    type=int,
                    help="The number of workers to use in data loading")
# epochs = 25 (https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/ExperimentsQA.py#L10)
parser.add_argument("--max-epochs",
                    action="store",
                    default=25,
                    type=int,
                    help="The max number of epochs to use")
parser.add_argument("--use-gpu",
                    action="store",
                    default=torch.cuda.is_available(),
                    type=bool,
                    help=f"Should GPUs be used (detected automatically as {torch.cuda.is_available()})")

if __name__ == "__main__":
    args = parser.parse_args()
    seed_everything(args.seed)

    thisdir = path.dirname(path.realpath(__file__))
    experiment_id = f"snli_roar_s-{args.seed}_-k{args.k}_r-{int(args.random_masking)}"

    # Create ROAR dataset
    base_dataset = SNLIDataModule(
        cachedir=thisdir + "/../cache", num_workers=args.num_workers
    )
    base_dataset.prepare_data()
    base_model = MultipleSequenceToClass.load_from_checkpoint(
        checkpoint_path=thisdir + f"/../checkpoints/snli_s-{args.seed}/checkpoint.ckpt",
        embedding=base_dataset.embedding()
    )
    roar_dataset = ROARDataset(
        model=base_model,
        base_dataset=base_dataset,
        k=args.k,
        random_masking=args.random_masking,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=128,
    )

    logger = TensorBoardLogger(thisdir + "/../tensorboard", name=experiment_id)
    model = MultipleSequenceToClass(base_dataset.embedding())

    """
    Original implementation chooses the best checkpoint on the basis of accuracy
    https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/Trainers/TrainerQA.py#L12
    """
    checkpoint_callback = ModelCheckpoint(
        monitor="acc_val",
        dirpath=thisdir + f"/../checkpoints/{experiment_id}",
        filename="checkpoint-{epoch:02d}-{acc_val:.2f}",
        mode="max",
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
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

    os.makedirs(thisdir + '/../results', exist_ok=True)
    with open(thisdir + f"/../results/{experiment_id}.json", "w") as f:
        json.dump({"seed": args.seed, "dataset": "snli", "roar": True,
                   "k": args.k, "random_masking": args.random_masking,
                   **results}, f)
