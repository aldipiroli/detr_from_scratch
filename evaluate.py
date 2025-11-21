import argparse
import os

from detr.dataset.voc_dataset import VOCDataset
from detr.model.loss import DetrLoss
from detr.model.model import DetrModelDYI
from detr.utils.misc import get_logger, load_config, make_artifacts_dirs
from detr.utils.trainer import Trainer


def get_all_ckps(path):
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a valid directory.")
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return sorted(files)


def train(args):
    config = load_config(args.config)
    config = make_artifacts_dirs(config, log_datetime=True)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    train_dataset = VOCDataset(cfg=config, mode="train", logger=logger)
    val_dataset = VOCDataset(cfg=config, mode="val", logger=logger)
    trainer.set_dataset(
        train_dataset, val_dataset, data_config=config["DATA"], val_set_batch_size=1, shuffle_valset_once=True
    )
    trainer.set_loss_function(DetrLoss(config, logger))
    model = DetrModelDYI(config)
    trainer.set_model(model)

    if args.ckpt is not None:
        trainer.load_checkpoint(args.ckpt)
        trainer.evaluate_model(save_plots=True)

    if args.ckpt_folder is not None:
        all_ckpts = get_all_ckps(args.ckpt_folder)
        for curr_ckpt in all_ckpts:
            trainer.load_checkpoint(curr_ckpt)
            trainer.evaluate_model(save_plots=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="config/config.yaml", help="Config path")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ckpt_folder", type=str, default=None)
    args = parser.parse_args()
    train(args)
