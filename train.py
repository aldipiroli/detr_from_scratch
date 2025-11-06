import argparse

from detr.dataset.voc_dataset import VOCDataset
from detr.model.loss import DetrLoss
from detr.model.model import DetrModelDYI
from detr.utils.misc import get_logger, load_config, make_artifacts_dirs
from detr.utils.trainer import Trainer


def train(args):
    config = load_config(args.config)
    config = make_artifacts_dirs(config, log_datetime=True)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    train_dataset = VOCDataset(cfg=config, mode="train", logger=logger)
    val_dataset = VOCDataset(cfg=config, mode="val", logger=logger)

    model = DetrModelDYI(config)
    trainer.set_model(model)

    trainer.set_dataset(train_dataset, val_dataset, data_config=config["DATA"], val_set_batch_size=1)
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.set_loss_function(DetrLoss(config, logger))

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="config/config.yaml", help="Config path")
    args = parser.parse_args()
    train(args)
