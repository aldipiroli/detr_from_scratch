import torch
from tqdm import tqdm

from detr.utils.plotters import plot_img_with_boxes
from detr.utils.trainer_base import TrainerBase


class Trainer(TrainerBase):
    def __init__(self, config, logger):
        super().__init__(config, logger)

    def train(self):
        self.logger.info("Started training..")
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.config["OPTIM"]["num_epochs"]):
            self.epoch = epoch
            self.train_one_epoch()
            self.evaluate_model()
            if epoch % self.config["OPTIM"]["save_ckpt_every"] == 0:
                self.save_checkpoint()

    def train_one_epoch(self):
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for n_iter, (img, targets) in pbar:
            img = img.to(self.device)
            preds = self.model(img)
            pred_boxes, pred_cls = self.post_processor(preds)
            loss, loss_dict = self.loss_fn(preds, targets)
            self.write_dict_to_tb(loss_dict, self.total_iters_train, prefix="train")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.total_iters_train += 1
            pbar.set_postfix(
                {
                    "mode": "train",
                    "epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}",
                    "loss": loss.item(),
                }
            )
        pbar.close()

    @torch.no_grad()
    def evaluate_model(self, save_plots=False):
        self.model.eval()
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        val_loss = []
        for n_iter, (img, labels) in pbar:
            img = img.to(self.device)
            labels = labels.to(self.device)

            preds = self.model(img)
            loss, loss_dict = self.loss_fn(preds, labels)
            val_loss.append(loss)
            self.write_dict_to_tb(loss_dict, self.total_iters_val, prefix="val")
            self.total_iters_val += 1
            pbar.set_postfix(
                {
                    "mode": "val",
                    "epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}",
                    "loss": loss.item(),
                }
            )
        self.write_float_to_tb(torch.mean(loss).item(), name="val/avg_loss", step=self.epoch)
        pbar.close()

    def post_processor(self, preds):
        pred_boxes, pred_cls = preds
        B = pred_boxes.shape[0]

        pred_cls_idx = pred_cls.softmax(-1).argmax(-1)
        valid_mask = pred_cls_idx != self.config["MODEL"]["n_classes"]
        out_boxes = []
        out_labels = []
        for b in range(B):
            out_boxes.append(pred_boxes[b][valid_mask[b]])
            out_labels.append(pred_cls_idx[b][valid_mask[b]])
        return out_boxes, out_labels

    def plot_predictions(self, img, targets, pred_boxes, batch=0):
        plot_img_with_boxes(img[batch], targets[batch]["boxes"], pred_boxes[batch])
