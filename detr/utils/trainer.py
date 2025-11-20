import torch
import torchvision
from tqdm import tqdm

from detr.utils.misc import rescale_boxes
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
    def evaluate_model(self, save_plots=True):
        self.model.eval()
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        val_loss = []
        for n_iter, (img, targets) in pbar:
            img = img.to(self.device)
            preds = self.model(img)
            self.plot_predictions(img, targets, preds, iter=n_iter, save_plots=save_plots)
            loss, loss_dict = self.loss_fn(preds, targets)
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

        pred_cls_prob, pred_cls_idx = torch.max(pred_cls.softmax(-1), -1)
        valid_mask = pred_cls_idx != self.config["MODEL"]["n_classes"]
        out_boxes = []
        out_cls_idx = []
        out_cls_prob = []
        for b in range(B):
            curr_boxes = pred_boxes[b][valid_mask[b]]
            curr_cls_idx = pred_cls_idx[b][valid_mask[b]]
            curr_cls_prob = pred_cls_prob[b][valid_mask[b]]
            curr_boxes, curr_cls_idx, curr_cls_prob = self.apply_nms(curr_boxes, curr_cls_idx, curr_cls_prob)
            out_boxes.append(curr_boxes)
            out_cls_idx.append(curr_cls_idx)
            out_cls_prob.append(curr_cls_prob)
        return out_boxes, out_cls_idx, out_cls_prob

    def apply_nms(self, boxes, pred_cls_idx, pred_cls_prob, iou_threshold=0):
        unique_ids = torch.unique(pred_cls_idx)
        all_boxes = []
        all_cls_idx = []
        all_cls_prob = []
        for curr_unique_id in unique_ids:
            curr_maks = pred_cls_idx == curr_unique_id
            selected_ids = torchvision.ops.nms(boxes[curr_maks], pred_cls_prob[curr_maks], iou_threshold)
            all_boxes.append(boxes[curr_maks][selected_ids])
            all_cls_idx.append(pred_cls_idx[curr_maks][selected_ids])
            all_cls_prob.append(pred_cls_prob[curr_maks][selected_ids])
        all_boxes = torch.cat(all_boxes, 0) if len(all_boxes) > 0 else []
        all_cls_idx = torch.cat(all_cls_idx, 0) if len(all_cls_idx) > 0 else []
        all_cls_prob = torch.cat(all_cls_prob, 0) if len(all_cls_prob) > 0 else []
        return all_boxes, all_cls_idx, all_cls_prob

    def plot_predictions(self, img, targets, preds, batch=0, iter=0, save_plots=False):
        img_size = self.config["DATA"]["img_size"]
        pred_boxes, pred_cls_idx, pred_cls_prob = self.post_processor(preds)
        gt_boxes = rescale_boxes(targets[batch]["boxes"], img_size[0], img_size[1])
        gt_class = targets[batch]["class_id"]
        pred_boxes = rescale_boxes(pred_boxes[batch], img_size[0], img_size[1])
        imgs = plot_img_with_boxes(
            img[batch],
            gt_boxes,
            gt_class,
            pred_boxes,
            pred_cls_idx[batch],
            pred_cls_prob[batch],
            return_figure=False,
            output_path=f'{self.config["IMG_OUT_DIR"]}/{str(iter).zfill(3)}.png',
        )
        if not save_plots:
            self.write_images_to_tb(imgs, self.total_iters_train, f"img/{str(iter).zfill(4)}")
