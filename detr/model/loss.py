import torch
import torch.nn as nn
import torchvision
from scipy.optimize import linear_sum_assignment

from detr.utils.misc import get_device


class BaseLoss(nn.Module):
    def __init__(self, config, logger):
        super(BaseLoss, self).__init__()
        self.config = config
        self.logger = logger

    def forward(self, preds, labels):
        pass


class DetrLoss(BaseLoss):
    def __init__(self, config, logger):
        super(DetrLoss, self).__init__(config, logger)
        self.n_queries = config["MODEL"]["dec_n_queries"]
        self.valid_weight = config["LOSS"]["valid_weight"]
        self.invalid_weight = config["LOSS"]["invalid_weight"]

    def forward(self, preds, targets):
        gt_boxes, gt_cls, gt_validity = self.get_targets(targets)
        assignments = self.find_optimal_assignment(preds[0], preds[1], gt_boxes, gt_cls, gt_validity)
        # loss_cls, loss_box = self.compute_loss(assignments, preds[0], preds[1], gt_boxes, gt_cls, gt_validity)
        loss_cls, loss_box = 0, 0
        total_loss = loss_cls + loss_box

        loss_dict = {}
        loss_dict["loss_cls"] = loss_cls
        loss_dict["loss_box"] = loss_box
        loss_dict["total_loss"] = total_loss
        return total_loss, loss_dict

    @torch.no_grad()
    def find_optimal_assignment(self, pred_boxes, pred_cls, gt_boxes, gt_cls, gt_validity):
        B = pred_boxes.shape[0]
        l1_weight = self.config["ASSIGNMENT"]["l1_weight"]
        giou_weight = self.config["ASSIGNMENT"]["giou_weight"]
        all_assignments = []
        for batch_id in range(B):
            curr_pred_boxes = pred_boxes[batch_id]
            curr_pred_cls = pred_cls[batch_id].softmax(-1)
            curr_gt_boxes = gt_boxes[batch_id].to(get_device())
            curr_gt_cls = gt_cls[batch_id].to(get_device())
            curr_gt_validity = gt_validity[batch_id]
            if sum(curr_gt_validity) == 0:
                all_assignments.append(([], []))
                continue

            cls_cost = -curr_pred_cls[:, curr_gt_cls.int()][:, curr_gt_validity]
            giou_cost = 1 - torchvision.ops.generalized_box_iou(curr_pred_boxes, curr_gt_boxes[curr_gt_validity])
            l1_cost = torch.cdist(curr_pred_boxes, curr_gt_boxes[curr_gt_validity])
            cost_matrix = cls_cost + (giou_weight * giou_cost) + (l1_weight * l1_cost)
            assignment = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
            all_assignments.append(assignment)
        return all_assignments

    def compute_loss(self, assignments, pred_boxes, pred_cls, gt_boxes, gt_cls, gt_validity):
        B = pred_boxes.shape[0]
        all_loss_cls = []
        all_loss_box = []
        for batch_id in range(B):
            curr_assignment = assignments[batch_id]
            curr_pred_boxes = pred_boxes[batch_id]
            curr_pred_cls = pred_cls[batch_id]
            curr_gt_boxes = gt_boxes[batch_id].to(get_device())
            curr_gt_cls = gt_cls[batch_id].to(get_device())
            curr_gt_validity = gt_validity[batch_id]

            loss_cls = self.compute_cls_loss(curr_assignment, curr_pred_cls, curr_gt_cls, curr_gt_validity)
            all_loss_cls.append(loss_cls)

            loss_box = self.compute_box_loss(curr_assignment, curr_pred_boxes, curr_gt_boxes, curr_gt_validity)
            all_loss_box.append(loss_box)
        avg_loss_cls = torch.stack(all_loss_cls).mean()
        avg_loss_box = torch.stack(all_loss_box).mean()
        return avg_loss_cls, avg_loss_box

    def compute_box_loss(self, curr_assignment, curr_pred_boxes, curr_gt_boxes, gt_validity):
        gt_idx = curr_assignment[0]
        pred_idx = curr_assignment[1]
        gt_boxes = curr_gt_boxes[gt_idx]
        gt_validity = gt_validity[gt_idx]
        pred_boxes = curr_pred_boxes[pred_idx]
        loss_box = self.compute_box_cost(pred_boxes, gt_boxes)

        loss_box = loss_box[gt_validity]
        loss_box = loss_box.mean()
        return loss_box

    def compute_cls_loss(self, curr_assignment, pred_cls, gt_cls, gt_validity):
        gt_idx = curr_assignment[0]
        pred_idx = curr_assignment[1]
        gt_cls = gt_cls[gt_idx]
        gt_validity = gt_validity[gt_idx]
        pred_cls = pred_cls[pred_idx]

        loss = nn.CrossEntropyLoss(reduction="none")
        loss_cls = loss(pred_cls, gt_cls)
        weight = self.get_weighting(gt_validity)
        loss_cls = loss_cls * weight
        loss_cls = loss_cls.mean()
        return loss_cls

    def get_weighting(self, gt_validity):
        weight = torch.ones(self.n_queries) * self.valid_weight
        weight[~gt_validity] *= self.invalid_weight
        return weight.to(get_device())

    def get_targets(self, targets):
        B = len(targets)
        gt_boxes = torch.zeros(B, self.n_queries, 4)
        gt_cls = torch.zeros(B, self.n_queries)
        gt_validity = torch.zeros(B, self.n_queries).bool()

        for batch_id in range(len(targets)):
            curr_gt_boxes = targets[batch_id]["boxes"]
            curr_gt_cls = targets[batch_id]["class_id"]
            N = self.get_n_elements(curr_gt_boxes)
            gt_boxes[batch_id][:N] = curr_gt_boxes[:N]
            gt_cls[batch_id][:N] = curr_gt_cls[:N]
            gt_validity[batch_id][:N] = torch.tensor(True).repeat(N)
        return gt_boxes, gt_cls, gt_validity

    def get_n_elements(self, elements):
        return min(len(elements), self.n_queries)


###########################################
import debugpy

debugpy.listen(("localhost", 6001))
print("Waiting for debugger attach...")
debugpy.wait_for_client()
###########################################
