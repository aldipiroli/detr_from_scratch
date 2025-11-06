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
        assignments = self.find_optimal_assignment(preds[0], preds[1], gt_boxes, gt_validity)
        loss = self.compute_loss(assignments, preds[0], preds[1], gt_boxes, gt_cls, gt_validity)

        loss_dict = {}
        loss_dict["loss"] = loss
        return loss, loss_dict

    def compute_loss(self, assignments, pred_boxes, pred_cls, gt_boxes, gt_cls, gt_validity):
        B = pred_boxes.shape[0]
        for batch_id in range(B):
            curr_assignment = assignments[batch_id]
            curr_pred_boxes = pred_boxes[batch_id]
            curr_pred_cls = pred_cls[batch_id]
            curr_gt_boxes = gt_boxes[batch_id].to(get_device())
            curr_gt_cls = gt_cls[batch_id].to(get_device())
            curr_gt_validity = gt_validity[batch_id]

            loss_cls = self.compute_cls_loss(curr_assignment, curr_pred_cls, curr_gt_cls, curr_gt_validity)

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

    def find_optimal_assignment(self, pred_boxes, pred_cls, gt_boxes, gt_validity):
        B = pred_boxes.shape[0]
        all_assignments = []
        for batch_id in range(B):
            cost_matrix = torch.zeros((self.n_queries, self.n_queries))
            curr_pred_boxes = pred_boxes[batch_id]
            curr_pred_cls = pred_cls[batch_id]
            curr_gt_boxes = gt_boxes[batch_id].to(get_device())
            curr_gt_validity = gt_validity[batch_id]

            for gt_idx in range(self.n_queries):
                for pred_idx in range(self.n_queries):
                    if curr_gt_validity[gt_idx]:
                        cls_cost = -curr_pred_cls[pred_idx][gt_idx]
                        box_cost = self.compute_box_cost(curr_pred_boxes[pred_idx], curr_gt_boxes[gt_idx])
                        cost_matrix[gt_idx, pred_idx] = cls_cost + box_cost
            assignment = linear_sum_assignment(cost_matrix.detach().numpy())
            all_assignments.append(assignment)
        return all_assignments

    def compute_box_cost(self, pred_boxes, gt_boxes):
        cost_l1 = self.compute_box_cost_l1(pred_boxes, gt_boxes)
        cost_giou = self.compute_box_cost_GIoU(pred_boxes, gt_boxes)
        cost_box = cost_l1 + cost_giou
        return cost_box

    def compute_box_cost_l1(self, pred_boxes, gt_boxes):
        cost_l1 = nn.functional.l1_loss(pred_boxes, gt_boxes)
        return cost_l1

    def compute_box_cost_GIoU(self, pred_boxes, gt_boxes):
        cost_giou = torchvision.ops.generalized_box_iou_loss(pred_boxes, gt_boxes)
        return cost_giou

    def get_targets(self, targets):
        B = len(targets)
        gt_boxes = torch.zeros(B, self.n_queries, 4)
        gt_cls = torch.zeros(B, self.n_queries)
        gt_validity = torch.zeros(B, self.n_queries).bool()

        for batch_id in range(len(targets)):
            curr_gt_boxes = targets[batch_id]["boxes"]
            curr_gt_cls = targets[batch_id]["class_id"]
            N = self.get_n_elements(curr_gt_boxes)
            gt_boxes[batch_id][:N] = curr_gt_boxes
            gt_cls[batch_id][:N] = curr_gt_cls
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
