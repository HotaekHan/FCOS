import torch
import torch.nn as nn
import torch.nn.functional as F


class IOULoss(nn.Module):
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, preds, targets, weight=None):
        pred_left = preds[:, 0]
        pred_top = preds[:, 1]
        pred_right = preds[:, 2]
        pred_bottom = preds[:, 3]

        target_left = targets[:, 0]
        target_top = targets[:, 1]
        target_right = targets[:, 2]
        target_bottom = targets[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()


class FocalLoss(nn.Module):
    def __init__(self, num_classes):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.iou_loss = IOULoss(loss_type='giou')

    def focal_loss(self, x, y):
        alpha = 0.25
        gamma = 2

        # target = one_hot_embedding(y, 1 + self.num_classes)
        target = y[:, :, 1:]  # exclude background

        prob = x.sigmoid()
        pred = prob*target + (1-prob)*(1-target)         # pt = p if t > 0 else 1-p
        weight = alpha*target + (1-alpha)*(1-target)  # w = alpha if t > 0 else 1-alpha
        weight = weight * (1-pred).pow(gamma)
        weight = weight.detach()

        loss = F.binary_cross_entropy_with_logits(input=x, target=target, weight=weight, reduction='sum')
        return loss

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets, center_preds, center_targets):
        # where is not back ground. 0 is background
        pos_ind = cls_targets[:, :, 0] <= 0
        num_pos = pos_ind.data.long().sum()

        cls_loss = self.focal_loss(cls_preds, cls_targets)
        # loc_loss = F.smooth_l1_loss(loc_preds[pos_ind], loc_targets[pos_ind], reduction='sum')
        loc_loss = self.iou_loss(loc_preds[pos_ind], loc_targets[pos_ind])
        center_loss = self.bce_loss(center_preds, center_targets)

        return loc_loss, cls_loss, center_loss, num_pos
