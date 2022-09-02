from torch import nn
from segmentation_models_pytorch.losses import DiceLoss


class BCEDiceLoss(nn.Module):

    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index=ignore_index
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(mode='binary', from_logits=True, ignore_index=ignore_index)

    def forward(self, y_pr, y_gt):
        if self.ignore_index:
            valid = y_gt != self.ignore_index
            bce_loss = self.bce(y_pr[valid], y_gt[valid])
        else:
            bce_loss = self.bce(y_pr, y_gt)
        return self.dice(y_pr, y_gt) + bce_loss


class CEDiceLoss(nn.Module):

    def __init__(self, ignore_index=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(mode='multiclass', from_logits=True, ignore_index=ignore_index)

    def forward(self, y_pr, y_gt):
        return self.ce(y_pr, y_gt) + self.dice(y_pr, y_gt)