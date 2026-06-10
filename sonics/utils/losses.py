import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, label_smoothing=0.0, **kwargs):
        super(BCEWithLogitsLoss, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        if self.label_smoothing:
            target = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return super(BCEWithLogitsLoss, self).forward(input, target)

class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, label_smoothing=0.0, reduction="mean"):
        """
        Args:
            alpha (float): Weighting factor in range (0,1) to balance positive vs negative examples.
            gamma (float): Focusing parameter to reduce the relative loss for well-classified examples.
            label_smoothing (float): Label smoothing factor to reduce the confidence of the true label.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                             'none': no reduction will be applied,
                             'mean': the sum of the output will be divided by the number of elements in the output,
                             'sum': the output will be summed.
        """
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, input, target):
        """
        Args:
            input (Tensor): Predicted logits for each example.
            target (Tensor): Ground truth binary labels (0 or 1) for each example.
        """
        if self.label_smoothing:
            target = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        p = torch.sigmoid(input)

        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss
