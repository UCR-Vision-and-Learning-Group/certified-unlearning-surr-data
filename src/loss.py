import torch.nn as nn


class L2RegularizedCrossEntropyLoss(nn.Module):
    def __init__(self, l2_lambda=0.01):
        """
        Wrapper for CrossEntropyLoss with L2 regularization.

        Args:
            l2_lambda (float): Regularization strength (default: 0.01).
        """
        super(L2RegularizedCrossEntropyLoss, self).__init__()
        self.l2_lambda = l2_lambda
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, targets, model):
        """
        Compute L2-regularized cross-entropy loss.

        Args:
            logits (torch.Tensor): Model outputs (logits).
            targets (torch.Tensor): Ground truth labels.
            model (torch.nn.Module): The model containing parameters to regularize.

        Returns:
            torch.Tensor: Total loss (cross-entropy + L2 regularization).
        """
        # Compute cross-entropy loss
        ce_loss = self.cross_entropy(logits, targets)

        # Compute L2 regularization
        l2_reg = sum(param.pow(2).sum() for param in model.parameters())

        # Total loss
        total_loss = ce_loss + self.l2_lambda * l2_reg
        return total_loss
