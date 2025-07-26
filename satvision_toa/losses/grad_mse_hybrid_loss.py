import torch
import torch.nn as nn


class GradMSEHybridLoss(nn.Module):
    def __init__(self, gradient_weight=0.1):
        super(GradMSEHybridLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.gradient_weight = gradient_weight

    def gradient_loss(self, pred, target):
        pred = pred.unsqueeze(1)    # becomes (b, 1, w, h)
        target = target.unsqueeze(1)  # becomes (b, 1, w, h)

        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        return torch.mean(
            torch.abs(pred_dx - target_dx)) + torch.mean(
                torch.abs(pred_dy - target_dy))

    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        grad_loss = self.gradient_loss(pred, target)
        return mse + self.gradient_weight * grad_loss
