import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLoss(nn.Module):
    def __init__(
                self,
                loss_type='combined',
                scales=[1, 2, 4],
                weights=[1.0, 0.5, 0.25]
            ):
        super().__init__()
        self.scales = scales
        self.weights = weights

        if loss_type == 'mse':
            self.base_loss = nn.MSELoss()
        elif loss_type == 'l1':
            self.base_loss = nn.L1Loss()
        elif loss_type == 'huber':
            self.base_loss = nn.SmoothL1Loss()
        elif loss_type == 'combined':
            self.mse_loss = nn.MSELoss()
            self.l1_loss = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        self.loss_type = loss_type

    def forward(self, pred, target):
        total_loss = 0

        for scale, weight in zip(self.scales, self.weights):
            if scale == 1:
                pred_scaled = pred
                target_scaled = target
            else:
                # Downsample using average pooling
                pred_scaled = F.avg_pool2d(
                    pred, kernel_size=scale, stride=scale)
                target_scaled = F.avg_pool2d(
                    target, kernel_size=scale, stride=scale)

            # Compute loss at this scale
            if self.loss_type == 'combined':
                scale_loss = 0.7 * self.mse_loss(
                    pred_scaled, target_scaled) + \
                    0.3 * self.l1_loss(pred_scaled, target_scaled)
            else:
                scale_loss = self.base_loss(pred_scaled, target_scaled)

            total_loss += weight * scale_loss

        return total_loss
