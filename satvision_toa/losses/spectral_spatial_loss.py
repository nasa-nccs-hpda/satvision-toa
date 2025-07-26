import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralSpatialLoss(nn.Module):
    def __init__(self, edge_weight=0.5, num_bands=14):
        super(SpectralSpatialLoss, self).__init__()
        self.edge_weight = edge_weight
        self.num_bands = num_bands

        # We'll create kernels dynamically in compute_edge_loss
        # No need to register buffers since we create them on-demand

    def _create_sobel_kernels(self, device, dtype):
        """Create Sobel kernels on the specified device and dtype"""
        sobel_x = torch.tensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ],
            dtype=dtype, device=device).view(1, 1, 3, 3)

        sobel_y = torch.tensor([
                [-1, -2, -1],
                [0,  0,  0],
                [1,  2,  1]
            ],
            dtype=dtype, device=device).view(1, 1, 3, 3)

        return sobel_x, sobel_y

    def compute_edge_loss(self, pred, target):
        # print(f"pred, target shapes: {pred.shape, target.shape}")
        """Compute edge-based loss using Sobel operators"""
        edge_loss = 0.0
        device = pred.device
        dtype = pred.dtype

        # Create Sobel kernels directly on the input device with matching dtype
        sobel_x, sobel_y = self._create_sobel_kernels(device, dtype)

        # Apply edge detection to each band separately

        # Apply Sobel X
        pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
        target_edge_x = F.conv2d(target, sobel_x, padding=1)

        # Apply Sobel Y
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
        target_edge_y = F.conv2d(target, sobel_y, padding=1)

        # Combine X and Y gradients (magnitude)
        pred_edge_mag = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-8)
        target_edge_mag = torch.sqrt(
            target_edge_x**2 + target_edge_y**2 + 1e-8)

        # Add L1 loss for this band's edges
        edge_loss += F.l1_loss(pred_edge_mag, target_edge_mag)

        return edge_loss / self.num_bands  # Average over bands

    def forward(self, pred, target):
        """
        Forward pass of the loss function

        Args:
            pred: Predicted tensor [B, C, H, W] where C = num_bands
            target: Target tensor [B, C, H, W] where C = num_bands

        Returns:
            Combined spectral-spatial loss
        """
        # Spectral loss (L1 across all bands and pixels)
        spectral_loss = F.l1_loss(pred, target)

        # Spatial (edge) loss
        edge_loss = self.compute_edge_loss(pred, target)

        # Combined loss
        total_loss = spectral_loss + self.edge_weight * edge_loss

        return total_loss

    def get_loss_components(self, pred, target):
        """
        Return individual loss components for monitoring

        Returns:
            dict with 'spectral_loss', 'edge_loss', and 'total_loss'
        """
        spectral_loss = F.l1_loss(pred, target)
        edge_loss = self.compute_edge_loss(pred, target)
        total_loss = spectral_loss + self.edge_weight * edge_loss

        return {
            'spectral_loss': spectral_loss.item(),
            'edge_loss': edge_loss.item(),
            'total_loss': total_loss.item()
        }

    def set_edge_weight(self, new_weight):
        """Allow dynamic adjustment of edge weight during training"""
        self.edge_weight = new_weight

    def get_edge_weight(
                epoch,
                initial_weight=0.5,
                max_weight=2.0,
                transition_epoch=50
            ):
        if epoch < transition_epoch:
            return initial_weight
        else:
            # Gradually increase edge weight for sharper predictions
            progress = min((epoch - transition_epoch) / 50, 1.0)
            return initial_weight + (max_weight - initial_weight) * progress

    def __repr__(self):
        return (
            'SpectralSpatialLoss(edge_weight='
            f'{self.edge_weight}, num_bands={self.num_bands})'
        )
