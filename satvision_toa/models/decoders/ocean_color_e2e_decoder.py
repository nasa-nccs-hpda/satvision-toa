import torch
import torch.nn as nn


class OceanColorFCN(nn.Module):
    """
    Basic 7-layer FCN for detecting Chlorophyll A content in oceans.
    """

    def __init__(self, in_channels=12, out_channels=1):
        """
        Initialize the FCN model for chlorophyll regression.

        Args:
            in_channels (int): bands from MODIS Aqua
            out_channels (int): target size, chlorophyll a (1,)
        """
        super(OceanColorFCN, self).__init__()
        self.in_channels = in_channels

        # Encoder wrapped in ModuleList
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        ])

        # Decoder wrapped in ModuleList
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 2, stride=2),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 2, stride=2),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_channels, 1),
                nn.Sigmoid()
            )
        ])

    def forward(self, x):
        """Forward pass of the model."""

        # Encoder forward pass
        for layer in self.encoder:
            x = layer(x)

        # Decoder forward pass
        for layer in self.decoder:
            x = layer(x)

        return x


class OceanColorUNET(nn.Module):
    """
    UNET deep learning model for detecting Chlorophyll A content in oceans.
    """

    def __init__(self, in_channels=12, out_channels=1):
        """
        Initialize the FCN model for chlorophyll regression.

        Args:
            in_channels (int): bands from MODIS Aqua
            out_channels (int): target size, chlorophyll a (1,)
        """
        super(OceanColorUNET, self).__init__()
        self.in_channels = in_channels

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Decoder with skip connections
        self.upconv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, 3, padding=1),  # +128 for skip connection
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(
            nn.Conv2d(64 + 64, 32, 3, padding=1),  # +64 for skip connection
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 12, 128, 128]
        Returns:
            torch.Tensor: Predicted chlorophyll a content of shape
            [batch_size, 1, 128, 128]
        """
        x1 = self.conv1(x)  # Save for skip
        x2 = self.conv2(x1)  # Save for skip
        x3 = self.conv3(x2)

        # Decoder with skip connections
        up2 = self.upconv2(x3)
        up2 = torch.cat([up2, x2], dim=1)  # Skip connection

        up1 = self.upconv1(up2)
        up1 = torch.cat([up1, x1], dim=1)  # Skip connection

        output = self.final(up1)
        return output
