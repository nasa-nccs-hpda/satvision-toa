import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------
# FREEZING ENCODER SHARED FCN
# ---------------------------------------------------

def freeze_encoder_selective_simple(encoder):
    unfrozen_count = 0

    for name, param in encoder.named_parameters():
        # Keep unfrozen if name contains "layers.3"
        # or is exactly "norm.weight" or "norm.bias"
        if ("layers.2" in name):
            param.requires_grad = False
            unfrozen_count += 1
        else:
            param.requires_grad = True
            unfrozen_count += 1


# ---------------------------------------------------
# ChatGSFC FCNV2.5
# ---------------------------------------------------

# Used for residual connections within UNET Architecture
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # Add residual connection
        return F.relu(out)


# Actual model class
class OceanColorFCNV2point5(nn.Module):
    def __init__(
        self,
        swin_encoder,
        freeze_encoder=False,
        img_size=128,
        num_inputs=14,
        num_targets=1,
        max_output_value=15.631364822387695,
    ):
        super(OceanColorFCNV2point5, self).__init__()

        self.num_inputs = num_inputs
        self.num_targets = num_targets
        self.max_output_value = max_output_value

        self.encoder = swin_encoder  # SWINV2
        if freeze_encoder:
            print("Freezing encoder")
            freeze_encoder_selective_simple(self.encoder)

        # Based on your encoder output shapes:
        # enc_features[0]: [B, 1024, 16, 16] - 1/8 resolution
        # enc_features[1]: [B, 2048, 8, 8] - 1/16 resolution
        # enc_features[2]: [B, 4096, 4, 4] - 1/32 resolution
        # enc_features[3]: [B, 4096, 4, 4] - 1/32 resolution (duplicate of [2])

        self.encoder_channels = [1024, 2048, 4096, 4096]

        # Fusion convolutions to match channel dimensions before concatenation
        self.fusion_conv3 = nn.Conv2d(
            self.encoder_channels[2], 1024, 1)  # 4096 -> 1024
        self.fusion_conv2 = nn.Conv2d(
            self.encoder_channels[1], 512, 1)   # 2048 -> 512
        self.fusion_conv1 = nn.Conv2d(
            self.encoder_channels[0], 256, 1)   # 1024 -> 256

        # Start from the deepest features [B, 4096, 4, 4]
        # Upconv4: 4x4 -> 8x8 (1/16 resolution)
        self.upconv4 = nn.Sequential(
            ResidualBlock(4096, 2048),  # Process deepest features
            ResidualBlock(2048, 1024)   # Reduce channels
        )

        # Upconv3: 8x8 -> 16x16 (1/8 resolution) with fusion
        self.upconv3 = nn.Sequential(
            # 1x1 conv to handle concatenated channels
            # (1024 from upconv4 + 1024 from fusion)
            nn.Conv2d(1024 + 1024, 1024, 1),
            ResidualBlock(1024, 1024),
            ResidualBlock(1024, 512)
        )

        # Upconv2: 16x16 -> 32x32 (1/4 resolution) with fusion
        self.upconv2 = nn.Sequential(
            # 1x1 conv to handle concatenated channels
            # (512 from upconv3 + 512 from fusion)
            nn.Conv2d(512 + 512, 512, 1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 256)
        )

        # Upconv1: 32x32 -> 64x64 (1/2 resolution) with fusion
        self.upconv1 = nn.Sequential(
            # 1x1 conv to handle concatenated channels
            #  (256 from upconv2 + 256 from fusion)
            nn.Conv2d(256 + 256, 256, 1),
            ResidualBlock(256, 256),
            ResidualBlock(256, 128)
        )

        # Final upconv: 64x64 -> 128x128 (full resolution)
        self.upconv0 = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 64)
        )

        # Final prediction layer
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),  # 1x1 conv for final prediction
            nn.Sigmoid()  # Keep sigmoid for bounded output [0, 1]
        )

    def forward(self, x, mask=None):

        # Get multi-scale features from encoder
        enc_features = self.encoder.extra_features(x)  # List of 4 feature maps

        # Start with the deepest features: [B, 4096, 4, 4]
        x = enc_features[-1]

        # Upconv4: 4x4 -> 8x8 (process deepest features)
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear",
            align_corners=False)  # [B, 4096, 8, 8]
        x = self.upconv4(x)  # [B, 1024, 8, 8]

        # Upconv3: 8x8 -> 16x16 with fusion from
        # enc_features[2] (also 4x4, need to upsample)
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear",
            align_corners=False)  # [B, 1024, 16, 16]
        # Fuse with enc_features[2] (4096 channels at 4x4, upsample to 16x16)
        enc_feat = F.interpolate(
            enc_features[2],
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )  # [B, 4096, 16, 16]
        enc_feat = self.fusion_conv3(enc_feat)  # [B, 1024, 16, 16]
        x = torch.cat([x, enc_feat], dim=1)  # [B, 2048, 16, 16]
        x = self.upconv3(x)  # [B, 512, 16, 16]

        # Upconv2: 16x16 -> 32x32 with fusion from
        # enc_features[1] (2048 channels at 8x8)
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear",
            align_corners=False)  # [B, 512, 32, 32]
        # Fuse with enc_features[1] (2048 channels at 8x8, upsample to 32x32)
        enc_feat = F.interpolate(
            enc_features[1],
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )  # [B, 2048, 32, 32]
        enc_feat = self.fusion_conv2(enc_feat)  # [B, 512, 32, 32]
        x = torch.cat([x, enc_feat], dim=1)  # [B, 1024, 32, 32]
        x = self.upconv2(x)  # [B, 256, 32, 32]

        # Upconv1: 32x32 -> 64x64 with fusion
        # from enc_features[0] (1024 channels at 16x16)
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear",
            align_corners=False)  # [B, 256, 64, 64]
        # Fuse with enc_features[0] (1024 channels at 16x16, upsample to 64x64)
        enc_feat = F.interpolate(
            enc_features[0],
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )  # [B, 1024, 64, 64]
        enc_feat = self.fusion_conv1(enc_feat)  # [B, 256, 64, 64]
        x = torch.cat([x, enc_feat], dim=1)  # [B, 512, 64, 64]
        x = self.upconv1(x)  # [B, 128, 64, 64]

        # Final upconv: 64x64 -> 128x128 (full resolution)
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear",
            align_corners=False)  # [B, 128, 128, 128]
        x = self.upconv0(x)  # [B, 64, 128, 128]

        # Final prediction with proper scaling and clamping
        x = self.final(x)  # [B, 1, 128, 128] with sigmoid output [0, 1]

        return x


# ---------------------------------------------------
# FCNV2 based on e2e training
# ---------------------------------------------------

class OceanColorFCNV2(nn.Module):
    def __init__(
        self,
        swin_encoder,
        freeze_encoder=False,
        img_size=128,
        num_inputs=14,
        num_targets=1,
        min_val=0.0,
        max_val=15.631364822387695
    ):  # noqa: E501

        super(OceanColorFCNV2, self).__init__()

        self.num_inputs = num_inputs
        self.num_targets = num_targets
        self.min_val = min_val
        self.max_val = max_val

        self.encoder = swin_encoder  # SWINV2
        if freeze_encoder:
            print("Freezing encoder")
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Modified OceanColorModel from end-to-end
        self.upconv5 = nn.Sequential(
            nn.Conv2d(4096, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        self.upconv4 = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.upconv3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.upconv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.upconv1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Final layer (128x128x64 → 128x128x1)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),  # 1x1 conv for final prediction
            nn.Sigmoid(),  # Remove if using BCE with logits loss
        )

    def forward(self, x):
        # x: input tensor from transformer encoder [batch_size, 4096, H, W]
        # No skip_connections parameter needed
        x = self.encoder.extra_features(x)[-1]

        # Upconv5: Upsample and process
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.upconv5(x)

        # Upconv4: Upsample and process
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.upconv4(x)

        # Upconv3: Upsample and process
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.upconv3(x)

        # Upconv2: Upsample and process
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.upconv2(x)

        # Upconv1: Upsample and process
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.upconv1(x)

        # Final: Upsample and predict
        x = self.final(x)
        x = x * self.max_val
        x = torch.clamp(x, min=self.min_val, max=self.max_val)

        return x


# ---------------------------------------------------
# UNET FROM CHATGSFC (based on FCNV2)
# ---------------------------------------------------

class OceanColorUNETV2(nn.Module):
    def __init__(
        self,
        swin_encoder,
        freeze_encoder=False,
        img_size=128,
        num_inputs=14,
        num_targets=1,
    ):  # noqa: E501

        super(OceanColorUNETV2, self).__init__()

        self.num_inputs = num_inputs
        self.num_targets = num_targets

        self.encoder = swin_encoder  # SWINV2
        if freeze_encoder:
            print("Freezing encoder")
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Upsampling layers - adjusted for 4x4 -> 128x128
        # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 (5 upsampling steps)
        self.up1 = nn.ConvTranspose2d(
            4096, 2048, kernel_size=4, stride=2, padding=1)  # 4x4 -> 8x8
        self.up2 = nn.ConvTranspose2d(
            2048, 1024, kernel_size=4, stride=2, padding=1)  # 8x8 -> 16x16
        self.up3 = nn.ConvTranspose2d(
            1024, 512, kernel_size=4, stride=2, padding=1)   # 16x16 -> 32x32
        self.up4 = nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1)    # 32x32 -> 64x64
        self.up5 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1)    # 64x64 -> 128x128

        # Decoder blocks - handle both with and without skip connections
        # Block 1: No skip connection, input = 2048
        # (after upsampling from 4096)
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        # Block 2: Optional skip connection,
        # input = 1024 or 1024 + skip_channels
        self.conv2 = nn.Sequential(
            # Will be adjusted dynamically if skip exists
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # Block 3: Optional skip connection
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Block 4: Optional skip connection
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Block 5: Optional skip connection
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Final layer
        self.final = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x, skip_connections=None):
        """
        Args:
            x: Input tensor from encoder (batch, 4096, 4, 4)
            skip_connections: Optional list of skip connection tensors
        """
        x = self.encoder.extra_features(x)[-1]  # (batch, 4096, 4, 4)

        # Level 1: 4096x4x4 -> 2048x8x8 (no skip)
        x = self.up1(x)  # 4096x4x4 -> 2048x8x8
        x = self.conv1(x)  # 2048x8x8 -> 2048x8x8

        # Level 2: 2048x8x8 -> 1024x16x16 + optional skip
        x = self.up2(x)  # 2048x8x8 -> 1024x16x16
        if skip_connections is not None \
                and len(skip_connections) > 0 \
                and skip_connections[0] is not None:
            x = torch.cat([x, skip_connections[0]], dim=1)
            # Need to adjust conv2 input channels dynamically
            # For now, let's use a 1x1 conv to match channels
            x = nn.Conv2d(x.shape[1], 1024, 1).to(x.device)(x)
        x = self.conv2(x)  # 1024x16x16 -> 1024x16x16

        # Level 3: 1024x16x16 -> 512x32x32 + optional skip
        x = self.up3(x)  # 1024x16x16 -> 512x32x32
        if skip_connections is not None \
                and len(skip_connections) > 1 \
                and skip_connections[1] is not None:
            x = torch.cat([x, skip_connections[1]], dim=1)
            x = nn.Conv2d(x.shape[1], 512, 1).to(x.device)(x)
        x = self.conv3(x)  # 512x32x32 -> 512x32x32

        # Level 4: 512x32x32 -> 256x64x64 + optional skip
        x = self.up4(x)  # 512x32x32 -> 256x64x64
        if skip_connections is not None \
                and len(skip_connections) > 2 \
                and skip_connections[2] is not None:
            x = torch.cat([x, skip_connections[2]], dim=1)
            x = nn.Conv2d(x.shape[1], 256, 1).to(x.device)(x)

        x = self.conv4(x)  # 256x64x64 -> 256x64x64

        # Level 5: 256x64x64 -> 128x128x128 + optional skip
        x = self.up5(x)  # 256x64x64 -> 128x128x128
        if skip_connections is not None \
                and len(skip_connections) > 3 \
                and skip_connections[3] is not None:
            x = torch.cat([x, skip_connections[3]], dim=1)
            x = nn.Conv2d(x.shape[1], 128, 1).to(x.device)(x)
        x = self.conv5(x)  # 128x128x128 -> 128x128x128

        # Final
        x = self.final(x)  # 128x128x128 -> 1x128x128

        return x


# ---------------------------------------------------
# FCN FROM 3D CLOUD
# ---------------------------------------------------
class OceanColorFCNV3(nn.Module):
    def __init__(
        self,
        swin_encoder,
        freeze_encoder=False,
        img_size=128,
        num_inputs=14,
        num_targets=1,
    ):  # noqa: E501

        super(OceanColorFCNV3, self).__init__()

        self.num_inputs = num_inputs
        self.num_targets = num_targets

        self.encoder = swin_encoder
        if freeze_encoder:
            print('Freezing encoder')
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                4096, 2048, kernel_size=3, stride=2,
                padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.25),  # Higher early in decoder

            nn.ConvTranspose2d(
                2048, 512, kernel_size=3, stride=2,
                padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.ConvTranspose2d(
                512, 256, kernel_size=3, stride=2,
                padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.15),

            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2,
                padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.05),  # Lower near output
        )

        self.final_layer = nn.Conv2d(64, num_targets, kernel_size=3,
                                     stride=1, padding=1)  # 128x128x1

    def forward(self, x):
        x = self.encoder.extra_features(x)[-1]
        x = self.decoder(x)
        x = self.final_layer(x)

        return x
