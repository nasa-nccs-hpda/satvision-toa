import os
import sys
import torch
import subprocess
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset, DataLoader

sys.path.append('/explore/nobackup/people/jacaraba/development/satvision-toa')
from satvision_toa.utils import load_config
from satvision_toa.models.mim import build_mim_model


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, C, H, W), targets: (B, H, W)
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score.mean()


class DummyCloudDataset(Dataset):
    def __init__(
                self,
                num_samples=100,
                input_dims=(14, 128, 128),
                mask_dims=(96, 40),
                num_classes=9
            ):
        self.num_samples = num_samples
        self.input_dims = input_dims
        self.mask_dims = mask_dims
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chip = torch.rand(self.input_dims, dtype=torch.float32)
        # multi-class mask (0–8)
        mask = torch.randint(0, self.num_classes, self.mask_dims, dtype=torch.long)
        return {"chip": chip, "mask": mask}


def get_dummy_dataloader(batch_size=8, num_workers=0):
    dataset = DummyCloudDataset()
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


class SatVisionUNet(nn.Module):
    def __init__(self, out_channels=9, freeze_encoder=True, input_channels=14, final_size=(96, 40)):
        super().__init__()
        self.final_size = final_size

        # Load Swin encoder
        config = load_config()
        backbone = build_mim_model(config)
        self.encoder = backbone.encoder

        # Adjust input channels if needed
        if input_channels != 14:
            self.encoder.patch_embed.proj = nn.Conv2d(
                input_channels,
                self.encoder.patch_embed.proj.out_channels,
                kernel_size=self.encoder.patch_embed.proj.kernel_size,
                stride=self.encoder.patch_embed.proj.stride,
                padding=self.encoder.patch_embed.proj.padding,
                bias=False
            )

        # Load pretrained weights
        checkpoint = torch.load(config.MODEL.RESUME, weights_only=False)
        checkpoint = checkpoint['module']
        checkpoint = {k.replace('model.encoder.', ''): v for k, v in checkpoint.items() if k.startswith('model.encoder')}
        self.encoder.load_state_dict(checkpoint, strict=False)

        # Freeze encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Decoder
        self.fusion_conv3 = nn.Conv2d(4096, 1024, 1)
        self.fusion_conv2 = nn.Conv2d(2048, 512, 1)
        self.fusion_conv1 = nn.Conv2d(1024, 256, 1)

        self.upconv4 = nn.Sequential(nn.Conv2d(4096, 2048, 3, padding=1), nn.ReLU(), nn.Conv2d(2048, 1024, 3, padding=1))
        self.upconv3 = nn.Sequential(nn.Conv2d(1024 + 1024, 1024, 3, padding=1), nn.ReLU(), nn.Conv2d(1024, 512, 3, padding=1))
        self.upconv2 = nn.Sequential(nn.Conv2d(512 + 512, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 256, 3, padding=1))
        self.upconv1 = nn.Sequential(nn.Conv2d(256 + 256, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 128, 3, padding=1))
        self.upconv0 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 64, 3, padding=1))

        # Final prediction layer (multi-class)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, mask=None):
        # Multi-scale encoder features
        enc_features = self.encoder.extra_features(x)  # returns [stage1, stage2, stage3, stage4]
        x = enc_features[-1]  # [B, 4096, 4, 4]

        # Upconv4: 4x4 -> 8x8
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.upconv4(x)

        # Upconv3: 8x8 -> 16x16 with fusion
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        enc_feat = F.interpolate(enc_features[2], size=x.shape[2:], mode="bilinear", align_corners=False)
        enc_feat = self.fusion_conv3(enc_feat)
        x = torch.cat([x, enc_feat], dim=1)
        x = self.upconv3(x)

        # Upconv2: 16x16 -> 32x32 with fusion
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        enc_feat = F.interpolate(enc_features[1], size=x.shape[2:], mode="bilinear", align_corners=False)
        enc_feat = self.fusion_conv2(enc_feat)
        x = torch.cat([x, enc_feat], dim=1)
        x = self.upconv2(x)

        # Upconv1: 32x32 -> 64x64 with fusion
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        enc_feat = F.interpolate(enc_features[0], size=x.shape[2:], mode="bilinear", align_corners=False)
        enc_feat = self.fusion_conv1(enc_feat)
        x = torch.cat([x, enc_feat], dim=1)
        x = self.upconv1(x)

        # Final upconv: 64x64 -> 128x128
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.upconv0(x)
        x = self.final(x)  # [B, 9, 128, 128]

        # Resize to target (96, 40)
        x = F.interpolate(x, size=self.final_size, mode="bilinear", align_corners=False)
        return x


class SatVisionUNetLightning(pl.LightningModule):
    def __init__(self, lr=1e-4, dice_weight=0.5, freeze_encoder=True, num_classes=9):
        super().__init__()
        self.save_hyperparameters()
        self.model = SatVisionUNet(out_channels=num_classes, freeze_encoder=freeze_encoder, input_channels=14, final_size=(96, 40))
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, stage):
        chips, masks = batch["chip"], batch["mask"]
        logits = self.forward(chips)  # (B, C, H, W)
        ce = self.ce_loss(logits, masks)
        dice = self.dice_loss(logits, masks)
        loss = self.dice_weight * dice + (1 - self.dice_weight) * ce
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, b, i): return self._common_step(b, 'train')
    def validation_step(self, b, i): return self._common_step(b, 'val')
    def test_step(self, b, i): return self._common_step(b, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == '__main__':
    model = SatVisionUNetLightning(
        lr=1e-4,
        dice_weight=0.5,
        freeze_encoder=True
    )

    train_loader = get_dummy_dataloader()
    val_loader = get_dummy_dataloader()

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1
    )
    trainer.fit(model, train_loader, val_loader)
