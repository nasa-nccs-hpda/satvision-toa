import os
import sys
import torch
import subprocess
import torch.nn as nn
import lightning as pl
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset, DataLoader

repo_dir = "satvision-toa"

if not os.path.exists(repo_dir):
    subprocess.run(["git", "clone", "https://github.com/nasa-nccs-hpda/satvision-toa"])
else:
    subprocess.run(["git", "-C", repo_dir, "pull"])

sys.path.append('satvision-toa')

from satvision_toa.utils import load_config
from satvision_toa.models.mim import build_mim_model


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth


    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        probs = F.softmax(logits, dim=1)
       
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
       
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score.mean()


class DummyCloudDataset(Dataset):
    def __init__(self, num_samples=100, in_channels=3, height=96, width=40, num_classes=9):
        self.num_samples = num_samples
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random "ABI-like" chip (float32, normalized 0–1)
        chip = torch.rand(self.in_channels, self.height, self.width, dtype=torch.float32)
        # Random segmentation mask (integer labels between 0 and num_classes-1)
        mask = torch.randint(0, self.num_classes, (self.height, self.width), dtype=torch.long)
        return {"chip": chip, "mask": mask}

def get_dummy_dataloader(batch_size=8, num_workers=0):
    dataset = DummyCloudDataset()
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


class SatVisionUNet(nn.Module):

    def __init__(self, num_classes=9, freeze_encoder=True, output_shape=(96, 40)):
        super().__init__()

        # Load SatVision encoder
        config = load_config()
        backbone = build_mim_model(config)
        self.encoder = backbone.encoder

        # Load pretrained weights
        checkpoint = torch.load(config.MODEL.RESUME, weights_only=False)
        checkpoint = checkpoint['module']
        checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items() if k.startswith('model')}
        self.encoder.load_state_dict(checkpoint, strict=False)

        # Freeze if requested
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Decoder (same as your U-Net upsampling part)
        self.up_trans = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        ])
        self.double_conv_ups = nn.ModuleList([
            self._double_conv(1024, 512),
            self._double_conv(512, 256),
            self._double_conv(256, 128),
            self._double_conv(128, 64),
        ])

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_upsample = nn.Upsample(size=output_shape, mode='bilinear', align_corners=False)

    def _double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        # SatVision encoder forward

        # returns multi-scale features: [stage1, stage2, stage3, stage4]
        features = self.encoder(x)

        # deepest feature
        x = features[-1]

        # U-Net decoder (skip connections with SatVision features)
        skips = features[:-1][::-1]
        for up, conv, skip in zip(self.up_trans, self.double_conv_ups, skips):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = conv(x)

        x = self.final_conv(x)
        x = self.final_upsample(x)
        return x

class SatVisionUNetLightning(pl.LightningModule):
    def __init__(self, num_classes=9, lr=1e-4, dice_weight=0.5, freeze_encoder=True, target_height=96, target_width=40):
        super().__init__()
        self.save_hyperparameters()
        self.model = SatVisionUNet(num_classes=num_classes, freeze_encoder=freeze_encoder, output_shape=(target_height, target_width))
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, stage):
        chips, masks = batch["chip"], batch["mask"]
        logits = self.forward(chips)
        ce = self.ce_loss(logits, masks)
        dice = self.dice_loss(logits, masks.long())
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
        num_classes=9,
        lr=0.00001,
        dice_weight=0.5,
        freeze_encoder=True  # set False if you want to fine-tune SatVision
    )

    # Dummy DataLoader
    train_loader = get_dummy_dataloader()
    val_loader = get_dummy_dataloader()

    # Trainer
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1
    )

    # Train with dummy data
    trainer.fit(model, train_loader, val_loader)
