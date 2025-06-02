#!/usr/bin/env python
# coding: utf-8
import os
import argparse

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from torch.utils.data import DataLoader
import torchmetrics

import torch.nn.functional as F
from sklearn.metrics import jaccard_score
from sklearn.metrics import roc_curve, auc
import joblib

import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pytorch_caney.data.datasets.abi_3dcloud_dataset import ABI3DCloudsDataset
from pytorch_caney.models.build import build_model
from pytorch_caney.config import _C, _update_config_from_file


def _vis_calibrate(data):
    """Calibrate visible channels to reflectance."""
    solar_irradiance = np.array(2017)
    esd = np.array(0.99)
    factor = np.pi * esd * esd / solar_irradiance

    return data * np.float32(factor) * 100
 

def _ir_calibrate(data):
    """Calibrate IR channels to BT."""
    fk1 = np.array(13432.1),
    fk2 = np.array(1497.61),
    bc1 = np.array(0.09102),
    bc2 = np.array(0.99971),

    # if self.clip_negative_radiances:
    #     min_rad = self._get_minimum_radiance(data)
    #     data = data.clip(min=data.dtype.type(min_rad))

    res = (fk2 / np.log(fk1 / data + 1) - bc1) / bc2
    return res


class ConvertABIToReflectanceBT(object):
    """
    Performs scaling of MODIS TOA data
    - Scales reflectance percentages to reflectance units (% -> (0,1))
    - Performs per-channel minmax scaling for emissive bands (k -> (0,1))
    """

    def __init__(self):
        
        self.reflectance_indices = [0, 1, 2, 3, 4, 6]
        self.emissive_indices = [5, 7, 8, 9, 10, 11, 12, 13]

    def __call__(self, img):
        
        # Reflectance % to reflectance units
        img[:, :, self.reflectance_indices] = \
            _vis_calibrate(img[:, :, self.reflectance_indices])
        
        # Brightness temp scaled to (0,1) range
        img[:, :, self.emissive_indices] = _ir_calibrate(img[:, :, self.emissive_indices])
        
        return img
    

class MinMaxEmissiveScaleReflectance(object):
    """
    Performs scaling of MODIS TOA data
    - Scales reflectance percentages to reflectance units (% -> (0,1))
    - Performs per-channel minmax scaling for emissive bands (k -> (0,1))
    """

    def __init__(self):
        
        self.reflectance_indices = [0, 1, 2, 3, 4, 6]
        self.emissive_indices = [5, 7, 8, 9, 10, 11, 12, 13]

        self.emissive_mins = np.array(
            [117.04327, 152.00592, 157.96591, 176.15349,
             210.60493, 210.52264, 218.10147, 225.9894],
            dtype=np.float32)

        self.emissive_maxs = np.array(
            [221.07022, 224.44113, 242.3326, 307.42004,
             290.8879, 343.72617, 345.72894, 323.5239],
            dtype=np.float32)

    def __call__(self, img):
        
        # Reflectance % to reflectance units
        img[:, :, self.reflectance_indices] = \
            img[:, :, self.reflectance_indices] * 0.01
        
        # Brightness temp scaled to (0,1) range
        img[:, :, self.emissive_indices] = \
            (img[:, :, self.emissive_indices] - self.emissive_mins) / \
                (self.emissive_maxs - self.emissive_mins)
        
        return img

    
class ABITransform:
    """
    torchvision transform which transforms the input imagery into
    addition to generating a MiM mask
    """

    def __init__(self, img_size):

        self.transform_img = \
            T.Compose([
                ConvertABIToReflectanceBT(), # New transform for MinMax
                MinMaxEmissiveScaleReflectance(),
                T.ToTensor(),
                T.Resize((img_size, img_size), antialias=True),
            ])

    def __call__(self, img):

        img = self.transform_img(img)

        return img


class FCN(nn.Module):
    def __init__(self, swin_encoder, num_output_channels=1, freeze_encoder=False, dropout_rate=0.2):
        super(FCN, self).__init__()

        # Define the encoder part (down-sampling)
        self.encoder = swin_encoder
        if freeze_encoder:
            print('Freezing encoder')
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Decoder (up-sampling) with dropout layers added after ReLU activations
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4096, 2048, kernel_size=3, stride=2, padding=1, output_padding=1), # 8x8x2048
            nn.ReLU(),
            # nn.Dropout(p=dropout_rate),
            nn.ConvTranspose2d(2048, 512, kernel_size=3, stride=2, padding=1, output_padding=1), # 16x16x512
            nn.ReLU(),
            # nn.Dropout(p=dropout_rate),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # 32x32x256
            nn.ReLU(),
            # nn.Dropout(p=dropout_rate),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 64x64x128
            nn.ReLU(),
            # nn.Dropout(p=dropout_rate),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x128x64
            nn.ReLU(),
            # nn.Dropout(p=dropout_rate),
        )

        self.final_layer = nn.Conv2d(64, num_output_channels, kernel_size=3, stride=1, padding=1)  # 128x128x1
        self.resize = nn.Upsample(size=(91, 40), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.encoder.extra_features(x)[-1]
        x = self.decoder(x)
        x = self.final_layer(x)
        x = self.resize(x)

        return x

class ThreeDCloudPTL(pl.LightningModule):
    def __init__(self, model, data_path, label_key, lr=1e-3, batch_size=64):
        super(ThreeDCloudPTL, self).__init__()
        
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = lr
        self.transform = ABITransform(img_size=128)
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_iou = torchmetrics.JaccardIndex(num_classes=2, task="binary")
        self.val_iou = torchmetrics.JaccardIndex(num_classes=2, task="binary")
        self.save_hyperparameters(ignore=['model'])
        
        self.trainset = ABI3DCloudsDataset(data_paths=data_path, label_key=label_key, split='train', transform=self.transform)
        self.validset = ABI3DCloudsDataset(data_paths=data_path, label_key=label_key, split='valid', transform=self.transform)

        self.train_loss_avg = torchmetrics.MeanMetric()
        self.val_loss_avg = torchmetrics.MeanMetric()

        self.train_iou_avg = torchmetrics.MeanMetric()
        self.val_iou_avg = torchmetrics.MeanMetric()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.unsqueeze(1)
        logits = self.forward(inputs)# .unsqueeze(1)
        loss = self.criterion(logits, targets.float())
        preds = torch.sigmoid(logits)
        iou = self.train_iou(preds, targets.int())

        self.train_loss_avg.update(loss)
        self.train_iou_avg.update(iou)
        self.log('train_loss', self.train_loss_avg, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', self.train_iou_avg, on_step=True, on_epoch=True, prog_bar=True)        
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.unsqueeze(1)
        logits = self.forward(inputs)# .unsqueeze(1)
        val_loss = self.criterion(logits, targets.float())
        preds = torch.sigmoid(logits)
        val_iou = self.val_iou(preds, targets.int())
        self.val_loss_avg.update(val_loss)
        self.val_iou_avg.update(val_iou)
        self.log('val_loss', self.val_loss_avg, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_iou', self.val_iou_avg, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        return optimizer

    def on_train_epoch_start(self):
        self.train_loss_avg.reset()
        self.train_iou_avg.reset()

    def on_validation_epoch_start(self):
        self.val_loss_avg.reset()
        self.val_iou_avg.reset()

    def train_dataloader(self):
        
        return DataLoader(self.trainset,
                          self.batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=10)
    
    def val_dataloader(self):
        
        return DataLoader(self.validset,
                          self.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=10)


def calculate_iou(pred, target):
    """
    Calculate Intersection over Union (IoU) between predicted and target masks.
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    if pred_flat.sum() == 0 and target_flat.sum() == 0:
        return None  # Mark for exclusion

    return jaccard_score(target_flat, pred_flat, average='binary')


def run_evaluation_and_boxplot(model, dataloader, tag, device='cuda', checkpoint_name='abi_model'):
    """
    Run prediction on the entire validation set, calculate mIOU and BCE loss, 
    and generate a box plot for both metrics.
    """
    model.eval()
    model.to(device)

    iou_scores = []
    bce_losses = []
    binaryAccuracyMetric = torchmetrics.classification.BinaryAccuracy(threshold=0.5).to(device)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = targets.unsqueeze(1)

            # Get model predictions
            logits = model(inputs)
            preds = torch.sigmoid(logits)

            binaryAccuracyMetric.update(preds, targets)

            # Binary predictions (threshold at 0.5)
            preds_binary = (preds > 0.5).float()

            # Compute BCE loss
            bce_loss = F.binary_cross_entropy(preds, targets.float())
            bce_losses.append(bce_loss.item())

            # Compute IoU for each prediction in the batch
            for i in range(preds_binary.shape[0]):
                iou = calculate_iou(preds_binary[i].cpu().numpy(), targets[i].cpu().numpy())
                # Only add the IoU score if it's not None (i.e., valid)
                if iou is not None:
                    iou_scores.append(iou)

    binaryAccuracy = binaryAccuracyMetric.compute()
    print(f'Accuracy: {binaryAccuracy}')
    mean_iou = np.mean(iou_scores)
    mean_bce_loss = np.mean(bce_losses)

    # Generate box plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Box plot for mIOU
    ax[0].boxplot(iou_scores)
    ax[0].set_title('mIOU for Validation Set')
    ax[0].set_ylabel('mIOU')
    ax[0].set_ylim(-0.1, 1.1)
    ax[0].text(0.2, 0.9, f'mIoU: {mean_iou:.4f}\nacc: {binaryAccuracy:.4f}', horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes, fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    # Box plot for BCE loss
    ax[1].boxplot(bce_losses)
    ax[1].set_title('BCE Loss for Validation Set')
    ax[1].set_ylabel('BCE Loss')
    ax[1].set_ylim(0, 0.5)
    ax[1].text(0.2, 0.9, f'BCE loss: {mean_bce_loss:.4f}',
               horizontalalignment='center', verticalalignment='center',
               transform=ax[1].transAxes, fontsize=12,
               bbox=dict(facecolor='white',
                         edgecolor='black',
                         boxstyle='round,pad=0.5')
                )

    plt.suptitle(f'{tag} Validation', fontsize=16)

    # Adjust layout and save the plot
    plt.tight_layout()
    savePath = f'{checkpoint_name}_boxplot.png'
    print(f'Saving box plot to {savePath}')
    plt.savefig(savePath)
    plt.show()


def save_roc_curve(model, dataloader, device='cuda', checkpoint_name='abi_model'):
    """
    Run prediction on the entire validation set and generate an ROC curve.
    """
    model.eval()
    model.to(device)

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get model predictions (before applying threshold, i.e., raw probabilities)
            logits = model(inputs)
            preds = torch.sigmoid(logits)  # Sigmoid for probability output

            # Move predictions and targets to CPU for further processing
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches into single arrays
    all_preds = np.concatenate(all_preds).ravel()  # Flatten to 1D array
    all_targets = np.concatenate(all_targets).ravel()  # Flatten to 1D array
    save_dict = {'preds': all_preds, 'targs': all_targets}
    joblib.dump(save_dict, 'svtoa_preds_targs.sav')


    # Compute ROC curve and ROC area (AUC)
    fpr, tpr, thresholds = roc_curve(all_targets, all_preds)
    roc_auc = auc(fpr, tpr)

    save_dict = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'roc_auc': roc_auc}
    joblib.dump(save_dict, f'{checkpoint_name}_auc_dict.sav')


def reverse_transform(image):
    minMaxTransform = MinMaxEmissiveScaleReflectance()
    image = image.transpose((1,2,0))
    
    image[:, :, minMaxTransform.reflectance_indices] = image[:, :, minMaxTransform.reflectance_indices] * 100
    image[:, :, minMaxTransform.emissive_indices] = (
        image[:, :, minMaxTransform.emissive_indices] * \
        (minMaxTransform.emissive_maxs - minMaxTransform.emissive_mins)) + minMaxTransform.emissive_mins

    image = image.transpose((2,0,1))
    return image


def minmax_norm(img_arr):
    arr_min = img_arr.min()
    arr_max = img_arr.max()
    img_arr_scaled = (img_arr - arr_min) / (arr_max - arr_min)
    img_arr_scaled = img_arr_scaled * 255
    img_arr_scaled = img_arr_scaled.astype(np.uint8)
    return img_arr_scaled


def run_prediction_and_plot(model, dataloader, tag, device='cuda', num_predictions=5, checkpoint_name='abi_model'):
    """
    Run prediction on a single batch from the dataloader and plot the input image and predicted output.
    """
    model.eval()
    model.to(device)

    inputs, targets = next(iter(dataloader))
    inputs = inputs.to(device)

    with torch.no_grad():
        logits = model(inputs)
        preds = torch.sigmoid(logits)

    preds_binary = (preds > 0.5).float()
    # Detach tensors and move them to the CPU for plotting
    inputs = inputs.cpu().numpy()
    preds_binary = preds_binary.cpu().numpy()
    targets = targets.cpu().numpy()


    preds_binary = np.rot90(preds_binary, k=1, axes=(2, 3)) # preds_binary.transpose(0, 1, 3, 2)  # Switch dimensions 2 and 3 (91x40 to 40x91)
    #np.save("test.npy", preds_binary)
    targets = np.rot90(targets, k=1, axes=(1, 2))  # Switch dimensions 1 and 2 (91x40 to 40x91)
    diffs = preds_binary[:, 0, :, :] - targets 

    num_predictions = min(num_predictions, inputs.shape[0])

    fig, axes = plt.subplots(num_predictions, 4, figsize=(20, 6 * num_predictions), dpi=300)

    save_dict = {
        'images': [],
        'preds': [],
        'truths': [],
        'diffs': [],
    }

    for i in range(num_predictions):

        image = reverse_transform(inputs[i])
        red_coi = 0.9 
        green_coi = 0.45
        blue_coi = 0.65
        rgb_index = [1, 2, 0]
        print(f'Red: {(image[rgb_index[0]]*red_coi).mean()}')
        print(f'Green: {(image[rgb_index[1]]*green_coi).mean()}')
        print(f'Blue: {(image[rgb_index[2]]*blue_coi).mean()}')
        rgb_image = np.stack((image[rgb_index[0], :, :]*red_coi,
                            image[rgb_index[1], :, :]*green_coi,
                            image[rgb_index[2], :, :]*blue_coi),
                            axis=-1)
        rgb_image = minmax_norm(rgb_image*1.1)
        save_dict['images'].append(rgb_image)
        axes[i, 0].imshow(rgb_image)  #  inputs[i, 0, :, :], cmap='magma')  # Assuming the first channel is relevant for visualization
        # axes[i, 0].plot([128/2+9,128/2-9],[0,128-1],'r-')
        # Line coordinates
        x_values = np.linspace(128/2+9, 128/2-9, 100)  # 100 points along the x-axis
        y_values = np.linspace(0, 128-1, 100)          # Corresponding y-values

        # Viridis colormap
        cmap = cm.get_cmap('cool')
        colors = cmap(np.linspace(0, 1, len(x_values)))  # Map positions along the line to the colormap

        # Plot the line with varying colors
        for j in range(len(x_values)-1):
            axes[i, 0].plot([x_values[j], x_values[j+1]], [y_values[j], y_values[j+1]], color=colors[j], lw=2)

        axes[i, 0].set_title(f'ABI image chip (channels [1, 2, 3])')
        axes[i, 0].axis('on')

        cmap = cm.get_cmap('cool')
        num_points = targets[i].shape[1]  # Number of points along the x-axis (image width)
        x_values = np.linspace(0, num_points-1, num_points)  # X-coordinates along the bottom
        y_value = -0.5  # Slightly below the ticks

        # Color map for the horizontal line
        colors = cmap(np.linspace(0, 1, num_points))  # Assign colors along the viridis colormap
        # Display the predicted binary mask
        save_dict['preds'].append(preds_binary[i, 0])
        axes[i, 1].matshow(preds_binary[i, 0], cmap='viridis')  # First channel contains the binary mask
        axes[i, 1].set_title(f'{tag} Predicted Mask {i+1}')
        axes[i, 1].axis('on')
        axes[i, 1].invert_yaxis()
        axes[i, 1].set_ylabel('km')
        axes[i, 1].xaxis.set_ticks_position('bottom')
        for j in range(len(x_values)-1):
            axes[i, 1].plot([x_values[j], x_values[j+1]], [y_value, y_value], color=colors[j], lw=7)

        # Display the predicted binary mask
        save_dict['truths'].append(targets[i])
        axes[i, 2].matshow(targets[i], cmap='viridis')  # First channel contains the binary mask
        axes[i, 2].set_title(f'CloudSat/CALIPSO "Truth"')
        axes[i, 2].axis('on')
        axes[i, 2].invert_yaxis()
        axes[i, 2].xaxis.set_ticks_position('bottom')
        # Viridis colormap for the horizontal line


        # Plot the horizontal line with color variation
        for j in range(len(x_values)-1):
            axes[i, 2].plot([x_values[j], x_values[j+1]], [y_value, y_value], color=colors[j], lw=7)


        print(np.unique(diffs[i], return_counts=True))
        save_dict['diffs'].append(diffs[i])
        axes[i, 3].matshow(diffs[i], cmap='bwr', vmin=-1, vmax=1)  # First channel contains the binary mask
        axes[i, 3].set_title(f'Prediction vs Truth Difference')
        axes[i, 3].axis('on')
        axes[i, 3].invert_yaxis()
        axes[i, 3].xaxis.set_ticks_position('bottom')
        for j in range(len(x_values)-1):
            axes[i, 3].plot([x_values[j], x_values[j+1]], [y_value, y_value], color=colors[j], lw=7)
        # legend_elements = [Patch(facecolor='red', edgecolor='r', label='False Positive'),
        #                 Patch(facecolor='blue', edgecolor='b', label='False Negative')]

        # Add the legend to the last plot or below all plots
        # axes[0, 3].legend(handles=legend_elements, loc='upper right')


    # Adjust layout
    plt.tight_layout()
    savePath = f'{checkpoint_name}_prediction_rgb.png'
    saveDictPath = f'{checkpoint_name}_save_dict.sav'
    joblib.dump(save_dict, saveDictPath)
    print(f'Saving save dict to {saveDictPath}')
    print(f'Saving plot to {savePath}')
    plt.savefig(savePath)


def main(args, config):
    DATA_PATH: str = '/explore/nobackup/people/jli30/workspace/cat-dog/chips/abiChipsPick'
    # DATA_PATH: str = '/explore/nobackup/people/jli30/workspace/cat-dog/chips/abiChipsNew'
    # DATA_PATH: str = '/explore/nobackup/projects/ilab/data/satvision-toa/3dcloud.data/abiChipsNew/'
    # DATA_PATH: str = '/explore/nobackup/projects/ilab/data/satvision-toa/3dcloud.data/handpicked/'
    IN_CHANNELS: int = 14
    OUTPUT_CHANNELS: int = 1

    pretrained_model = load_pretrained_model(config)
    model = FCN(swin_encoder=pretrained_model.encoder, num_output_channels=OUTPUT_CHANNELS, freeze_encoder=args.freeze_encoder)

    if args.inference:
        print(f'Performing inference')

        output_path = f'{args.checkpoint.replace(".ckpt", "")}.ptl.ckpt'

        if not os.path.exists(output_path):
            # Use this if performing inference from a PTL checkpoint that has the zero_* associated files
            # from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
            # print(f'Converting sharded checkpoint {args.checkpoint} to single PTL save {output_path}')
            # convert_zero_checkpoint_to_fp32_state_dict(args.checkpoint, output_path)
            # print(f'Saved single checkpoint file to {output_path}')

            # Use this if performing inference from a torch checkpoint w/oput the zero_* associated files
            checkpoint = torch.load(args.checkpoint)
            checkpoint_model = checkpoint['module']
            checkpoint_model = {k.replace('model.', ''): v for k, v in checkpoint_model.items() if k.startswith('model')}
            model.load_state_dict(checkpoint_model, strict=False)
            
        else:
            print(f'Using previously made single checkpoint file {output_path}')

        ptlCheckpoint = ThreeDCloudPTL(model, data_path=[DATA_PATH], label_key='Cloud_mask', lr=args.lr, batch_size=args.batch_size)

        # Loading from basic PTL checkpoint
        # ptlCheckpoint = ThreeDCloudPTL.load_from_checkpoint(output_path,
        #                                                     model=model,
        #                                                     data_path=[DATA_PATH],
        #                                                     label_key='Cloud_mask')

        run_prediction_and_plot(ptlCheckpoint,
                                ptlCheckpoint.train_dataloader(),
                                tag=args.tag,
                                num_predictions=args.samples,
                                checkpoint_name=os.path.basename(args.checkpoint).replace('.ckpt', ''))

        run_evaluation_and_boxplot(ptlCheckpoint,
                                   ptlCheckpoint.val_dataloader(),
                                   tag=args.tag,
                                   checkpoint_name=os.path.basename(args.checkpoint).replace('.ckpt', ''))

        save_roc_curve(ptlCheckpoint,
                       ptlCheckpoint.val_dataloader(),
                       checkpoint_name=os.path.basename(args.checkpoint).replace('.ckpt', ''))



    else:
        print('Training')

        ptlModel = ThreeDCloudPTL(model, data_path=[DATA_PATH], label_key='Cloud_mask', lr=args.lr, batch_size=args.batch_size)

        train_callbacks = [
            ModelCheckpoint(dirpath=args.checkpoint_dir,
                            monitor='val_iou',
                            mode='max',
                            save_top_k=5,
                            filename='{epoch}-{val_iou:.8f}.ckpt'),
            ModelCheckpoint(dirpath=args.checkpoint_dir,
                            mode='max',
                            save_top_k=-1,               # save all checkpoints
                            every_n_epochs=10,            # save a checkpoint every 5 epochs
                            save_on_train_epoch_end=True, # save at the end of each epoch
                            filename='{epoch}-{val_iou:.8f}.ckpt'),
            EarlyStopping("val_iou", patience=args.patience, mode='max'),

        ]

        trainer = Trainer(
            accelerator="gpu",
            devices=-1,
            strategy="deepspeed_stage_2",  # "ddp",
            min_epochs=1,
            max_epochs=args.epochs,
            callbacks=train_callbacks,
            # precision=16,
            # logger=CSVLogger(save_dir="logs/"),
            # precision=16 # makes loss nan, need to fix that
        )

        trainer.fit(model=ptlModel)


def load_pretrained_model(config):
    checkpoint = torch.load(config.MODEL.RESUME)
    pretrained_model = build_model(config, pretrain=True)
    pretrained_model.load_state_dict(checkpoint['module'])
    return pretrained_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--data_path", nargs='+', required=True,
    #     help="path where dataset is stored")
    # parser.add_argument('--ngpus', type=int,
    #                     default=torch.cuda.device_count(),
    #                     help='number of gpus to use')
    parser.add_argument(
        '--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument(
        '--patience', type=int, default=20, help='number of epochs for patience')
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="adam: learning rate")
    parser.add_argument(
        '--log-dir', type=str, default='.', help='Directory to log to'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default='.', help='Directory to save checkpoints to'
    )
    parser.add_argument(
        '--checkpoint', type=str, default='.', help='Path to PTL checkpoint'
    )
    parser.add_argument(
        '--inference', action='store_true'
    )
    parser.add_argument(
        '--samples', type=int, default=5, help='Number of samples to run inference on and plot'
    )
    parser.add_argument(
        '--pretrained-model', type=str, help='Path to pretrained model checkpoint'
    )
    parser.add_argument(
        '--config-path', type=str, help='Patht to pretrained model config'
    )
    parser.add_argument(
        '--freeze-encoder', action='store_true', help='Freeze the parameters of the encoder'
    )
    parser.add_argument(
        '--tag', type=str, default='svtoa-fcn-3dcloud', help='tag to use for plot titles'
    )
    hparams = parser.parse_args()

    config = _C.clone()
    _update_config_from_file(config, hparams.config_path)

    config.defrost()
    config.MODEL.RESUME = hparams.pretrained_model 
    config.OUTPUT = hparams.log_dir 
    config.TAG = hparams.tag 
    config.freeze()

    main(hparams, config)

