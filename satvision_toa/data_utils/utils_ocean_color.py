import os
import time
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
# from torchmetrics import structural_similarity_index_measure as ssim
# from torchmetrics import peak_signal_noise_ratio as psnr

from tqdm import tqdm
from datetime import datetime

from huggingface_hub import hf_hub_download
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split

from satvision_toa.configs.config import _C, _update_config_from_file
from satvision_toa.datasets.ocean_color_dataset import OceanColorDataset
from satvision_toa.losses.spectral_spatial_loss import SpectralSpatialLoss
from satvision_toa.transforms.ocean_color import PBMinMaxNorm, RandomFlipChoice
from satvision_toa.data_utils.ocean_color_metrics_logger import MetricsLogger


def load_config():
    """
    Loads the mim-model config for SatVision from HF.

    Returns:
        config: config file that can be used to load the model
    """

    # directories/URLs
    model_repo_id = (
        "nasa-cisto-data-science-group/"
        "downstream-satvision-toa-3dclouds"
    )
    config_filename = (
        "mim_pretrain_swinv2_satvision_giant"
        "_128_window08_50ep.yaml"
    )
    model_filename = "mp_rank_00_model_states.pt"

    # Extract filenames from HF to be used later
    config_filename = hf_hub_download(
        repo_id=model_repo_id, filename=config_filename)
    ckpt_model_filename = hf_hub_download(  # CHANGE
        repo_id=model_repo_id, filename=model_filename
    )

    # edit config so we can load mim model from it
    config = _C.clone()
    _update_config_from_file(config, config_filename)
    config.defrost()
    config.MODEL.RESUME = ckpt_model_filename
    config.freeze()

    return config


def gather_datasets(
            train_data_path,
            test_data_path,
            transform,
            augment=True,
            num_inputs=14
        ):
    """Split into train, val sets."""

    full_dataset = OceanColorDataset(
        data_path=train_data_path,
        transform=transform,
        num_inputs=num_inputs,
    )

    # Find indices to split the dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    print(f"train & val size: {train_size, val_size}")
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size]
    )

    # split into train and validset
    # trainset = Subset(full_dataset, train_indices.indices)
    trainset_plain = Subset(full_dataset, train_indices.indices)
    if (augment):
        augment_transform = transforms.Compose([
            PBMinMaxNorm(),
            RandomFlipChoice(p=1.0)
        ])
        augmented = OceanColorDataset(
            data_path=train_data_path,
            transform=augment_transform,
            num_inputs=num_inputs,
        )
        trainset = ConcatDataset([trainset_plain, augmented])
    else:
        trainset = trainset_plain

    validset = Subset(full_dataset, val_indices.indices)

    testset = OceanColorDataset(
        data_path=test_data_path,
        transform=transform,
        num_inputs=num_inputs,
    )

    return trainset, validset, testset


def get_dataloaders(
            train_data_path,
            test_data_path,
            augment=True,
            num_inputs=14,
            batch_size=32
        ):
    transform = transforms.Compose([PBMinMaxNorm()])

    trainset, validset, testset = gather_datasets(
        train_data_path, test_data_path, transform,
        augment, num_inputs=num_inputs
    )

    train_dataloader = DataLoader(
            trainset,
            batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=10,
    )
    val_dataloader = DataLoader(
            validset,
            batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=10,
    )
    test_dataloader = DataLoader(
            testset,
            batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=10,
    )
    return train_dataloader, val_dataloader, test_dataloader


def train_model(
            model,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            num_epochs=50,
            learning_rate=1e-4,
            weight_decay=1e-4,
            device='cuda',
            save_path='best_model.pth',
            log_interval=100,
            save_every=50,
            test_every=10,
            pdf_path="pred_pdfs",
            metrics_filename="metrics",
        ):
    """
    Training loop for OceanColorFCNV2 model - Jupyter notebook optimized
    """

    # Move model to device
    model = model.to(device)

    # Initialize weights (if not already done)
    if (save_path == "sv_unet"):
        initialize_decoder_weights(model)

    # Define loss function and optimizer
    criterion = SpectralSpatialLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6)

    # Training tracking variables
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # all_test_results = []
    logger = MetricsLogger(metrics_filename)

    print(f"Starting training for {num_epochs} epochs on {device}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Main epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in epoch_pbar:
        epoch_start_time = time.time()

        # new_edge_weight = criterion.get_edge_weight(epoch)
        # criterion.set_edge_weight(new_edge_weight)

        # Training phase
        model.train()
        train_loss_accum = 0.0
        train_batches = 0

        # Training progress bar with leave=False to not clutter output
        train_pbar = tqdm(
            train_dataloader,
            desc=f'Epoch {epoch+1}/{num_epochs} [Train]',
            leave=False  # This prevents the bar from staying after completion
        )

        for batch_idx, (data, target) in enumerate(train_pbar):
            # Move data to device
            data, target = data.to(device), target.to(device)
            # print(torch.cuda.memory_summary())

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            # Accumulate loss
            train_loss_accum += loss.item()
            train_batches += 1

            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg': f'{train_loss_accum/train_batches:.6f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        # Calculate average training loss
        avg_train_loss = train_loss_accum / train_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        val_mae_accum = 0.0

        # Validation progress bar
        val_pbar = tqdm(
            val_dataloader,
            desc=f'Epoch {epoch+1}/{num_epochs} [Val]',
            leave=False  # This prevents the bar from staying after completion
        )

        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)

                # Forward pass
                output = model(data)
                loss = criterion(output, target)
                mae = nn.functional.l1_loss(output, target)

                # Accumulate losses
                val_loss_accum += loss.item()
                val_mae_accum += mae.item()
                val_batches += 1

                # Update progress bar
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'MAE': f'{mae.item():.6f}'
                })

        # Calculate average validation loss
        avg_val_loss = val_loss_accum / val_batches
        avg_val_mae = val_mae_accum / val_batches
        val_losses.append(avg_val_loss)

        # Learning rate step
        scheduler.step()

        # TESTING
        test_results = test_model_comprehensive(
            model, test_dataloader, epoch+1, pdf_path)
        epoch_metrics = test_results['epoch_metrics']
        individual_metrics = test_results['individual_metrics']
        logger.log_epoch_metrics(epoch_metrics, individual_metrics, epoch)
        # all_test_results.append(test_results)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Update main epoch progress bar
        epoch_pbar.set_postfix({
            'Train_Loss': f'{avg_train_loss:.4f}',
            'Val_Loss': f'{avg_val_loss:.4f}',
            'Val_MAE': f'{avg_val_mae:.4f}',
            'Time': f'{epoch_time:.1f}s'
        })

        # Save best model
        if (epoch + 1) % save_every == 0:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_mae': avg_val_mae,
            }, save_path)
            # Only print when we save a new best model
            tqdm.write(
                'New best model saved at epoch '
                f'{epoch+1}! Val Loss: {avg_val_loss:.6f}')

    print('\nTraining completed!')
    print(f'Best validation loss: {best_val_loss:.6f}')

    # print(f'Saving test results to csv.')
    # _save_metrics_to_csv(all_test_results, metrics_filename)
    logger.close()

    return train_losses, val_losses


def initialize_decoder_weights(model):
    """Initialize decoder weights with Kaiming initialization"""
    decoder_modules = [
        model.fusion_conv3, model.fusion_conv2, model.fusion_conv1,
        model.upconv4, model.upconv3, model.upconv2,
        model.upconv1, model.upconv0
    ]

    for module in decoder_modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # Special initialization for final layer
    for m in model.final.modules():
        if isinstance(m, nn.Conv2d):
            if m.out_channels == 1:  # Final prediction layer
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out',
                    nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    print("Decoder weights initialized with Kaiming/Xavier initialization")


def log_gradient_norms(model, epoch, batch_idx):
    """Log gradient norms for monitoring"""
    total_norm = 0
    max_norm = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            max_norm = max(max_norm, param_norm)

    total_norm = total_norm ** 0.5

    print(
        f'Epoch {epoch+1}, Batch {batch_idx}: '
        f'Total grad norm: {total_norm:.6f}, '
        f'Max grad norm: {max_norm:.6f}')
    return {"total": total_norm, "max": max_norm}


def test_model_comprehensive(
            model,
            test_dataloader,
            epoch,
            pdf_path,
            device='cuda'
        ):
    """
    Pure PyTorch testing function that replicates your
    Lightning testing functionality
    """
    model.eval()
    model.to(device)

    # Storage for predictions and targets (like your Lightning attributes)
    test_predictions = []
    test_targets = []
    all_individual_metrics = []

    # Metrics storage for epoch-level aggregation
    # epoch_metrics = {'r2': [], 'rmse': [], 'ssim': [], 'psnr': []}
    epoch_metrics = {'r2': [], 'rmse': []}

    print("Starting testing...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(
                tqdm(test_dataloader, desc="Testing")):
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Forward pass
            y_hat = model(x)

            # Store predictions and targets
            test_predictions.append(y_hat.cpu())
            test_targets.append(y.cpu())

            batch_size = y.shape[0]

            # Process all samples at once (same as your original logic)
            sample_data = [
                (
                    _calculate_sample_metrics(y[i:i+1], y_hat[i:i+1]),
                    i
                )
                for i in range(batch_size)
            ]

            # Extract metrics and create results
            batch_metrics = {
                metric: [data[metric] for data, _ in sample_data]
                for metric in epoch_metrics.keys()
            }

            individual_metrics = [
                _create_individual_result(metrics, batch_idx, i, batch_size)
                for metrics, i in sample_data
            ]

            # Store individual results
            all_individual_metrics.extend(individual_metrics)

            # Accumulate metrics for epoch-level calculation
            for metric_name, values in batch_metrics.items():
                epoch_metrics[metric_name].extend(
                    [
                        v.item()
                        if torch.is_tensor(v)
                        else v for v in values
                    ])

    # Save individual results to CSV
    # _save_individual_metrics_to_csv(all_individual_metrics)

    # Calculate and print epoch-level metrics
    # _print_epoch_metrics(epoch_metrics)

    # Concatenate all predictions and targets
    all_predictions = torch.cat(test_predictions, dim=0)
    all_targets = torch.cat(test_targets, dim=0)

    # Create plots
    _plot_and_save_predictions(all_predictions, all_targets, epoch, pdf_path)

    return {
        'epoch': epoch,
        'predictions': all_predictions,
        'targets': all_targets,
        'epoch_metrics': epoch_metrics,
        'individual_metrics': all_individual_metrics
    }


def _calculate_sample_metrics(y_true, y_pred):
    """Calculate all metrics for a single sample."""
    # Flatten for R² and RMSE
    y_flat = y_true.flatten()
    y_hat_flat = y_pred.flatten()

    # Calculate R² manually (since torchmetrics
    # R2Score might not be directly available)
    ss_res = torch.sum((y_flat - y_hat_flat) ** 2)
    ss_tot = torch.sum((y_flat - torch.mean(y_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        'r2': r2,
        'rmse': torch.sqrt(F.mse_loss(y_hat_flat, y_flat)),
        # 'ssim': ssim(y_pred, y_true, data_range=1.0),
        # 'psnr': psnr(y_pred, y_true)
    }


def _create_individual_result(metrics, batch_idx, sample_idx, batch_size):
    """Create individual result dictionary."""
    return {
        "batch_idx": batch_idx,
        "sample_idx": sample_idx,
        "global_idx": batch_idx * batch_size + sample_idx,
        **{
            k: v.item()
            if torch.is_tensor(v)
            else v for k, v in metrics.items()
        }
    }


def _save_metrics_to_csv(metrics_list, base_filename):
    # Extract epoch metrics from all dictionaries
    epoch_metrics_list = []
    individual_metrics_list = []

    for metrics in metrics_list:
        # Add epoch number to the metrics for reference
        epoch_num = metrics.get('epoch', None)

        # Extract and flatten epoch_metrics
        if 'epoch_metrics' in metrics:
            epoch_row = {'epoch': epoch_num}
            epoch_row.update(metrics['epoch_metrics'])
            epoch_metrics_list.append(epoch_row)

        # Extract and flatten individual_metrics
        if 'individual_metrics' in metrics:
            individual_row = {'epoch': epoch_num}
            individual_row.update(metrics['individual_metrics'])
            individual_metrics_list.append(individual_row)

    # Convert to DataFrames and save to CSV
    if epoch_metrics_list:
        epoch_df = pd.DataFrame(epoch_metrics_list)
        epoch_filename = f"{base_filename}_epoch_metrics.csv"
        epoch_df.to_csv(epoch_filename, index=False)
        print(f"Epoch metrics saved to: {epoch_filename}")

    if individual_metrics_list:
        individual_df = pd.DataFrame(individual_metrics_list)
        individual_filename = f"{base_filename}_individual_metrics.csv"
        individual_df.to_csv(individual_filename, index=False)
        print(f"Individual metrics saved to: {individual_filename}")


def _print_epoch_metrics(epoch_metrics):
    """Print epoch-level metrics"""
    print("\nTest Results (Epoch-level averages):")
    print("-" * 40)
    for metric_name, values in epoch_metrics.items():
        avg_value = sum(values) / len(values)
        print(f"test_{metric_name}: {avg_value:.4f}")


def _plot_and_save_predictions(preds, targets, epoch, pdf_path):
    """Create and save prediction plots"""
    fig_1, axes_1 = plt.subplots(nrows=1, ncols=5, figsize=(40, 40))
    fig_2, axes_2 = plt.subplots(nrows=1, ncols=5, figsize=(40, 40))

    preds = torch.squeeze(preds, dim=1)
    targets = torch.squeeze(targets, dim=1)

    # Only plot first 5 samples (as in your original code)
    num_plots = min(5, len(preds))

    for idx in range(num_plots):
        # Plot truth chip
        im_1 = axes_1[idx].imshow(targets[idx], cmap="viridis", vmin=0, vmax=1)
        axes_1[idx].set_title(f"Epoch {epoch+1} Chlor A")
        fig_1.colorbar(im_1, ax=axes_1[idx], shrink=0.14)

        # Plot prediction of chip
        im_2 = axes_2[idx].imshow(preds[idx], cmap="viridis", vmin=0, vmax=1)
        axes_2[idx].set_title(f"Epoch {epoch+1} Pred")
        fig_2.colorbar(im_2, ax=axes_2[idx], shrink=0.15)

    # Tight layout
    fig_1.tight_layout()
    fig_2.tight_layout()

    # Save plots
    _save_plots(fig_1, fig_2, epoch, pdf_path)

    # Show figures
    plt.close(fig_1)
    plt.close(fig_2)


def _save_plots(fig_1, fig_2, epoch, pdf_path):
    """Save plots to PDF"""
    now = datetime.now()
    datetime_str = now.strftime("day_%Y_%m_%d_time_%H_%M")
    pdf_name = f"{pdf_path}/preds_{datetime_str}_{epoch}ep.pdf"

    # Create directory if it doesn't exist
    os.makedirs(pdf_path, exist_ok=True)

    with PdfPages(pdf_name) as pdf:
        pdf.savefig(fig_1, bbox_inches='tight')
        pdf.savefig(fig_2, bbox_inches='tight')

    print(f"Plots saved to {pdf_name}")
