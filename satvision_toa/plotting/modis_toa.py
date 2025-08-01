import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

from ..transforms.modis_toa_scale import MinMaxEmissiveScaleReflectance


# -----------------------------------------------------------------------------
# MODIS Reconstruction Visualization Pipeline
# -----------------------------------------------------------------------------
# This script processes MODIS TOA images and model reconstructions, generating
# comparison visualizations in a PDF format. It contains several functions that
# interact to prepare, transform, and visualize MODIS image data, applying
# necessary transformations for reflective and emissive band scaling, masking,
# and normalization. The flow is as follows:
#
# 1. `plot_export_pdf`: Main function that generates PDF visualizations.
#    It uses other functions to process and organize data.
# 2. `process_reconstruction_prediction`: Prepares images and masks for
#    visualization, applying transformations and normalization.
# 3. `pb_minmax_norm`: Scales image arrays to [0,1] range for display.
# 4. `process_mask`: Prepares mask images to match the input image dimensions.
# 5. `reverse_transform`: Applies band-specific scaling to MODIS data.
#
# ASCII Diagram:
#
# plot_export_pdf
#      └── process_reconstruction_prediction
#            ├── process_mask
#            └── process_img
#               ├── reverse_transform
#               ├── pb_minmax_norm
#               ├── stack
#
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# plot_export_pdf
# -----------------------------------------------------------------------------
# Generates a multi-page PDF with visualizations of original, reconstructed,
# and masked MODIS images. Uses the `process_reconstruction_prediction` funct
# to prepare data for display and organizes subplots for easy comparison.
# -----------------------------------------------------------------------------
def plot_export_pdf(path, inputs, outputs, masks, rgb_index, save_to_pdf=True):

    if save_to_pdf:
        pdf_plot_obj = PdfPages(path)

    # clone model tensors to prevent mutation
    model_inputs = [elem.detach().clone() for elem in inputs]
    model_outputs = [elem.detach().clone() for elem in outputs]
    model_masks = [elem.detach().clone() for elem in masks]

    # process and plot each image
    for i in range(len(inputs)):
        chip_input = model_inputs[i]
        chip_recon = model_outputs[i]
        chip_mask = model_masks[i]

        input_p, output_p, mask_p = process_recon_pred(
            chip_input, chip_recon, chip_mask, rgb_index)

        # plot (input, output, mask) in a line
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 20))

        axes[0].imshow(input_p)
        input_title = f"MOD021KM v6.1 Bands: {rgb_index}, Image #{i+1}"
        axes[0].set_title(input_title, fontsize=30)
        axes[0].axis('off')

        axes[1].imshow(output_p)
        recon_title = f"Model reconstruction #{i+1}"
        axes[1].set_title(recon_title, fontsize=30)
        axes[1].axis('off')

        # custom colormap for mask
        mask_colors = [[0.2, 0.65, 0.2, 1], [0.5, 0.5, 0.5, 1]]
        mask_cmap = ListedColormap(mask_colors)

        axes[2].matshow(mask_p[:, :, 0], cmap=mask_cmap)
        mask_title = f"Mask #{i+1}"
        axes[2].set_title(mask_title, fontsize=30)
        axes[2].axis('off')

        plt.tight_layout()

        # save this figure to pdf
        if (save_to_pdf):
            pdf_plot_obj.savefig()

    if (save_to_pdf):
        pdf_plot_obj.close()


# -----------------------------------------------------------------------------
# process_reconstruction_prediction
# -----------------------------------------------------------------------------
# Prepares RGB images, reconstructions, and masked versions by extracting and
# normalizing specific bands based on the provided RGB indices. Returns masked
# images and the processed mask for visualization in the PDF.
# -----------------------------------------------------------------------------

def process_recon_pred(image, recon, mask, rgb_index):
    # process mask
    mask_p = process_mask(mask)

    # clip if necessary
    image = image.numpy()
    recon = recon.numpy()

    # stack bands properly, normalize
    image_p = process_img(image, rgb_index)
    recon_processed = process_img(recon, rgb_index)

    # apply image masking to actual, model recon
    recon_masked = np.where(mask_p == 0, image_p, recon_processed)

    return image_p, recon_masked, mask_p


# -----------------------------------------------------------------------------
# pb_minmax_norm
# -----------------------------------------------------------------------------
# Normalizes an image array to a range of [0,1] for consistent display.
# -----------------------------------------------------------------------------

def pb_minmax_norm(img):
    normalized = np.zeros_like(img, dtype=float)

    for i in range(3):
        band = img[:, :, i]
        min_val = band.min()
        max_val = band.max()
        normalized[:, :, i] = (band - min_val) / (max_val - min_val)

    return normalized


# -----------------------------------------------------------------------------
# process_img
# -----------------------------------------------------------------------------
# Call three processing functions to get image scaled correctly and in correct
# shape.
# -----------------------------------------------------------------------------

def process_img(img, rgb_index):
    transformed = reverse_transform(img)
    stacked = stack(transformed, rgb_index)
    return pb_minmax_norm(stacked)


# -----------------------------------------------------------------------------
# stack
# -----------------------------------------------------------------------------
# Orders bands in RGB order.
# -----------------------------------------------------------------------------

def stack(img, rgb_index):
    red_idx = rgb_index[0]
    blue_idx = rgb_index[1]
    green_idx = rgb_index[2]

    return np.stack((img[red_idx, :, :],
                     img[green_idx, :, :],
                     img[blue_idx, :, :]), axis=-1)


# -----------------------------------------------------------------------------
# process_mask
# -----------------------------------------------------------------------------
# Adjusts the dimensions of a binary mask to match the input image shape,
# replicating mask values across the image.
# -----------------------------------------------------------------------------

def process_mask(mask):
    mask_img = mask.unsqueeze(0)
    mask_img = mask_img.repeat_interleave(4, 1).repeat_interleave(4, 2)
    mask_img = mask_img.unsqueeze(1).contiguous()[0, 0]
    return np.stack([mask_img] * 3, axis=-1)


# -----------------------------------------------------------------------------
# reverse_transform
# -----------------------------------------------------------------------------
# Reverses scaling transformations applied to the original MODIS data to
# prepare the image for RGB visualization.
# -----------------------------------------------------------------------------

def reverse_transform(image):
    minMaxTransform = MinMaxEmissiveScaleReflectance()
    image = image.transpose((1, 2, 0))
    image[:, :, minMaxTransform.reflectance_indices] *= 100
    emis_min, emis_max = \
        minMaxTransform.emissive_mins, minMaxTransform.emissive_maxs
    image[:, :, minMaxTransform.emissive_indices] *= (emis_max - emis_min)
    image[:, :, minMaxTransform.emissive_indices] += emis_min
    return image.transpose((2, 0, 1))
