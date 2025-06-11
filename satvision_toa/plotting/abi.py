import numpy as np
from satvision_toa.transforms.abi_toa_scale \
    import MinMaxEmissiveScaleReflectance


def pb_minmax_norm(img):
    """Normalize an image using per-band min/max."""
    normalized = np.zeros_like(img, dtype=float)

    for i in range(3):
        band = img[:, :, i]
        min_val = band.min()
        max_val = band.max()
        normalized[:, :, i] = (band - min_val) / (max_val - min_val)

    return normalized


def reverse_transform(image):
    minMaxTransform = MinMaxEmissiveScaleReflectance()
    image = image.transpose((1, 2, 0))

    image[:, :, minMaxTransform.reflectance_indices] = image[
        :, :, minMaxTransform.reflectance_indices] * 100
    image[:, :, minMaxTransform.emissive_indices] = (
        image[:, :, minMaxTransform.emissive_indices] *
        (minMaxTransform.emissive_maxs - minMaxTransform.emissive_mins))
    + minMaxTransform.emissive_mins

    image = image.transpose((2, 0, 1))
    return image
