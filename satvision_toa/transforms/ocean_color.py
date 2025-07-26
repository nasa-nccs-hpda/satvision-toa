import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_augments():
    return A.Compose(
            [

                PBMinMaxNorm(p=1.0),
                # Geometric transformations - common for remote sensing
                A.OneOf([
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0)
                ], p=1.0),
                ToTensorV2(),
            ]
        )


class ScaleAndOffset(A.ImageOnlyTransform):
    def __init__(
                self,
                always_apply=False,
                p=1.0,
                num_inputs=14,
                num_targets=1
            ):
        super().__init__(always_apply=always_apply, p=p)
        self.num_inputs = num_inputs
        self.num_targets = num_targets
        self.ref_offset = 316.9721985
        # ref_scales correspond to bands 8-17, at indices 0-11 here
        self.ref_scales = [
            2.051673619e-05, 1.100883856e-05,
            2.051673619e-05, 1.100883856e-05,  # duplicated bands 8-9
            6.804773875e-06, 5.185710506e-06,
            4.120129688e-06, 2.385006837e-06,
            1.149707714e-06, 2.436357136e-06,
            8.907225606e-07, 2.297678748e-06,
            2.036709247e-06, 2.206565841e-05
        ]

    def apply(self, image, **params):
        # print(f"before transform, image shape: {image.shape}")
        scaled = np.zeros_like(
            image[:, :, :self.num_inputs], dtype=np.float32
        )

        # scale and offset inputs
        for i in range(self.num_inputs):
            band = image[:, :, i]
            scaled[:, :, i] = \
                (band - self.ref_offset) * self.ref_scales[i] * 100

        # add target(s) back in
        targets = image[:, :, i+1:i+1+self.num_targets]
        scaled = np.concatenate((scaled, targets), axis=2)
        # print(f"after transform, image shape: {image.shape}")
        return scaled.astype(np.float32)


class GlobalMinMaxNorm(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

        self.mins = np.array([
                0.5109237432479858, 3.9489009380340576,
                0.5109237432479858, 3.9489009380340576,
                3.5895371437072754, 9.76937484741211,
                2.8478450775146484, 3.649782657623291,
                0.8498671650886536, 2.169339179992676,
                0.7050984501838684, 1.9017950296401978,
                1.4424031972885132, 10.635708808898926, 0.0
            ],
            dtype=np.float32)
        self.maxs = np.array([
                10.178409576416016, 35.72371292114258,
                10.178409576416016, 35.72371292114258,
                22.081510543823242, 16.82764434814453,
                13.369832038879395, 7.739353656768799,
                3.730804681777954, 7.9059858322143555,
                2.511127471923828, 5.556712627410889,
                3.455486536026001, 22.03703498840332,
                15.631364822387695
            ],
            dtype=np.float32)

    def apply(self, image, **params):
        num_channels = image.shape[2]
        normalized = []
        for band_idx in range(num_channels):
            band_min = self.mins[band_idx]
            band_max = self.maxs[band_idx]

            band = image[:, :, band_idx]
            normalized_band = (band - band_min) / (band_max - band_min)
            normalized.append(normalized_band)

        norm = np.stack(normalized, axis=2)  # Shape: (H, W, C)
        return norm.astype(np.float32)


class PBMinMaxNorm(A.ImageOnlyTransform):
    """Albumentations wrapper class for a per-band minmax norm."""
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply_3d(self, image: np.array, **params):
        """For calling transform on a 3D numpy array (image)."""
        # Assume image is (H, W, C)
        normalized_channels = []

        for c in range(image.shape[2]):
            channel = image[:, :, c]

            # Find min/max for this specific channel
            channel_min = np.min(channel)
            channel_max = np.max(channel)

            # Normalize this channel
            if channel_max > channel_min:
                norm_channel = \
                    (channel - channel_min) / (channel_max - channel_min)
            else:
                norm_channel = channel * 0.0

            normalized_channels.append(norm_channel)

        # Stack channels back together
        norm = np.stack(normalized_channels, axis=2)  # Shape: (H, W, C)
        return norm.astype(np.float32)

    def apply_2d(self, image, **params):
        """For calling transform on a 2D numpy array (image)."""
        # For 2D input, treat as single channel
        channel_min = np.min(image)
        channel_max = np.max(image)

        if channel_max > channel_min:
            norm = (image - channel_min) / (channel_max - channel_min)
        else:
            norm = image * 0.0

        return norm.astype(np.float32)

    def apply(self, image: np.array, **params):
        """Transforms input image based on dimensionality.
        Both are per-channel minmax,
        2d case treats whole image as 1 channel."""
        if (image.ndim == 3):
            return self.apply_3d(image)
        elif (image.ndim == 2):
            return self.apply_2d(image)
        else:
            raise ValueError(
                f"Input image must be 2D or 3D, got {image.ndim}D")

    def get_transform_init_args_names(self):
        return ()
