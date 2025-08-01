import numpy as np


# def get_augments():
#     return transforms.Compose([
#         PBMinMaxNorm(),
#         RandomFlipChoice(p=1.0),
#         transforms.ToTensor(),
#     ])


class RandomFlipChoice:
    """Custom random flip that works on HWC numpy arrays"""
    def __init__(self, p=1.0):
        self.p = p
    
    def __call__(self, image):
        if np.random.random() < self.p:
            # Randomly choose horizontal or vertical flip
            if np.random.random() < 0.5:
                # Horizontal flip
                return np.fliplr(image).copy()
            else:
                # Vertical flip
                return np.flipud(image).copy()
        return image


class ScaleAndOffset:
    def __init__(self, num_inputs=14, num_targets=1):
        self.num_inputs = num_inputs
        self.num_targets = num_targets
        self.ref_offset = 316.9721985
        self.ref_scales = [
            2.051673619e-05, 1.100883856e-05,
            2.051673619e-05, 1.100883856e-05,
            6.804773875e-06, 5.185710506e-06,
            4.120129688e-06, 2.385006837e-06,
            1.149707714e-06, 2.436357136e-06,
            8.907225606e-07, 2.297678748e-06,
            2.036709247e-06, 2.206565841e-05
        ]

    def __call__(self, image):
        # Expects CHW numpy array, returns scaled CHW numpy array
        assert isinstance(image, np.ndarray), f"Expected numpy array, got {type(image)}"
        assert image.ndim == 3, f"Expected 3D array (HWC), got {image.ndim}D"
        
        scaled = np.zeros_like(image[:self.num_inputs, :, :], dtype=np.float32)

        # scale and offset inputs
        for c in range(self.num_inputs):
            band = image[c, :, :]
            scaled[c, :, :] = (band - self.ref_offset) * self.ref_scales[c] * 100

        # add target(s) back in
        targets = image[c+1:c+1+self.num_targets, :, :]
        scaled = np.concatenate((scaled, targets), axis=2)
        
        return scaled.astype(np.float32)


class GlobalMinMaxNorm:
    def __init__(self):
        self.mins = np.array([
                0.5109237432479858, 3.9489009380340576,
                0.5109237432479858, 3.9489009380340576,
                3.5895371437072754, 9.76937484741211,
                2.8478450775146484, 3.649782657623291,
                0.8498671650886536, 2.169339179992676,
                0.7050984501838684, 1.9017950296401978,
                1.4424031972885132, 10.635708808898926, 0.0
            ], dtype=np.float32)
        self.maxs = np.array([
                10.178409576416016, 35.72371292114258,
                10.178409576416016, 35.72371292114258,
                22.081510543823242, 16.82764434814453,
                13.369832038879395, 7.739353656798799,
                3.730804681777954, 7.9059858322143555,
                2.511127471923828, 5.556712627410889,
                3.455486536026001, 22.03703498840332,
                15.631364822387695
            ], dtype=np.float32)

    def __call__(self, image):
        # Expects CHW numpy array, CHW numpy array
        assert isinstance(image, np.ndarray), f"Expected numpy array, got {type(image)}"
        
        normalized = []
        for c in range(image.shape[0]):
            band_min = self.mins[c]
            band_max = self.maxs[c]

            band = image[c, :, :]
            normalized_band = (band - band_min) / (band_max - band_min)
            normalized.append(normalized_band)

        norm = np.stack(normalized, axis=0)  # Shape: (H, W, C)
        return norm.astype(np.float32)


class PBMinMaxNorm:
    """Per-band minmax norm that maintains HWC format."""
    def __init__(self):
        pass

    def __call__(self, image: np.ndarray):
        """Transforms input image, maintaining numpy array format. 
        Expects CHW format, returns CHW format."""
        assert isinstance(image, np.ndarray), f"Expected numpy array, got {type(image)}"
        
        normalized_channels = []

        for c in range(image.shape[0]):
            channel = image[c, :, :]
            channel_min = np.min(channel)
            channel_max = np.max(channel)

            if channel_max > channel_min:
                norm_channel = (channel - channel_min) / (channel_max - channel_min)
            else:
                norm_channel = channel * 0.0

            normalized_channels.append(norm_channel)

        # Stack channels back together, maintaining CHW format
        norm = np.stack(normalized_channels, axis=0)
        return norm.astype(np.float32)