import os
import torch
import numpy as np
from torch.utils.data import Dataset


class OceanColorDataset(Dataset):
    """
    Dataset of MOD021KM Aqua Data. For now this uses .npy chip files.
    """

    def __init__(
        self,
        data_path,
        split: str = "train",
        val_split: float = 0.2,
        random_split: int = 42,
        config=None,
        transform=None,
        num_inputs: int = 12,
        num_targets: int = 1
    ):
        self.samples = self.gather_files(data_path)
        self.config = config
        self.transform = transform
        self.num_inputs = num_inputs
        self.num_targets = num_targets

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the next item in the dataset. Load sample from file,
        apply transforms to the entire sample, then extract inputs and targets.
        """
        sample = self.samples[index].astype(np.float32)  # NumPy array

        # apply transform
        if self.transform is not None:
            sample = self.transform(sample)

        # extract inputs and target(s)
        sample = torch.from_numpy(sample)
        inputs = sample[:self.num_inputs]
        target = sample[
            self.num_inputs:self.num_inputs + self.num_targets]

        return inputs, target

    def gather_files(self, data_path: str) -> list[str]:
        """
        Finds all filenames in data_path and all its subdirs.
        Loads them into a numpy array of samples.
        Only looks 1 subdirectory deep (e.g. doesn't look recursively
        in directories of directories).

        Args:
            self: self
            data_path: string filepath where data is stored
        Returns:
            numpy.array of loaded samples from data_path and all of its subdirs
        """
        filenames = self.examine_dir(data_path)
        for subdir_name in self.find_subdirs(data_path):
            filenames = filenames + self.examine_dir(subdir_name)

        samples = [np.load(fn) for fn in filenames if fn.endswith('.npy')]
        return np.array(samples)

    def examine_dir(self, path: str) -> list[str]:
        """Finds all filenames in a given path."""
        filenames = []
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                filenames.append(item_path)
        return filenames

    def find_subdirs(self, path: str) -> list[str]:
        """Finds all directories in a given path."""
        return [
            os.path.join(path, item)
            for item in os.listdir(path)
            if os.path.isdir(os.path.join(path, item))
        ]
