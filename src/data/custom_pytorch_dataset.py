"""A container for the custom pytorch dataset class."""

import numpy as np
from torch.utils.data import Dataset


class PytorchDataset(Dataset):
    """A custom PyTorch dataset class for handling geospatial data."""

    def __init__(self, data: dict[str, np.array]):
        """
        Initialize the dataset with the provided data.

        :param data: The data to be used in creating the dataset.
        """
        self.data = data

        # some safety checks
        dataset_lengths = set()
        for data in self.data.values():
            dataset_lengths.add(len(data))

        assert len(dataset_lengths) == 1, (
            "All data arrays must have the same length. "
            f"Found lengths: {dataset_lengths}"
        )

    def __len__(self):
        """Return the length of the dataset.

        We assume that all data arrays have the same length, as checked in __init__.
        """

        first_key = list(self.data.keys())[0]  # pick an arbitrary data key
        return len(self.data[first_key])

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        :param idx: The index of the item to retrieve.
        :return: The item at the specified index.
        """
        return {feature_type: data[idx] for feature_type, data in self.data.items()}
