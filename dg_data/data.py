import torch
import gpytorch
import RatInteractionDataLoader as ridl
from torch.utils.data import Dataset


class GaussianData(Dataset):

    def __init__(self, data, gaussian_size=120):
        """
    initialize Dataset for Gaussian Process
    :param data:
    """
        super(GaussianData, self).__init__()
        self.raw_data = data
        self.size = gaussian_size

    def __getitem__(self, item):
        return self.raw_data[item:item + self.size, :]

    def __len__(self):
        return self.raw_data.shape[0] - self.size

    def __setitem__(self, key, value):
        # if isinstance(value, type(self.raw_data[key])):
        #   self.raw_data[key] = value
        # else:
        #   self.raw_data[key] = type(self.raw_data[key])(value)
        self.raw_data[key] = value
