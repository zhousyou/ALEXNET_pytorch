import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        self.image_path = []
        self.labels = []
        with open(data_root, 'r') as f :
            lines = f.readlines()
        nums = len(lines)
        tra
