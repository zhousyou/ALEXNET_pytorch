import torch
from torch.utils.data import Dataset
import os
import cv2

class CustomDataset(Dataset):
    def __init__(self, img_dir, data_txt, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform

        self.img_names = []
        self.labels = []

        with open(data_txt, 'r') as f:
            lines = f.readlines()
        for l in lines:
            img = l.split(';')
            self.img_names.append(img[0])
            self.labels.append(int(img[1]))
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        label = self.labels[index]

        return img, label
        
