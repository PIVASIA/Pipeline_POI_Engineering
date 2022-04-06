import torch
from PIL import Image
from torch.utils.data import Dataset
class Labelizer():
    def __init__(self):
        super().__init__()
        self.labels = {'background': 0, 'bien': 1}
        self.inv_labels = {0: 'background', 1: 'bien'}
    def transform(self, label):
        return self.labels[label]
    
    def inverse_transform(self, ys):
        return self.inv_labels(ys)
    
    def num_classes(self):
        return len(self.labels)

class PoIDataset(Dataset):
    def __init__(self,
                 data_path,
                 transforms=None):
        self.data_path = data_path
        self.transforms = transforms
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        image = Image.open(self.data_path).convert('RGB')
        target = {}
        if self.transforms is not None:
            image = self.transforms(image)
        return image, target