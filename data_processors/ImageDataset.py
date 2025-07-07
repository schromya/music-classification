import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):

    def __init__(self, data_path:str, transform:transforms=None):
        self.data_path = data_path
        self.classes = []
        self.img_paths = []
        self.labels = []
        self.transform = transform

        self.create_directory()
        
        
    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image, label


    def create_directory(self):
        os.makedirs(self.data_path, exist_ok=True)
        

    def is_data_downloaded(self):
        return len(os.listdir(self.data_path)) > 0
        

    def download_data():
        pass



