from data_processors.ImageDataset import ImageDataset

import subprocess
import os
from torchvision import transforms


class GtzanDataset(ImageDataset): 
    def __init__(self, data_path: str, transform=None):
        super().__init__(data_path, transform)  
        
        self.img_directory = data_path + '/Data/images_original'
        self.transform = self.transform if self.transform else transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),  # Converts to [0,1], shape: (1, 128, 128)
            transforms.Normalize((0.5,), (0.5,))
        ])
    
        self.download_data()

        
    def download_data(self):
        if not self.is_data_downloaded():    
            print("Downloading dataset...")
            subprocess.run([
                "kaggle", "datasets", "download",
                "-d", "andradaolteanu/gtzan-dataset-music-genre-classification",
                "-p", self.data_path,
                "--unzip"
            ], check=True)

            print("Download complete. Saved to:", self.data_path)

        # Set classes and labels
        self.classes = sorted(os.listdir(self.img_directory))  # genre folders
        for idx, label in enumerate(self.classes):
            class_dir = os.path.join(self.img_directory, label)
            for fname in os.listdir(class_dir):
                if fname.endswith(".png"):
                    self.img_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(idx)
