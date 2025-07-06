from data_processors.DataProcessor import DataProcessor

import subprocess

class GtzanProcessor(DataProcessor): 
    def __init__(self, data_path: str):
        super().__init__(data_path)  # Call the superclass's __init__ method
        

    def download_data(self):
        if self.is_data_downloaded():
            print("Dataset already downloaded.")
            return
        
        print("Downloading dataset...")
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "andradaolteanu/gtzan-dataset-music-genre-classification",
            "-p", self.data_path,
            "--unzip"
        ], check=True)

        print("Download complete. Saved to:", self.data_path)