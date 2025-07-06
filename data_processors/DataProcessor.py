import os

class DataProcessor():

    def __init__(self, data_path:str):
        self.data_path = data_path
        self.create_directory()
        

    def create_directory(self):
        os.makedirs(self.data_path, exist_ok=True)
        

    def is_data_downloaded(self):
        return len(os.listdir(self.data_path)) > 0
        

    def download_data():
        pass


    def load_data(self):
        pass
