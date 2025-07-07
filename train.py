from data_processors.GtzanDataset import GtzanDataset
from models.MusicRecNet import MusicRecNet

import yaml
from torch.utils.data import DataLoader


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    path = config["data"]["path"]
    batch_size = config["training"]["batch_size"]

    dataset = GtzanDataset(path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MusicRecNet()

    for batch_images, batch_labels in dataloader:
        outputs, features = model(batch_images)
        print("Outputs:", outputs.shape)     # (batch_size, 10)
        print("Features:", features.shape)   # (batch_size, 128)
        break


if __name__ == "__main__":
    main()


    