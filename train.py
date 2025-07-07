from data_processors.GtzanDataset import GtzanDataset
from models.MusicRecNet import MusicRecNet
from torch.utils.data import random_split


import os
import yaml
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

def train(model:nn.Module, dataloader:DataLoader, device:torch.device,
          optimizer:optim, loss:torch.nn.modules.loss, num_epochs:int, 
          trained_weight_path:str):
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(batch_images)
            loss_val = loss(outputs, batch_labels)
            loss_val.backward()
            optimizer.step()

            running_loss += loss_val.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), trained_weight_path)
    print("Training complete. Model saved to:", trained_weight_path)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.2%}")
    return accuracy


def main():

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    path = config["data"]["directory"]
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    learning_rate = config["training"]["learning_rate"]
    train_test_ratio = config["training"]["train_test_ratio"]
    trained_weight_dir = config["training"]["weight_directory"]
    trained_weight_name = config["training"]["weight_file_name"]
    trained_weight_path = trained_weight_dir + '/' + trained_weight_name

    os.makedirs(trained_weight_dir, exist_ok=True)


    ############################### Device ###############################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    ############################### Dataset ###############################
    dataset = GtzanDataset(path)
    
    total_size = len(dataset)
    train_size = int(train_test_ratio * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)


    ############################### Model ###############################
    model = MusicRecNet().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ######################### Train and Eval #############################
    train(model, train_loader, device, optimizer, loss, num_epochs, 
          trained_weight_path)
    
    evaluate(model, val_loader, device)

if __name__ == "__main__":
    main()
