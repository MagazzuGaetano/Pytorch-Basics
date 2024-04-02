from pathlib import Path
import torch
import torch.nn as nn

from src.models.mlp import MLP

import torchvision.transforms as transforms
from torchvision.transforms import Lambda

from torch.utils.data import DataLoader
from src.datasets.animal_dataset import AnimalDataset

from src.train import train_loop
from src.test import test_loop

from src.config import DATA_FOLDER


def main():
    train_print_freq = 50
    val_freq = 1
    checkpoint_path = "model.pth"
    learning_rate = 1e-3
    batch_size = 64
    epochs = 10

    # set reproducibility
    random_seed = 42
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = MLP().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    target_transform = Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            dim=0, index=torch.tensor(y), value=1
        )
    )

    train_set = AnimalDataset(
        DATA_FOLDER / "train.csv", DATA_FOLDER / "images/", transform, target_transform
    )
    val_set = AnimalDataset(
        DATA_FOLDER / "val.csv", DATA_FOLDER / "images/", transform, target_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=True, drop_last=True
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    start_epoch = 0

    # load checkpoint
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, device, train_print_freq)

        # validation
        if epoch % val_freq == 0:
            test_loop(val_loader, model, loss_fn, device)

    print("Done!")

    # save the model
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, "model.pth")


if __name__ == "__main__":
    main()
