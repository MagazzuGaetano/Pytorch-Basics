from pathlib import Path
import torch
import torch.nn as nn

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from src.datasets.animal_dataset import AnimalDataset

from src.models.vgg16 import VGG16

from torchmetrics.classification import MulticlassF1Score

from src.train import train_loop
from src.test import test_loop

from src.config import DATA_FOLDER

import matplotlib.pyplot as plt


def plot_loss(train_metric, test_metric):
    plt.plot(test_metric, label="test")
    plt.plot(train_metric, label="train")
    plt.title("Loss per epoch")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()


def main():
    train_print_freq = 50
    val_freq = 1
    checkpoint_path = "model.pth"
    learning_rate = 1e-3
    weight_decay = 5e-4
    batch_size = 64
    epochs = 20
    num_classes = 10

    # set reproducibility
    random_seed = 42
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    train_metric = MulticlassF1Score(num_classes).to(device)
    val_metric = MulticlassF1Score(num_classes).to(device)
    train_results = {}
    train_results["train_loss"] = []
    train_results["train_metric"] = []
    train_results["val_loss"] = []
    train_results["val_metric"] = []

    model = VGG16(num_classes=num_classes).to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # target_transform = Lambda(
    #     lambda y: torch.zeros(num_classes, dtype=torch.float).scatter_(
    #         dim=0, index=torch.tensor(y), value=1
    #     )
    # )

    target_transform = None

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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    start_epoch = 0

    # load checkpoint
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        train_loss, train_metric_value = train_loop(
            train_loader,
            model,
            loss_fn,
            optimizer,
            device,
            train_metric,
            train_print_freq,
        )
        train_results["train_loss"].append(train_loss)
        train_results["train_metric"].append(train_metric_value.detach().cpu().numpy())

        # validation
        if epoch % val_freq == 0:
            val_loss, val_metric_value = test_loop(
                val_loader, model, loss_fn, device, val_metric
            )
            train_results["val_loss"].append(val_loss)
            train_results["val_metric"].append(val_metric_value.detach().cpu().numpy())

    # save the model
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, "model.pth")

    print("Done!")

    plot_loss(train_results["train_loss"], train_results["val_loss"])
    plot_loss(train_results["train_metric"], train_results["val_metric"])


if __name__ == "__main__":
    main()
