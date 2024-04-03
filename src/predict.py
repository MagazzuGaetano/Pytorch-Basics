from pathlib import Path
import torch

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from datasets.animal_dataset import AnimalDataset

from models.vgg16 import VGG16

from config import DATA_FOLDER

import matplotlib.pyplot as plt


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image


checkpoint_path = "model.pth"
classes = [
    "cane",
    "cavallo",
    "elefante",
    "farfalla",
    "gallina",
    "gatto",
    "mucca",
    "pecora",
    "ragno",
    "scoiattolo",
]
num_classes = len(classes)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

target_transform = None

test_set = AnimalDataset(
    DATA_FOLDER / "test.csv", DATA_FOLDER / "images/", transform, target_transform
)
test_loader = DataLoader(test_set, batch_size=1)


# load model
model = VGG16(num_classes=num_classes).to(device)

if Path(checkpoint_path).exists():
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])


# inference on new images
model.eval()
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        pred_class = classes[torch.argmax(torch.softmax(pred, dim=1), dim=1)]
        label = classes[y]

        X = normalize_image(X.squeeze().permute(1, 2, 0).cpu())
        plt.imshow(X)
        plt.title("pred: {} - label: {}".format(pred_class, label))
        plt.show()
