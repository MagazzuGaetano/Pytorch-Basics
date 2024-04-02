import torch
from torch import nn
from torchsummary import summary


class MLP(nn.Module):
    def __init__(self, width, height, channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(0.5)
        self.fc1 = nn.Linear(width * height * channels, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.drop_out(self.fc3(x))
        return logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(width=224, height=224, channels=3, num_classes=10).to(device)
    summary(model, (3, 224, 224), 64, device)
