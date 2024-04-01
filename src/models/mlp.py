from torch import nn
from torchsummary import summary


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(224 * 224 * 3, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


if __name__ == "__main__":
    model = MLP()
    summary(model, (3, 224, 224), 16)
