import torch.nn as nn
import torchvision


def VGG16(num_classes):
    weigths = torchvision.models.VGG16_Weights.DEFAULT
    model = torchvision.models.vgg16(weights=weigths)

    # Freeze early layers
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[-1] = nn.Linear(4096, num_classes)
    return model
