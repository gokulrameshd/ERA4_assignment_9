import torch.nn as nn
from torchvision import models


def create_model(num_classes, pretrained=None):
    """Return a ResNet-34 with custom output layer."""
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
