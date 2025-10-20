import torch.nn as nn
from torchvision import models
import torch
import os

def create_model_1(num_classes, pretrained=None):
    """Return a ResNet-50 with custom output layer."""
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def create_model(num_classes, pretrained=False):
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.resnet50(weights=weights)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def create_finetuned_model(num_classes, weights_path="./best_weights.pth"):
    """
    Create a ResNet-50 for fine-tuning with selective layer freezing.
    """
    # 1️⃣ Initialize model (you can also use pretrained=IMAGENET1K_V1)
    model = models.resnet50(weights=None)  # don't pass custom path here

    # 2️⃣ Replace classifier for your dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 3️⃣ Load your custom weights if available
    if weights_path and os.path.exists(weights_path):
        print(f"✅ Loading custom weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        # if saved with model.state_dict() only
        if "state_dict" in state_dict:  # handle checkpoints with wrapped dicts
            state_dict = state_dict["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"ℹ️ Missing keys: {missing}")
        print(f"ℹ️ Unexpected keys: {unexpected}")
    else:
        print("⚠️ No pretrained weights found — training from scratch")

    # 4️⃣ Freeze lower layers (conv1 through layer2)
    for name, child in model.named_children():
        if name in ["conv1", "bn1", "layer1", "layer2"]:
            for param in child.parameters():
                param.requires_grad = False

    # 5️⃣ Unfreeze deeper layers
    for name, child in model.named_children():
        if name in ["layer3", "layer4", "fc"]:
            for param in child.parameters():
                param.requires_grad = True

    return model
