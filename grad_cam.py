#pip install grad-cam
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Example: visualize GradCAM on a few test images
def visualize_grad_cam(model, testloader, device):
    model.eval()
    target_layer = model.layer4[-1]  # last conv block in ResNet-34
    cam = GradCAM(model=model, target_layers=[target_layer])

    # get one sample from test set
    images, labels = next(iter(testloader))
    image = images[0].unsqueeze(0).to(device)
    rgb_img = np.transpose(images[0].cpu().numpy(), (1, 2, 0))
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

    # forward pass
    pred = model(image)
    pred_class = pred.argmax(dim=1).item()

    # generate GradCAM
    grayscale_cam = cam(input_tensor=image, targets=[ClassifierOutputTarget(pred_class)])
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.title(f"True: {labels[0].item()}, Pred: {pred_class}")
    plt.axis('off')
    plt.show()

def denormalize(img, mean, std):
    """Undo normalization for display."""
    img = img.clone().cpu()
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    return img.clamp(0, 1)


def get_misclassified_images(model, device, test_loader):
    """Return list of misclassified (image, pred, true) tuples."""
    model.eval()
    misclassified = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            mismatch = preds != labels
            for i in range(len(imgs)):
                if mismatch[i]:
                    misclassified.append((imgs[i].cpu(), preds[i].item(), labels[i].item()))
    return misclassified


def plot_gradcam_for_misclassified(model, device, test_loader, class_names, mean, std, num_images=5,layer=None):
    """Plot Grad-CAM results for misclassified samples."""
    # 1️⃣ Collect misclassified samples
    misclassified = get_misclassified_images(model, device, test_loader)
    print(f"Found {len(misclassified)} misclassified images.")

    if len(misclassified) == 0:
        print("No misclassified images found!")
        return

    # 2️⃣ Select target layer (usually last conv layer in ResNet)
    target_layer = model.layer4[-1] if layer == 4 else model.layer3[-1] if layer == 3 else model.layer2[-1] if layer == 2 else model.layer1[-1] if layer == 1 else model.layer4[-1]

    # 3️⃣ Create GradCAM object
    cam = GradCAM(model=model, target_layers=[target_layer])

    # 4️⃣ Pick a few misclassified samples
    samples = misclassified[:num_images]
    fig, axes = plt.subplots(len(samples), 3, figsize=(10, 3 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (img, pred, true) in enumerate(samples):
        input_tensor = img.unsqueeze(0).to(device)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]  # first batch element

        # Convert to numpy & prepare for overlay
        rgb_img = denormalize(img, mean, std).permute(1, 2, 0).numpy()
        heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # 5️⃣ Display
        axes[idx, 0].imshow(rgb_img)
        axes[idx, 0].set_title(f"True: {class_names[true]}")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(grayscale_cam, cmap='jet')
        axes[idx, 1].set_title("Grad-CAM Heatmap")
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(heatmap)
        axes[idx, 2].set_title(f"Pred: {class_names[pred]}")
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.show()


"""
Example for CIFAR-100
mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)
class_names = test_dataset.classes  # from torchvision.datasets.CIFAR100

plot_gradcam_for_misclassified(
    model, 
    device, 
    test_loader, 
    class_names=class_names, 
    mean=mean, 
    std=std, 
    num_images=8  # show 8 misclassified samples
)
"""

def plot_and_save_gradcam_for_misclassified(
    model,
    device,
    test_loader,
    class_names,
    mean,
    std,
    num_images=10,
    save_dir="./gradcam_results",
):
    """
    Generates Grad-CAM for misclassified CIFAR images.
    Saves and displays original, heatmap, and overlayed images side-by-side.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Get misclassified samples
    misclassified = get_misclassified_images(model, device, test_loader)
    print(f"Found {len(misclassified)} misclassified images.")

    if len(misclassified) == 0:
        print("✅ Model classified all correctly (no misclassified samples).")
        return

    # Target last conv layer for GradCAM
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device.type == "cuda"))

    # Select few samples
    samples = misclassified[:num_images]
    for idx, (img, pred, true) in enumerate(samples):
        input_tensor = img.unsqueeze(0).to(device)

        # Generate GradCAM
        grayscale_cam = cam(input_tensor=input_tensor)[0, :]
        rgb_img = denormalize(img, mean, std).permute(1, 2, 0).numpy()
        heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Plot side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"True: {class_names[true]}")
        axes[0].axis("off")

        axes[1].imshow(grayscale_cam, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(heatmap)
        axes[2].set_title(f"Pred: {class_names[pred]}")
        axes[2].axis("off")

        plt.tight_layout()

        # Save image
        save_path = os.path.join(
            save_dir, f"misclassified_{idx+1:02d}_true_{class_names[true]}_pred_{class_names[pred]}.png"
        )
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        print(f"💾 Saved Grad-CAM result: {save_path}")

    print(f"\n✅ All {len(samples)} Grad-CAM visualizations saved to: {os.path.abspath(save_dir)}")