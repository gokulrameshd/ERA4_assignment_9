import torch
from torchvision.transforms import v2

print("CUDA available:", torch.cuda.is_available())

x = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
t = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

# Apply transforms on CPU, then move to GPU
y = t(x).to("cuda", non_blocking=True)

print("Output device:", y.device)
print("Mean:", y.mean().item(), "Std:", y.std().item())
