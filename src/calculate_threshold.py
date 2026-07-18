from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((128, 192)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)

model = Autoencoder().to(device)
model.load_state_dict(
    torch.load("models/autoencoder.pth", map_location=device)
)
model.eval()

frames = sorted(Path("data/UCSDped2/Train").glob("Train*/*.tif"))
errors = []

with torch.no_grad():
    for frame_path in frames:
        image = Image.open(frame_path)
        tensor = transform(image).unsqueeze(0).to(device)

        reconstructed = model(tensor)
        error = torch.mean((tensor - reconstructed) ** 2).item()
        errors.append(error)

mean_error = np.mean(errors)
std_error = np.std(errors)
threshold = mean_error + (3 * std_error)

print(f"Mean error: {mean_error:.6f}")
print(f"Standard deviation: {std_error:.6f}")
print(f"Recommended threshold: {threshold:.6f}")