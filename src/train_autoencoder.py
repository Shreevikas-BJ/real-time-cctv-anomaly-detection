from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class Ped2Dataset(Dataset):
    def __init__(self, folder: str):
        self.frames = sorted(Path(folder).glob("Train*/*.tif"))

        if not self.frames:
            raise FileNotFoundError(f"No training frames found in {folder}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 192)),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        image = Image.open(self.frames[index])
        return self.transform(image)


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

dataset = Ped2Dataset("data/UCSDped2/Train")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = Autoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 10

for epoch in range(epochs):
    total_loss = 0

    for images in loader:
        images = images.to(device)

        reconstructed = model(images)
        loss = criterion(reconstructed, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.6f}")

Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/autoencoder.pth")

print("Model saved to models/autoencoder.pth")