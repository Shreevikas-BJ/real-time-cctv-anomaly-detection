from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from datetime import datetime
from dotenv import load_dotenv

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

test_folder = Path("data/UCSDped2/Test/Test012")
frames = sorted(test_folder.glob("*.tif"))

threshold = 0.001148

consecutive_anomalies = 0
required_frames = 10
alert_sent = False

Path("alerts").mkdir(exist_ok=True)

with torch.no_grad():
    for frame_path in frames:
        image = Image.open(frame_path)
        tensor = transform(image).unsqueeze(0).to(device)

        reconstructed = model(tensor)
        error = torch.mean((tensor - reconstructed) ** 2).item()

        frame = np.array(image.convert("RGB"))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        status = "ANOMALY" if error > threshold else "NORMAL"

        if status == "ANOMALY":
            consecutive_anomalies += 1
        else:
            consecutive_anomalies = 0
            alert_sent = False

        if consecutive_anomalies >= required_frames and not alert_sent:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alert_path = f"alerts/anomaly_{timestamp}.jpg"

            cv2.imwrite(alert_path, frame)

            print(f"ALERT: Anomaly detected at {frame_path.name}")
            print(f"Screenshot saved: {alert_path}")

            alert_sent = True

        cv2.putText(
            frame,
            f"{status} | Error: {error:.5f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255) if status == "ANOMALY" else (0, 255, 0),
            2,
        )

        cv2.imshow("Anomaly Detection", frame)

        if cv2.waitKey(50) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()