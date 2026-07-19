from multiprocessing import freeze_support

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18

from ucf_sequence_dataset import UCFSequenceDataset


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = UCFSequenceDataset(
        r"C:\Users\savij\Downloads\UCF\test"
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(
        torch.load(
            "models/ucf_r3d18_binary.pth",
            map_location=device,
        )
    )

    model = model.to(device)
    model.eval()

    threshold = 0.50
    stop_requested = False

    with torch.inference_mode():
        for clips, labels in loader:
            clips = clips.to(device)

            predictions = model(clips)

            anomaly_probability = torch.softmax(
                predictions,
                dim=1,
            )[0, 1].item()

            status = (
                "ANOMALY"
                if anomaly_probability >= threshold
                else "NORMAL"
            )

            actual = (
                "ANOMALY"
                if labels.item() == 1
                else "NORMAL"
            )

            # Display all 16 frames from the current clip.
            for frame_index in range(clips.shape[2]):
                frame = clips[0, :, frame_index]
                frame = frame.detach().cpu()

                frame = frame.permute(1, 2, 0).numpy()
                frame = np.clip(frame * 255, 0, 255)
                frame = frame.astype(np.uint8)

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGB2BGR,
                )

                color = (
                    (0, 0, 255)
                    if status == "ANOMALY"
                    else (0, 255, 0)
                )

                cv2.putText(
                    frame,
                    f"Predicted: {status}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

                cv2.putText(
                    frame,
                    f"Actual: {actual}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Anomaly probability: {anomaly_probability:.2f}",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    color,
                    2,
                )

                cv2.imshow(
                    "UCF Anomaly Detection",
                    frame,
                )

                if cv2.waitKey(80) & 0xFF == ord("q"):
                    stop_requested = True
                    break

            if stop_requested:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    freeze_support()
    main()