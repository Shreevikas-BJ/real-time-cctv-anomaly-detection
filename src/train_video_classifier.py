from collections import Counter
from multiprocessing import freeze_support
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.video import R3D_18_Weights, r3d_18

from ucf_sequence_dataset import UCFSequenceDataset


def main():
    # Use NVIDIA GPU when CUDA is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load 16-frame training clips.
    dataset = UCFSequenceDataset(
        r"C:\Users\savij\Downloads\UCF\train"
    )

    # Count normal and anomaly clips.
    label_counts = Counter(
        label for _, label in dataset.samples
    )

    print(f"Class counts: {label_counts}")
    print(f"Total clips: {len(dataset)}")

    # Give the smaller class a higher sampling probability.
    class_weights = {
        label: 1.0 / count
        for label, count in label_counts.items()
    }

    sample_weights = [
        class_weights[label]
        for _, label in dataset.samples
    ]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Load frames in parallel and send balanced batches to the GPU.
    loader = DataLoader(
        dataset,
        batch_size=8,
        sampler=sampler,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Load pretrained R3D-18 video features.
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)

    # Freeze the pretrained feature extractor.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace the final layer with normal/anomaly output.
    model.fc = nn.Linear(
        model.fc.in_features,
        2,
    )

    model = model.to(device)

    # Measure prediction error.
    criterion = nn.CrossEntropyLoss()

    # Train only the new final classification layer.
    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=0.001,
    )

    models_folder = Path("models")
    models_folder.mkdir(exist_ok=True)

    checkpoint_path = models_folder / "r3d18_checkpoint.pth"
    final_model_path = models_folder / "ucf_r3d18_binary.pth"

    start_epoch = 0
    epochs = 3

    # Resume from the latest completed epoch when available.
    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1

        print(f"Resuming from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, epochs):
        # Keep frozen pretrained layers in evaluation mode.
        model.eval()
        model.fc.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_number, (clips, labels) in enumerate(loader):
            clips = clips.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            # Predict normal or anomaly.
            predictions = model(clips)

            # Compare predictions with correct labels.
            loss = criterion(predictions, labels)

            # Update the final-layer weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted_labels = predictions.argmax(dim=1)

            correct += (
                predicted_labels == labels
            ).sum().item()

            total += labels.size(0)

            if batch_number % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {batch_number}/{len(loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        average_loss = running_loss / len(loader)
        accuracy = 100 * correct / total

        print(
            f"Epoch {epoch + 1} complete | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

        # Save progress after every completed epoch.
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            checkpoint_path,
        )

        print(f"Checkpoint saved after epoch {epoch + 1}")

    # Save the final trained weights.
    torch.save(
        model.state_dict(),
        final_model_path,
    )

    print(f"Model saved to {final_model_path}")


if __name__ == "__main__":
    # Required for DataLoader multiprocessing on Windows.
    freeze_support()
    main()