from collections import Counter
from multiprocessing import freeze_support

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18

from ucf_sequence_dataset import UCFSequenceDataset


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    test_dataset = UCFSequenceDataset(
        r"C:\Users\savij\Downloads\UCF\test"
    )

    print(f"Test clips: {len(test_dataset)}")
    print(
        "Test labels:",
        Counter(label for _, label in test_dataset.samples),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
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

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    with torch.no_grad():
        for clips, labels in test_loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            predictions = model(clips)
            loss = criterion(predictions, labels)

            predicted_labels = predictions.argmax(dim=1)

            total_loss += loss.item()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

            true_positive += (
                (predicted_labels == 1) & (labels == 1)
            ).sum().item()

            false_positive += (
                (predicted_labels == 1) & (labels == 0)
            ).sum().item()

            true_negative += (
                (predicted_labels == 0) & (labels == 0)
            ).sum().item()

            false_negative += (
                (predicted_labels == 0) & (labels == 1)
            ).sum().item()

    accuracy = correct / total
    precision = true_positive / max(
        true_positive + false_positive, 1
    )
    recall = true_positive / max(
        true_positive + false_negative, 1
    )
    f1_score = (
        2 * precision * recall / max(precision + recall, 1e-8)
    )

    print(f"Test loss: {total_loss / len(test_loader):.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 score: {f1_score * 100:.2f}%")

    print(f"True positives: {true_positive}")
    print(f"False positives: {false_positive}")
    print(f"True negatives: {true_negative}")
    print(f"False negatives: {false_negative}")


if __name__ == "__main__":
    freeze_support()
    main()