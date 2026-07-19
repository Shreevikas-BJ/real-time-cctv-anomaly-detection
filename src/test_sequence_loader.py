from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ucf_sequence_dataset import UCFSequenceDataset


dataset = UCFSequenceDataset(
    r"C:\Users\savij\Downloads\UCF\train"
)

label_counts = Counter(label for _, label in dataset.samples)

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

loader = DataLoader(
    dataset,
    batch_size=8,
    sampler=sampler,
)

clips, labels = next(iter(loader))

print(f"Batch shape: {clips.shape}")
print(f"Labels: {labels}")