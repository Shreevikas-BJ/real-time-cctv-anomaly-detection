from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UCFSequenceDataset(Dataset):
    def __init__(self, root_folder, sequence_length=16):
        self.root_folder = Path(root_folder)
        self.sequence_length = sequence_length
        self.samples = []

        self.transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
            ]
        )
        for class_folder in self.root_folder.iterdir():
            if not class_folder.is_dir():
                continue

            label = 0 if class_folder.name == "NormalVideos" else 1

        for class_folder in self.root_folder.iterdir():
            if not class_folder.is_dir():
                continue

            label = 0 if class_folder.name == "NormalVideos" else 1
            videos = defaultdict(list)

            for frame_path in class_folder.glob("*.png"):
                video_name, frame_number = frame_path.stem.rsplit("_", 1)

                videos[video_name].append(
                    (int(frame_number), frame_path)
                )

            for frames in videos.values():
                frames.sort(key=lambda item: item[0])
                ordered_paths = [path for _, path in frames]

                for start in range(
                    0,
                    len(ordered_paths) - self.sequence_length + 1,
                    self.sequence_length,
                ):
                    clip = ordered_paths[start : start + self.sequence_length]
                    self.samples.append((clip, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        frame_paths, label = self.samples[index]

        frames = []

        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            frames.append(self.transform(image))

        clip = torch.stack(frames, dim=1)

        return clip, torch.tensor(label, dtype=torch.long)