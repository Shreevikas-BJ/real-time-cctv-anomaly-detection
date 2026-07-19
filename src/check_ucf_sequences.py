from collections import defaultdict
from pathlib import Path

folder = Path(r"C:\Users\savij\Downloads\UCF\train\Fighting")

videos = defaultdict(list)

for frame_path in folder.glob("*.png"):
    video_name, frame_number = frame_path.stem.rsplit("_", 1)
    videos[video_name].append((int(frame_number), frame_path))

for video_name in videos:
    videos[video_name].sort(key=lambda item: item[0])

print(f"Original videos found: {len(videos)}")

first_video = next(iter(videos))
first_frames = videos[first_video]

print(f"Example video: {first_video}")
print(f"Frames found: {len(first_frames)}")
print("First 10 ordered frames:")

for frame_number, frame_path in first_frames[:10]:
    print(frame_number, frame_path.name)