from pathlib import Path

train_path = Path("data/UCSDped2/Train")
test_path = Path("data/UCSDped2/Test")

train_frames = list(train_path.glob("Train*/*.tif"))
test_frames = list(test_path.glob("Test*/*.tif"))

print(f"Training frames: {len(train_frames)}")
print(f"Testing frames: {len(test_frames)}")

if not train_frames:
    print("Training path is incorrect.")

if not test_frames:
    print("Testing path is incorrect.")