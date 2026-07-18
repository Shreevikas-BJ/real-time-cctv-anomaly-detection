from pathlib import Path
import cv2

folder = Path("data/UCSDped2/Train/Train001")
frames = sorted(folder.glob("*.tif"))

if not frames:
    raise FileNotFoundError(f"No frames found inside {folder}")

for frame_path in frames:
    frame = cv2.imread(str(frame_path))

    cv2.imshow("UCSD Ped2 Dataset", frame)

    if cv2.waitKey(50) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()