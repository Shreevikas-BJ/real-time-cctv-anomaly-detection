# Converts paired RGB.mp4 + MASK.mp4 into YOLO box labels (class 0 = pothole)
# Output: images/.jpg + labels/.txt per frame, for train/val/test splits.

import os, glob, cv2

DATA_DIR = r"C:\Users\savij\OneDrive\Desktop\yolo-trt-project\data\pothole_video"
SPLITS = ["train", "val", "test"]
IMG_SUB = "rgb"     # where your RGB mp4s are
MSK_SUB = "mask"    # where your MASK mp4s are
OUT_IMG = "images"  # YOLO expects images/
OUT_LBL = "labels"  # and labels/

def ensure(p): 
    os.makedirs(p, exist_ok=True); return p

def process_pair(rgb_path, msk_path, out_img_dir, out_lbl_dir, stem):
    vr = cv2.VideoCapture(rgb_path)
    vm = cv2.VideoCapture(msk_path)
    if not vr.isOpened() or not vm.isOpened():
        print("!! cannot open:", rgb_path, "or", msk_path); return

    # choose common frame count
    n_r = int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
    n_m = int(vm.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_r, n_m)

    idx = 0
    while idx < n:
        ok_r, fr = vr.read()
        ok_m, fm = vm.read()
        if not ok_r or not ok_m: break

        h, w = fr.shape[:2]

        # mask → binary (any non-zero = pothole)
        gray = cv2.cvtColor(fm, cv2.COLOR_BGR2GRAY)
        _, binm = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        # clean tiny specks
        binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

        contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # save image
        img_name = f"{stem}_f{idx:06d}.jpg"
        lbl_name = f"{stem}_f{idx:06d}.txt"
        cv2.imwrite(os.path.join(out_img_dir, img_name), fr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # write YOLO boxes
        lines = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 50:  # ignore very small blobs
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            cx = (x + bw/2) / w
            cy = (y + bh/2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        with open(os.path.join(out_lbl_dir, lbl_name), "w") as f:
            f.write("\n".join(lines))

        idx += 1

    vr.release(); vm.release()
    print(f"ok: {stem}  frames: {idx}")

def run_split(split_dir):
    rgb_dir = os.path.join(split_dir, IMG_SUB)
    msk_dir = os.path.join(split_dir, MSK_SUB)
    out_img = ensure(os.path.join(split_dir, OUT_IMG))
    out_lbl = ensure(os.path.join(split_dir, OUT_LBL))

    # match videos by stem (filename without extension)
    rgb_videos = {os.path.splitext(os.path.basename(p))[0]: p
                  for p in glob.glob(os.path.join(rgb_dir, "*.mp4"))}
    msk_videos = {os.path.splitext(os.path.basename(p))[0]: p
                  for p in glob.glob(os.path.join(msk_dir, "*.mp4"))}

    if not rgb_videos:
        print("No RGB mp4s in", rgb_dir); return

    for stem, rpath in rgb_videos.items():
        mpath = msk_videos.get(stem)
        if not mpath:
            print("!! no matching mask mp4 for", rpath); 
            continue
        process_pair(rpath, mpath, out_img, out_lbl, stem)

if __name__ == "__main__":
    for s in SPLITS:
        sd = os.path.join(DATA_DIR, s)
        if os.path.isdir(sd):
            print("\n=== split:", s, "===")
            run_split(sd)
        else:
            print("skip (no folder):", sd)
    print("\nDone.")
