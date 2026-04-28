import os
import time
import threading
import cv2
from flask import Flask, render_template, request, Response, redirect, url_for
from ultralytics import YOLO

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
MODEL_PATH = os.path.join(APP_DIR, "models", "best_fp16.engine")

# --- tunables ---
CONF_THRESH = float(os.getenv("CONF", "0.35"))  # detection confidence
JPEG_QUALITY = int(os.getenv("JPEGQ", "80"))    # stream quality (lower = faster)
HUD_COLOR = (0, 255, 0)                         # BGR for overlay text

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)

# Load TensorRT engine
model = YOLO(MODEL_PATH)
# Ensure class name is visible on overlays
if not getattr(model, "names", None) or 0 not in model.names:
    model.names = {0: "pothole"}

current_video_path = None
lock = threading.Lock()

def frame_generator():
    """Yields annotated frames as an MJPEG stream."""
    global current_video_path
    cap = None
    last_path = None
    t_prev = None

    while True:
        with lock:
            path = current_video_path

        if not path or not os.path.exists(path):
            time.sleep(0.1)
            continue

        # (Re)open only if the file changed
        if path != last_path:
            if cap: cap.release()
            cap = cv2.VideoCapture(path)
            last_path = path

        # Try to keep playback near source FPS
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        delay = 1.0 / float(fps)

        ret, frame = cap.read()
        if not ret:
            # loop video: reopen from start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.05)
            continue

        # Inference (TensorRT engine) with confidence threshold
        # NOTE: Ultralytics auto-selects GPU for .engine
        results = model(frame, conf=CONF_THRESH)
        r0 = results[0]
        boxes = r0.boxes

        # Count & (optional) print to console
        count = 0 if boxes is None else len(boxes)
        if count:
            # confidences rounded for readability
            confs = [] if boxes is None else [round(float(c), 3) for c in boxes.conf.tolist()]
            print(f"[DETECT] {count} pothole(s) | conf={confs}")

        # Draw YOLO overlays
        annotated = r0.plot()

        # Add HUD: count + FPS
        # compute simple instantaneous FPS
        now = time.time()
        inst_fps = 0.0 if t_prev is None else (1.0 / max(1e-6, (now - t_prev)))
        t_prev = now

        cv2.putText(annotated, f"Potholes: {count}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, HUD_COLOR, 2, cv2.LINE_AA)
        cv2.putText(annotated, f"FPS: {inst_fps:.1f}  (conf>={CONF_THRESH})", (12, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, HUD_COLOR, 2, cv2.LINE_AA)

        # Encode as JPEG for MJPEG streaming
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            # skip this frame if encoding fails
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + bytearray(buf) + b"\r\n")

        # Pace: roughly match original fps (reduce if you want max throughput)
        time.sleep(delay)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global current_video_path
    file = request.files.get("video")
    if not file or file.filename == "":
        return redirect(url_for("index"))

    # save upload
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    # switch the stream to this file
    with lock:
        current_video_path = save_path

    return redirect(url_for("index"))

@app.route("/use_sample", methods=["POST"])
def use_sample():
    """Optional: point to a sample video in uploads/ (copy one there)."""
    global current_video_path
    sample = request.form.get("name", "")
    path = os.path.join(UPLOAD_DIR, sample)
    if os.path.exists(path):
        with lock:
            current_video_path = path
    return redirect(url_for("index"))

@app.route("/video_feed")
def video_feed():
    return Response(frame_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    # Run on local network too; debug off for stability
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
