# Real-Time CCTV Anomaly Detection

An unsupervised video-anomaly detection pipeline built with PyTorch and OpenCV. The project trains a convolutional autoencoder on **normal** surveillance frames, treats unusually high reconstruction error as an anomaly, and can monitor either the UCSD Ped2 test set or a live webcam feed.

The repository includes trained checkpoints at `models/autoencoder.pth` and `models/ucf_r3d18_binary.pth`, so you can run inference without retraining once the Python dependencies and input data are available.

## What the project does

- Trains a convolutional autoencoder on normal UCSD Ped2 frames.
- Calculates a reconstruction-error threshold from the training set.
- Labels each test or webcam frame as `NORMAL` or `ANOMALY`.
- Requires 10 consecutive anomalous frames before creating an alert.
- Saves one timestamped JPEG for each continuous anomaly event in `alerts/`.
- Uses CUDA automatically when PyTorch detects a compatible GPU; otherwise it runs on CPU.
- Includes utilities for checking and previewing the expected dataset layout.
- Includes optional experimental UCF video-classification utilities for a separate normal/anomaly classifier.

This is **frame-level anomaly detection**. It does not identify an anomaly class or draw a bounding box around the anomalous object.

## Detection pipeline

```text
Normal training frames
        |
        v
Resize to 128 x 192 -> grayscale -> tensor
        |
        v
Convolutional autoencoder learns to reconstruct normal scenes
        |
        +--------------------------+
        |                          |
        v                          v
Threshold calibration       Test frame / webcam frame
(mean error + 3 std. dev.)          |
                                   v
                         Mean squared reconstruction error
                                   |
                                   v
                      error > threshold for 10 frames?
                              |              |
                             no             yes
                              |              |
                           NORMAL     ANOMALY + JPEG alert
```

The current inference scripts use a threshold of `0.001148`. `src/calculate_threshold.py` calculates a recommended value as:

```text
threshold = mean(training reconstruction errors) + 3 * standard deviation
```

For a new camera, scene, or retrained checkpoint, recalculate the threshold and update it in both inference scripts.

## Model architecture

All frames are resized to `128 x 192`, converted to one grayscale channel, and normalized to `[0, 1]` by `ToTensor()`.

| Stage | Layer | Output channels | Kernel / stride | Activation |
| --- | --- | ---: | --- | --- |
| Encoder | `Conv2d` | 16 | `3 x 3 / 2` | ReLU |
| Encoder | `Conv2d` | 32 | `3 x 3 / 2` | ReLU |
| Encoder | `Conv2d` | 64 | `3 x 3 / 2` | ReLU |
| Decoder | `ConvTranspose2d` | 32 | `4 x 4 / 2` | ReLU |
| Decoder | `ConvTranspose2d` | 16 | `4 x 4 / 2` | ReLU |
| Decoder | `ConvTranspose2d` | 1 | `4 x 4 / 2` | Sigmoid |

Training uses mean squared error, Adam with a learning rate of `0.001`, a batch size of `16`, and `10` epochs.

## Repository layout

```text
.
|-- models/
|   |-- autoencoder.pth          # Included trained autoencoder weights
|   `-- ucf_r3d18_binary.pth      # Optional UCF video-classifier weights
|-- src/
|   |-- calculate_threshold.py   # Derive a threshold from training errors
|   |-- check_dataset.py         # Count the expected train/test frames
|   |-- live_detection.py        # Run anomaly detection on webcam index 0
|   |-- test_autoencoder.py      # Play and score UCSD Ped2 Test012
|   |-- train_autoencoder.py     # Train the autoencoder on normal frames
|   |-- view_dataset.py          # Preview UCSD Ped2 Train001
|   |-- check_ucf_sequences.py   # Inspect UCF-shaped sequence files
|   |-- evaluate_video_classifier.py # Evaluate the UCF R3D-18 model
|   |-- test_sequence_loader.py  # Debug UCF sequence loading
|   |-- train_video_classifier.py # Train the UCF R3D-18 classifier
|   |-- ucf_sequence_dataset.py  # UCF frame sequence dataset helper
|   `-- visualize_predictions.py # Display UCF clip predictions
|-- requirements.txt             # Pinned Python environment
`-- README.md
```

`data/`, `alerts/`, virtual environments, and `.env` files are ignored by Git.

## Running the optional UCF video classifier utilities

The repository also contains experimental UCF video-classification utilities under `src/`. These scripts are separate from the UCSD Ped2 autoencoder pipeline and use `models/ucf_r3d18_binary.pth`.

- `src/train_video_classifier.py` — train a binary R3D-18 anomaly classifier.
- `src/evaluate_video_classifier.py` — evaluate the trained UCF classifier on a test dataset.
- `src/visualize_predictions.py` — display frame-by-frame predictions for UCF clips.
- `src/ucf_sequence_dataset.py` — helper dataset implementation for UCF frame sequences.
- `src/check_ucf_sequences.py` and `src/test_sequence_loader.py` — dataset inspection and loader-debugging utilities.

These UCF utilities currently use local Windows paths and are not configured with command-line options.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Shreevikas-BJ/real-time-cctv-anomaly-detection.git
cd real-time-cctv-anomaly-detection
```

Run the commands below from the repository root because the scripts use paths such as `data/UCSDped2/...` and `models/autoencoder.pth` relative to the current working directory.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install python-dotenv
```

`python-dotenv` is installed separately because `src/live_detection.py` and `src/test_autoencoder.py` import it, but it is not currently listed in `requirements.txt`.

For GPU execution, install the PyTorch build appropriate for your operating system and CUDA environment by following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Dataset setup

Download **UCSD Ped2** from the [official UCSD Anomaly Detection Dataset page](https://www.svcl.ucsd.edu/projects/anomaly/dataset.htm). Extract it so the repository has this structure:

```text
data/
`-- UCSDped2/
    |-- Train/
    |   |-- Train001/
    |   |   `-- *.tif
    |   `-- Train016/
    |       `-- *.tif
    `-- Test/
        |-- Test001/
        |   `-- *.tif
        `-- Test012/
            `-- *.tif
```

The scripts use only the frame images. The dataset's test ground-truth masks are not consumed by the current implementation.

Confirm that the paths are correct:

```bash
python src/check_dataset.py
```

The command should report nonzero training and testing frame counts.

To preview the first training sequence, run:

```bash
python src/view_dataset.py
```

Press `q` in the OpenCV window to stop playback.

## Run with the included checkpoint

### Evaluate a recorded test sequence

```bash
python src/test_autoencoder.py
```

This script processes `data/UCSDped2/Test/Test012`, overlays the reconstruction error and current status, and writes alert images to `alerts/`. Press `q` to stop.

### Monitor a live camera

```bash
python src/live_detection.py
```

The live script opens camera index `0`. When the error remains above the threshold for 10 consecutive frames, it prints an alert and saves a frame such as:

```text
alerts/anomaly_YYYYMMDD_HHMMSS.jpg
```

Press `q` to release the camera and close the window. A desktop session with an available webcam and OpenCV GUI support is required.

## Train and calibrate a model

Train on all normal frames under `data/UCSDped2/Train/Train*`:

```bash
python src/train_autoencoder.py
```

The command replaces `models/autoencoder.pth` after training completes.

Then calculate a threshold for that checkpoint:

```bash
python src/calculate_threshold.py
```

Copy the printed recommended threshold into the `threshold` variable in:

- `src/test_autoencoder.py`
- `src/live_detection.py`

## Alert behavior

The two inference scripts share the same persistence logic:

1. A frame is anomalous when its reconstruction error exceeds the threshold.
2. The consecutive-anomaly counter resets whenever a normal frame appears.
3. An alert is created after 10 consecutive anomalous frames.
4. Only one screenshot is saved during a continuous anomaly event.
5. Alerting is re-armed after the stream returns to normal.

## Current limitations

- The detector reports only a frame-level status; it does not localize or classify anomalies.
- The hard-coded threshold is scene-dependent and may not transfer to a different camera.
- The test sequence, webcam index, training settings, and threshold are not exposed as command-line options.
- Test and live inference require an OpenCV display, so the scripts are not headless-ready.
- There is no evaluation script for precision, recall, ROC-AUC, or ground-truth localization.
- The autoencoder class is duplicated across the training and inference scripts.
- Alert images are stored locally without retention, notification, or database support.

## Possible next steps

- Move the shared model and preprocessing code into reusable modules.
- Add command-line or configuration options for input source, threshold, checkpoint, and persistence length.
- Evaluate against UCSD Ped2 ground truth and report standard anomaly-detection metrics.
- Add spatial anomaly localization using reconstruction-error maps.
- Support video files, RTSP streams, and configurable camera indices.
- Add notifications, alert metadata, retention policies, and a review interface.
- Add automated tests, reproducible environment files, and headless inference support.

## Responsible use

This repository is an educational prototype, not a production surveillance system. Validate it on the intended scene, measure false-positive and false-negative rates, secure stored footage and alerts, and follow applicable privacy, consent, and data-retention requirements before deployment.

## Author

Shreevikas Bangalore Jagadish

Graduate Student, Information Technology and Management, Illinois Institute of Technology

- GitHub: [Shreevikas-BJ](https://github.com/Shreevikas-BJ)
- LinkedIn: [shreevikasbj](https://www.linkedin.com/in/shreevikasbj/)
- Portfolio: [datascienceportfol.io/shreevikasbj](https://datascienceportfol.io/shreevikasbj)


