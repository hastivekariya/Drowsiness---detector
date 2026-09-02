# 😴 Drowsiness Detector

A real-time eye-closure / drowsiness detector built with Python, OpenCV, and MediaPipe. It watches your face through a webcam and sounds an alarm if your eyes stay closed for too long — useful for staying alert while working, studying, or driving.

## How It Works

The script uses **MediaPipe FaceMesh** to detect facial landmarks around each eye, then calculates the **Eye Aspect Ratio (EAR)** — a well-known metric (Soukupová & Čech, 2016) that measures how open or closed an eye is based on landmark geometry.

- When EAR drops below a threshold, the eyes are considered "closed"
- If the eyes stay closed continuously for a configurable duration (default: **20 seconds**), an alarm sound is triggered
- A short cooldown prevents the alarm from spamming repeatedly

## Tech Stack

- Python 3.11
- [OpenCV](https://opencv.org/) — video capture & display
- [MediaPipe](https://developers.google.com/mediapipe) — facial landmark detection
- NumPy — EAR calculation

## Installation

```bash
pip install opencv-python mediapipe==0.10.21 numpy
```

> Note: newer MediaPipe versions (0.10.30+) removed the legacy `mediapipe.solutions` API used here. Pin to `0.10.21` or earlier if you hit an `AttributeError`.

## Usage

```bash
python drowsiness_detector.py
```

Press **`q`** to quit the webcam window.

## Configuration

Tweak these constants at the top of the script:

| Variable | Description | Default |
|---|---|---|
| `EAR_THRESHOLD` | EAR value below which eyes are considered closed | `0.21` |
| `CLOSED_EYES_DURATION_SEC` | How many seconds eyes must stay closed before the alarm fires | `20` |
| `ALARM_COOLDOWN_SEC` | Minimum time between repeated alarm triggers | `3` |

## Future Improvements

- Yawn detection as an additional drowsiness signal
- Head-tilt / nodding detection
- Mobile app version (camera-based, browser or native)
- Logging drowsiness events over a session for analysis

## License

MIT — free to use and modify.
