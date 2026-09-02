"""
Drowsiness Detector
--------------------
Watches your eyes through the webcam. If your eyes stay closed for too long
(you're falling asleep on your phone/laptop), it plays a loud alarm sound.

HOW IT WORKS
Uses MediaPipe FaceMesh to find eye landmarks, then computes the
Eye Aspect Ratio (EAR) — a standard measure of how open the eyes are.
When EAR drops below a threshold and stays there for CLOSED_EYES_DURATION_SEC
(default 20 seconds), an alarm is triggered.

INSTALL (run once in a terminal):
    pip install opencv-python mediapipe numpy playsound==1.2.2

RUN:
    python drowsiness_detector.py

Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time

# ---------------------------------------------------------------------------
# CONFIG — tweak these to taste
# ---------------------------------------------------------------------------
EAR_THRESHOLD = 0.21              # below this = eyes considered "closed"
CLOSED_EYES_DURATION_SEC = 20     # eyes must stay closed this many seconds to trigger alarm
ALARM_COOLDOWN_SEC = 3            # don't re-trigger alarm sound every single frame

# ---------------------------------------------------------------------------
# ALARM SOUND
# ---------------------------------------------------------------------------
def play_alarm():
    """Plays a beep using the system. Falls back to a printed bell char."""
    try:
        import winsound  # Windows
        winsound.Beep(2500, 700)
    except ImportError:
        try:
            # macOS / Linux: use the terminal bell + 'afplay'/'aplay' if available
            import os
            if os.uname().sysname == "Darwin":
                os.system("afplay /System/Library/Sounds/Sosumi.aiff")
            else:
                os.system('play -nq -t alsa synth 0.7 sine 880 2>/dev/null || echo -e "\\a"')
        except Exception:
            print("\a")  # terminal bell as last resort


def alarm_thread_worker():
    play_alarm()


# ---------------------------------------------------------------------------
# EYE ASPECT RATIO (EAR)
# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark indices for the eyes (6 points per eye,
# matching the classic EAR formula from Soukupová & Čech, 2016)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts
    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check camera permissions/index.")
        return

    eyes_closed_start = None
    last_alarm_time = 0.0

    print("Drowsiness detector running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status_text = "No face detected"
        status_color = (0, 165, 255)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                if eyes_closed_start is None:
                    eyes_closed_start = time.time()
                closed_duration = time.time() - eyes_closed_start
            else:
                eyes_closed_start = None
                closed_duration = 0.0

            if closed_duration >= CLOSED_EYES_DURATION_SEC:
                status_text = "DROWSY! WAKE UP!"
                status_color = (0, 0, 255)
                now = time.time()
                if now - last_alarm_time > ALARM_COOLDOWN_SEC:
                    last_alarm_time = now
                    threading.Thread(target=alarm_thread_worker, daemon=True).start()
            elif closed_duration > 0:
                status_text = f"Eyes closed: {closed_duration:.1f}s / {CLOSED_EYES_DURATION_SEC}s"
                status_color = (0, 165, 255)
            else:
                status_text = f"Eyes open (EAR: {avg_ear:.2f})"
                status_color = (0, 255, 0)

            # Draw eye landmarks
            for idx in LEFT_EYE + RIGHT_EYE:
                x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, status_color, 2)

        cv2.imshow("Drowsiness Detector - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
