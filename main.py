"""
Eye Tracking Mouse Control
Controls your mouse cursor with eye gaze + blink-to-click functionality.

Calibration circle is drawn directly on a full-screen OpenCV window so
keyboard input (SPACE, C, ESC) always works without focus issues.
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time


# ==================== CONFIG ====================
class Config:
    CAMERA_INDEX    = 1       # 0 = built-in webcam, 1 = external
    BLINK_THRESHOLD = 0.20    # lower = harder blink needed to click
    BLINK_COOLDOWN  = 15      # frames to wait between clicks
    SMOOTHING       = 0.15    # 0..1, lower = smoother but laggier
    SHOW_IRIS_DOT   = True


# ==================== CALIBRATION ====================
class CalibrationManager:

    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h

        mx, my = screen_w // 5, screen_h // 5
        self.targets = [
            (mx,              my),
            (screen_w - mx,   my),
            (mx,              screen_h - my),
            (screen_w - mx,   screen_h - my),
        ]

        self.data          = []
        self.current_point = 0
        self.calibrated    = False
        self.coef_x        = None
        self.coef_y        = None

    def current_target(self):
        if self.current_point < len(self.targets):
            return self.targets[self.current_point]
        return None

    def progress(self):
        return self.current_point / len(self.targets)

    def record(self, eye_x, eye_y):
        target = self.current_target()
        if target is None:
            return False
        self.data.append((eye_x, eye_y, target[0], target[1]))
        self.current_point += 1
        if self.current_point >= len(self.targets):
            self._finalize()
        return True

    def _finalize(self):
        ex = np.array([d[0] for d in self.data], dtype=float)
        ey = np.array([d[1] for d in self.data], dtype=float)
        sx = np.array([d[2] for d in self.data], dtype=float)
        sy = np.array([d[3] for d in self.data], dtype=float)
        self.coef_x     = np.polyfit(ex, sx, 1)
        self.coef_y     = np.polyfit(ey, sy, 1)
        self.calibrated = True

    def reset(self):
        self.data          = []
        self.current_point = 0
        self.calibrated    = False
        self.coef_x        = None
        self.coef_y        = None

    def draw_target(self, canvas):
        target = self.current_target()
        if target is None:
            return
        x, y = target
        r = 40
        cv2.circle(canvas, (x, y), r,  (0, 220, 0), 3, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 6,  (0, 255, 0), -1, cv2.LINE_AA)
        cv2.line(canvas, (x - r - 15, y), (x - r + 5,  y), (0, 220, 0), 2, cv2.LINE_AA)
        cv2.line(canvas, (x + r - 5,  y), (x + r + 15, y), (0, 220, 0), 2, cv2.LINE_AA)
        cv2.line(canvas, (x, y - r - 15), (x, y - r + 5),  (0, 220, 0), 2, cv2.LINE_AA)
        cv2.line(canvas, (x, y + r - 5),  (x, y + r + 15), (0, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas,
                    f"Point {self.current_point + 1}/{len(self.targets)}",
                    (x - 60, y + r + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


# ==================== EYE TRACKER ====================
class EyeTracker:

    LEFT_EYE = [33, 160, 158, 133, 153, 144]

    def __init__(self):
        self.mp_mesh   = mp.solutions.face_mesh
        self.face_mesh = self.mp_mesh.FaceMesh(
            refine_landmarks=True, max_num_faces=1)

        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {Config.CAMERA_INDEX}. "
                "Change Config.CAMERA_INDEX and try again.")

        self.screen_w, self.screen_h = pyautogui.size()
        self.cal = CalibrationManager(self.screen_w, self.screen_h)

        self.prev_x         = self.screen_w // 2
        self.prev_y         = self.screen_h // 2
        self.blink_cooldown = 0

        pyautogui.FAILSAFE = False

    def _ear(self, lm, w, h):
        def d(a, b):
            return math.hypot((lm[a].x - lm[b].x) * w,
                              (lm[a].y - lm[b].y) * h)
        idx = self.LEFT_EYE
        v1  = d(idx[1], idx[5])
        v2  = d(idx[2], idx[4])
        hd  = d(idx[0], idx[3])
        return (v1 + v2) / (2.0 * hd) if hd > 0 else 0.0

    def _smooth(self, x, y):
        a  = Config.SMOOTHING
        sx = self.prev_x + a * (x - self.prev_x)
        sy = self.prev_y + a * (y - self.prev_y)
        sx = max(0, min(self.screen_w - 1, sx))
        sy = max(0, min(self.screen_h - 1, sy))
        self.prev_x, self.prev_y = sx, sy
        return int(sx), int(sy)

    def process_frame(self):
        """Returns (cam_frame, landmarks_or_None, blink_bool)."""
        ret, frame = self.cap.read()
        if not ret:
            return None, None, False

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        results   = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarks = None
        blink     = False

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            iris      = landmarks[468]

            ear      = self._ear(landmarks, w, h)
            is_blink = ear < Config.BLINK_THRESHOLD

            if is_blink and self.blink_cooldown == 0:
                pyautogui.click()
                self.blink_cooldown = Config.BLINK_COOLDOWN
                blink = True
            elif not is_blink and self.blink_cooldown > 0:
                self.blink_cooldown -= 1

            if self.cal.calibrated:
                sx, sy = self._smooth(
                    np.polyval(self.cal.coef_x, iris.x),
                    np.polyval(self.cal.coef_y, iris.y),
                )
                pyautogui.moveTo(sx, sy)

            if Config.SHOW_IRIS_DOT:
                ix, iy = int(iris.x * w), int(iris.y * h)
                cv2.circle(frame, (ix, iy), 5, (0, 255, 0),   -1, cv2.LINE_AA)
                cv2.circle(frame, (ix, iy), 5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame, landmarks, blink

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()


# ==================== DISPLAY ====================
def build_display(cam_frame, tracker, landmarks, blink):
    sw, sh = tracker.screen_w, tracker.screen_h

    if not tracker.cal.calibrated:
        # Black canvas with calibration target drawn on it
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)

        # Small camera preview (top-left)
        cam_h, cam_w = cam_frame.shape[:2]
        pw = sw // 4
        ph = int(cam_h * pw / cam_w)
        canvas[:ph, :pw] = cv2.resize(cam_frame, (pw, ph))

        tracker.cal.draw_target(canvas)

        if landmarks is not None:
            msg   = "Look at the circle, then press  SPACE  to record"
            color = (180, 255, 180)
        else:
            msg   = "No face detected  --  adjust your camera"
            color = (80, 80, 255)

        cv2.putText(canvas, msg,
                    (sw // 2 - 300, sh - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # Progress bar
        bx, by, bw = sw // 2 - 200, sh - 28, 400
        filled = int(bw * tracker.cal.progress())
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + 12), (60, 60, 60), -1)
        if filled > 0:
            cv2.rectangle(canvas, (bx, by), (bx + filled, by + 12), (0, 200, 100), -1)

        return canvas

    else:
        canvas = cv2.resize(cam_frame, (sw, sh))

        if blink:
            cv2.rectangle(canvas, (sw - 120, 10), (sw - 10, 60), (0, 80, 220), -1)
            cv2.putText(canvas, "CLICK!", (sw - 110, 47),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(canvas,
                    "Blink = click  |  C = recalibrate  |  ESC = exit",
                    (10, sh - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        return canvas


# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("Eye Tracking Mouse Control")
    print("=" * 60)

    tracker = None
    try:
        tracker = EyeTracker()
        sw, sh  = tracker.screen_w, tracker.screen_h
        print(f"Camera {Config.CAMERA_INDEX} opened  |  screen {sw}x{sh}")
        print()
        print("CALIBRATION:")
        print("  Look at the GREEN CIRCLE, then press SPACE to record.")
        print("  Repeat for all 4 corners.")
        print()
        print("TRACKING:")
        print("  Blink = click  |  C = recalibrate  |  ESC = exit")
        print("-" * 60)

        WIN = "Eye Mouse Control"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            cam_frame, landmarks, blink = tracker.process_frame()

            if cam_frame is None:
                print("Camera read failed.")
                break

            if not tracker.cal.calibrated:
                cv2.imshow(WIN, build_display(cam_frame, tracker, landmarks, blink))

            key = cv2.waitKey(1) & 0xFF

            if key == 27:                           # ESC
                print("Exiting.")
                break

            elif key == ord(' '):                   # SPACE
                if not tracker.cal.calibrated:
                    if landmarks is not None:
                        tracker.cal.record(landmarks[468].x, landmarks[468].y)
                        if tracker.cal.calibrated:
                            print("Calibration complete — tracking active.")
                            cv2.destroyWindow(WIN)
                        else:
                            print(f"  Point {tracker.cal.current_point}/4 recorded.")
                    else:
                        print("  No face detected — move into frame first.")

            elif key == ord('c'):                   # C
                tracker.cal.reset()
                print("Recalibrating...")
                cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    except RuntimeError as e:
        print(f"Error: {e}")
    finally:
        if tracker:
            tracker.cleanup()
        print("Done.")


if __name__ == "__main__":
    main()