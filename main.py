"""
Eye Tracking Mouse Control - With Visible Calibration Circles
Controls your mouse cursor with eye gaze + blink-to-click functionality
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
from collections import deque
import threading
import time

# ==================== CONFIG ====================
class Config:
    """User-configurable settings"""
    CAMERA_INDEX = 1  # Change to 0 if using laptop camera
    BLINK_THRESHOLD = 0.20
    BLINK_COOLDOWN = 15
    SMOOTHING_FACTOR = 0.15
    SHOW_IRIS_MARKER = True


# ==================== CALIBRATION CIRCLE WINDOW ====================
class CalibrationCircleWindow:
    """Displays calibration circles on screen"""
    
    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.current_x = None
        self.current_y = None
        self.active = False
        self.thread = threading.Thread(target=self._window_loop, daemon=True)
        self.thread.start()
        
    def _window_loop(self):
        """Run in separate thread to display circles"""
        import tkinter as tk
        from PIL import Image, ImageDraw, ImageTk
        
        root = tk.Tk()
        root.attributes('-topmost', True)
        root.attributes('-alpha', 0.3)  # Semi-transparent
        root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        
        # Make window click-through
        try:
            import ctypes
            # Windows - set as layered window with transparency
            ctypes.windll.user32.SetWindowLong(
                root.winfo_id(), -20, 
                ctypes.windll.user32.GetWindowLong(root.winfo_id(), -20) | 0x80000
            )
        except:
            try:
                # Linux - set skip taskbar
                root.attributes('-type', 'dock')
            except:
                pass
        
        canvas = tk.Canvas(root, bg='black', highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = canvas
        self.root = root
        
        def update_canvas():
            canvas.delete('all')
            if self.current_x is not None and self.current_y is not None:
                r = 50
                canvas.create_oval(
                    self.current_x - r, self.current_y - r,
                    self.current_x + r, self.current_y + r,
                    outline='#00FF00', width=3
                )
                # Crosshair
                canvas.create_line(
                    self.current_x - 20, self.current_y,
                    self.current_x + 20, self.current_y,
                    fill='#00FF00', width=2
                )
                canvas.create_line(
                    self.current_x, self.current_y - 20,
                    self.current_x, self.current_y + 20,
                    fill='#00FF00', width=2
                )
            root.after(100, update_canvas)
        
        update_canvas()
        root.mainloop()
    
    def show_point(self, x, y):
        """Show a calibration point"""
        self.current_x = x
        self.current_y = y
        self.active = True
    
    def hide_point(self):
        """Hide the calibration point"""
        self.current_x = None
        self.current_y = None
        self.active = False


# ==================== CALIBRATION MANAGER ====================
class CalibrationManager:
    """Handles calibration with visual feedback"""
    
    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.screen_points = [
            (screen_w // 4, screen_h // 4),           # Top-left
            (3 * screen_w // 4, screen_h // 4),       # Top-right
            (screen_w // 4, 3 * screen_h // 4),       # Bottom-left
            (3 * screen_w // 4, 3 * screen_h // 4)    # Bottom-right
        ]
        self.calibration_data = []
        self.current_point = 0
        self.calibrated = False
        self.coef_x = None
        self.coef_y = None
        
        # Create circle window
        self.circle_window = CalibrationCircleWindow(screen_w, screen_h)
        
    def get_current_target(self):
        """Get the current calibration target point"""
        if self.current_point < len(self.screen_points):
            return self.screen_points[self.current_point]
        return None
    
    def show_current_point(self):
        """Display the current calibration point"""
        target = self.get_current_target()
        if target:
            self.circle_window.show_point(target[0], target[1])
    
    def add_calibration_point(self, eye_x, eye_y):
        """Record eye position for current target"""
        target = self.get_current_target()
        if target:
            self.calibration_data.append((eye_x, eye_y, target[0], target[1]))
            self.current_point += 1
            
            if self.current_point == len(self.screen_points):
                self.circle_window.hide_point()
                return self._finalize_calibration()
            else:
                self.show_current_point()
            return True
        return False
    
    def _finalize_calibration(self):
        """Compute mapping coefficients"""
        eye_x = np.array([d[0] for d in self.calibration_data])
        eye_y = np.array([d[1] for d in self.calibration_data])
        screen_x = np.array([d[2] for d in self.calibration_data])
        screen_y = np.array([d[3] for d in self.calibration_data])
        
        self.coef_x = np.polyfit(eye_x, screen_x, 1)
        self.coef_y = np.polyfit(eye_y, screen_y, 1)
        self.calibrated = True
        return True
    
    def get_progress(self):
        """Returns calibration progress (0.0 to 1.0)"""
        return self.current_point / len(self.screen_points)
    
    def reset(self):
        """Reset calibration"""
        self.calibration_data = []
        self.current_point = 0
        self.calibrated = False
        self.coef_x = None
        self.coef_y = None
        self.show_current_point()


# ==================== EYE TRACKER ====================
class EyeTracker:
    """Main eye tracking and control logic"""
    
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    def __init__(self, camera_index=1):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1
        )
        
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        
        self.screen_w, self.screen_h = pyautogui.size()
        self.calibration = CalibrationManager(self.screen_w, self.screen_h)
        
        self.prev_x, self.prev_y = self.screen_w // 2, self.screen_h // 2
        self.blink_cooldown = 0
        
        pyautogui.FAILSAFE = False
        
    def _eye_aspect_ratio(self, landmarks, w, h):
        """Calculate eye aspect ratio for blink detection"""
        def dist(p1, p2):
            return math.hypot(
                (landmarks[p1].x - landmarks[p2].x) * w,
                (landmarks[p1].y - landmarks[p2].y) * h
            )
        
        v1 = dist(self.LEFT_EYE_INDICES[1], self.LEFT_EYE_INDICES[5])
        v2 = dist(self.LEFT_EYE_INDICES[2], self.LEFT_EYE_INDICES[4])
        h_dist = dist(self.LEFT_EYE_INDICES[0], self.LEFT_EYE_INDICES[3])
        
        return (v1 + v2) / (2.0 * h_dist) if h_dist > 0 else 0
    
    def _smooth_position(self, x, y):
        """Apply exponential smoothing to cursor position"""
        alpha = Config.SMOOTHING_FACTOR
        smooth_x = self.prev_x + alpha * (x - self.prev_x)
        smooth_y = self.prev_y + alpha * (y - self.prev_y)
        
        smooth_x = max(0, min(self.screen_w - 1, smooth_x))
        smooth_y = max(0, min(self.screen_h - 1, smooth_y))
        
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return int(smooth_x), int(smooth_y)
    
    def process_frame(self):
        """Process single frame"""
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        status = {
            'calibrating': not self.calibration.calibrated,
            'blink_detected': False,
            'iris_detected': False,
            'cursor_x': self.prev_x,
            'cursor_y': self.prev_y
        }
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_iris = landmarks[468]
            
            ear = self._eye_aspect_ratio(landmarks, w, h)
            is_blinking = ear < Config.BLINK_THRESHOLD
            
            if is_blinking and self.blink_cooldown == 0:
                pyautogui.click()
                self.blink_cooldown = Config.BLINK_COOLDOWN
                status['blink_detected'] = True
            elif not is_blinking and self.blink_cooldown > 0:
                self.blink_cooldown -= 1
            
            status['iris_detected'] = True
            
            if not self.calibration.calibrated:
                # Calibration mode - just detect face
                pass
            else:
                # Tracking mode
                screen_x = self.calibration.coef_x[0] * left_iris.x + self.calibration.coef_x[1]
                screen_y = self.calibration.coef_y[0] * left_iris.y + self.calibration.coef_y[1]
                
                screen_x, screen_y = self._smooth_position(screen_x, screen_y)
                status['cursor_x'] = screen_x
                status['cursor_y'] = screen_y
                
                pyautogui.moveTo(screen_x, screen_y)
                
                if Config.SHOW_IRIS_MARKER:
                    iris_x = int(left_iris.x * w)
                    iris_y = int(left_iris.y * h)
                    cv2.circle(frame, (iris_x, iris_y), 6, (0, 255, 0), -1)
                    cv2.circle(frame, (iris_x, iris_y), 6, (255, 255, 255), 1)
        
        return frame, status, results.multi_face_landmarks[0].landmark if results.multi_face_landmarks else None
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()


# ==================== UI RENDERER ====================
class UIRenderer:
    """Renders on-screen UI"""
    
    @staticmethod
    def render_calibration_ui(frame, calibration):
        """Render calibration mode UI"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        progress = calibration.get_progress()
        bar_y = 30
        bar_h = 15
        bar_w = int(w * 0.6)
        bar_x = (w - bar_w) // 2
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                     (50, 50, 50), 2)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + int(bar_w * progress), bar_y + bar_h), 
                     (0, 200, 100), -1)
        
        point_num = calibration.current_point + 1
        cv2.putText(frame, f"Point {point_num}/4: Look at GREEN CIRCLE on your screen", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1)
        cv2.putText(frame, "Press SPACE to record", (20, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
        
        return frame
    
    @staticmethod
    def render_tracking_ui(frame, status):
        """Render tracking mode UI"""
        h, w = frame.shape[:2]
        
        cv2.putText(frame, f"Cursor: ({status['cursor_x']}, {status['cursor_y']})", 
                   (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "Blink to click | Press C to recalibrate | ESC to exit", 
                   (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if status['blink_detected']:
            cv2.rectangle(frame, (w - 80, 10), (w - 10, 50), (0, 100, 255), -1)
            cv2.putText(frame, "CLICK!", (w - 70, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    @staticmethod
    def render_no_face_warning(frame):
        """Render warning when face not detected"""
        h, w = frame.shape[:2]
        cv2.putText(frame, "No face detected - adjust camera", 
                   (w // 2 - 200, h // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame


# ==================== MAIN APPLICATION ====================
def main():
    """Main application loop"""
    print("=" * 60)
    print("👁️  Eye Tracking Mouse Control - Starting")
    print("=" * 60)
    
    try:
        tracker = EyeTracker(camera_index=Config.CAMERA_INDEX)
        print(f"✓ Camera opened (index {Config.CAMERA_INDEX})")
        print(f"✓ Screen resolution: {tracker.screen_w}x{tracker.screen_h}")
        print(f"✓ Calibration circle window opened (semi-transparent overlay)")
        
        print("\n📍 CALIBRATION MODE:")
        print("   1. You will see a GREEN CIRCLE on your screen")
        print("   2. Look directly at the circle")
        print("   3. Press SPACE to record your eye position")
        print("   4. Repeat for 4 points")
        
        print("\n🖱️  TRACKING MODE:")
        print("   • Move mouse: Just look where you want")
        print("   • Click: Blink your eyes quickly")
        print("   • Reset: Press 'C' to recalibrate")
        print("   • Exit: Press ESC")
        print("-" * 60)
        
        # Show first calibration point
        tracker.calibration.show_current_point()
        time.sleep(0.5)  # Let window render
        
        # Focus the OpenCV window for keyboard input
        cv2.namedWindow("👁️  Eye Mouse Control")
        
        frame_count = 0
        
        while True:
            frame, status, landmarks = tracker.process_frame()
            
            if frame is None:
                print("✗ Failed to read frame")
                break
            
            if status['iris_detected']:
                if tracker.calibration.calibrated:
                    frame = UIRenderer.render_tracking_ui(frame, status)
                else:
                    frame = UIRenderer.render_calibration_ui(frame, tracker.calibration)
            else:
                frame = UIRenderer.render_no_face_warning(frame)
            
            # Display - click on the OpenCV window to focus it for SPACE input
            cv2.imshow("👁️  Eye Mouse Control", frame)
            
            # This ensures the OpenCV window gets keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("\n✓ Exiting...")
                break
            
            elif key == ord(' ') and not tracker.calibration.calibrated:  # SPACE
                if landmarks:
                    iris = landmarks[468]
                    tracker.calibration.add_calibration_point(iris.x, iris.y)
                    if tracker.calibration.calibrated:
                        print(f"\n✓✓✓ Calibration COMPLETE! ✓✓✓")
                        print("Starting eye tracking mode...")
                        print("-" * 60)
                    else:
                        print(f"✓ Point {tracker.calibration.current_point}/4 recorded")
            
            elif key == ord('c'):  # C
                if tracker.calibration.calibrated:
                    tracker.calibration.reset()
                    print("\n✓ Recalibrating...")
                    print("-" * 60)
            
            frame_count += 1
    
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        print("  Make sure camera index is correct (change Config.CAMERA_INDEX)")
    finally:
        tracker.cleanup()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    main()
