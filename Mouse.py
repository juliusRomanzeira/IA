from pathlib import Path
from urllib.request import urlopen

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


class MouseController:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.PAUSE = 0
        self.last_click_time = 0.0
        self.click_cooldown = 0.35
        self.click_threshold = 0.05
        self.left_button_down = False
        self.right_gesture_active = False
        self.commands_enabled = True
        self.mode_toggle_armed = False
        self.edge_margin = 1
        self.smoothing = 0.35
        self.input_margin_ratio_x = 0.22
        self.input_margin_ratio_y = 0.22
        self.mouse_x = self.screen_width / 2
        self.mouse_y = self.screen_height / 2
        self.model_path = self.ensure_model()
        self.hand_landmarker = self.create_hand_landmarker()
        self.connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    def ensure_model(self):
        model_path = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
        if model_path.exists():
            return model_path

        model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = model_path.with_suffix(".task.part")

        try:
            with urlopen(MODEL_URL, timeout=30) as response, temp_path.open("wb") as model_file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    model_file.write(chunk)
        except Exception as exc:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError("Nao foi possivel baixar o modelo do MediaPipe.") from exc

        temp_path.replace(model_path)
        return model_path

    def create_hand_landmarker(self):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def draw_hand_landmarks(self, frame, hand_landmarks):
        frame_height, frame_width = frame.shape[:2]

        for connection in self.connections:
            start_landmark = hand_landmarks[connection.start]
            end_landmark = hand_landmarks[connection.end]
            start_point = (
                int(start_landmark.x * frame_width),
                int(start_landmark.y * frame_height),
            )
            end_point = (
                int(end_landmark.x * frame_width),
                int(end_landmark.y * frame_height),
            )
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        for landmark in hand_landmarks:
            point = (
                int(landmark.x * frame_width),
                int(landmark.y * frame_height),
            )
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

    def normalize_with_margin(self, value, margin_ratio):
        min_value = margin_ratio
        max_value = 1.0 - margin_ratio
        normalized_value = (value - min_value) / (max_value - min_value)
        return float(np.clip(normalized_value, 0.0, 1.0))

    def is_pinching(self, thumb_tip, finger_tip):
        distance = np.hypot(thumb_tip.x - finger_tip.x, thumb_tip.y - finger_tip.y)
        return distance < self.click_threshold

    def is_finger_extended(self, hand_landmarks, tip_index, pip_index, mcp_index):
        tip = hand_landmarks[tip_index]
        pip = hand_landmarks[pip_index]
        mcp = hand_landmarks[mcp_index]
        return tip.y < pip.y < mcp.y

    def classify_hand_state(self, hand_landmarks):
        finger_joints = (
            (8, 6, 5),
            (12, 10, 9),
            (16, 14, 13),
            (20, 18, 17),
        )
        extended_fingers = sum(
            self.is_finger_extended(hand_landmarks, tip_index, pip_index, mcp_index)
            for tip_index, pip_index, mcp_index in finger_joints
        )

        if extended_fingers == 4:
            return "open"
        if extended_fingers == 0:
            return "closed"
        return "other"

    def release_left_button(self):
        if self.left_button_down:
            pyautogui.mouseUp(button="left")
            self.left_button_down = False

    def update_command_mode(self, hand_state):
        if self.commands_enabled:
            if hand_state == "open":
                self.mode_toggle_armed = True
            elif hand_state == "closed" and self.mode_toggle_armed:
                self.commands_enabled = False
                self.mode_toggle_armed = False
                self.release_left_button()
                self.right_gesture_active = False
                return True
            elif hand_state == "other":
                self.mode_toggle_armed = False
        else:
            if hand_state == "closed":
                self.mode_toggle_armed = True
            elif hand_state == "open" and self.mode_toggle_armed:
                self.commands_enabled = True
                self.mode_toggle_armed = False
                return True
            elif hand_state == "other":
                self.mode_toggle_armed = False

        return False

    def control_mouse(self, frame, timestamp_ms):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            self.draw_hand_landmarks(frame, hand_landmarks)
            hand_state = self.classify_hand_state(hand_landmarks)
            mode_changed = self.update_command_mode(hand_state)

            if self.commands_enabled and not mode_changed:
                normalized_x = self.normalize_with_margin(
                    hand_landmarks[8].x,
                    self.input_margin_ratio_x,
                )
                normalized_y = self.normalize_with_margin(
                    hand_landmarks[8].y,
                    self.input_margin_ratio_y,
                )
                target_x = float(
                    np.clip(
                        normalized_x * self.screen_width,
                        self.edge_margin,
                        self.screen_width - self.edge_margin,
                    )
                )
                target_y = float(
                    np.clip(
                        normalized_y * self.screen_height,
                        self.edge_margin,
                        self.screen_height - self.edge_margin,
                    )
                )
                self.mouse_x += (target_x - self.mouse_x) * self.smoothing
                self.mouse_y += (target_y - self.mouse_y) * self.smoothing
                pyautogui.moveTo(int(self.mouse_x), int(self.mouse_y))

                thumb_tip = hand_landmarks[4]
                middle_tip = hand_landmarks[12]
                pinky_tip = hand_landmarks[20]
                current_time = time.monotonic()
                left_gesture_active = self.is_pinching(thumb_tip, middle_tip)
                right_gesture_active = self.is_pinching(thumb_tip, pinky_tip) and not left_gesture_active

                if left_gesture_active:
                    if not self.left_button_down:
                        pyautogui.mouseDown(button="left")
                        self.left_button_down = True
                else:
                    self.release_left_button()

                if (
                    right_gesture_active
                    and not self.right_gesture_active
                    and current_time - self.last_click_time > self.click_cooldown
                ):
                    pyautogui.click(button="right")
                    self.last_click_time = current_time
                self.right_gesture_active = right_gesture_active
            else:
                self.release_left_button()
                self.right_gesture_active = False
        else:
            self.release_left_button()
            self.right_gesture_active = False
            self.mode_toggle_armed = False

        cv2.putText(
            frame,
            "",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Comandos: ATIVOS" if self.commands_enabled else "Comandos: PAUSADOS",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if self.commands_enabled else (0, 0, 255),
            2,
        )
        return frame

    def open_camera(self):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                return cap
            cap.release()

        raise RuntimeError("Nao foi possivel abrir a camera.")

    def run(self):
        cap = self.open_camera()

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                timestamp_ms = time.monotonic_ns() // 1_000_000
                frame = self.control_mouse(frame, timestamp_ms)

                cv2.imshow("Controle de Mouse com a Mao", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.release_left_button()
            cap.release()
            self.hand_landmarker.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    MouseController().run()
