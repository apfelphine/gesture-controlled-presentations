import enum

from PyQt5 import QtWidgets, QtCore, QtGui
import mediapipe as mp

from action_controller.action_controller import ActionClassificationResult
from pointing.pointer_controller import PointerMode, PointerState


class OverlayWindow(QtWidgets.QWidget):
    _CALIB_CORNERS = [
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
    ]
    _CORNER_DELAY_MS = 2000

    _ACTION_TRANSLATION_DICT = {
        "point": "Laserpointer",
        "next": "Nächste Folie",
        "prev": "Vorherige Folie"
    }
    _HAND_TRANSLATION_DICT = {
        "right": "Rechts",
        "left": "Links"
    }

    class ActionState(str, enum.Enum):
        NONE = "none"
        RECOGNIZED = "recognized"
        TRIGGERED = "triggered"

    _COLOR_SCHEMES = [
        # 🎨 Cool Tones (ruhig, modern)
        {
            ActionState.NONE: (66, 135, 245),  # Cool Blue
            ActionState.RECOGNIZED: (54, 209, 153),  # Aqua Green
            ActionState.TRIGGERED: (175, 255, 87),  # Lime Yellow
        },
        # 🌅 Sunset Palette (warm, organisch)
        {
            ActionState.NONE: (255, 94, 87),  # Coral Red
            ActionState.RECOGNIZED: (255, 165, 89),  # Soft Orange
            ActionState.TRIGGERED: (255, 221, 89),  # Golden Yellow
        },
        # 🧊 Monochrome Blue Gradient (minimalistisch)
        {
            ActionState.NONE: (75, 123, 236),  # Medium Blue
            ActionState.RECOGNIZED: (70, 130, 180),  # Steel Blue (gut sichtbar)
            ActionState.TRIGGERED: (153, 199, 255),  # Light Blue
        },
        # 🔋 High Contrast (barrierefrei, gut sichtbar)
        {
            ActionState.NONE: (128, 170, 255),  # Hellblau
            ActionState.RECOGNIZED: (255, 255, 0),  # Leuchtgelb
            ActionState.TRIGGERED: (0, 255, 0),  # Neongrün
        },
        # 🌈 Playful Vibrant (lebendig, verspielt)
        {
            ActionState.NONE: (255, 64, 129),  # Pink
            ActionState.RECOGNIZED: (255, 196, 0),  # Amber
            ActionState.TRIGGERED: (0, 230, 118),  # Spring Green
        },
    ]

    def __init__(self):
        super().__init__()

        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.Tool
            | QtCore.Qt.WindowTransparentForInput
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        screen_geometry = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(screen_geometry)

        self._action_color_scheme = self._COLOR_SCHEMES[3]
        self.action_font = QtGui.QFont("Arial", 18, QtGui.QFont.Bold)
        self.action_text = ""

        self._last_action_hand = None
        self._last_gesture_detection_result = None
        self._action_state = self.ActionState.NONE

        self.instruction_text = ""
        self.previous_instruction_text = ""
        self.instruction_font = QtGui.QFont("Arial", 32, QtGui.QFont.Bold)
        self.instruction_color = (255, 255, 255)
        self.keep_instruction_visible = True
        self.progress = 0

        self.pointer_pos = None
        self.pointer_mode = PointerMode.DOT
        self.pointing_target = None

        self.show()

    @staticmethod
    def sort_gesture_recognizer_result_by_min_x(
            result: mp.tasks.vision.GestureRecognizerResult
    ) -> mp.tasks.vision.GestureRecognizerResult:
        # Extract min x for each hand
        min_x_with_index = []
        for i, landmarks in enumerate(result.hand_landmarks):
            xs = [lm.x for lm in landmarks]
            min_x = min(xs)
            min_x_with_index.append((min_x, i))

        # Sort by min x ascending
        min_x_with_index.sort(key=lambda x: x[0], reverse=True)

        # Reorder hand_landmarks and handedness
        sorted_hand_landmarks = [result.hand_landmarks[i] for _, i in min_x_with_index]
        sorted_handedness = [result.handedness[i] for _, i in min_x_with_index]

        # Similarly, you might want to reorder gestures and hand_world_landmarks if needed
        sorted_gestures = [result.gestures[i] for _, i in min_x_with_index] if hasattr(result, "gestures") else []
        sorted_hand_world_landmarks = [result.hand_world_landmarks[i] for _, i in min_x_with_index] if hasattr(result,
                                                                                                               "hand_world_landmarks") else []

        # Create new result object (assuming you can instantiate like this)
        sorted_result = mp.tasks.vision.GestureRecognizerResult(
            gestures=sorted_gestures,
            hand_landmarks=sorted_hand_landmarks,
            hand_world_landmarks=sorted_hand_world_landmarks,
            handedness=sorted_handedness
        )

        return sorted_result

    def update_action_text(self, gesture_detection_result: mp.tasks.vision.GestureRecognizerResult,
                           action_result: ActionClassificationResult):
        gesture_detection_result = self.sort_gesture_recognizer_result_by_min_x(gesture_detection_result)
        self._last_gesture_detection_result = gesture_detection_result

        gesture = None
        hands = [
            self._HAND_TRANSLATION_DICT.get(h[0].category_name.lower(), "Unbekannt")
            for h in gesture_detection_result.handedness
        ]

        if len(hands) == 0:
            hand = "Keine"
        else:
            hand = " und ".join(hands)
            gesture = "Keine"

        self._last_action_hand = None

        if action_result.action is not None:
            gesture = self._ACTION_TRANSLATION_DICT.get(action_result.action.value, "Unbekannt")
            self._last_action_hand = action_result.hand.value.lower()
            gesture += (
                f" ({str(abs(round(action_result.trigger_value, 2)))}/"
                f"{action_result.trigger_threshold})"
            )

            if action_result.triggered:
                self._action_state = self.ActionState.TRIGGERED
            elif self._action_state == self.ActionState.NONE:
                self._action_state = self.ActionState.RECOGNIZED
        else:
            self._action_state = self.ActionState.NONE
        if hand is not None:
            text = f"Erkannte Hand: {hand}"
            if gesture is not None:
                text += f"\nErkannte Geste: {gesture}"
            self.action_text = text

    def update_pointer(self, pointer_pos: tuple[int, int] | None, mode: PointerMode):
        self.pointer_pos = pointer_pos
        self.pointer_mode = mode

    def update_instruction(self, instruction_text: str, pointing_controller, progress=None):
        if progress:
            self.progress = progress
        if instruction_text in self._CALIB_CORNERS:
            if instruction_text != self.previous_instruction_text:
                self.progress = 0
                self.previous_instruction_text = instruction_text
                if instruction_text != self._CALIB_CORNERS[0]:
                    self.__set_instruction("corner completed", color=(0, 255, 0))
                    pointing_controller.state = PointerState.IDLE
                    QtCore.QTimer.singleShot(
                        self._CORNER_DELAY_MS,
                        lambda: (
                            pointing_controller.set_state(PointerState.CALIBRATING),
                            self.__set_instruction(instruction_text)
                        )
                    )
                else:
                    self.__set_instruction(instruction_text)
        else:
            if instruction_text == 'complete':
                self.keep_instruction_visible = False
            self.__set_instruction(instruction_text)

    def update_pointing_target(self, target: tuple[int, int]):
        self.pointing_target = target

    def paintEvent(self, event):
        self.__draw_action_result()
        self.__draw_hand_skeleton()

        self.__draw_instruction()
        self.__draw_pointing_target()
        self.__draw_pointer()

    def __set_instruction(self, text: str, color: tuple[int, int] | None = None):
        if not text and self.keep_instruction_visible:
            return
        self.instruction_text = f"Point at {text}" if text in self._CALIB_CORNERS else text
        if color is not None:
            self.instruction_color = color
        else:
            # Autoset colour: green for final "complete", white otherwise
            self.instruction_color = (0, 255, 0) if text == "complete" else (255, 255, 255)
        self.update()

    def __draw_action_result(self):
        if not self.action_text:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        margin, padding = 20, 10
        painter.setFont(self.action_font)
        fm = QtGui.QFontMetrics(self.action_font)

        rect_txt = fm.boundingRect(0, 0, self.width() - 2 * margin, 1000,  # large height to allow for multiple lines
                                   QtCore.Qt.TextWordWrap, self.action_text)
        rect_bg = QtCore.QRect(margin, margin,
                               rect_txt.width() + 2 * padding,
                               rect_txt.height() + 2 * padding)

        painter.setBrush(QtGui.QColor(0, 0, 0, 120))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(rect_bg, 10, 10)

        painter.setPen(QtGui.QColor(*self._action_color_scheme[self._action_state]))
        painter.drawText(rect_bg.adjusted(padding, padding, -padding, -padding),
                         QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop | QtCore.Qt.TextWordWrap,
                         self.action_text)

    def __draw_instruction(self):
        if not self.instruction_text:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 80))

        target = self.instruction_text[len("Point at "):] if self.instruction_text.startswith("Point at ") else ""
        show_corner = target in self._CALIB_CORNERS
        if show_corner:
            painter.setBrush(QtGui.QColor(255, 255, 0, 220))
            painter.setPen(QtCore.Qt.NoPen)
            x_margin, y_margin, r = 100, 100, 50
            w, h = self.width(), self.height()
            match target:
                case "top-left":
                    pos = (x_margin, y_margin)
                case "top-right":
                    pos = (w - x_margin, y_margin)
                case "bottom-left":
                    pos = (x_margin, h - y_margin)
                case "bottom-right":
                    pos = (w - x_margin, h - y_margin)
                case _:
                    pos = (w // 2, h // 2)
            painter.drawEllipse(QtCore.QPoint(*pos), r, r)

        padding = 20
        fm = QtGui.QFontMetrics(self.instruction_font)
        rect_txt = fm.boundingRect(self.instruction_text)
        rect_bg = QtCore.QRect((self.width() - rect_txt.width() - 2 * padding) // 2,
                               (self.height() - rect_txt.height() - 2 * padding) // 2,
                               rect_txt.width() + 2 * padding,
                               rect_txt.height() + 2 * padding)
        painter.setBrush(QtGui.QColor(0, 0, 0, 180))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(rect_bg, 12, 12)
        painter.setPen(QtGui.QColor(*self.instruction_color))
        painter.setFont(self.instruction_font)
        painter.drawText(rect_bg.adjusted(padding, padding, -padding, -padding),
                         QtCore.Qt.AlignCenter, self.instruction_text)

        if show_corner:
            bar_w, bar_h = 440, 56
            bar_x = (self.width() - bar_w) // 2
            bar_y = rect_bg.bottom() + 16
            painter.setBrush(QtGui.QColor(255, 255, 255, 60))
            painter.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 6, 6)
            painter.setBrush(QtGui.QColor(0, 180, 0, 230))
            painter.drawRoundedRect(bar_x, bar_y,
                                    int(bar_w * self.progress / 100), bar_h, 6, 6)

    def __draw_pointer(self):
        if self.pointer_pos is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        if self.pointer_mode == PointerMode.DOT:
            r = 12
            painter.setBrush(QtGui.QColor(255, 0, 0, 200))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(QtCore.QPoint(*self.pointer_pos), r, r)
        else:
            grad = QtGui.QRadialGradient(QtCore.QPointF(*self.pointer_pos), 120)
            grad.setColorAt(0, QtGui.QColor(255, 255, 255, 0))
            grad.setColorAt(1, QtGui.QColor(0, 0, 0, 180))
            painter.setBrush(QtGui.QBrush(grad))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawRect(self.rect())

    def __draw_pointing_target(self):
        if self.pointing_target is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        r = 36
        painter.setBrush(QtGui.QColor(0, 0, 255, 200))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(QtCore.QPoint(*self.pointing_target), r, r)

    def __draw_hand_skeleton(self):
        if not self._last_gesture_detection_result or not self._last_gesture_detection_result.hand_landmarks:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        target_wrist_height = 30
        hand_spacing = 50
        offset_y = 325
        current_x = 50
        connections = mp.solutions.hands.HAND_CONNECTIONS

        def distance3D(a, b):
            return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5

        for idx, landmarks in enumerate(self._last_gesture_detection_result.hand_landmarks):
            # Mirror x-coordinates
            mirrored_landmarks = [
                type(lm)(x=1.0 - lm.x, y=lm.y, z=lm.z) for lm in landmarks
            ]

            # Reference bone scaling
            lm0, lm1 = mirrored_landmarks[0], mirrored_landmarks[1]
            wrist_height_norm = distance3D(lm0, lm1)
            if wrist_height_norm == 0:
                print("Skipping landmarks due to invalid wrist height...", landmarks)
                continue

            scale = target_wrist_height / wrist_height_norm

            # Compute bounds and offsets
            min_x = min(lm.x for lm in mirrored_landmarks)
            min_y = min(lm.y for lm in mirrored_landmarks)
            wrist_offset_y = (lm0.y - min_y) * scale

            bg_pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 120), 20, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
            painter.setPen(bg_pen)
            for start_idx, end_idx in connections:
                start = mirrored_landmarks[start_idx]
                end = mirrored_landmarks[end_idx]

                sx = (start.x - min_x) * scale
                sy = (start.y - min_y) * scale
                ex = (end.x - min_x) * scale
                ey = (end.y - min_y) * scale

                pt_start = QtCore.QPointF(current_x + sx, offset_y + sy - wrist_offset_y)
                pt_end = QtCore.QPointF(current_x + ex, offset_y + ey - wrist_offset_y)

                # Background line
                painter.drawLine(pt_start, pt_end)

            bg_brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 120))
            painter.setBrush(bg_brush)
            painter.setPen(QtCore.Qt.NoPen)

            for lm in mirrored_landmarks:
                cx = current_x + (lm.x - min_x) * scale
                cy = offset_y + (lm.y - min_y) * scale - wrist_offset_y

                # Draw background circle (e.g., radius 7, semi-transparent black)

                painter.drawEllipse(QtCore.QPointF(cx, cy), 14, 14)

            bone_pen = QtGui.QPen(QtGui.QColor(224, 224, 224), 3, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
            painter.setPen(bone_pen)

            for start_idx, end_idx in connections:
                start = mirrored_landmarks[start_idx]
                end = mirrored_landmarks[end_idx]

                sx = (start.x - min_x) * scale
                sy = (start.y - min_y) * scale
                ex = (end.x - min_x) * scale
                ey = (end.y - min_y) * scale

                pt_start = QtCore.QPointF(current_x + sx, offset_y + sy - wrist_offset_y)
                pt_end = QtCore.QPointF(current_x + ex, offset_y + ey - wrist_offset_y)

                painter.drawLine(pt_start, pt_end)

            # Draw joints
            hand = self._last_gesture_detection_result.handedness[idx][0].category_name.lower()
            active_hand = self._last_action_hand == hand
            joint_brush = QtGui.QBrush(
                QtGui.QColor(*self._action_color_scheme[self._action_state])
                if active_hand else
                QtGui.QColor(*self._action_color_scheme[self.ActionState.NONE])
            )
            joint_pen = QtGui.QPen(QtGui.QColor(224, 224, 224), 2)
            painter.setBrush(joint_brush)
            painter.setPen(joint_pen)

            for lm in mirrored_landmarks:
                cx = current_x + (lm.x - min_x) * scale
                cy = offset_y + (lm.y - min_y) * scale - wrist_offset_y

                # Draw foreground circle (actual joint)
                painter.setBrush(joint_brush)
                painter.setPen(joint_pen)
                painter.drawEllipse(QtCore.QPointF(cx, cy), 5, 5)

            # Advance for next hand
            hand_width_px = (max(lm.x for lm in mirrored_landmarks) - min_x) * scale
            current_x += hand_width_px + hand_spacing


class OverlayContextManager:

    def __init__(self):
        self.app = QtWidgets.QApplication.instance()
        self.created_app = False
        if self.app is None:
            self.app = QtWidgets.QApplication([])
            self.created_app = True
        self.overlay = OverlayWindow()

    def __enter__(self):
        return self.overlay

    def __exit__(self, exc_type, exc_value, traceback):
        self.overlay.close()
        if self.created_app:
            self.app.quit()
