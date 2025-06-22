from PyQt5 import QtWidgets, QtCore, QtGui
import sys

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

        self.action_text = ""
        self.last_action = None
        self.action_color = (255, 0, 0)
        self.action_font = QtGui.QFont("Arial", 24, QtGui.QFont.Bold)

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

    def update_action_result(self, action_result: ActionClassificationResult):
        if action_result.action != self.last_action:
            self.action_color = (255, 0, 0)
        self.last_action = action_result.action

        if not action_result.action:
            self.action_text = ""
            return

        if self.action_color != (0, 255, 0):
            text = f"Recognized action: {action_result.action.value.capitalize()} "
            if action_result.swipe_distance is not None:
                text += (
                    f"({str(abs(round(action_result.swipe_distance, 2)))}/"
                    f"{action_result.min_swipe_distance})"
                )
            elif action_result.count is not None:
                text += f"({action_result.count}/{action_result.min_count})"
            self.action_text = text

        if action_result.triggered:
            self.action_color = (0, 255, 0)

    def update_pointer(self, pointer_pos: tuple[int, int] | None, mode: PointerMode):
        self.pointer_pos = pointer_pos
        self.pointer_mode = mode

    def update_instruction(self, instruction_text: str, pointing_controller, progress = None):
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
        self.__draw_instruction()
        self.__draw_action_result()
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
        fm = QtGui.QFontMetrics(self.action_font)
        rect_txt = fm.boundingRect(self.action_text)
        rect_bg = QtCore.QRect(margin, margin, rect_txt.width() + 2 * padding, rect_txt.height() + 2 * padding)
        painter.setBrush(QtGui.QColor(0, 0, 0, 120))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(rect_bg, 10, 10)
        painter.setPen(QtGui.QColor(*self.action_color))
        painter.setFont(self.action_font)
        painter.drawText(rect_bg.adjusted(padding, padding, -padding, -padding), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self.action_text)

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