from PyQt5 import QtWidgets, QtCore, QtGui
import sys

from action_controller.action_controller import ActionClassificationResult
from pointing.pointer_controller import PointerMode


class OverlayWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Fullscreen window flags: transparent, frameless, always-on-top, click-through
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.Tool
            | QtCore.Qt.WindowTransparentForInput
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # Show fullscreen on the primary screen
        screen_geometry = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(screen_geometry)

        self.action_text = ""
        self.last_action = None
        self.action_color = (255, 0, 0)
        self.action_font = QtGui.QFont("Arial", 24, QtGui.QFont.Bold)

        self.instruction_text = ""
        self.instruction_font = QtGui.QFont("Arial", 32, QtGui.QFont.Bold)
        self.instruction_color = (255, 255, 255)

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
            text = f"{action_result.action.value.upper()} ({action_result.gesture} / {action_result.hand.value})"

            if action_result.swipe_distance is not None:
                text += (
                    f" - swipe distance: {str(abs(round(action_result.swipe_distance, 2)))}/"
                    f"{action_result.min_swipe_distance}"
                )
            elif action_result.count is not None:
                text += f" - count: {action_result.count}/{action_result.min_count}"

            self.action_text = text

        if action_result.triggered:
            self.action_color = (0, 255, 0)

    def update_pointer(self, pointer_pos: tuple[int, int] | None, mode: PointerMode):
        self.pointer_pos = pointer_pos
        self.pointer_mode = mode

    def update_instruction(self, instruction_text: str):
        self.instruction_text = instruction_text

    def update_pointing_target(self, target: (int, int)):
        self.pointing_target = target

    def paintEvent(self, event):
        self.__draw_action_result()
        self.__draw_instruction()
        self.__draw_pointing_target()
        self.__draw_pointer()

    def __draw_action_result(self):
        if not self.action_text:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        margin = 20
        padding = 10

        fm = QtGui.QFontMetrics(self.action_font)
        # Calculate text size (width & height)
        text_rect = fm.boundingRect(self.action_text)

        # Create background rectangle with padding + margin from top-left screen corner
        rect = QtCore.QRect(
            margin,
            margin,
            text_rect.width() + 2 * padding,
            text_rect.height() + 2 * padding,
        )

        # Draw translucent background box
        painter.setBrush(QtGui.QColor(0, 0, 0, 120))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(rect, 10, 10)

        # Draw text inside the box
        painter.setPen(QtGui.QColor(*self.action_color))
        painter.setFont(self.action_font)
        painter.drawText(
            rect.adjusted(padding, padding, -padding, -padding),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            self.action_text,
        )

    def __draw_instruction(self):
        if not self.instruction_text:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        padding = 20
        fm = QtGui.QFontMetrics(self.instruction_font)
        txt_rect = fm.boundingRect(self.instruction_text)

        # centred rectangle
        rect = QtCore.QRect(
            (self.width() - txt_rect.width() - 2 * padding) // 2,
            (self.height() - txt_rect.height() - 2 * padding) // 2,
            txt_rect.width() + 2 * padding,
            txt_rect.height() + 2 * padding,
        )

        # background
        painter.setBrush(QtGui.QColor(0, 0, 0, 160))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(rect, 12, 12)

        # text
        painter.setPen(QtGui.QColor(*self.instruction_color))
        painter.setFont(self.instruction_font)
        painter.drawText(
            rect.adjusted(padding, padding, -padding, -padding),
            QtCore.Qt.AlignCenter,
            self.instruction_text,
        )

    def __draw_pointer(self):
        if self.pointer_pos is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        if self.pointer_mode is PointerMode.DOT:
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
