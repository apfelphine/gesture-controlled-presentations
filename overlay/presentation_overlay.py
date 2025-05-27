from PyQt5 import QtWidgets, QtCore, QtGui
import sys

from action_controller.action_controller import ActionClassificationResult


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

    def paintEvent(self, event):
        self.__draw_action_result()

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
