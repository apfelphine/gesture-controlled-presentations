import sys
from PyQt5 import QtCore, QtGui, QtWidgets


class SpotlightOverlay(QtWidgets.QWidget):
    SPOT_RADIUS = 150
    DIM_OPACITY = 0.75

    def __init__(self):
        super().__init__()

        # window is borderless, transparent, and always on top
        flags = (
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool # hide task-bar icon
        )
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)  # let clicks pass through

        # full screen
        screen = QtWidgets.QApplication.primaryScreen()
        self.setGeometry(screen.geometry())

        self.setAutoFillBackground(False)
        self.show()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """
            dim whole screen except for a circle in the center
        """
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        dim_color = QtGui.QColor(0, 0, 0, int(255 * self.DIM_OPACITY))
        painter.fillRect(self.rect(), dim_color)

        centre = self.rect().center()
        spotlight_path = QtGui.QPainterPath()
        spotlight_path.addEllipse(centre, self.SPOT_RADIUS, self.SPOT_RADIUS)

        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
        painter.fillPath(spotlight_path, QtCore.Qt.transparent)

        painter.end()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)

    QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), None, app.quit)

    overlay = SpotlightOverlay()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()