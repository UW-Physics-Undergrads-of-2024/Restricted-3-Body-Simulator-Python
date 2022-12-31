from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
import PyQt6.QtGui as QtG
import matplotlib.pyplot as plt
from sys import argv


class MainWindow(QMainWindow):
    def __int__(self) -> None:
        """
        Creates the main window of the application

        :return: None
        """
        super().__init__()

        # Set basic dimensions and attributes
        self.setWindowTitle("SPARC Visualizer")
        self.setGeometry(100, 100, 600, 600)
        self.centralWidget = QWidget()
        palette = self.centralWidget.palette()
        palette.setColor(self.centralWidget.backgroundRole(), QtG.QColor(0, 0, 139, 255))
        self.centralWidget.setPalette(palette)


if __name__ == '__main__':
    app = QApplication(argv)
    window = MainWindow()
    window.show()
    app.exec()
