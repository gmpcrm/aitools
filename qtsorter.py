import sys
import os
from PyQt6.QtCore import QObject, pyqtSlot, QUrl
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtQml import QQmlApplicationEngine
from PyQt6.QtWidgets import QFileDialog


class ImageSorter(QObject):
    rukiModel = []
    perchatkiModel = []
    drugoeModel = []

    def __init__(self):
        super().__init__()
        self.images = []
        self.current_image_index = 0

    @pyqtSlot()
    def loadImages(self):
        dialog = QFileDialog()
        files, _ = dialog.getOpenFileNames(
            None, "Open Images", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)"
        )
        if files:
            self.images = files
            self.current_image_index = 0
            self.showImage()

    @pyqtSlot(str)
    def sortImage(self, category):
        if self.images:
            current_image_path = self.images[self.current_image_index]
            if category == "Ruki":
                self.rukiModel.append(current_image_path)
            elif category == "Perchatki":
                self.perchatkiModel.append(current_image_path)
            elif category == "Drugoe":
                self.drugoeModel.append(current_image_path)
            self.current_image_index += 1
            if self.current_image_index < len(self.images):
                self.showImage()
            else:
                self.current_image_index = 0
                self.showImage()

    def showImage(self):
        if self.images:
            image_path = self.images[self.current_image_index]
            context.setContextProperty("currentImagePath", image_path)


app = QGuiApplication(sys.argv)
engine = QQmlApplicationEngine()

sorter = ImageSorter()
context = engine.rootContext()
context.setContextProperty("sorter", sorter)
context.setContextProperty("currentImagePath", "")

qml_file_path = os.path.join(os.path.dirname(__file__), "qsorter.qml")
engine.load(QUrl.fromLocalFile(qml_file_path))

if not engine.rootObjects():
    sys.exit(-1)

sys.exit(app.exec())
