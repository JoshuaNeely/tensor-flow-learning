# importing modules
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys

# creating class for window
class Window(QMainWindow):
  def __init__(self):
    super().__init__()

    title = "Paint"

    self.scale = 10
    self.mode = 1

    top = 400
    left = 400
    width = 800
    height = 600

    self.setWindowTitle(title)
    self.setGeometry(top, left, width, height)
    self.image = QImage(self.size(), QImage.Format_RGB32)
    self.image.fill(Qt.white)

    mainMenu = self.menuBar()
    fileMenu = mainMenu.addMenu("File")
    saveAction = QAction("Save", self)
    saveAction.setShortcut("Ctrl + S")
    fileMenu.addAction(saveAction)
    saveAction.triggered.connect(self.save)

  # this name is baked in to the Window api...
  def paintEvent(self, event):
    painter = QPainter(self)
    painter.drawImage(self.rect(), self.image, self.image.rect())

  def _draw_pixel(self, scaled_x, scaled_y):
    painter = QPainter(self.image)
    rect = QRect(scaled_x, scaled_y, self.scale, self.scale)

    color = QBrush(Qt.black) if self.mode == 1 else QBrush(Qt.white)

    painter.fillRect(rect, color)
    painter.end()
    self.update()

  def mouseMoveEvent(self, e):
    scaled_x = int((e.x())/self.scale) * self.scale
    scaled_y = int((e.y())/self.scale) * self.scale
    self._draw_pixel(scaled_x, scaled_y)

  def mousePressEvent(self, e):
    self.mode = e.button()
    scaled_x = int(e.x()/self.scale) * self.scale
    scaled_y = int(e.y()/self.scale) * self.scale
    self._draw_pixel(scaled_x, scaled_y)

  def mouseReleaseEvent(self, e):
    self.save()

  def save(self):
    self.image.save('/tmp/paint.png', "PNG")
    print('saved')

# main method
if __name__ == "__main__":
  app = QApplication(sys.argv)
  window = Window()
  window.show()

  # looping for window
  sys.exit(app.exec())
