# importing modules
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

app = None

# creating class for window
class Window(QMainWindow):
  def __init__(self, on_save_callback):
    super().__init__()

    self.on_save_callback = on_save_callback

    title = "Paint"

    self.scale = 10
    self.mode = 1

    self.output_size = 28
    size = self.output_size * self.scale

    self.setWindowTitle(title)
    self.setGeometry(size, size, size, size)
    self.image = QImage(self.size(), QImage.Format_RGB32)
    self.image.fill(Qt.black)

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

    color = QBrush(Qt.white) if self.mode == 1 else QBrush(Qt.black)

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
    scaled_image = self.image.scaled(self.output_size, self.output_size)
    scaled_image.save('/tmp/paint2.png', "PNG")
    self.on_save_callback('/tmp/paint2.png')

def launch_paint_loop(on_save_callback):
  app = QApplication(sys.argv)
  window = Window(on_save_callback)
  window.show()
  sys.exit(app.exec())


if __name__ == "__main__":
  def noop():
    pass
  launch_paint_loop(noop)
