# -*- coding: utf-8 -*-

import cv2
import msgpack
import numpy as np
import os
import sys
import threading
import time

from PyQt4.QtGui import *
from PyQt4.QtCore import *

import capture
import timer
import util
import vision


RUN = 0
STOP = 1
PAUSE = 2

TRACKING = 0
CALIBRATING = 1

_ORIGNAL_WIDTH = 480
_CROP_SIZE = 24  # Divisible by 8.
_BLUR = 1
_LARGEST_BOARDSIZE = 19

_LAYOUT_PARAM_OFFSET = 360
_LAYOUT_PARAM_HEIGHT = 20
_LAYOUT_LABEL_WIDTH = 100
_LAYOUT_MARGIN = 5


def LoadModel(filename):
  unpacker = msgpack.Unpacker(open(filename, 'rb'))
  coef, intercept = tuple(unpacker)
  coef = np.array(coef, dtype=np.float32)
  return coef, intercept


class MainWindow(QDialog):
  def __init__(self, app, parent=None):
    super(MainWindow, self).__init__(parent)
    self._app = app

    # Try to load the detection models.
    if os.path.isfile('models/empty_full.bin') and os.path.isfile('models/black_white.bin'):
      self._empty_full_model = LoadModel('models/empty_full.bin')
      self._black_white_model = LoadModel('models/black_white.bin')
      print 'Loaded detection models.'

    # Display images.
    self._display_step_lock = threading.Lock()
    self._display_step = vision.CALIBRATION_ORIGINAL
    self._display_image_lock = threading.Lock()
    self._display_image_left = None
    self._display_image_right = None
    self._boardsize = None
    self._processed_image = None

    # Vision processing.
    self._vision = vision.Vision(debug=True)

    # Sliders.
    self._param_offset = _LAYOUT_PARAM_OFFSET
    self._params = {}
    self._params_lock = threading.Lock()
    # These parameters are the same as the ones in vision.Vision.
    self.AddParameter('blur', min_value=1, max_value=11, step_size=2, default_value=3, step=vision.CALIBRATION_BLUR)
    self.AddParameter('close', min_value=3, max_value=19, step_size=2, default_value=11, step=vision.CALIBRATION_DIV)
    self.AddParameter('th_block', min_value=11, max_value=27, step_size=2, default_value=11, step=vision.CALIBRATION_THRESHOLD)
    self.AddParameter('th_mean', min_value=0, max_value=50, step_size=1, default_value=3, step=vision.CALIBRATION_THRESHOLD)
    self.AddParameter('ctr_size', min_value=10, max_value=300, step_size=10, default_value=200, step=vision.CALIBRATION_CONTOUR)
    self.AddParameter('approx', min_value=0, max_value=1, step_size=0.01, default_value=0.01, step=vision.CALIBRATION_CONTOUR)
    self.AddParameter('vote', min_value=20, max_value=200, step_size=10, default_value=50, step=vision.CALIBRATION_VOTE)
    self.AddParameter('match_k', min_value=1, max_value=5, step_size=1, default_value=2, step=vision.TRACKING_MATCHES)
    self.AddParameter('match_r', min_value=4, max_value=100, step_size=2, default_value=10, step=vision.TRACKING_MATCHES)
    self.AddParameter('ransac_th', min_value=1, max_value=10, step_size=1, default_value=5, step=vision.TRACKING_MATCHES)

    # Buttons (these are invisible originally).
    self._dataset_lock = threading.Lock()
    self._dataset = []
    self._groundtruth_lock = threading.Lock()
    self._groundtruth = np.zeros((_LARGEST_BOARDSIZE, _LARGEST_BOARDSIZE), dtype=np.int8)
    self._button_size = int(vision.SQUARE_SIZE * 0.7)
    self._buttons = []
    for i in range(_LARGEST_BOARDSIZE):
      self._buttons.append([])
      x = vision.MARGIN + i * vision.SQUARE_SIZE - self._button_size / 2
      for j in range(_LARGEST_BOARDSIZE):
        y = vision.MARGIN + j * vision.SQUARE_SIZE - self._button_size / 2
        button = QPushButton('', self)
        button.move(x + _ORIGNAL_WIDTH, y)
        button.clicked.connect(self.GetToggleFunction(i, j))
        button.setFixedSize(self._button_size, self._button_size)
        button.setStyleSheet('background-color: rgba(255, 255, 255, 10); '
                             'border: 1px solid rgb(150, 150, 150); '
                             'border-radius: %dpx; ' % (self._button_size / 2))
        button.hide()
        self._buttons[-1].append(button)
    # Timing information.
    self._timings_label = QLabel('', self)
    self._timings_label.move(_LAYOUT_MARGIN, _LAYOUT_MARGIN)
    self._timings_label.setFixedSize(_ORIGNAL_WIDTH - 2 * _LAYOUT_MARGIN, _LAYOUT_PARAM_OFFSET - 2 * _LAYOUT_MARGIN - _LAYOUT_PARAM_HEIGHT)
    self._timings_label.setStyleSheet('color: green')
    self._timings_label.setAlignment(Qt.AlignRight | Qt.AlignTop)

    # Capturing thread.
    self._capture = capture.VideoStream().Start()
    self._timer = timer.Timer()

    # Processing thread.
    self._status_lock = threading.Lock()
    self._status = RUN
    self._mode_lock = threading.Lock()
    self._mode = CALIBRATING
    self._display_groundtruth_lock = threading.Lock()
    self._display_groundtruth = False
    self._must_reprocess_lock = threading.Lock()
    self._must_reprocess = False
    self._flip_lock = threading.Lock()
    self._flip = False
    threading.Thread(target=self.Process, args=()).start()

  def Detect(self, img, i, j):
    def CropLocalRegion(img, i, j):
      x = vision.MARGIN + i * vision.SQUARE_SIZE - _CROP_SIZE / 2
      y = vision.MARGIN + j * vision.SQUARE_SIZE - _CROP_SIZE / 2
      return img[y:y + _CROP_SIZE, x:x + _CROP_SIZE]

    def ComputeHoG(img):
      gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
      gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
      mag, ang = cv2.cartToPolar(gx, gy)
      bin_n = 16  # Number of orientation bins.
      bin = np.int32(bin_n * ang / (2 * np.pi))
      bin_cells = []
      mag_cells = []
      cellx = celly = 8
      for i in range(img.shape[0] / celly):
        for j in range(img.shape[1] / cellx):
          bin_cells.append(bin[i * celly:i * celly + celly, j * cellx:j * cellx + cellx])
          mag_cells.append(mag[i * celly:i * celly + celly, j * cellx:j * cellx + cellx])
      hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
      hist = np.hstack(hists)
      # Transform to Hellinger kernel.
      eps = 1e-7
      hist /= hist.sum() + eps
      hist = np.sqrt(hist)
      hist /= np.linalg.norm(hist) + eps
      return hist.astype(np.float32, copy=False)

    hog = ComputeHoG(CropLocalRegion(img, i, j))
    if self._empty_full_model[0].dot(hog) + self._empty_full_model[1] < 0.:
      return 0  # Empty.
    if self._black_white_model[0].dot(hog) + self._black_white_model[1] < 0.:
      return 1  # Black.
    return 2  # White.

  def AddParameter(self, param_name, min_value, max_value, step_size, default_value, step):
    self._vision.SetParameter(param_name, default_value)
    label = QLabel('%s: %g' % (param_name, default_value), self)
    label.move(_LAYOUT_MARGIN, self._param_offset)
    label.setFixedSize(_LAYOUT_LABEL_WIDTH, _LAYOUT_PARAM_HEIGHT)
    slider = QSlider(Qt.Horizontal, self)
    slider.move(_LAYOUT_LABEL_WIDTH + _LAYOUT_MARGIN, self._param_offset)
    slider.setFixedSize(_ORIGNAL_WIDTH - _LAYOUT_LABEL_WIDTH - 2 * _LAYOUT_MARGIN, _LAYOUT_PARAM_HEIGHT)
    slider.setMinimum(0)
    slider.setMaximum(int((max_value - min_value) / step_size))
    slider.setValue(int((default_value - min_value) / step_size))
    slider.valueChanged.connect(self.GetParameterChangeFunction(slider, param_name, label, min_value, step_size))
    slider.sliderPressed.connect(self.GetParameterPressedFunction(step))
    slider.sliderReleased.connect(self.GetParameterReleasedFunction())
    self._param_offset += _LAYOUT_PARAM_HEIGHT + _LAYOUT_MARGIN

  def GetParameterChangeFunction(self, slider, param_name, label, min_value, step_size):
    def ChangeParameter():
      with self._params_lock:
        value = slider.value() * step_size + min_value
        self._vision.SetParameter(param_name, value)
      label.setText('%s: %g' % (param_name, value))
      with self._must_reprocess_lock:
        self._must_reprocess = True
    return ChangeParameter

  def GetParameterPressedFunction(self, step):
    def PressedParameter():
      with self._display_step_lock:
        self._display_step = step
      with self._must_reprocess_lock:
        self._must_reprocess = True
    return PressedParameter

  def GetParameterReleasedFunction(self):
    def ReleasedParameter():
      with self._display_step_lock:
        self._display_step = vision.CALIBRATION_OR_TRACKING_ORIGINAL
      with self._must_reprocess_lock:
        self._must_reprocess = True
    return ReleasedParameter

  def GetToggleFunction(self, i, j):
    def ToggleGroundtruth():
      with self._display_image_lock:
        boardsize = self._boardsize
      with self._groundtruth_lock:
        status = (self._groundtruth[i, j] + 1) % 3
        self._groundtruth[i, j] = status
        UpdateButton(i, j, self.sender(), boardsize)
    return ToggleGroundtruth

  # Hold locks!
  def UpdateButton(self, i, j, button, boardsize):
    if i >= self._boardsize or j >= self._boardsize:
      button.hide()
      return
    status = self._groundtruth[i, j]
    color = ((255, 255, 255, 10),
             (0, 0, 0, 200),
             (255, 255, 255, 200))[status]
    border = ((150, 150, 150),
              (255, 255, 255),
              (0, 0, 0))[status]
    button.setStyleSheet('background-color: rgba(%d, %d, %d, %d); '
                         'border: 1px solid rgb(%d, %d, %d); '
                         'border-radius: %dpx; ' % (color[0], color[1], color[2], color[3],
                                                    border[0], border[1], border[2],
                                                    self._button_size / 2))

  # Must hold self._display_image_lock.
  def UpdateGroundtruthButtons(self):
    with self._display_groundtruth_lock:
      if self._display_groundtruth:
        with self._groundtruth_lock:
          for i in range(_LARGEST_BOARDSIZE):
            for j in range(_LARGEST_BOARDSIZE):
              if i < self._boardsize and j < self._boardsize:
                self.UpdateButton(i, j, self._buttons[i][j], self._boardsize)
                self._buttons[i][j].show()
              else:
                self._buttons[i][j].hide()
      else:
        for i in range(_LARGEST_BOARDSIZE):
          for j in range(_LARGEST_BOARDSIZE):
            self._buttons[i][j].hide()

  def ConvertToQImage(self, img, width=None, height=None):
    img = util.ResizeImage(img, width, height)
    if len(img.shape) == 2:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, num_bytes = img.shape
    return QImage(img, width, height, num_bytes * width, QImage.Format_RGB888)

  def Process(self):
    with self._status_lock:
      status = self._status
    while status != STOP:
      with self._must_reprocess_lock:
        must_reprocess = self._must_reprocess
        self._must_reprocess = False
      if status != PAUSE or must_reprocess:
        img, is_old = self._capture.Read()
        if not is_old or must_reprocess:
          with self._flip_lock:
            if self._flip:
              img = cv2.flip(img, 0)
          with self._mode_lock:
            if self._mode == CALIBRATING:
              self._vision.Calibrate(img)
              with self._display_step_lock:
                right_image, boardsize, left_image, timings = self._vision.GetCalibrationResult(self._display_step)
            elif self._mode == TRACKING:
              self._vision.Track(img)
              with self._display_step_lock:
                right_image, left_image, timings = self._vision.GetTrackingResult(self._display_step)
          with self._display_image_lock:
            self._display_image_left = self.ConvertToQImage(left_image, width=480) if left_image is not None else None
            self._display_image_right = self.ConvertToQImage(right_image) if right_image is not None else None
            self._processed_image = right_image
            self._boardsize = boardsize
          # Update timings.
          self._timings_label.setText('\n'.join(('%s: %d [ms]' % (k, int(v))) for k, v in timings))
          self.update()
      time.sleep(0.01)
      with self._status_lock:
        status = self._status

  def paintEvent(self, QPaintEvent):
    painter = QPainter()
    painter.begin(self)
    with self._display_image_lock:
      self.UpdateGroundtruthButtons()
      if self._display_image_left is not None:
        painter.drawImage(0, 0, self._display_image_left)
      if self._display_image_right is not None:
        painter.drawImage(480, 0, self._display_image_right)
    painter.end()

  def Quit(self):
    with self._dataset_lock:
      print 'Writing dataset...'
      if self._dataset:
        with open('data/data-%s.bin' % time.strftime('%Y%m%d-%H%M%S'), 'wb') as fp:
          for img, gt in self._dataset:
            fp.write(msgpack.packb((img.tolist(), gt.tolist())))
    with self._status_lock:
      self._status = STOP
    print 'Stopping capture...'
    self._capture.Stop()
    self._vision.Stop()
    print 'Quitting...'

  def keyPressEvent(self, QKeyEvent):
    super(MainWindow, self).keyPressEvent(QKeyEvent)
    if 'q' == QKeyEvent.text():
      self.Quit()
      self._app.exit(1)
    if 'f' == QKeyEvent.text():
      with self._flip_lock:
        self._flip = not self._flip
    elif ' ' == QKeyEvent.text():
      with self._status_lock:
        self._status = PAUSE if self._status == RUN else RUN
        if self._status == PAUSE:
          self._capture.Pause()
        else:
          self._capture.Continue()
        # Show groundtruth buttons.
        with self._display_image_lock:
          show_groundtruth = self._status == PAUSE and self._boardsize
          with self._display_groundtruth_lock:
            self._display_groundtruth = show_groundtruth
          self.UpdateGroundtruthButtons()
    elif 'd' == QKeyEvent.text():
      with self._display_image_lock:
        with self._display_groundtruth_lock:
          if self._display_groundtruth:
            print 'Detecting stones...'
            img = cv2.GaussianBlur(self._processed_image, (_BLUR, _BLUR), 0)
            with self._groundtruth_lock:
              for i in range(self._boardsize):
                for j in range(self._boardsize):
                  self._groundtruth[i, j] = self.Detect(img, i, j)
          print 'Done'
        self.UpdateGroundtruthButtons()
    elif 'n' == QKeyEvent.text():
      print 'Getting next static image...'
      self._capture.Next()
      with self._must_reprocess_lock:
        self._must_reprocess = True
    elif 't' == QKeyEvent.text():
      with self._mode_lock:
        self._mode = TRACKING
      with self._must_reprocess_lock:
        self._must_reprocess = True
    elif 'c' == QKeyEvent.text():
      with self._mode_lock:
        self._mode = CALIBRATING
      with self._must_reprocess_lock:
        self._must_reprocess = True
    elif 'p' == QKeyEvent.text():
      # Do a screenshot.
      with self._status_lock:
        if self._status != PAUSE:
          print 'Cannot screenshot when not paused.'
          return
        with self._latest_input_image_lock:
          cv2.imwrite('screenshots/image-%s.jpg' % time.strftime('%Y%m%d-%H%M%S'), self._latest_input_image)
        print 'Saved image.'
    elif 's' == QKeyEvent.text():
      with self._status_lock:
        if self._status != PAUSE:
          print 'Cannot save configuration when not paused.'
          return
      with self._display_image_lock:
        if self._processed_image is None:
          print 'Board was not found. Cannot save.'
          return
        with self._groundtruth_lock:
          with self._dataset_lock:
            self._dataset.append((self._processed_image.copy(),
                                  self._groundtruth[:self._boardsize, :self._boardsize].copy()))
      print 'Saved in database.'

  def closeEvent(self, event):
    super(MainWindow, self).closeEvent(event)
    self.Quit()


if __name__ == '__main__':
  app = QApplication(sys.argv)
  w = MainWindow(app)
  w.resize(480 * 2, 640)
  w.show()
  app.exec_()
