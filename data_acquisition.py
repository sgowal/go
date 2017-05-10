# -*- coding: utf-8 -*-

import copy
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
import vision
import util


RUN = 0
STOP = 1
PAUSE = 2

_ORIGNAL_WIDTH = 480
_CROP_SIZE = 24  # Divisible by 8.
_BLUR = 1
_LARGEST_BOARDSIZE = 19

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
    self._display_step = vision.CALIBRATION_ORIGNAL
    self._display_image_lock = threading.Lock()
    self._display_original_image = None
    self._display_result_image = None

    # Vision processing.
    self._vision = vision.Vision(debug=True)

    # Sliders.
    self._param_offset = 360
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

    # Buttons (these are invisible originally).
    self._dataset_lock = threading.Lock()
    self._dataset = []
    self._groundtruth_lock = threading.Lock()
    self._groundtruth = np.zeros((_LARGEST_BOARDSIZE, _LARGEST_BOARDSIZE), dtype=np.int8)
    self._button_size = int(_SQUARE_SIZE * 0.7)
    self._buttons = []
    for i in range(_LARGEST_BOARDSIZE):
      self._buttons.append([])
      x = _MARGIN + i * _SQUARE_SIZE - self._button_size / 2
      for j in range(_LARGEST_BOARDSIZE):
        y = _MARGIN + j * _SQUARE_SIZE - self._button_size / 2
        button = QPushButton('', self)
        button.move(x + _ORIGNAL_WIDTH, y)
        button.clicked.connect(self.GetToggleFunction(i, j))
        button.setFixedSize(self._button_size, self._button_size)
        button.setStyleSheet('background-color: rgba(255, 255, 255, 10); '
                             'border: 1px solid rgb(150, 150, 150); '
                             'border-radius: %dpx; ' % (self._button_size / 2))
        button.hide()
        self._buttons[-1].append(button)

    # Capturing thread.
    self._capture = capture.VideoStream().Start()

    # Processing thread.
    self._status_lock = threading.Lock()
    self._status = RUN
    self._must_reprocess_lock = threading.Lock()
    self._must_reprocess = False
    self._flip_lock = threading.Lock()
    self._flip = False
    threading.Thread(target=self.Process, args=()).start()

  def Detect(self, img, i, j):
    def CropLocalRegion(img, i, j):
      x = _MARGIN + i * _SQUARE_SIZE - _CROP_SIZE / 2
      y = _MARGIN + j * _SQUARE_SIZE - _CROP_SIZE / 2
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

  def AddParameter(self, param_name, min_value, max_value, step_size, default_value, step='original'):
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
        self._display_step = vision.CALIBRATION_ORIGNAL
      with self._must_reprocess_lock:
        self._must_reprocess = True
    return ReleasedParameter

  def GetToggleFunction(self, i, j):
    def ToggleGroundtruth():
      with self._groundtruth_lock:
        status = (self._groundtruth[i, j] + 1) % 3
        self._groundtruth[i, j] = status
        color = ((255, 255, 255, 10),
                 (0, 0, 0, 200),
                 (255, 255, 255, 200))[status]
        border = ((150, 150, 150),
                  (255, 255, 255),
                  (0, 0, 0))[status]
        self.sender().setStyleSheet('background-color: rgba(%d, %d, %d, %d); '
                                    'border: 1px solid rgb(%d, %d, %d); '
                                    'border-radius: %dpx; ' % (color[0], color[1], color[2], color[3],
                                                               border[0], border[1], border[2],
                                                               self._button_size / 2))
    return ToggleGroundtruth

  def UpdateGroundtruth(self, i, j, status):
    # Lock must be hold.
    self._groundtruth[i, j] = status
    color = ((255, 255, 255, 10),
             (0, 0, 0, 200),
             (255, 255, 255, 200))[status]
    border = ((150, 150, 150),
              (255, 255, 255),
              (0, 0, 0))[status]
    self._buttons[i][j].setStyleSheet('background-color: rgba(%d, %d, %d, %d); '
                                      'border: 1px solid rgb(%d, %d, %d); '
                                      'border-radius: %dpx; ' % (color[0], color[1], color[2], color[3],
                                                                 border[0], border[1], border[2],
                                                                 self._button_size / 2))

  def ConvertToQImage(self, img, width=None, height=None):
    img = util.ResizeImage(img, width, height)
    if len(img.shape) == 2:
      height, width = img.shape
      num_bytes = 1
      return QImage(img, width, height, num_bytes * width, QImage.Format_Indexed8)
    height, width, num_bytes = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
          ret = self._vision.Calibrate(img)
      time.sleep(0.01)
      with self._status_lock:
        status = self._status

  def ProcessImage(self, input_image):
    # Keep local parameter copy.
    with self._params_lock:
      params = copy.deepcopy(self._params)

    # Keep track of all images.
    pipeline = {}

    # Store.
    with self._latest_input_image_lock:
      self._latest_input_image = input_image
    pipeline['original'] = self.ResizeImage(input_image, height=720)

    # Blur + Grayscale.
    pipeline['gray'] = cv2.cvtColor(pipeline['original'], cv2.COLOR_BGR2GRAY)
    pipeline['blur'] = cv2.GaussianBlur(pipeline['gray'], (params['blur'], params['blur']), 0)

    # The following sequence removes small lines.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params['close'], params['close']))
    pipeline['close'] = cv2.morphologyEx(pipeline['blur'], cv2.MORPH_CLOSE, kernel)

    # Small lines that were removed will pop.
    div = np.float32(pipeline['blur']) / (pipeline['close'] + 1e-4)
    pipeline['div'] = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))

    # Threshold.
    output = cv2.adaptiveThreshold(pipeline['div'], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                   params['th_block'], params['th_mean'])
    pipeline['threshold'] = output.copy()

    # Find the largest contour.
    contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    all_contours = []
    best_non_approx_contour = None
    for contour in contours:
      area = cv2.contourArea(contour)
      if area > params['ctr_size'] * params['ctr_size']:
        all_contours.append(contour)
        if area > max_area:
          best_non_approx_contour = contour
          max_area = area
    # Fit a square to the best contour.
    best_contour = None
    approximated_contour = None
    if best_non_approx_contour is not None:
      hull = cv2.convexHull(best_non_approx_contour)
      perimeter = cv2.arcLength(hull, True)
      approximated_contour = cv2.approxPolyDP(hull, params['approx'] * perimeter, True)
      # Unfortunately, we still get 1 or 2 extra points sometimes.
      num_points = approximated_contour.shape[0]
      if 4 < num_points < 7:
        keep_indices = []
        for i in range(num_points):
          prev = (i + num_points - 1) % num_points
          next = (i + 1) % num_points
          v_prev = np.float32(np.squeeze(approximated_contour[prev] - approximated_contour[i]))
          v_next = np.float32(np.squeeze(approximated_contour[next] - approximated_contour[i]))
          v_prev /= np.linalg.norm(v_prev)
          v_next /= np.linalg.norm(v_next)
          if v_prev.dot(v_next) > -0.95:
            keep_indices.append(i)
        approximated_contour = approximated_contour[np.array(keep_indices)]
      if approximated_contour.shape[0] == 4:
        best_contour = approximated_contour
    # For display only.
    img = pipeline['threshold'].copy()
    pipeline['contour'] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(pipeline['contour'], all_contours, -1, (255, 0, 0), 2)
    if approximated_contour is not None:
      cv2.drawContours(pipeline['contour'], [approximated_contour], -1, (0, 0, 255), 4)
    # We found nothing.
    if best_contour is None:
      with self._display_image_lock:
        with self._display_step_lock:
          if self._display_step in ('vote',):
            img = pipeline['original']
          else:
            img = pipeline[self._display_step]
          self._display_original_image = self.ConvertToQImage(img, width=480)
        # We do not update the result image.
      self.update()
      return

    original_points = OrderPoints(np.squeeze(best_contour))

    # We detect the board size by first projecting the detected and counting the number of lines.
    destination_points = OrderPoints(np.float32([
        [0, 0], [0, _VOTE_SIZE], [_VOTE_SIZE, _VOTE_SIZE],
        [_VOTE_SIZE, 0]])) + _VOTE_MARGIN
    M = cv2.getPerspectiveTransform(original_points, destination_points)
    pipeline['vote'] = cv2.warpPerspective(
        pipeline['threshold'], M, (_VOTE_SIZE + 2 * _VOTE_MARGIN,
                                   _VOTE_SIZE + 2 * _VOTE_MARGIN))
    best_vote, best_size = max((float(np.sum(pipeline['vote'] * m)) / float(np.sum(m)), s) for s, m in self._board_masks.iteritems())
    # For display only.
    pipeline['vote'] *= self._board_masks[best_size]
    if best_vote < params['vote']:
      with self._display_image_lock:
        with self._display_step_lock:
          self._display_original_image = self.ConvertToQImage(pipeline[self._display_step], width=480)
        # We do not update the result image.
      self.update()
      return
    with self._boardsize_lock:
      self._boardsize = best_size
      boardsize_pixel = _SQUARE_SIZE * (self._boardsize - 1)

    # Reproject to the right size.
    image_pixel = boardsize_pixel + 2 * _MARGIN
    destination_points = OrderPoints(np.float32([[0, 0], [0, boardsize_pixel], [boardsize_pixel, boardsize_pixel],
                                                 [boardsize_pixel, 0]])) + _MARGIN
    M = cv2.getPerspectiveTransform(original_points, destination_points)
    with self._processed_image_lock:
      self._processed_image = cv2.warpPerspective(pipeline['gray'], M, (image_pixel, image_pixel))

    # Display.
    with self._display_step_lock:
      img = pipeline[self._display_step].copy()
      if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
      if self._display_step not in ('hough',):
        cv2.drawContours(img, [best_contour], -1, (0, 255, 0), 2)
    with self._display_image_lock:
      self._display_original_image = self.ConvertToQImage(img, width=480)
      self._display_result_image = self.ConvertToQImage(self._processed_image, width=image_pixel)
    self.update()

  def paintEvent(self, QPaintEvent):
    painter = QPainter()
    painter.begin(self)
    with self._display_image_lock:
      if self._display_original_image is not None:
        painter.drawImage(0, 0, self._display_original_image)
      if self._display_result_image is not None:
        painter.drawImage(480, 0, self._display_result_image)
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
        with self._boardsize_lock:
          if self._status == PAUSE and self._boardsize:
            for i in range(self._boardsize):
              for j in range(self._boardsize):
                self._buttons[i][j].show()
          else:
            for i in range(_LARGEST_BOARDSIZE):
              for j in range(_LARGEST_BOARDSIZE):
                self._buttons[i][j].hide()
    elif 'd' == QKeyEvent.text():
      with self._status_lock:
        with self._boardsize_lock:
          if self._status == PAUSE and self._boardsize:
            print 'Detecting stones...'
            with self._processed_image_lock:
              img = cv2.GaussianBlur(self._processed_image, (_BLUR, _BLUR), 0)
            with self._groundtruth_lock:
              for i in range(self._boardsize):
                for j in range(self._boardsize):
                  self.UpdateGroundtruth(i, j, self.Detect(img, i, j))
            print 'Done'
    elif 'n' == QKeyEvent.text():
      print 'Getting next static image...'
      self._capture.Next()
      # We unpause if paused.
      with self._status_lock:
        if self._status == PAUSE:
          self._status = RUN
          # Hide groundtruth buttons.
          for i in range(_LARGEST_BOARDSIZE):
            for j in range(_LARGEST_BOARDSIZE):
              self._buttons[i][j].hide()
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
      with self._processed_image_lock:
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
