import copy
import cv2
import numpy as np
import rw_lock
import util


# Calibration processing steps.
CALIBRATION_ORIGINAL = 0
CALIBRATION_GRAY = 1
CALIBRATION_BLUR = 2
CALIBRATION_CLOSE = 3
CALIBRATION_DIV = 4
CALIBRATION_THRESHOLD = 5
CALIBRATION_CONTOUR = 6
CALIBRATION_VOTE = 7
CALIBRATION_FINAL = 8
ALL_CALIBRATION_STEPS = range(9)
ALL_UNPROJECTED_CALIBRATION_STEPS = [
    CALIBRATION_ORIGINAL,
    CALIBRATION_GRAY,
    CALIBRATION_BLUR,
    CALIBRATION_CLOSE,
    CALIBRATION_DIV,
    CALIBRATION_THRESHOLD,
    CALIBRATION_CONTOUR,
]

# Fixed parameters.
_VOTE_SIZE = 200
_VOTE_MARGIN = 5
_SQUARE_SIZE = 25
_MARGIN = 13


class Vision(object):

  def __init__(self, debug=False):
    # Default fine-tuned parameters.
    self._parameters_lock = rw_lock.RWLock()
    self._parameters = {
        'blur': 3,
        'close': 11,
        'th_block': 11,
        'th_mean': 3,
        'ctr_size': 200,
        'approx': 0.01,
        'vote': 50,
    }
    # Masks used to detect the board size.
    self._board_masks = {}
    for size in (9, 13, 19):
      self._board_masks[size] = _BuildGridImage(size)
    self._calibration_result_lock = rw_lock.RWLock()
    self._boardsize = None
    self._calibration_image = None
    # Debugging mode keeps track of all intermediate processing steps.
    self._debug = debug
    self._debug_pipeline_lock = rw_lock.RWLock()
    self._debug_pipeline = {}
    for step in ALL_CALIBRATION_STEPS:
      self._debug_pipeline[step] = None

  def SetParameter(self, param_name, param_value):
    with self._parameters_lock(rw_lock.WRITE_LOCKED):
      self._parameters[param_name] = param_value

  def _StoreCalibrationResult(self, pipeline, boardsize):
    with self._calibration_result_lock(rw_lock.WRITE_LOCKED):
      # Only store a new calibration image if there is one.
      if pipeline[CALIBRATION_FINAL]:
        self._calibration_image = pipeline[CALIBRATION_FINAL]
        self._boardsize = boardsize
      if self._debug:
        with self._debug_pipeline_lock(rw_lock.WRITE_LOCKED):
          self._debug_pipeline = pipeline

  def GetCalibrationResult(self, debug_step=CALIBRATION_ORIGINAL):
    # This function should really only be used in debug mode.
    assert self._debug
    with self._calibration_result_lock(rw_lock.READ_LOCKED):
      with self._debug_pipeline_lock(rw_lock.READ_LOCKED):
        return self._calibration_image.copy(), self._boardsize, self._debug_pipeline[debug_step].copy()

  def Calibrate(self, input_image):
    """Finds an almost-empty Go board in an image."""
    # Keep local parameter copy.
    with self._parameters_lock(rw_lock.READ_LOCKED):
      params = copy.deepcopy(self._parameters)

    # Reset parameters.
    with self._boardsize_lock(rw_lock.WRITE_LOCKED):
      self._boardsize = None

    # Keep track of all images. In debug mode, these images are
    # then copied atomically. Otherwise they are discarded.
    pipeline = {}
    pipeline[CALIBRATION_ORIGINAL] = util.ResizeImage(input_image, height=720)

    # Blur + Grayscale.
    pipeline[CALIBRATION_GRAY] = cv2.cvtColor(pipeline[CALIBRATION_ORIGINAL], cv2.COLOR_BGR2GRAY)
    pipeline[CALIBRATION_BLUR] = cv2.GaussianBlur(pipeline[CALIBRATION_GRAY], (params['blur'], params['blur']), 0)

    # The following sequence removes small lines.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params['close'], params['close']))
    pipeline[CALIBRATION_CLOSE] = cv2.morphologyEx(pipeline[CALIBRATION_BLUR], cv2.MORPH_CLOSE, kernel)

    # Small lines that were removed will pop.
    div = np.float32(pipeline[CALIBRATION_BLUR]) / (pipeline[CALIBRATION_CLOSE] + 1e-4)
    pipeline[CALIBRATION_DIV] = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))

    # Threshold.
    output = cv2.adaptiveThreshold(pipeline[CALIBRATION_DIV], 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, params['th_block'], params['th_mean'])
    pipeline[CALIBRATION_THRESHOLD] = output.copy()  # Reused later. However findContours overwrites it :(

    # Find the largest contour.
    contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    all_valid_contours = []  # Used for debug only.
    best_contour = None
    for contour in contours:
      area = cv2.contourArea(contour)
      if area > params['ctr_size'] * params['ctr_size']:
        all_valid_contours.append(contour)  # Used for debug only.
        if area > max_area:
          best_contour = contour
          max_area = area
    # Fit a square to the best contour.
    if best_contour:
      best_contour = util.SimplifyContour(best_contour, approximation_eps=params['approx'])
    # For display only.
    if self._debug:
      pipeline[CALIBRATION_CONTOUR] = cv2.cvtColor(pipeline[CALIBRATION_THRESHOLD].copy(), cv2.COLOR_GRAY2BGR)
      cv2.drawContours(pipeline[CALIBRATION_CONTOUR], all_valid_contours, -1, (255, 0, 0), 2)
      if best_contour is not None:
        cv2.drawContours(pipeline[CALIBRATION_CONTOUR], [best_contour], -1, (0, 0, 255), 4)
    # We found nothing.
    if best_contour is None:
      pipeline[CALIBRATION_VOTE] = None
      pipeline[CALIBRATION_FINAL] = None
      self._StoreCalibrationResult(pipeline)
      return False

    original_points = _OrderPoints(np.squeeze(best_contour))

    # We detect the board size by first projecting the detected and counting the number of lines.
    destination_points = _OrderPoints(np.float32([
        [0, 0], [0, _VOTE_SIZE], [_VOTE_SIZE, _VOTE_SIZE],
        [_VOTE_SIZE, 0]])) + _VOTE_MARGIN
    M = cv2.getPerspectiveTransform(original_points, destination_points)
    pipeline[CALIBRATION_VOTE] = cv2.warpPerspective(
        pipeline[CALIBRATION_THRESHOLD], M, (_VOTE_SIZE + 2 * _VOTE_MARGIN,
                                             _VOTE_SIZE + 2 * _VOTE_MARGIN))
    best_vote, best_size = max((float(np.sum(pipeline[CALIBRATION_VOTE] * m)) / float(np.sum(m)), s) for s, m in self._board_masks.iteritems())
    # For display only.
    if self._debug:
      pipeline[CALIBRATION_VOTE] *= self._board_masks[best_size]
    # We cannot detect the size.
    if best_vote < params['vote']:
      pipeline[CALIBRATION_FINAL] = None
      self._StoreCalibrationResult(pipeline)
      return False
    # Set boardsize.
    boardsize_pixel = _SQUARE_SIZE * (best_size - 1)
    # Reproject to the right size.
    image_pixel = boardsize_pixel + 2 * _MARGIN
    destination_points = _OrderPoints(np.float32([[0, 0], [0, boardsize_pixel], [boardsize_pixel, boardsize_pixel],
                                                  [boardsize_pixel, 0]])) + _MARGIN
    M = cv2.getPerspectiveTransform(original_points, destination_points)
    pipeline[CALIBRATION_FINAL] = cv2.warpPerspective(pipeline[CALIBRATION_GRAY], M, (image_pixel, image_pixel))
    # For display only.
    if self._debug:
      for step in ALL_UNPROJECTED_CALIBRATION_STEPS:
        cv2.drawContours(pipeline[step], [best_contour], -1, (0, 255, 0), 2)
    self._StoreCalibrationResult(pipeline, best_size)
    return True


def _BuildGridImage(size):
  mask = np.zeros((_VOTE_MARGIN * 2 + _VOTE_SIZE, _VOTE_MARGIN * 2 + _VOTE_SIZE), dtype=np.uint8)
  cell_size = float(_VOTE_SIZE) / float(size - 1)
  for i in range(size):
    x = int(i * cell_size + _VOTE_MARGIN)
    mask[x - 1:x + 2, _VOTE_MARGIN:_VOTE_SIZE + _VOTE_MARGIN] = 1
  for i in range(size):
    x = int(i * cell_size + _VOTE_MARGIN)
    mask[_VOTE_MARGIN:_VOTE_SIZE + _VOTE_MARGIN, x - 1:x + 2] = 1
  return mask


def _OrderPoints(pts):
  center = np.mean(pts, axis=0)
  tpts = pts - center
  vecs_angle = [np.arctan2(pt[1], pt[0]) for pt in tpts]
  pts, _ = zip(*sorted(zip(pts, vecs_angle), key=lambda x: x[1]))
  pts = np.float32(pts)
  # Starting corner is the most top corner.
  i = np.argmax([y for x, y in pts])
  i = np.array((i + np.arange(len(pts))) % len(pts))
  return np.float32(pts[i, :])
