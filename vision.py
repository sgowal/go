import collections
import copy
import cv2
import numpy as np

import nn
import rw_lock
import timer
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
ALL_CALIBRATION_STEPS = set(range(9))
ALL_UNPROJECTED_CALIBRATION_STEPS = set([
    CALIBRATION_ORIGINAL,
    CALIBRATION_GRAY,
    CALIBRATION_BLUR,
    CALIBRATION_CLOSE,
    CALIBRATION_DIV,
    CALIBRATION_THRESHOLD,
    CALIBRATION_CONTOUR,
])

# Tracking steps.
TRACKING_ORIGINAL = 20
TRACKING_GRAY = 21
TRACKING_RESIZE = 22
TRACKING_GRAY_REDUCED = 23
TRACKING_FINAL = 24
ALL_TRACKING_STEPS = set(range(20, 25))

# For API clarity.
CALIBRATION_OR_TRACKING_ORIGINAL = 100

# Fixed parameters.
_VOTE_SIZE = 200
_VOTE_MARGIN = 5
SQUARE_SIZE = 25
MARGIN = 13

# Reduced image sizes passed to ECC by this factor.
_ECC_REDUCE = 4


# This class is mostly NOT multi-threaded. Only the parameters can be changed atomically.
# In particular: Calibrate/Track/GetCalibrationResults and GetTrackingResults cannot be called concurrently.
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
        'match_k': 2,
        'match_sim': 0.75,
        'match_min': 30,
        'ransac_th': 5,
    }
    # Masks used to detect the board size.
    self._board_masks = {}
    for size in (9, 13, 19):
      self._board_masks[size] = _BuildGridImage(size)
    # Calibration.
    self._boardsize = None
    self._calibration_image = None
    # Tracking.
    self._tracking_image = None
    self._transform_matrix = None
    # Debugging mode keeps track of all intermediate processing steps.
    self._debug = debug
    self._debug_calibration_pipeline = {}
    for step in ALL_CALIBRATION_STEPS:
      self._debug_calibration_pipeline[step] = None
    self._debug_tracking_pipeline = {}
    for step in ALL_TRACKING_STEPS:
      self._debug_tracking_pipeline[step] = None
    # Statistics.
    self._timers = collections.defaultdict(timer.Timer)

  def Stop(self):
    pass

  def SetParameter(self, param_name, param_value):
    with self._parameters_lock(rw_lock.WRITE_LOCKED):
      self._parameters[param_name] = param_value

  def _CopyTimingsMs(self):
    return sorted((k, t.GetMs()) for k, t in self._timers.iteritems())

  def _StoreCalibrationResult(self, pipeline, boardsize=0, transform=None):
    if self._debug:
      self._debug_calibration_pipeline = pipeline
    if pipeline[CALIBRATION_FINAL] is None:
      return False
    # Only store a new calibration image if there is one.
    self._calibration_image = pipeline[CALIBRATION_FINAL]
    self._boardsize = boardsize
    self._transform_matrix = transform
    return True

  def GetCalibrationResult(self, debug_step=CALIBRATION_ORIGINAL):
    # This function should really only be used in debug mode.
    assert self._debug
    if debug_step not in ALL_CALIBRATION_STEPS:
      debug_step = CALIBRATION_ORIGINAL
    return (self._calibration_image.copy() if self._calibration_image is not None else None,
            self._boardsize,
            self._debug_calibration_pipeline[debug_step].copy() if self._debug_calibration_pipeline[debug_step] is not None else None,
            self._CopyTimingsMs())

  def Calibrate(self, input_image):
    """Finds an almost-empty Go board in an image."""
    # Keep local parameter copy.
    with self._parameters_lock(rw_lock.READ_LOCKED):
      params = copy.deepcopy(self._parameters)

    # Keep track of all images. In debug mode, these images are
    # then copied atomically. Otherwise they are discarded.
    pipeline = {}
    with self._timers['c_resize']:
      pipeline[CALIBRATION_ORIGINAL] = util.ResizeImage(input_image, height=720)

    # Blur + Grayscale.
    with self._timers['c_gray_blur']:
      gray = cv2.cvtColor(pipeline[CALIBRATION_ORIGINAL], cv2.COLOR_BGR2GRAY)
      pipeline[CALIBRATION_GRAY] = gray.copy() if self._debug else gray
      pipeline[CALIBRATION_BLUR] = cv2.GaussianBlur(pipeline[CALIBRATION_GRAY], (params['blur'], params['blur']), 0)

    # The following sequence removes small lines.
    with self._timers['c_close']:
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params['close'], params['close']))
      pipeline[CALIBRATION_CLOSE] = cv2.morphologyEx(pipeline[CALIBRATION_BLUR], cv2.MORPH_CLOSE, kernel)

    # Small lines that were removed will pop.
    with self._timers['c_div']:
      div = np.float32(pipeline[CALIBRATION_BLUR]) / (pipeline[CALIBRATION_CLOSE] + 1e-4)
      pipeline[CALIBRATION_DIV] = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))

    # Threshold.
    with self._timers['c_threshold']:
      output = cv2.adaptiveThreshold(pipeline[CALIBRATION_DIV], 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, params['th_block'], params['th_mean'])
      pipeline[CALIBRATION_THRESHOLD] = output.copy()  # Reused later. However findContours overwrites it :(

    # Find the largest contour.
    with self._timers['c_contour']:
      _, contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    if best_contour is not None:
      with self._timers['c_simplify']:
        best_contour = util.SimplifyContour(best_contour, approximation_eps=params['approx'])
    # For display only.
    if self._debug:
      with self._timers['c_dbg']:
        pipeline[CALIBRATION_CONTOUR] = cv2.cvtColor(pipeline[CALIBRATION_THRESHOLD].copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(pipeline[CALIBRATION_CONTOUR], all_valid_contours, -1, (255, 0, 0), 2)
        if best_contour is not None:
          cv2.drawContours(pipeline[CALIBRATION_CONTOUR], [best_contour], -1, (0, 0, 255), 4)
    # We found nothing.
    if best_contour is None:
      pipeline[CALIBRATION_VOTE] = None
      pipeline[CALIBRATION_FINAL] = None
      return self._StoreCalibrationResult(pipeline)

    original_points = _OrderPoints(np.squeeze(best_contour))

    # We detect the board size by first projecting the detected and counting the number of lines.
    with self._timers['c_wrap_1']:
      destination_points = _OrderPoints(np.float32([
          [0, 0], [0, _VOTE_SIZE], [_VOTE_SIZE, _VOTE_SIZE],
          [_VOTE_SIZE, 0]])) + _VOTE_MARGIN
      M = cv2.getPerspectiveTransform(original_points, destination_points)
      pipeline[CALIBRATION_VOTE] = cv2.warpPerspective(
          pipeline[CALIBRATION_THRESHOLD], M, (_VOTE_SIZE + 2 * _VOTE_MARGIN,
                                               _VOTE_SIZE + 2 * _VOTE_MARGIN))
    with self._timers['c_template']:
      best_vote, best_size = max((float(np.sum(pipeline[CALIBRATION_VOTE] * m)) / float(np.sum(m)), s) for s, m in self._board_masks.iteritems())
    # For display only.
    if self._debug:
      pipeline[CALIBRATION_VOTE] *= self._board_masks[best_size]
    # We cannot detect the size.
    if best_vote < params['vote']:
      pipeline[CALIBRATION_FINAL] = None
      return self._StoreCalibrationResult(pipeline)
    # Set boardsize.
    boardsize_pixel = SQUARE_SIZE * (best_size - 1)
    # Reproject to the right size.
    with self._timers['c_wrap_2']:
      image_pixel = boardsize_pixel + 2 * MARGIN
      destination_points = _OrderPoints(np.float32([[0, 0], [0, boardsize_pixel], [boardsize_pixel, boardsize_pixel],
                                                    [boardsize_pixel, 0]])) + MARGIN
      M = cv2.getPerspectiveTransform(original_points, destination_points)
      pipeline[CALIBRATION_FINAL] = cv2.warpPerspective(pipeline[CALIBRATION_GRAY], M, (image_pixel, image_pixel))
    # For display only.
    if self._debug:
      with self._timers['c_dbg']:
        for step in ALL_UNPROJECTED_CALIBRATION_STEPS:
          if len(pipeline[step].shape) == 2:
            pipeline[step] = cv2.cvtColor(pipeline[step], cv2.COLOR_GRAY2BGR)
          cv2.drawContours(pipeline[step], [best_contour], -1, (0, 255, 0), 2)
    return self._StoreCalibrationResult(pipeline, best_size, M)

  def _StoreTrackingResult(self, pipeline):
    if self._debug:
      self._debug_tracking_pipeline = pipeline
    # Only store a new calibration image if there is one.
    if pipeline[TRACKING_FINAL] is None:
      return False
    self._tracking_image = pipeline[TRACKING_FINAL]
    return True

  def GetTrackingResult(self, debug_step=TRACKING_ORIGINAL):
    # This function should really only be used in debug mode.
    assert self._debug
    if debug_step not in ALL_TRACKING_STEPS:
      debug_step = TRACKING_ORIGINAL
    return (self._tracking_image.copy() if self._tracking_image is not None else None,
            self._debug_tracking_pipeline[debug_step].copy() if self._debug_tracking_pipeline[debug_step] is not None else None,
            self._CopyTimingsMs())

  def Track(self, input_image):
    # Keep local parameter copy.
    with self._parameters_lock(rw_lock.READ_LOCKED):
      params = copy.deepcopy(self._parameters)

    assert self._calibration_image is not None
    pattern = util.ResizeImage(self._calibration_image, height=self._calibration_image.shape[0] // _ECC_REDUCE)
    print pattern.shape

    # Calibration was successful.

    # DEBUG.
    img = input_image.copy()
    inverse_M = np.array([[1, 0.1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    input_image = cv2.warpPerspective(img, inverse_M, img.shape[:2][::-1])

    # Keep track of all images. In debug mode, these images are
    # then copied atomically. Otherwise they are discarded.
    pipeline = {}
    with self._timers['t_resize_1']:
      pipeline[TRACKING_ORIGINAL] = util.ResizeImage(input_image, height=720)
    with self._timers['t_gray']:
      pipeline[TRACKING_GRAY] = cv2.cvtColor(pipeline[TRACKING_ORIGINAL], cv2.COLOR_BGR2GRAY)
      pipeline[TRACKING_GRAY_REDUCED] = util.ResizeImage(pipeline[TRACKING_GRAY], height=pipeline[TRACKING_GRAY].shape[0] // _ECC_REDUCE)
    print pipeline[TRACKING_GRAY_REDUCED].shape

    # We perform ECC matching, but providing the warp_matrix that transforms
    # reduced pattern into our reduced gray image.
    with self._timers['t_ecc']:
      warp_mode = cv2.MOTION_HOMOGRAPHY
      S1 = np.eye(3, 3, dtype=np.float32)
      S1[0, 0] = float(_ECC_REDUCE)
      S1[1, 1] = float(_ECC_REDUCE)
      S2 = S1.copy()
      S2[np.diag_indices_from(S2)] = 1./ S2[np.diag_indices_from(S2)]
      M1 = S2.dot(self._transform_matrix).dot(S1)
      warp_matrix = cv2.invert(M1)[1].astype(np.float32)
      number_of_iterations = 100
      termination_eps = 1e-3
      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
      try:
        cc, M2 = cv2.findTransformECC(pattern, pipeline[TRACKING_GRAY_REDUCED], warp_matrix, warp_mode, criteria)
        M = S1.dot(cv2.invert(M2)[1]).dot(S2)
        print cc
      except Exception:
        print 'Cannot track'
        pipeline[TRACKING_FINAL] = None
        return self._StoreTrackingResult(pipeline)

    with self._timers['t_warp2']:
      pipeline[TRACKING_FINAL] = cv2.warpPerspective(
          pipeline[TRACKING_GRAY], M, self._calibration_image.shape)

    if self._debug:
      with self._timers['t_dbg']:
        h, w = self._calibration_image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, cv2.invert(M)[1])
        cv2.polylines(pipeline[TRACKING_ORIGINAL], [np.int32(dst)], True, (255, 0, 0), 3)
        dst = cv2.perspectiveTransform(pts, cv2.invert(self._transform_matrix)[1])
        cv2.polylines(pipeline[TRACKING_ORIGINAL], [np.int32(dst)], True, (255, 255, 255), 1)

    # TODO: Update tracking
    # self._transform_matrix = M

    return self._StoreTrackingResult(pipeline)


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


def _DrawMatches(img1, img2, matches):
  rows1 = img1.shape[0]
  cols1 = img1.shape[1]
  rows2 = img2.shape[0]
  cols2 = img2.shape[1]
  out = np.ones((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8') * 200
  out[:rows1, :cols1] = np.dstack([img1, img1, img1])
  out[:rows2, cols1:] = np.dstack([img2, img2, img2])
  for m in matches:
    x1, y1 = m.xy_database
    x2, y2 = m.xy_query
    cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
    cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)
    cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)
  return out
