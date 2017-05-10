import cv2
import numpy as np


def ResizeImage(img, width=None, height=None):
  if width:
    if not height:
      height = int(img.shape[0] * float(width) / img.shape[1])
  elif height:
    width = int(img.shape[1] * float(height) / img.shape[0])
  if width or height:
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
  return img


def SimplifyContour(contour, approximation_eps=0.01):
  """Approximates a contour by a bounding parallelogram."""
  hull = cv2.convexHull(contour)
  perimeter = cv2.arcLength(hull, True)
  contour = cv2.approxPolyDP(hull, approximation_eps * perimeter, True)
  # Unfortunately, we still get 1 or 2 extra points sometimes.
  num_points = contour.shape[0]
  if 4 < num_points < 7:
    keep_indices = []
    for i in range(num_points):
      prev = (i + num_points - 1) % num_points
      next = (i + 1) % num_points
      v_prev = np.float32(np.squeeze(contour[prev] - contour[i]))
      v_next = np.float32(np.squeeze(contour[next] - contour[i]))
      v_prev /= np.linalg.norm(v_prev)
      v_next /= np.linalg.norm(v_next)
      if v_prev.dot(v_next) > -0.95:  # Less than 180deg.
        keep_indices.append(i)
    contour = contour[np.array(keep_indices)]
  if contour.shape[0] == 4:
    return contour
  return None
