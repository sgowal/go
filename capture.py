import argparse
import cv2
import threading
import time

import rw_lock


class VideoStream(object):
  def __init__(self, camera_index=0):
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--movie', help='Movie filename')
    parser.add_argument('--image', nargs='+', help='Image or list of image filenames')
    args, _ = parser.parse_known_args()
    assert args.image is None or args.movie is None, 'Cannot specify both --movie and --image'

    self._static_images = None
    self._static_images_lock = rw_lock.RWLock()
    self._static_image_index = 0
    self._movie_file = None
    if args.image is None and args.movie is None:
      self._stream = cv2.VideoCapture(args.camera)
      _, self._frame = self._stream.read()
    elif args.image:
      self._static_images = [cv2.imread(f) for f in args.image]
      self._frame = self._static_images[0]
    else:
      assert args.movie is not None
      self._movie_file = args.movie
      self._stream = cv2.VideoCapture(args.movie)
      _, self._frame = self._stream.read()
    assert self._frame is not None, 'Unable to open movie, camera feed or image'

    # Locks.
    self._frame_lock = rw_lock.RWLock()
    self._already_read_lock = rw_lock.RWLock()
    self._already_read = False
    self._status_lock = rw_lock.RWLock()
    self._stopped = False
    self._paused = False

  def Start(self):
    threading.Thread(target=self._Update, args=()).start()
    return self

  def _Update(self):
    while True:
      with self._status_lock(rw_lock.READ_LOCKED):
        if self._stopped:
          return
        if self._paused:
          time.sleep(0.001)
          continue
      if self._static_images is not None:
        with self._frame_lock(rw_lock.WRITE_LOCKED):
          with self._static_images_lock(rw_lock.READ_LOCKED):
            self._frame = self._static_images[self._static_image_index]
      else:
        _, frame = self._stream.read()
        if frame is None and self._movie_file:
          print 'Restarting movie'
          self._stream = cv2.VideoCapture(self._movie_file)
          _, frame = self._stream.read()
        with self._frame_lock(rw_lock.WRITE_LOCKED):
          self._frame = frame
          with self._already_read_lock(rw_lock.WRITE_LOCKED):
            self._already_read = False
      if self._movie_file:
        time.sleep(.03)
      else:
        time.sleep(0.001)

  def Read(self):
    with self._frame_lock(rw_lock.READ_LOCKED):
      with self._already_read_lock(rw_lock.WRITE_LOCKED):
        ret = self._already_read
        self._already_read = True
      return self._frame.copy(), ret

  def Stop(self):
    with self._status_lock(rw_lock.WRITE_LOCKED):
      self._stopped = True

  def Pause(self):
    with self._status_lock(rw_lock.WRITE_LOCKED):
      self._paused = True

  def Continue(self):
    with self._status_lock(rw_lock.WRITE_LOCKED):
      self._paused = False

  def Next(self):
    if not self._static_images:
      return
    with self._static_images_lock(rw_lock.WRITE_LOCKED):
      self._static_image_index = (self._static_image_index + 1) % len(self._static_images)
      self._frame = self._static_images[self._static_image_index]
      with self._already_read_lock(rw_lock.WRITE_LOCKED):
        self._already_read = False
