import time


class Timer:
  def __init__(self):
    self._average_time = 0.

  def __enter__(self):
    self._start = time.clock()
    return self

  def __exit__(self, *args):
    dt = time.clock() - self._start
    self._average_time = .9 * self._average_time + .1 * dt

  def GetMs(self):
    return self._average_time * 1000.
