import numpy as np
import scipy.spatial


EUCLIDEAN_DISTANCE = 0
HAMMING_DISTANCE = 1


class Match(object):
  def __init__(self, kp1, kp2, spatial_distance, descriptor_distance):
    self.spatial_distance = spatial_distance
    self.descriptor_distance = descriptor_distance
    self.xy_database = kp1.pt
    self.xy_query = kp2.pt

  def __repr__(self):
    return '(%g,%g) <- (%g,%g): %g %g' % (self.xy_database[0], self.xy_database[1], self.xy_query[0], self.xy_query[1], self.spatial_distance, self.descriptor_distance)


class NearestNeighborSearcher(object):
  def __init__(self, keypoints, descriptors, descriptor_distance=EUCLIDEAN_DISTANCE):
    self._keypoints = keypoints
    self._descriptors = descriptors
    self._kdtree = scipy.spatial.cKDTree([kp.pt for kp in keypoints], 10)
    self._distance = (scipy.spatial.distance.euclidean if descriptor_distance == EUCLIDEAN_DISTANCE else
                      scipy.spatial.distance.hamming)

  def Search(self, keypoints, descriptors, k=1, r=1e-8):
    # Returns the closest descriptor in descriptor-space among the closest k neighbors in keypoint-space.
    matches = []
    distances, indices = self._kdtree.query([kp.pt for kp in keypoints], k=k)
    if k == 1:
      distances = distances.reshape(-1, 1)
      indices = indices.reshape(-1, 1)
    for i in range(len(keypoints)):
      descriptor_distances = [self._distance(descriptors[i], self._descriptors[indices[i, j]]) for j in range(k) if distances[i, j] < r]
      if descriptor_distances:
        j = np.argmin(descriptor_distances)
        matches.append(Match(self._keypoints[indices[i, j]], keypoints[i], distances[i, j], descriptor_distances[j]))
    return matches
