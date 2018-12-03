import numpy as np
import cv2

from histogram_manager import HistogramManager, MatchData
from utils import rectifier


class HistogramLogicFilter:

    def __init__(self):
        self.angle_fitness = 0
        self.length_fitness = 0
        self.angle_histogram = None
        self.length_histogram = None
        self.saved_configuration = None

        self.match_vector = []
        self.angle_threshold = None
        self.length_threshold = None

    def fit_gaussian(self, match_vector, kp1, kp2, angle_threshold, length_threshold):

        self.match_vector = match_vector
        self.angle_threshold = angle_threshold
        self.length_threshold = length_threshold

        # Initialize and fill angle and length arrays of the vectors of the matched points
        angle_vec = np.zeros([len(match_vector), 1])
        length_vec = np.zeros([len(match_vector), 1])

        for i, match_object in enumerate(match_vector):
            dist = [a_i - b_i for a_i, b_i in zip(kp2[match_object.trainIdx].pt, kp1[match_object.queryIdx].pt)]
            angle_vec[i] = np.arctan(dist[1] / max(dist[0], 0.01))
            length_vec[i] = np.sqrt(dist[0] ** 2 + dist[1] ** 2)

        # Compute histograms of the two distributions (angle and length of the displacement vectors)
        self.angle_histogram = HistogramManager(angle_vec, 8)
        self.length_histogram = HistogramManager(length_vec, 8)

        # Fit gaussian functions to distribution and remove outliers
        self.angle_histogram.fit_gaussian(angle_threshold)
        self.length_histogram.fit_gaussian(length_threshold)

        # Calculate fitness
        if self.angle_histogram.success and self.length_histogram.success:
            self.angle_fitness = self.angle_histogram.area_under_curve / rectifier(self.angle_histogram.fano_factor)
            self.length_fitness = self.length_histogram.area_under_curve / rectifier(self.length_histogram.fano_factor)

            print("Fano factors: angle = ", self.angle_histogram.fano_factor, "  length = ",
                  self.length_histogram.fano_factor)
            print("Total fitness: ", self.angle_fitness + self.length_fitness, " Split attributes: angle = ",
                  self.angle_fitness, "  length = ", self.length_fitness)

    def save_configuration(self):
        self.saved_configuration = MatchData(self.match_vector, self.angle_histogram, self.length_histogram)


class RansacFilter:
    @staticmethod
    def ransac_homography(kp1, kp2, match_vector, data_plotter, plot_data):

        src_pts = np.float32([kp1[m.queryIdx].pt for m in match_vector]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in match_vector]).reshape(-1, 1, 2)

        h_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        if data_plotter is not None and plot_data:
            data_plotter.plot_ransac_homography(match_vector, h_matrix, matches_mask)

        return h_matrix
