#!/usr/bin/env python

from __future__ import division

import time
import cv2
import numpy as np
import rospy
import tf2_ros as tf2
import tf.transformations

from cv_bridge import CvBridge

from feature_visual_odometry.data_plotter import DataPlotter
from feature_visual_odometry.match_geometric_filters import HistogramLogicFilter
from feature_visual_odometry.utils import knn_match_filter, rotation_matrix_to_euler_angles, qv_multiply
from feature_visual_odometry.image_manager import ImageManager

from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import Vector3, Quaternion, PoseStamped, TransformStamped

############################################
#              HYPERPARAMETERS             #
############################################


class AlignmentParameters:
    def __init__(self):
        self.threshold_angle = 0
        self.threshold_length = 0
        self.shrink_x_ratio = 0
        self.shrink_y_ratio = 0
        self.plot_images = False
        self.feature_extractor = 'ORB'
        self.matcher = 'BF'
        self.knn_neighbors = 0
        self.filter_by_histogram = False

        self.knn_weight = [1.5]


class VisualOdometry:
    def __init__(self, parameters):
        self.parameters = parameters
        self.images = np.array([ImageManager(), ImageManager()])
        self.bridge = CvBridge()
        self.camera_K = None
        self.publisher = rospy.Publisher("visual_odometry_pose", PoseStamped, queue_size=2)
        self.transform_broadcaster = tf2.TransformBroadcaster()

        self.stacked_position = Vector3(0, 0, 0)
        self.stacked_rotation = tf.transformations.quaternion_from_euler(0, 0, 0)

        # Initiate the feature detector
        if parameters.feature_extractor == 'SURF':
            self.cv2_detector = cv2.xfeatures2d.SURF_create()
        elif parameters.feature_extractor == 'ORB':
            self.cv2_detector = cv2.ORB_create(nfeatures=100)
        else:
            self.cv2_detector = cv2.xfeatures2d.SIFT_create()

        # aux_image_manager = ImageManager()
        # aux_image_manager.read_image(
        #     '/home/guillem/Documents/feature_alignment/catkin_ws/src/image_provider/Images/IMG_0568.JPG')
        # image1 = self.bridge.cv2_to_compressed_imgmsg(aux_image_manager.image)
        # self.save_image_and_trigger_vo(image1)
        # aux_image_manager.read_image(
        #     '/home/guillem/Documents/feature_alignment/catkin_ws/src/image_provider/Images/IMG_0570.JPG')
        # image2 = self.bridge.cv2_to_compressed_imgmsg(aux_image_manager.image)
        # self.save_image_and_trigger_vo(image2)

    def save_camera_calibration(self, data):
        self.camera_K = np.resize(data.K, [3, 3])

    def save_image_and_trigger_vo(self, data):
        start = time.time()
        cv_image = self.bridge.compressed_imgmsg_to_cv2(data)

        # Read new image, extract features, and flip vector to place it in the first position
        self.images[1].load_image(cv_image, gray_scale=True)
        self.extract_image_features(self.images[1])
        self.images = np.flip(self.images)

        if self.images[1].height > 0:
            self.visual_odometry_core()

        rospy.logwarn("TIME: Total time: %s", time.time() - start)
        rospy.logwarn("===================================================")

    def extract_image_features(self, image):
        parameters = self.parameters

        # Down-sample image
        start = time.time()
        image.downsample(parameters.shrink_x_ratio, parameters.shrink_y_ratio)
        end = time.time()
        rospy.logwarn("TIME: Down-sample image. Elapsed time: %s", end - start)

        # Find the key points and descriptors for train image
        start = time.time()
        image.find_keypoints(self.cv2_detector)
        rospy.logwarn("Number or keypoints: %s", len(image.keypoints))
        end = time.time()
        rospy.logwarn("TIME: Extract features of image. Elapsed time: %s", end - start)

    def visual_odometry_core(self):

        parameters = self.parameters
        train_image = self.images[1]
        query_image = self.images[0]

        ############################################
        #                MAIN BODY                 #
        ############################################

        processed_data_plotter = DataPlotter(train_image, query_image)

        # Instantiate histogram logic filter
        histogram_filter = HistogramLogicFilter()

        start = time.time()
        if parameters.matcher == 'KNN':
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(train_image.descriptors, query_image.descriptors, k=parameters.knn_neighbors)
        else:
            # Brute force matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(train_image.descriptors, query_image.descriptors)
        end = time.time()
        rospy.logwarn("TIME: Matching done. Elapsed time: %s", end - start)

        # Initialize fitness trackers
        fitness = float('-inf')
        max_fit = fitness
        max_weight = parameters.knn_weight[0]

        # Explore all the weight values
        for weight in parameters.knn_weight:

            # Filter knn matches by best to second best match ratio
            if parameters.matcher == 'KNN':
                start = time.time()
                good_matches = knn_match_filter(matches, weight)
                end = time.time()
                rospy.logwarn("TIME: Distance filtering of matches done. Elapsed time: %s", end - start)
            else:
                good_matches = matches

            # Filter histograms by gaussian function fitting
            if parameters.filter_by_histogram:
                start = time.time()
                histogram_filter.fit_gaussian(good_matches, train_image.keypoints,
                                              query_image.keypoints, parameters.angle_th, parameters.length_th)
                end = time.time()
                rospy.logwarn("TIME: Histogram filtering done. Elapsed time: %s", end - start)

                fitness = histogram_filter.angle_fitness + histogram_filter.length_fitness

                # Store current configuration as best configuration if fitness is new maximum
                if fitness > max_fit:
                    histogram_filter.save_configuration()
                    max_fit = fitness
                    max_weight = weight

        # Recover best configuration from histogram filtering (for best weight)
        if parameters.filter_by_histogram and histogram_filter.saved_configuration is not None:
            best_matches = histogram_filter.saved_configuration.filter_data_by_histogram()

            if False:  # parameters.plot_images:
                processed_data_plotter.plot_histogram_filtering(good_matches, best_matches, histogram_filter,
                                                                max_weight, max_fit)
        else:
            best_matches = good_matches

        n_final_matches = len(best_matches)

        # Initialize final displacement vectors; x and y will contain the initial points and Dx and Dy the
        # corresponding deformations
        x = np.zeros([n_final_matches, 1])
        y = np.zeros([n_final_matches, 1])
        displacement_x = np.zeros([n_final_matches, 1])
        displacement_y = np.zeros([n_final_matches, 1])

        matched_train_points = [None] * n_final_matches
        matched_query_points = [None] * n_final_matches

        troublesome_matches = np.array([])
        # Proceed to calculate deformations and point maps
        for match_index, match_object in enumerate(best_matches):
            dist = [a_i - b_i for a_i, b_i in zip(
                query_image.keypoints[match_object.trainIdx].pt,
                train_image.keypoints[match_object.queryIdx].pt)]
            x[match_index] = int(round(train_image.keypoints[match_object.queryIdx].pt[0]))
            y[match_index] = int(round(train_image.keypoints[match_object.queryIdx].pt[1]))
            displacement_x[match_index] = dist[0]
            displacement_y[match_index] = dist[1]

            matched_train_points[match_index] = train_image.keypoints[match_object.queryIdx].pt
            matched_query_points[match_index] = query_image.keypoints[match_object.trainIdx].pt

        # Remove matches that are not in list (TODO: why is this even happening?)
        for index in sorted(troublesome_matches, reverse=True):
            del matched_train_points[int(index)]
            del matched_query_points[int(index)]

        try:

            # Extract essential matrix
            start = time.time()
            [h_matrix, mask] = cv2.findEssentialMat(np.array(matched_train_points, dtype=float),
                                                    np.array(matched_query_points, dtype=float), self.camera_K,
                                                    method=cv2.RANSAC, prob=0.999, threshold=1.0)

            # Remove RANSAC outliers
            matched_query_points = np.reshape(
                np.array(matched_query_points, dtype=float)[mask * np.ones([1, 2]) == 1], [-1, 2])
            matched_train_points = np.reshape(
                np.array(matched_train_points, dtype=float)[mask * np.ones([1, 2]) == 1], [-1, 2])

            # Stream points used for essential matrix calculation for debugging
            if parameters.plot_images:
                processed_data_plotter.plot_point_correspondances(
                    matched_train_points, matched_query_points)

            # Recover rotation and translation from essential matrix
            [n_values, rot_mat, t_vec, pose_mask] = \
                cv2.recoverPose(h_matrix, matched_train_points, matched_query_points, self.camera_K)

            t = TransformStamped()
            t.header.frame_id = "world"
            t.child_frame_id = "axis"
            t.header.stamp = rospy.Time.now()

            # Rotate displacement vector by duckiebot rotation wrt world frame and add it to stacked displacement
            t_vec = qv_multiply(self.stacked_rotation, t_vec)
            translation = Vector3(t_vec[0], t_vec[1], t_vec[2])
            self.stacked_position = Vector3(
                self.stacked_position.x + translation.x,
                self.stacked_position.y + translation.y,
                self.stacked_position.z + translation.z)

            # Calculate euler rotation from matrix, and quaternion from euler rotation
            [roll, pitch, yaw] = rotation_matrix_to_euler_angles(rot_mat)
            quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

            # Add quaternion rotation to stacked rotation to obtain the rotation transformation between world and
            # duckiebot frames
            quaternion = tf.transformations.quaternion_multiply(self.stacked_rotation, quaternion)

            # Store quaternion and transform it to geometry message
            self.stacked_rotation = quaternion
            quaternion = Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])

            # Broadcast transform
            t.transform.translation = self.stacked_position
            t.transform.rotation = quaternion
            self.transform_broadcaster.sendTransform(t)

            end = time.time()
            rospy.logwarn("TIME: RANSAC homography done. Elapsed time: %s", end - start)

        except Exception as e:
            rospy.logerr(e)
            rospy.logwarn("Not enough matches for RANSAC homography")
            raise


def define_parameters():
    parameters = AlignmentParameters()

    # Accepted std deviations from average
    parameters.angle_th = 1.5  # Angular distribution
    parameters.length_th = 1.5  # Length distribution

    # Knn neighbors used. Cannot be changed from 2 right now
    parameters.knn_neighbors = 2

    # Image shrink factors in both dimensions
    parameters.shrink_x_ratio = 1 / 2
    parameters.shrink_y_ratio = 1 / 2

    # Publish debug images
    parameters.plot_images = False

    # Knn weight ratio exploration. Relates how bigger must the first match be wrt the second to be considered a match
    # parameters.histogram_weigh = np.arange(1.9, 1.3, -0.05)
    parameters.knn_weight = [1.4]

    # Crop iterations counter. During each iteration the area of matching is reduced based on the most likely
    # region of last iteration
    parameters.crop_iterations = 1

    # Feature extraction and matching parameters
    parameters.matcher = 'BF'
    parameters.feature_extractor = 'ORB'

    parameters.filter_by_histogram = False

    return parameters


def call_save_image(data, vo_object):
    rospy.logwarn("Image obtained")
    vo_object.save_image_and_trigger_vo(data)


def call_save_camera_calibration(data, vo_object):
    vo_object.save_camera_calibration(data)


if __name__ == '__main__':
    rospy.init_node("image_listener", anonymous=True, log_level=rospy.INFO)

    try:
        input_parameters = define_parameters()

        visual_odometry = VisualOdometry(input_parameters)
        rospy.Subscriber("/maserati/camera_node/image/compressed", CompressedImage, call_save_image, visual_odometry,
                         queue_size=2)
        camera_info_sub = rospy.Subscriber("/maserati/camera_node/camera_info", CameraInfo,
                                           call_save_camera_calibration, visual_odometry)

    except rospy.ROSInterruptException:
        pass

    rospy.spin()
