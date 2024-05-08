#!/usr/bin/env python3

import numpy as np
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid

from tf_transformations import euler_from_quaternion
from fp_pkg.og_process import process_occ_grid

import cv2

# class def for RRT
class OG(Node):
    def __init__(self):
       
        super().__init__('occ_grid_node')

        self.sim = True

        scan_topic = "/scan"
        if not self.sim:
            pose_topic = 'pf/viz/inferred_pose'
        else:
            pose_topic = '/ego_racecar/odom'
        

        # Create subscribers
        self.scan_sub_ = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 1)
        self.scan_sub_

        self.pose_sub_ = self.create_subscription(PoseStamped if not self.sim else Odometry, 
                                                  pose_topic, self.pose_callback, 1)
        self.pose_sub_

        # Publishers
        self.og_pub_ = self.create_publisher(OccupancyGrid, '/og', 10)

        # class attributes
        self.occupancy_grid = np.ones((36, 100)) * -1.0
        self.dilate_occupancy_grid = np.zeros((36, 100))

        self.lidar_max = 6.0
        self.resolution = 0.1
        self.angle_min = 0.0
        self.angle_max = 0.0

        self.pose = None

    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """

        self.angle_increment = scan_msg.angle_increment
        self.angle_max = scan_msg.angle_max
        self.angle_min = scan_msg.angle_min

        ranges, angles, mr, ma = self.preprocess_lidar(scan_msg.ranges)

        self.occupancy_grid = process_occ_grid(self.occupancy_grid, self.resolution, ranges, angles, mr, ma, False)
        self.dilate_occupancy_grid = process_occ_grid(self.dilate_occupancy_grid, self.resolution, ranges, angles, mr, ma, True)

        # Dilation
        dilation_size = 2
        kernel = np.ones((2 * dilation_size + 1, 2 * dilation_size + 1), np.uint8)
        self.dilate_occupancy_grid = cv2.dilate(self.dilate_occupancy_grid.astype(np.uint8), kernel, iterations=1)
        self.dilate_occupancy_grid = self.dilate_occupancy_grid.astype(np.int8)
        # copy the dilated occupied part to the occupancy grid
        self.occupancy_grid[np.where(self.dilate_occupancy_grid == 1)] = 1
        self.occupancy_grid = self.occupancy_grid.astype(np.int8)

        self.occ_grid() # Publish the occupancy grid

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        ranges = np.array(ranges)
        angles = np.arange(self.angle_min, self.angle_max, self.angle_increment)

        # remove indices where lidar is inf or too far
        idx = (~np.isfinite(ranges)) | (ranges > self.lidar_max)
        proc_ranges = ranges[~ idx]
        proc_angles = angles[~ idx]

        idx_maxed = (np.isfinite(ranges)) & (ranges > self.lidar_max)

        maxed_angles = angles[idx_maxed]
        maxed_ranges = np.ones_like(maxed_angles) * self.lidar_max
        
        return proc_ranges, proc_angles, maxed_ranges, maxed_angles
    
    def occ_grid(self):

        if self.pose is None:
            return

        og = OccupancyGrid()
        og.header.frame_id = "map"

        og.data = np.ravel(self.occupancy_grid, order='C').tolist() 

        og.info.origin = self.pose
        vec_length = math.sqrt((self.occupancy_grid.shape[1] // 2 * self.resolution) ** 2 + 
                                (self.occupancy_grid.shape[0] // 2 * self.resolution) ** 2)
        
        to_center_angle = np.arctan2((self.occupancy_grid.shape[0] // 2 * self.resolution), 
                                     (self.occupancy_grid.shape[1] // 2 * self.resolution))
        
        vec_angle = to_center_angle + self.theta
        og.info.origin.position.x = self.pose.position.x - vec_length * math.cos(vec_angle)
        og.info.origin.position.y = self.pose.position.y - vec_length * math.sin(vec_angle)

        og.info.width = self.occupancy_grid.shape[1]
        og.info.height = self.occupancy_grid.shape[0]
        og.info.resolution = self.resolution

        self.og_pub_.publish(og)

        return

    def pose_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:

        """
        if self.sim:
            quat = pose_msg.pose.pose.orientation
            self.pose = pose_msg.pose.pose
        else:
            quat = pose_msg.pose.orientation
            self.pose = pose_msg.pose
        # print("pose:", self.pose.position.x, self.pose.position.y)
        quat = [quat.x, quat.y, quat.z, quat.w]
        euler = euler_from_quaternion(quat)
        theta = euler[2]
        self.theta = theta

        return
    
def main(args=None):
    rclpy.init(args=args)
    print("Occupancy Grid Initialized")
    og_node = OG()
    rclpy.spin(og_node)

    og_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()