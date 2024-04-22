#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid

class ReactiveFollowGap(Node):
    """ 
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('reactive_node')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: Subscribe to LIDAR
        self.laser_sub = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.laser_sub

        # TODO: Publish to drive
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 100)

        self.occupancy_grid = None
        self.bias = None

        self.angle_increment = None
        self.acc_ranges = None
        self.set_speed = .5

        self.car_width = .2

        self.lidar_window = 60
        self.lidar_max = 5.0
        self.dist_threshold = 1.7
        self.cliff_threshold = .07
        self.straight_threshold = 3.0
        self.rb = 5
        self.gap_width_threshold = 80

        self.noturn_threshold = .2

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        ranges = np.array(ranges)

        # replace indices where lidar is inf with lidar_max
        idx = (~np.isfinite(ranges)) | (ranges > self.lidar_max)
        ranges[idx] = self.lidar_max

        kernel = np.ones(self.lidar_window)
        proc_ranges = np.convolve(ranges, kernel, 'same') / self.lidar_window
        
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ 
        Return the start index & end index of the max gap in free_space_ranges
        """
        gap_width, start_index, end_index = 0, 0, 0
        save_gap = False

        gap_list = []

        for i in range(len(free_space_ranges)):
            if free_space_ranges[i] > self.dist_threshold:
                gap_width += 1
                if (gap_width > self.gap_width_threshold):
                    save_gap = True
                    end_index = i + 1
                    start_index = end_index - gap_width

            else:
                gap_width = 0
                if save_gap:
                    gap_list.append((start_index, end_index))
                    save_gap = False
                    start_index, end_index = 0, 0

        bias_start_index, bias_end_index = 0, 0

        if len(gap_list) > 0:
            if self.bias == 0:
                bias_start_index, bias_end_index = gap_list[0]
            if self.bias == 1:
                bias_start_index, bias_end_index = gap_list[-1]

        return bias_start_index, bias_end_index
    
    def disparity_extender(self, ranges, start_i, end_i):
        # Find Disparities
        diff = np.diff(ranges[start_i:end_i+1])
        diff_left = np.where(diff > self.cliff_threshold)[0] + start_i
        diff_right = np.where(diff < -self.cliff_threshold)[0] + start_i

        # print("Min", np.min(diff))
        # print("Max", np.max(diff))
        
        # Extend Disparities
        if np.size(diff_left) != 0:
            # print("Ldiff")
            for idx in diff_left:
                idx = int(idx)
                extension = np.ceil((self.car_width / self.acc_ranges[idx]) / self.angle_increment)
                ranges[idx : idx + int(extension) + 1] = ranges[idx]
                
        if np.size(diff_right) != 0:
            # print("Rdiff")
            for idx in diff_right:
                idx = int(idx)
                extension = np.ceil((self.car_width / self.acc_ranges[idx + 1]) / self.angle_increment)
                ranges[idx - int(extension) - 1 : idx + 1] = ranges[idx + 1]
        
        return ranges

    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """

        ranges = self.disparity_extender(ranges, start_i - 40, end_i + 40)

        # Get Max Index
        # max_idx = np.argmax(ranges[start_i:end_i+1]) + start_i

        max_idx = int((start_i + end_i) / 2)

        return max_idx
    
    def check_back(self, ranges, angle_min, angle_max):
        l_bound = int((-np.pi/2 - angle_min) / self.angle_increment)
        r_bound = int((angle_max - np.pi/2) / self.angle_increment)
        lback_min = np.min(ranges[:l_bound])
        rback_min = np.min(ranges[-r_bound:])

        #print(lback_min)
        #print(rback_min)

        return (lback_min < self.noturn_threshold) | (rback_min < self.noturn_threshold)

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        self.angle_increment = data.angle_increment
        ranges = data.ranges
        self.acc_ranges = ranges
        proc_ranges = self.preprocess_lidar(ranges)
        
        # TODO:
        #Find closest point to LiDAR
        closest_point = np.argmin(proc_ranges)
        #Eliminate all points inside 'bubble' (set them to zero) 
        
        proc_ranges[max(0, closest_point - int(self.rb)):min(len(proc_ranges), closest_point + int(self.rb))] = 0
        """
        range indices: right to left - 0 to 1080
        """

        #Set bias
        # self.bias: 0 = right, 1 = left
        self.bias = 1
        #Find max length gap 
        start_i, end_i = self.find_max_gap(proc_ranges)
        #Find the best point in the gap 
        best_point_index = self.find_best_point(start_i, end_i, proc_ranges)
        #Publish Drive message
        angle = data.angle_min + best_point_index * data.angle_increment

        # if self.check_back(ranges, data.angle_min, data.angle_max) :
        #     #print("Straight Drive")
        #     angle = 0.0

        if 0 < abs(angle) <= np.pi/18:
            self.set_speed = 3.0
        elif np.pi/18 < abs(angle) <= np.pi/9:
            self.set_speed = 2.0
        else:
            self.set_speed = 1.5

        drive_msg = AckermannDriveStamped()
        # drive_msg.drive.speed = self.set_speed
        drive_msg.drive.speed = 1.0
        drive_msg.drive.steering_angle = angle
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    print("GapFollow Initialized")
    follow_gap_node = ReactiveFollowGap()
    rclpy.spin(follow_gap_node)

    follow_gap_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
