#!/usr/bin/env python3
import rclpy
import numpy as np
import tf2_ros
import math
import csv
import os
from time import gmtime, strftime
from rclpy.node import Node
from numpy import linalg as LA
from os.path import expanduser
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
# TODO CHECK: include needed ROS msg type headers and libraries
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import yaml

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        config_path = '/home/angli/sim_ws/src/fp_pkg/config/fp_params.yaml'
        # config_path = '/home/nvidia/f1tenth_ws/src/fp_pkg/config/fp_params.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.sim = True

        drive_topic = '/drive'
        if not self.sim:
            odom_topic = 'pf/viz/inferred_pose'
        else:
            odom_topic = '/ego_racecar/odom'    

        # TODO: create ROS subscribers and publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.drive_pub
        self.odom_sub = self.create_subscription(PoseStamped if not self.sim else Odometry, odom_topic, self.pose_callback, 10)
        self.odom_sub
        self.astar_sub = self.create_subscription(MarkerArray, '/astar_path', self.astar_callback, 10)
        self.astar_sub
        self.viz_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)
        self.viz_target_pub = self.create_publisher(Marker, '/visualization_marker', 10)

        
        self.steering_min = -1.571/2  # TODO: Change to actual value
        self.steering_max = 1.571/2 # TODO: Change to actual value

        self.speed_min = config.get('speed_min', 0.5)
        self.speed_max = config.get('speed_max', 4.5)

        self.speed_min_linear = config.get('speed_min_linear', 0.5)
        self.speed_max_linear = config.get('speed_max_linear', 4.0)

        # Parameters
        self.Kp = config.get('Kp', 0.35)
        self.Kd = config.get('Kd', 0.0)
        self.L = config.get('L', 1.0) # lookahead distance
        self.L_speed = config.get('L_speed', 0.5)
        
        # store history
        self.prev_time = self.get_clock().now()
        self.time_gap = None
        self.prev_gamma = 0.0

        # self.prev_waypoint_angle = 0.0

        self.np_waypoints = None # 2*N np array


    def visualize_all_waypoints(self, waypoints):
        marker_array = MarkerArray()
        for i, waypoint in enumerate(waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.150
            marker.scale.y = 0.15
            marker.scale.z = 0.5
            marker.color.a = 1.0  # Alpha is set to 1 to ensure the marker is not transparent
            marker.color.r = 1.0  
            marker.color.g = 1.0
            marker.color.b = 1.0  
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = waypoint['posX']
            marker.pose.position.y = waypoint['posY']
            marker.pose.position.z = 0.0
            # marker.lifetime = rclpy.duration.Duration()  # Setting to zero so it never auto-deletes
            marker_array.markers.append(marker)

        self.viz_pub.publish(marker_array)
            

    def visualize_target_waypoint(self, waypoints, current_waypoint):

        waypoint = waypoints[current_waypoint]
        target_marker = Marker()
        target_marker.header.frame_id = "map"
        target_marker.id = 0 # Only one target marker, thus we can see the target waypoint is being updated
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.scale.x = 0.15
        target_marker.scale.y = 0.15
        target_marker.scale.z = 0.5
        target_marker.color.a = 1.0  # Alpha is set to 1 to ensure the marker is not transparent
        target_marker.color.r = 255.0 
        target_marker.color.g = 0.0
        target_marker.color.b = 0.0 
        target_marker.pose.orientation.w = 1.0
        target_marker.pose.position.x = waypoint['posX']
        target_marker.pose.position.y = waypoint['posY']
        target_marker.pose.position.z = 0.0
        # marker.lifetime = rclpy.duration.Duration()  # Setting to zero so it never auto-deletes

        self.viz_target_pub.publish(target_marker)

    def pure_pursuit(self, goal_pose):
        '''
        Returns a steering angle given a goal pose in terms of the car's reference frame.

        goal_pose: Tuple (x, y) of the goal pose
        return: Float of desired steering angle
        '''

        L2 = (goal_pose[0] ** 2) + (goal_pose[1] ** 2)
        gamma = 2 * abs(goal_pose[1]) / L2

        # angle = self.Kp * gamma

        # PD controller for high-speed pure_pursuit
        P = self.Kp * gamma
        D =  self.Kd * (gamma - self.prev_gamma) / self.time_gap
        angle = P + D
        self.prev_gamma = gamma

        angle *= np.sign(goal_pose[1])

        angle = max(self.steering_min, angle)
        angle = min(self.steering_max, angle)

        return angle
    
    def map_angle_to_speed(self, angle):
        speed = self.speed_max_linear - (self.speed_max_linear - self.speed_min_linear) * abs(angle) / self.waypoint_angle_max
        speed = max(self.speed_min_linear, speed)
        return speed
    
    def astar_callback(self, msg):
        # print("Astar callback")
        self.waypoints = []
        for marker in msg.markers:
            self.waypoints.append({
                'posX': marker.pose.position.x,
                'posY': marker.pose.position.y
            })
        self.np_waypoints = np.array([np.array([wp['posX'], wp['posY']]) for wp in self.waypoints])
        self.visualize_all_waypoints(self.waypoints)
    
    def pose_callback(self, pose_msg):
        # TODO: find the current waypoint to track using methods mentioned in lecture
        if self.sim:
            posX = pose_msg.pose.pose.position.x
            posY = pose_msg.pose.pose.position.y
            quat = pose_msg.pose.pose.orientation
        else:
            posX = pose_msg.pose.position.x
            posY = pose_msg.pose.position.y
            quat = pose_msg.pose.orientation

        quat = [quat.x, quat.y, quat.z, quat.w]
        euler = euler_from_quaternion(quat)
        theta = euler[2]

        if self.np_waypoints is not None:

            lookahead_pos = (posX + self.L*math.cos(theta), posY + self.L*math.sin(theta))
            np_current_pos = np.zeros_like(self.np_waypoints)
            print("np_current_pos: ", np_current_pos)
            np_current_pos[:, 0] = lookahead_pos[0]
            np_current_pos[:, 1] = lookahead_pos[1]
            dist = np.linalg.norm(self.np_waypoints - np_current_pos, axis=1)

            closest_wp_index = np.argmin(dist)
            closest_wp = self.waypoints[closest_wp_index]

            # self.visualize_target_waypoint(self.waypoints, closest_wp_index)

            # TODO: transform goal point to vehicle frame of reference 
            x_diff = closest_wp['posX'] - posX
            y_diff = closest_wp['posY'] - posY
            x_goal_car = x_diff * math.cos(theta) + y_diff * math.sin(theta)
            y_goal_car = -x_diff * math.sin(theta) + y_diff * math.cos(theta)
            
            # TODO: calculate steering angle
            current_time = self.get_clock().now()
            self.time_gap = (current_time - self.prev_time).nanoseconds / 1e9
            steering_angle = self.pure_pursuit((x_goal_car, y_goal_car))
            self.prev_time = current_time

            # TODO: publish drive message, don't forget to limit the steering angle.
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = steering_angle

            # Speed control (steering_angle dependent)
            if 0 < abs(steering_angle) <= np.pi/18:
                speed = 1.0
            elif np.pi/18 < abs(steering_angle) <= np.pi/9:
                speed = 0.5
            else:
                speed = 0.2
            drive_msg.drive.speed = speed

            ### Speed control (waypoints_angle dependent)
            # speed = self.map_angle_to_speed(waypoints_angle)
            
            # print("speed: ", speed)
            drive_msg.drive.speed = speed
            # self.prev_waypoint_angle  = waypoints_angle
            self.drive_pub.publish(drive_msg)
        
        else:
            # print("No waypoints received yet")
            pass

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

