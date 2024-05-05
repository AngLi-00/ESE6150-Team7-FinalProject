#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PoseStamped
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray

from heapq import heappop, heappush
import math

class Astar(Node):
    def __init__(self):
        super().__init__('astar_node')

        self.sim = True

        if not self.sim:
            odom_topic = 'pf/viz/inferred_pose'
        else:
            odom_topic = '/ego_racecar/odom'
        
        # Subscribers
        self.og_sub = self.create_subscription(OccupancyGrid, '/og', self.og_callback, 10)
        self.odom_sub = self.create_subscription(PoseStamped if not self.sim else Odometry, odom_topic, self.pose_callback, 10)
        # Publishers
        # self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.traj_pub = self.create_publisher(MarkerArray, '/path', 10)
        
        self.current_posX = 0.0
        self.current_posY = 0.0
        self.current_theta = 0.0

        self.resolution = 0.0
        self.width = 0
        self.height = 0

    def pose_callback(self, pose_msg):
        # Update the current pose from the odometry message
        #self.current_pose = msg.pose.pose

        if self.sim:
            self.current_posX = pose_msg.pose.pose.position.x
            self.current_posY = pose_msg.pose.pose.position.y
            quat = pose_msg.pose.pose.orientation
        else:
            self.current_posX = pose_msg.pose.position.x
            self.current_posY = pose_msg.pose.position.y
            quat = pose_msg.pose.orientation

        quat = [quat.x, quat.y, quat.z, quat.w]
        euler = euler_from_quaternion(quat)
        self.current_theta = euler[2]

    def og_callback(self, msg):
        self.resolution = msg.info.resolution
        origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.width = msg.info.width # 100
        self.height = msg.info.height # 36
        grid = np.array(msg.data).reshape((self.height, self.width))
        
        # #print occupied cells
        # print("occ", np.argwhere(grid == 1))

        start_point = [18,50]
        goal_point = [18,99]
        start_point = tuple(start_point)
        goal_point = tuple(goal_point)

        #TODO:Searching in Occupancy Grid Frame
        path_og = self.astar_search(grid, start_point, goal_point)
        """
        path_og: list of tuples (y,x) in the occupancy grid frame 
        """

        #TODO: Transform path to world frame and publish
        if path_og is not False:
            # print("Path found!")
            path_world = [self.og_coords_to_world(p[1], p[0]) for p in path_og]

            # Publish the path 
            marker_array = MarkerArray()
            for i, point in enumerate(path_world):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.scale.x = 0.15
                marker.scale.y = 0.15
                marker.scale.z = 0.5
                marker.color.a = 1.0  # Alpha is set to 1 to ensure the marker is not transparent
                marker.color.r = 1.0  
                marker.color.g = 1.0
                marker.color.b = 1.0  
                marker.pose.orientation.w = 1.0
                marker.pose.position.x = point[0]
                marker.pose.position.y = point[1]
                marker.pose.position.z = 0.0
                # marker.lifetime = rclpy.duration.Duration()  # Setting to zero so it never auto-deletes
                marker_array.markers.append(marker)
                self.traj_pub.publish(marker_array)

        else:
            print("No path found")

    def astar_search(self, grid, start, goal):
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal)}
        oheap = []

        heappush(oheap, (fscore[start], start))
        
        while oheap:
            current = heappop(oheap)[1]

            if current == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                # data.append(start) # add start point if needed
                return data[::-1] # return path in form of a list of np.array (x,y) indices

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                tentative_g_score = gscore[current] + heuristic(current, neighbor)
                if 0 <= neighbor[0] < grid.shape[0]:
                    if 0 <= neighbor[1] < grid.shape[1]:
                        if grid[neighbor[0]][neighbor[1]] == 1:
                            continue
                    else:
                        continue
                else:
                    continue
                
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue
                
                if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1]for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(oheap, (fscore[neighbor], neighbor))
                    
        return False
    
    def world_to_og_coords(self, x_w, y_w):
        """
        world coordinates:
            (At the start position and orientation of the car:)
            x-axis: Pointing the same as the front of the car
            y-axis: Pointing to the left of the car
        occupancy grid coordinates:
            (occupancy grid is 36*100(y*x, height*width) np array, resolution is 0.1)
            (origin at right bottom corner of the occupancy grid, the car is always at the center of the occupancy grid(50,10) )
            x-axis: Pointing the same as the front of the car
            y-axis: Pointing to the left of the car
        """

        dist = math.sqrt(((self.current_posX - x_w)) ** 2 + 
                       ((self.current_posY - y_w)) ** 2) / self.resolution
        angle = math.atan2((self.current_posY - y_w), (self.current_posX - x_w))
        angle -= self.current_theta

        x_og = math.floor(self.width // 2 - dist * math.cos(angle))
        y_og = math.floor(self.height // 2 - dist * math.sin(angle))

        return x_og, y_og
    
    def og_coords_to_world(self, x_og, y_og):
        """
        world coordinates:
            (At the start position and orientation of the car:)
            x-axis: Pointing the same as the front of the car
            y-axis: Pointing to the left of the car
        occupancy grid coordinates:
            (occupancy grid is 36*100(y*x, height*width) np array, resolution is 0.1)
            (origin at right bottom corner of the occupancy grid, the car is always at the center of the occupancy grid(50,10) )
            x-axis: Pointing the same as the front of the car
            y-axis: Pointing to the left of the car
        """
        x_diff = x_og - self.width // 2
        y_diff = y_og - self.height // 2

        dist_x = x_diff * self.resolution
        dist_y = y_diff * self.resolution

        x_w = self.current_posX + (dist_x * math.cos(self.current_theta) + dist_y * math.sin(self.current_theta))
        y_w = self.current_posY + (dist_x * math.sin(self.current_theta) + dist_y * math.cos(self.current_theta))

        return x_w, y_w

        

def main(args=None):
    rclpy.init(args=args)
    print("Astar Initialized")
    astar_node = Astar()
    rclpy.spin(astar_node)
    astar_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

