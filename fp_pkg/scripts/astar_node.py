#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PoseStamped
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray

import math
import heapq
import scipy.ndimage

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
        self.astar_path_pub = self.create_publisher(MarkerArray, '/astar_path', 10)
        
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
        
        #find free cells in [0:,50:]
        free_cells = np.argwhere(grid == 0)
        free_cells = free_cells[free_cells[:, 1] >= 50]
        if free_cells.size == 0:
            # print("No free cells found")
            self.publish_path([])
            return
        else:
            # print("Free cells found")
            #find all free cells which have biggest colomn) in [0:,50:]
            farthest = np.argmax(free_cells[:, 1])
            farthest_col = free_cells[farthest][1]
            sample_range = free_cells[free_cells[:, 1] >= farthest_col - 15]
            goal_point = sample_range[np.random.choice(sample_range.shape[0])]
        
        start_point = [18,50]
        # goal_point = [35,70]
        start_point = tuple(start_point)
        goal_point = tuple(goal_point)

        #TODO:Searching in Occupancy Grid Frame
        path_og = self.astar_search(grid, start_point, goal_point)
            
        """
        path_og: list of tuples (y,x) in the occupancy grid frame 
        """

        #TODO: Transform path to world frame and publish
        if path_og:
            path_world = [self.og_coords_to_world(x, y) for y, x in path_og]
            path_smooth = self.bezier_curve(path_world, nTimes=500)  # Increase points for smoother curve

            # Publish the path
            self.publish_path(path_smooth)
        else:
            print("No path found")
            self.publish_path([])  # Clear the path


    # def astar_search(self, grid, start, goal):

    #     # def heuristic(a, b):
    #         # return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    #     def heuristic(current, goal, min_turn_radius=0.5):
    #             # use Euclidean distance and curvature cost as heuristic
    #             straight_distance = math.sqrt((goal[0] - current[0]) ** 2 + (goal[1] - current[1]) ** 2)
    #             # Assuming that the vehicle needs to turn an angle from the current direction to the target direction, 
    #             # this angle is calculated by the difference between the target and the current position
    #             theta = math.atan2(goal[1] - current[1], goal[0] - current[0])
    #             # The influence of curvature can be simulated by adding a term related to rotation angle. 
    #             #   Here, we simply multiply the sine value of the distance between the straight line and the angle difference 
    #             #   to simulate the cost increase when more curvature is required
    #             curvature_cost = straight_distance * abs(math.sin(theta))
    #             return straight_distance + min_turn_radius * curvature_cost

    #     neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    #     open_list = []
    #     heapq.heappush(open_list, (0 + heuristic(start, goal), 0, start))
    #     came_from = {}
    #     g_cost = {start: 0}

    #     while open_list:
    #         _, cost, current = heapq.heappop(open_list)

    #         if current == goal:
    #             path = []
    #             while current in came_from:
    #                 path.append(current)
    #                 current = came_from[current]
    #             path.append(start)  # optional: add start point
    #             path.reverse()  # optional: reverse path
    #             return path

    #         for dx, dy in neighbors:
    #             neighbor = (current[0] + dx, current[1] + dy)
    #             if 0 <= neighbor[0] < self.height and 0 <= neighbor[1] < self.width and grid[neighbor[0]][neighbor[1]] == 0: # Check if neighbor is within bounds and is free
    #                 tentative_g_cost = g_cost[current] + heuristic(current, neighbor)

    #                 if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
    #                     g_cost[neighbor] = tentative_g_cost
    #                     f_cost = tentative_g_cost + heuristic(neighbor, goal)
    #                     heapq.heappush(open_list, (f_cost, tentative_g_cost, neighbor))
    #                     came_from[neighbor] = current

    #     return False  # No path found

    def compute_cost(self, grid, inflation_radius):
        obstacle_grid = (grid == 1).astype(int)
        free_space_grid = 1 - obstacle_grid
        distance_transform = scipy.ndimage.distance_transform_edt(free_space_grid)
        inflated_costs = np.exp(-distance_transform / inflation_radius) * 10  # Scale factor for visibility
        inflated_costs[grid == 1] = np.max(inflated_costs)  # Assign highest cost to obstacles
        return inflated_costs

    def astar_search(self, cost_grid, start, goal):
        def heuristic(a, b):
            return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        open_list = []
        heapq.heappush(open_list, (0 + heuristic(start, goal), 0, start))
        came_from = {}
        g_cost = {start: 0}

        while open_list:
            _, cost, current = heapq.heappop(open_list)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.height and 0 <= neighbor[1] < self.width and cost_grid[neighbor[0]][neighbor[1]] == 0:
                    tentative_g_cost = g_cost[current] + heuristic(current, neighbor) + cost_grid[neighbor[0]][neighbor[1]]
                    if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                        g_cost[neighbor] = tentative_g_cost
                        f_cost = tentative_g_cost + heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_cost, tentative_g_cost, neighbor))
                        came_from[neighbor] = current
        return False

    def bezier_curve(self, points, nTimes=100):
        """
        Smoothen the path using Bezier curve
        """
        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        t = np.linspace(0.0, 1.0, nTimes)
        polynomial_array = np.array([math.comb(nPoints-1, i) * (t**(nPoints-1-i)) * ((1-t)**i) for i in range(nPoints)])
        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)
        return list(zip(xvals, yvals))

    def publish_path(self, path):
        marker_array = MarkerArray()
        for i, point in enumerate(path):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.15
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = 0.0
            marker_array.markers.append(marker)
        self.astar_path_pub.publish(marker_array)
    
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

        x_og = math.floor(self.height // 2 - dist * math.sin(angle))
        y_og = math.floor(self.width // 2 - dist * math.cos(angle))

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

        x_w = self.current_posX + (dist_x * math.cos(self.current_theta) - dist_y * math.sin(self.current_theta))
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

