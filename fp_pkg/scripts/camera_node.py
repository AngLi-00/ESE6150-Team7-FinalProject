#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import cv2

from cv_bridge import CvBridge


class CamNode(Node):
    def __init__(self):
       
        super().__init__('camera_node')

        cam_topic = "/cam_frame"

        device = "video4"

        self.publisher_ = self.create_publisher(Image, cam_topic , 10)
        self.timer = self.create_timer(1.0 / 30.0, self.cam_callback)

        self.cap = cv2.VideoCapture(f"v4l2src device=/dev/{device} extra-controls=\"c,exposure_auto=3\" ! "
                       "video/x-raw, width=1280, height=720, framerate=60/1 ! "
                       "videoconvert ! video/x-raw, format=BGR ! appsink")

        self.cap = cv2.VideoCapture(0)
        
        self.br = CvBridge()


    def cam_callback(self):
        ret, frame = self.cap.read()     
        if ret == True:
            self.publisher_.publish(self.br.cv2_to_imgmsg(frame))
        
        return
    
def main(args=None):
    rclpy.init(args=args)
    print("Camera Node Initialized")
    cam_node = CamNode()
    rclpy.spin(cam_node)

    cam_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()