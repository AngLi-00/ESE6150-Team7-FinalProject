#include "rclcpp/rclcpp.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "fp_pkg/fp_node.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("fp_node");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}