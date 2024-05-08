import os
from ament_index_python.packages import get_package_share_directory

import launch
import launch.actions
import launch.substitutions
import launch_ros.actions

from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    ld = LaunchDescription()
    launch_file_path = os.path.join(get_package_share_directory('scatcat_bringup'), 'launch')
    imu_file = PathJoinSubstitution([FindPackageShare("imu_filter_madgwick"), "config", "imu_filter.yaml"])
    rviz_config_file = PathJoinSubstitution([FindPackageShare("scatcat_bringup"), "rviz", "D435i.rviz"])
    start_rs = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(launch_file_path, 'scatcat_rs_launch.py')))

    imu = Node(
        package = "imu_filter_madgwick",
        executable = "imu_filter_madgwick_node",
        name = "imu_filter",
        output = "screen",
        parameters = [{'use_mag':False}, {'yaml_filename':imu_file}],
        remappings = [( '/imu/data_raw', '/camera/imu' )]
    )

    tf_imu_camera = Node(
        package = "tf2_ros",
        executable = "static_transform_publisher",
        arguments = ["0.0" , "0.0", "0.0", "0.0", "0.0", "0.0" ,"camera_imu_optical_frame", "camera_link"]
        )

    visu_rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        # output="log",
        output="screen",
        arguments=["-d", rviz_config_file]
    )

    ld.add_action(start_rs)
    ld.add_action(imu)
    ld.add_action(tf_imu_camera)
    ld.add_action(visu_rviz)

    return ld