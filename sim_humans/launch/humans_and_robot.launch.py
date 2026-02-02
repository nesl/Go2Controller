from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Default to: <pkg_share>/config/two_humans_plus_robot.yaml
    default_params = PathJoinSubstitution([
        FindPackageShare("sim_humans"),
        "config",
        "humans_and_robot.yaml",
    ])

    params_file = LaunchConfiguration("params_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Absolute path to params YAML"
        ),

        # Human A
        Node(
            package="sim_humans",
            executable="sim_human_agent",
            name="sim_human_a",
            output="screen",
            parameters=[params_file],
        ),

        # Human B
        Node(
            package="sim_humans",
            executable="sim_human_agent",
            name="sim_human_b",
            output="screen",
            parameters=[params_file],
        ),

        # Robot acting as a human-like agent
        Node(
            package="sim_humans",
            executable="sim_human_agent",
            name="sim_human_robot",
            output="screen",
            parameters=[params_file,{"think_sim_enable": False}],
        ),
    ])

