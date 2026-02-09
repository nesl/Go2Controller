from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Default to: <pkg_share>/config/two_humans.yaml
    default_params = PathJoinSubstitution([
        FindPackageShare("sim_humans"),
        "config",
        "two_humans.yaml",
    ])

    params_file = LaunchConfiguration("params_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Absolute path to params YAML"
        ),

        Node(
            package="sim_humans",
            executable="sim_human_agent",
            name="sim_human_a",
            output="screen",
            parameters=[params_file,
            {"agent_id": "human_a", "goal_property": "X", "prior_X": 0.75, "prior_Y": 0.40},],
        ),

        Node(
            package="sim_humans",
            executable="sim_human_agent",
            name="sim_human_b",
            output="screen",
            parameters=[params_file,{"prior_Y": 0.75, "prior_X": 0.40}],
        ),
    ])

