from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('architecture')
    return LaunchDescription([
        Node(
            package='architecture',
            executable='event_layer_node',
            name='event_layer_node',
            output='screen',
            parameters=[{
                'registry_path': os.path.join(pkg_share, 'config', 'task_registry.yaml'),
                'rules_path':    os.path.join(pkg_share, 'config', 'rules.yaml'),
                'enabled': True,
            }]
        ),
        Node(
            package='architecture',
            executable='skills_node',
            name='skills_node',
            output='screen',
            parameters=[{
                'skills_path': os.path.join(pkg_share, 'config', 'skills.yaml'),
            }]
        )
    ])
