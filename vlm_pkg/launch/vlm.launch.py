from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vlm_pkg',
            executable='qwen_vlm_server',
            name='qwen_vlm_server',
            output='screen',
            parameters=[
                {'model_id': 'Qwen/Qwen2.5-VL-7B-Instruct'},
                {'int4': False},
                {'bf16': True},
                {'max_new_tokens': 192},
                {'temperature': 0.0},
            ],
        ),
    ])
