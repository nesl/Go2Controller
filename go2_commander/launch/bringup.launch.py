from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    model = LaunchConfiguration('model', default='gpt-4.1-mini')
    trigger_words = LaunchConfiguration('trigger_words', default='["bob"]')
    params_file = LaunchConfiguration('params_file', default='')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('model', default_value='gpt-4.1-mini'),
        DeclareLaunchArgument('trigger_words', default_value='["bob"]'),
        DeclareLaunchArgument('params_file', default_value=''),

        Node(
            package='go2_commander',
            executable='openai_command_parser',
            name='openai_command_parser',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'model': model},
                {'trigger_words': ['bob']},   # overridden by params_file if set
                params_file
            ],
            # set API key in env; don’t hardcode it
            additional_env={'OPENAI_API_KEY': ''}  # leave empty here; export before launch
        ),

        Node(
            package='go2_commander',
            executable='llm_task_manager',
            name='llm_task_manager',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                params_file
            ],
        ),
    ])
