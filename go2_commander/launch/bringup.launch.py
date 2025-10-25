from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    # Common args
    use_sim_time   = LaunchConfiguration('use_sim_time', default='false')

    # OpenAI parser args
    model          = LaunchConfiguration('model', default='gpt-4.1-mini')
    trigger_word   = LaunchConfiguration('trigger_word', default='bob')  # single name → we pass [name]

    # Feature toggles
    use_task_manager = LaunchConfiguration('use_task_manager', default='true')
    use_bt_rssi      = LaunchConfiguration('use_bt_rssi',      default='true')
    use_coco         = LaunchConfiguration('use_coco',         default='true')
    use_stt          = LaunchConfiguration('use_stt',          default='true')
    use_doa          = LaunchConfiguration('use_doa',          default='true')
    use_tts          = LaunchConfiguration('use_tts',          default='true')
    use_vlm          = LaunchConfiguration('use_vlm',          default='true')

    # I/O topics
    cam_img      = LaunchConfiguration('cam_img',      default='/camera/image_raw')
    cam_info     = LaunchConfiguration('cam_info',     default='/camera/camera_info')
    cloud_topic  = LaunchConfiguration('cloud_topic',  default='/point_cloud2')

    return LaunchDescription([
        # ---- Declare args
        DeclareLaunchArgument('use_sim_time',   default_value='false'),
        DeclareLaunchArgument('model',          default_value='gpt-4.1-mini'),
        DeclareLaunchArgument('trigger_word',   default_value='bob'),
        DeclareLaunchArgument('use_task_manager', default_value='true'),
        DeclareLaunchArgument('use_bt_rssi',    default_value='true'),
        DeclareLaunchArgument('use_coco',       default_value='true'),
        DeclareLaunchArgument('use_stt',        default_value='true'),
        DeclareLaunchArgument('use_doa',        default_value='true'),
        DeclareLaunchArgument('use_tts',        default_value='true'),
        DeclareLaunchArgument('use_vlm',        default_value='true'),
        DeclareLaunchArgument('cam_img',        default_value='/camera/image_raw'),
        DeclareLaunchArgument('cam_info',       default_value='/camera/camera_info'),
        DeclareLaunchArgument('cloud_topic',    default_value='/point_cloud2'),
        DeclareLaunchArgument('eleven_api_key', default_value=EnvironmentVariable('ELEVEN_API_KEY', default_value='')),

        # ---- OpenAI command parser
        Node(
            package='go2_commander',
            executable='openai_command_parser',
            name='openai_command_parser',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'model': model},
                # pass as a one-element list; node expects an array
                {'trigger_word': trigger_word},
            ],
        ),

        # ---- LLM task manager (executor) — toggleable
        Node(
            condition=IfCondition(use_task_manager),
            package='go2_commander',
            executable='llm_task_manager',
            name='llm_task_manager',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),

        # ---- Bluetooth RSSI mapper (bt_rssi_mapper.py)
        Node(
            condition=IfCondition(use_bt_rssi),
            package='bt_rssi_mapper',        # adjust if your pkg differs
            executable='bt_rssi_mapper',     # script/entrypoint name
            name='bt_rssi_mapper',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),

        # ---- COCO detector (coco_detector_node.py)
        Node(
            condition=IfCondition(use_coco),
            package='coco_detector',         # adjust to your package
            executable='coco_detector_node', # script/entrypoint
            name='coco_detector',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'image_topic': cam_img},
                {'camera_info_topic': cam_info},
                {'pointcloud_topic': cloud_topic},
                {'publish_annotated_image': True},
            ],
        ),

        # ---- SSL DoA reader (ssldoareader.py)
        Node(
            condition=IfCondition(use_doa),
            package='odas_bridge',           # adjust to your package
            executable='ssl_doa',
            name='ssl_doa',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),

        # ---- STT angle node (stt_angle.py)
        Node(
            condition=IfCondition(use_stt),
            package='odas_bridge',              # adjust to your package
            executable='stt_angle',
            name='stt_angle',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),

        # ---- TTS node (tts_node.py)
        Node(
            condition=IfCondition(use_tts),
            package='speech_processor',      # adjust to your package
            executable='tts_node',
            name='tts_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}, {'api_key': EnvironmentVariable('ELEVEN_API_KEY', default_value='')}],
        ),

        # ---- Qwen VLM server (qwen_vlm_server.py)
        Node(
            condition=IfCondition(use_vlm),
            package='vlm_pkg',               # adjust to your package
            executable='qwen_vlm_server',
            name='qwen_vlm_server',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'model_id': 'Qwen/Qwen2.5-VL-7B-Instruct'},
                {'int4': False},
                {'bf16': True},
                {'max_new_tokens': 256},
                {'temperature': 0.1},
            ],
            # remappings=[('/camera/image_raw', cam_img)],
        ),
    ])

