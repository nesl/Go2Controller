from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import OpaqueFunction
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

from launch.actions import RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown


pkg_share = get_package_share_directory('architecture')

RULES_DYNAMIC_PATH = os.path.join(pkg_share, 'config', 'rules.yaml')
SKILLS_COMPOSITE_PATH = os.path.join(pkg_share, 'config', 'skills_composite.yaml')

def _clear_runtime_files(context, *args, **kwargs):
    # ensure directories exist
    for path in [RULES_DYNAMIC_PATH, SKILLS_COMPOSITE_PATH]:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                # initialize empty valid YAML
                f.write(
                    "version: 1\nrules: []\n"
                    if "rules" in path else
                    "version: 2\ndefaults:\n  window_ms: 3000\nskills: []\n"
                )
            print(f"[launch] Cleared file: {path}")
        except Exception as e:
            print(f"[launch] ERROR clearing {path}: {e}")
    return []

def generate_launch_description():
    plan_accept_policy = LaunchConfiguration("plan_accept_policy")

    taskstatemonitor = Node(
        package='architecture',
        executable='taskstatemonitor_node',
        name='taskstatemonitor_node',
        output='screen',
        parameters=[{
            'task_registry_path': os.path.join(pkg_share, 'config', 'task_registry.yaml'),
            'sim_mode': True,
            "plan_accept_policy": plan_accept_policy
        }]
    )

    shutdown_on_deadline = RegisterEventHandler(
        OnProcessExit(
            target_action=taskstatemonitor,
            on_exit=[EmitEvent(event=Shutdown(reason="Deadline reached: taskstatemonitor exited"))],
        )
    )

    return LaunchDescription([
        OpaqueFunction(function=_clear_runtime_files),

        DeclareLaunchArgument(
            "plan_accept_policy",
            default_value="normal",
            description="Policy for plan acceptance"
        ),

        Node(
            package='architecture',
            executable='event_layer_node',
            name='event_layer_node',
            output='screen',
            parameters=[{
                'registry_path': os.path.join(pkg_share, 'config', 'task_registry.yaml'),
                'rules_path':    RULES_DYNAMIC_PATH,
                'rules_init_path': os.path.join(pkg_share, 'config', 'rules_init.yaml'),
                'skills_base_path': os.path.join(pkg_share, 'config', 'skills.yaml'),
                'skills_composite_path': SKILLS_COMPOSITE_PATH,
                'enabled': True,
            }]
        ),

        Node(
            package='architecture',
            executable='skills_node',
            name='skills_node',
            output='screen',
            parameters=[{
                'skills_base_path': os.path.join(pkg_share, 'config', 'skills.yaml'),
                'skills_composite_path': SKILLS_COMPOSITE_PATH,
                'sim_mode': True,
            }]
        ),

        taskstatemonitor,

        Node(
            package='architecture',
            executable='reactive_node',
            name='reactive_node',
            output='screen',
            parameters=[{
                'llm_enabled': True,
                'model': 'gpt-5-mini',
                'skills_base_path': os.path.join(pkg_share, 'config', 'skills.yaml'),
                'skills_composite_path': SKILLS_COMPOSITE_PATH,
            }]
        ),

        shutdown_on_deadline,
    ])


