from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import OpaqueFunction
from ament_index_python.packages import get_package_share_directory
import os

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
                f.write("version: 1\nrules: []\n" if "rules" in path else
                        "version: 2\ndefaults:\n  window_ms: 3000\nskills: []\n")
            print(f"[launch] Cleared file: {path}")
        except Exception as e:
            print(f"[launch] ERROR clearing {path}: {e}")
    return []

def generate_launch_description():
    
    return LaunchDescription([
        OpaqueFunction(function=_clear_runtime_files),
        
        Node(
            package='architecture',
            executable='event_layer_node',
            name='event_layer_node',
            output='screen',
            parameters=[{
                'registry_path': os.path.join(pkg_share, 'config', 'task_registry.yaml'),
                'rules_path':    RULES_DYNAMIC_PATH,
                'rules_init_path':    os.path.join(pkg_share, 'config', 'rules_init.yaml'),
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
            }]
        ),
        Node(
            package='architecture',
            executable='broker_node',
            name='broker_node',
            output='screen',
            parameters=[{
                
            }]
        ),
        Node(
            package='architecture',
            executable='planner_node',
            name='planner_node',
            output='screen',
            parameters=[{
                
            }]
        ),
        Node(
            package='architecture',
            executable='orchestrator_node',
            name='orchestrator_node',
            output='screen',
            parameters=[{
                'skills_base_path': os.path.join(pkg_share, 'config', 'skills.yaml'),
                'skills_composite_path': SKILLS_COMPOSITE_PATH,
                'registry_path': os.path.join(pkg_share, 'config', 'task_registry.yaml'),
                'rules_path':    RULES_DYNAMIC_PATH,
                'rules_init_path':    os.path.join(pkg_share, 'config', 'rules_init.yaml'),
            }]
        ),
    ])
