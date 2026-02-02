#!/usr/bin/env python3
import json

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_json_list(s: str):
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _build_nodes(context, *args, **kwargs):
    package = LaunchConfiguration("package").perform(context)
    executable = LaunchConfiguration("executable").perform(context)

    # agent selection
    num_agents = int(LaunchConfiguration("num_agents").perform(context))
    agent_prefix = LaunchConfiguration("agent_prefix").perform(context)
    agent_ids = _load_json_list(LaunchConfiguration("agent_ids").perform(context))

    # reward_correct (applied to both X and Y for this optimizer)
    reward_correct = float(LaunchConfiguration("reward_correct").perform(context))

    if not agent_ids:
        if num_agents <= 0:
            agent_ids = ["robot"]
        else:
            agent_ids = [f"{agent_prefix}{i}" for i in range(num_agents)]

    nodes = []
    for aid in agent_ids:
        aid = str(aid)
        nodes.append(
            Node(
                package=package,
                executable=executable,
                name=f"single_agent_optimizer_{aid}",
                output="screen",
                parameters=[{
                    "agent_id": aid,
                    "reward_correct_X": reward_correct,
                    "reward_correct_Y": reward_correct,
                }],
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription([
        # Set these to whatever you installed your node as (ros2 run <package> <executable>)
        DeclareLaunchArgument("package", default_value="sim_humans"),
        DeclareLaunchArgument("executable", default_value="sim_optimzed_agent"),

        # Option A: spawn N agents: agent_prefix0..N-1
        DeclareLaunchArgument("num_agents", default_value="0"),
        DeclareLaunchArgument("agent_prefix", default_value="robot_"),

        # Option B: explicit list (overrides num_agents if non-empty)
        DeclareLaunchArgument("agent_ids", default_value="[]"),  # JSON list like '["robot","a","b"]'

        # Only knob you care about
        DeclareLaunchArgument("reward_correct", default_value="1.0"),

        OpaqueFunction(function=_build_nodes),
    ])

