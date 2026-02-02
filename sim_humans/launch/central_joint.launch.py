#!/usr/bin/env python3
from __future__ import annotations

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    agent_ids = ["robot", "human_a", "human_b"]

    central = Node(
        package="sim_humans",
        executable="sim_centralized_optimizer",  # or console script name
        name="central_joint_optimizer",
        output="screen",
        parameters=[{
            "agent_ids": agent_ids,
        }],
    )

    followers = [
        Node(
            package="sim_humans",
            executable="sim_centralized_agent",  # or console script name
            name=f"plan_follower_{aid}",
            output="screen",
            parameters=[{
                "agent_id": aid,
            }],
        )
        for aid in agent_ids
    ]

    return LaunchDescription([
        central,
        *followers,
    ])

