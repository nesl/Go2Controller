#!/bin/bash

echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=78" >> ~/.bashrc
echo "export CONN_TYPE='webrtc'" >> ~/.bashrc
echo "export ROBOT_IP='ROBOT_IP_ADDRESS'" >> ~/.bashrc

/bin/bash
