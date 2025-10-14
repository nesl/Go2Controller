#!/usr/bin/env python3
import math, time
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3Stamped
from visualization_msgs.msg import Marker

def quat_from_yaw_pitch(yaw, pitch, roll=0.0):
    """Build quaternion from ZYX (yaw, pitch, roll)."""
    cy = math.cos(yaw * 0.5);  sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5);  sr = math.sin(roll * 0.5)
    # Z (yaw) * Y (pitch) * X (roll)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return qx, qy, qz, qw

class DoAMarker(Node):
    def __init__(self):
        super().__init__('doa_marker')

        # Params
        self.declare_parameter('frame_id', 'mic_array')
        self.declare_parameter('marker_topic', '/audio/doa_marker')
        self.declare_parameter('arrow_length', 0.5)     # meters
        self.declare_parameter('shaft_diameter', 0.02)  # meters
        self.declare_parameter('head_diameter', 0.06)   # meters
        self.declare_parameter('color_rgba', [0.1, 0.8, 0.1, 0.9])  # green-ish
        self.declare_parameter('listen_vector_topic', True)         # prefer /audio/odas/doa
        self.declare_parameter('ttl_sec', 1.5)  # hide arrow if stale

        self.frame_id  = self.get_parameter('frame_id').value
        self.topic     = self.get_parameter('marker_topic').value
        self.length    = float(self.get_parameter('arrow_length').value)
        self.shaft_d   = float(self.get_parameter('shaft_diameter').value)
        self.head_d    = float(self.get_parameter('head_diameter').value)
        self.r, self.g, self.b, self.a = [float(x) for x in self.get_parameter('color_rgba').value]
        self.use_vec   = bool(self.get_parameter('listen_vector_topic').value)
        self.ttl       = float(self.get_parameter('ttl_sec').value)

        self.pub = self.create_publisher(Marker, self.topic, 10)

        # Latest direction (unit vector)
        self.dir = None
        self.dir_ts = 0.0

        if self.use_vec:
            self.create_subscription(Vector3Stamped, '/audio/odas/doa', self.on_vec, 10)
            self.get_logger().info("Listening to /audio/odas/doa (Vector3Stamped)")
        self.create_subscription(Float32, '/audio/active_azimuth_deg', self.on_az, 10)

        # Publish at ~15 Hz so RViz stays updated/visible
        self.create_timer(1.0/15.0, self.tick)

        self.get_logger().info(f"Publishing DoA marker on {self.topic} in frame '{self.frame_id}'")

    def on_vec(self, msg: Vector3Stamped):
        # Use vector directly (assumed already in mic_array frame)
        vx, vy, vz = float(msg.vector.x), float(msg.vector.y), float(msg.vector.z)
        n = math.sqrt(vx*vx + vy*vy + vz*vz)
        if n < 1e-6:
            return
        self.dir = (vx/n, vy/n, vz/n)
        self.dir_ts = time.time()

    def on_az(self, msg: Float32):
        # Fallback: have azimuth only (degrees), assume zero elevation
        az = math.radians(float(msg.data))
        vx = math.cos(az); vy = math.sin(az); vz = 0.0
        self.dir = (vx, vy, vz)
        self.dir_ts = time.time()

    def tick(self):
        now = time.time()
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "doa"
        m.id = 1
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.lifetime.sec = 0  # persistent until replaced

        # Hide if stale
        if self.dir is None or (now - self.dir_ts) > self.ttl:
            # Publish a transparent (alpha=0) tiny marker to effectively hide
            m.scale.x = 0.0001
            m.color.a = 0.0
            self.pub.publish(m)
            return

        vx, vy, vz = self.dir

        # Arrow points along +X in marker frame; rotate to (vx,vy,vz):
        # yaw from x,y; pitch from z
        yaw = math.atan2(vy, vx)
        pitch = math.atan2(-vz, math.sqrt(vx*vx + vy*vy))  # negative to tilt down when z>0
        qx, qy, qz, qw = quat_from_yaw_pitch(yaw, pitch, roll=0.0)

        # Pose at origin
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.0
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw

        # Size
        m.scale.x = self.length          # arrow length
        m.scale.y = self.shaft_d         # shaft diameter
        m.scale.z = self.head_d          # head diameter

        # Color
        m.color.r = self.r
        m.color.g = self.g
        m.color.b = self.b
        m.color.a = self.a

        self.pub.publish(m)

def main():
    rclpy.init()
    n = DoAMarker()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
