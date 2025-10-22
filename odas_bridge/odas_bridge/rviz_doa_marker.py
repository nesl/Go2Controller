#!/usr/bin/env python3
import math, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Vector3Stamped
from visualization_msgs.msg import Marker

EPS = 1e-4

def quat_from_yaw_pitch(yaw, pitch, roll=0.0):
    cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cr, sr = math.cos(roll*0.5), math.sin(roll*0.5)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return qx, qy, qz, qw

class DoAMarker(Node):
    def __init__(self):
        super().__init__('doa_marker')
        self.declare_parameter('frame_id', 'mic_array')
        self.declare_parameter('marker_topic', '/audio/doa_marker')
        self.declare_parameter('arrow_length', 0.6)
        self.declare_parameter('shaft_diameter', 0.03)
        self.declare_parameter('head_diameter', 0.08)
        self.declare_parameter('color_rgba', [0.1, 0.8, 0.1, 0.9])
        self.declare_parameter('ttl_sec', 1.5)
        self.declare_parameter('prefer_vector', True)

        self.frame_id  = self.get_parameter('frame_id').value or 'mic_array'
        self.topic     = self.get_parameter('marker_topic').value
        self.length    = max(EPS, float(self.get_parameter('arrow_length').value))
        self.shaft_d   = max(EPS, float(self.get_parameter('shaft_diameter').value))
        self.head_d    = max(EPS, float(self.get_parameter('head_diameter').value))
        self.r, self.g, self.b, self.a = [float(x) for x in self.get_parameter('color_rgba').value]
        self.ttl       = float(self.get_parameter('ttl_sec').value)
        self.prefer_vec= bool(self.get_parameter('prefer_vector').value)

        self.pub = self.create_publisher(Marker, self.topic, 10)

        self.dir = None
        self.dir_ts = 0.0
        self.voice_active = False

        # Inputs
        if self.prefer_vec:
            self.create_subscription(Vector3Stamped, '/audio/odas/doa', self.on_vec, 10)
        self.create_subscription(Float32, '/audio/active_azimuth_deg', self.on_az, 10)
        self.create_subscription(Bool, '/audio/voice_active_speech', self.on_vact, 10)

        self.create_timer(1.0/15.0, self.tick)
        self.get_logger().info(f"RViz DoA marker → {self.topic}, frame='{self.frame_id}'")

    def on_vact(self, msg: Bool):
        self.voice_active = bool(msg.data)

    def on_vec(self, msg: Vector3Stamped):
        vx, vy, vz = float(msg.vector.x), float(msg.vector.y), float(msg.vector.z)
        n = math.sqrt(vx*vx + vy*vy + vz*vz)
        if n < 1e-6:
            return
        self.dir = (vx/n, vy/n, vz/n)
        self.dir_ts = time.time()

    def on_az(self, msg: Float32):
        # fallback if vector not available; assumes zero elevation
        az = math.radians(float(msg.data))
        self.dir = (math.cos(az), math.sin(az), 0.0)
        self.dir_ts = time.time()

    def tick(self):
        now = time.time()
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "doa"
        m.id = 1

        # Hide if inactive or stale
        if (not self.voice_active) or (self.dir is None) or ((now - self.dir_ts) > self.ttl):
            m.action = Marker.DELETE
            self.pub.publish(m)
            return

        vx, vy, vz = self.dir
        yaw = math.atan2(vy, vx)
        pitch = math.atan2(-vz, math.sqrt(vx*vx + vy*vy))
        qx, qy, qz, qw = quat_from_yaw_pitch(yaw, pitch)

        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.0
        m.scale.x = self.length
        m.scale.y = self.shaft_d
        m.scale.z = self.head_d
        m.color.r, m.color.g, m.color.b, m.color.a = self.r, self.g, self.b, self.a
        self.pub.publish(m)

def main():
    rclpy.init()
    n = DoAMarker()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
