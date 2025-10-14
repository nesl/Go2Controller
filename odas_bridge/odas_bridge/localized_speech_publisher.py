#!/usr/bin/env python3
import rclpy, time
from rclpy.node import Node
from std_msgs.msg import String, Int32, Float32

class LocalizedSpeechPublisher(Node):
    def __init__(self):
        super().__init__('localized_speech_publisher')
        self.sub_text = self.create_subscription(String, '/stt/text', self.on_text, 10)
        self.sub_idx  = self.create_subscription(Int32, '/audio/active_channel', self.on_idx, 10)
        self.sub_az   = self.create_subscription(Float32, '/audio/active_azimuth_deg', self.on_az, 10)
        self.pub      = self.create_publisher(String, '/stt/localized', 10)
        self.last_idx = None
        self.last_az  = None
        self.last_az_t= 0.0

    def on_idx(self, m: Int32):
        self.last_idx = int(m.data)

    def on_az(self, m: Float32):
        self.last_az  = float(m.data)
        self.last_az_t= time.time()

    def on_text(self, m: String):
        # Pair the text with the most recent channel + azimuth
        payload = {
            "text": m.data,
            "channel": self.last_idx,
            "az_deg": self.last_az,
            "az_ts": self.last_az_t
        }
        out = String(); out.data = str(payload)
        self.pub.publish(out)
        self.get_logger().info(out.data)

def main():
    rclpy.init()
    n = LocalizedSpeechPublisher()
    try: rclpy.spin(n)
    finally: n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
