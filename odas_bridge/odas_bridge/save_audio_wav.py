# stt_node/tools/save_audio_raw.py
#!/usr/bin/env python3
import rclpy, os
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray
PATH = '/tmp/odas_capture.raw'
class Saver(Node):
    def __init__(self):
        super().__init__('odas_audio_raw_saver')
        if os.path.exists(PATH): os.remove(PATH)
        self.f = open(PATH, 'ab', buffering=0)
        self.sub = self.create_subscription(UInt8MultiArray, '/mic/audio', self.cb, 50)
        self.get_logger().info(f'Writing RAW bytes to {PATH} … Ctrl+C to stop.')
    def cb(self, msg): self.f.write(bytes(msg.data))
    def destroy_node(self):
        try: self.f.close()
        finally: super().destroy_node()
def main():
    rclpy.init(); n = Saver()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally: n.destroy_node(); rclpy.shutdown()
if __name__ == '__main__': main()
