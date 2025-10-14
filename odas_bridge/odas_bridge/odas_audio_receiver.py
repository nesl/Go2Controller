#!/usr/bin/env python3
import rclpy, socket
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray

class AudioTCPServer(Node):
    def __init__(self):
        super().__init__('odas_audio_tcp_server')
        self.declare_parameter('port', 9004)  # sss.postfiltered.interface port
        self.port = int(self.get_parameter('port').value)

        self.pub = self.create_publisher(UInt8MultiArray, '/mic/audio', 50)

        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(('0.0.0.0', self.port))
        self.srv.listen(1)
        self.srv.setblocking(False)
        self.conn = None

        self.get_logger().info(f'Audio TCP listening on :{self.port}')
        self.create_timer(0.001, self.loop)

    def loop(self):
        if self.conn is None:
            try:
                self.conn, _ = self.srv.accept()
                self.conn.setblocking(False)
                self.get_logger().info('Audio: ODAS connected')
            except BlockingIOError:
                return
        try:
            data = self.conn.recv(8192)  # raw PCM16 LE bytes
            if not data:
                self.conn.close(); self.conn = None
                self.get_logger().info('Audio: client disconnected')
                return
            msg = UInt8MultiArray()
            msg.data = list(data)  # keep as raw bytes
            self.pub.publish(msg)
        except BlockingIOError:
            pass

def main():
    rclpy.init()
    n = AudioTCPServer()
    try:
        rclpy.spin(n)
    finally:
        if n.conn: n.conn.close()
        n.srv.close()
        n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
