#!/usr/bin/env python3
import rclpy, socket, json
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Bool

class TrackedTCPServer(Node):
    def __init__(self):
        super().__init__('odas_tracked_tcp_server')
        self.declare_parameter('port', 9000)
        self.port = int(self.get_parameter('port').value)

        self.declare_parameter('min_activity', 0.2)  # tweak 0..1
        self.min_activity = float(self.get_parameter('min_activity').value)

        self.pub_json = self.create_publisher(String, '/audio/odas/tracked_json', 10)
        self.pub_vec  = self.create_publisher(Vector3Stamped, '/audio/odas/doa', 10)
        self.pub_vact  = self.create_publisher(Bool, '/audio/voice_active', 10)

        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(('0.0.0.0', self.port))
        self.srv.listen(1)
        self.srv.setblocking(False)
        self.conn = None

        # streaming JSON reassembly
        self.buf = bytearray()
        self.depth = 0            # brace depth for {...}
        self.in_obj = False

        self.get_logger().info(f'Tracked TCP listening on :{self.port}')
        self.create_timer(0.001, self.loop)

    def loop(self):
        if self.conn is None:
            try:
                self.conn, _ = self.srv.accept()
                self.conn.setblocking(False)
                self.get_logger().info('Tracked: ODAS connected')
                self.buf.clear(); self.depth = 0; self.in_obj = False
            except BlockingIOError:
                return

        try:
            data = self.conn.recv(65535)
            if not data:
                self.conn.close(); self.conn = None
                self.get_logger().info('Tracked: client disconnected')
                return
            self._feed(data)
        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().warn(f'Tracked recv error: {e}')

    def _feed(self, data: bytes):
        for b in data:
            ch = chr(b)
            if ch == '{':
                self.depth += 1
                self.in_obj = True
            if self.in_obj:
                self.buf.append(b)
            if ch == '}':
                self.depth -= 1
                if self.in_obj and self.depth == 0:
                    # end of one complete JSON object
                    blob = bytes(self.buf)
                    self.buf.clear(); self.in_obj = False
                    self._publish_one(blob)

    def _publish_one(self, blob: bytes):
        try:
            s = blob.decode('utf-8', errors='ignore')
            j = json.loads(s)

            # publish full JSON once per object
            self.pub_json.publish(String(data=s))

            # derive a simple DoA vector from the most active source (activity > 0)
            srcs = j.get('src', j.get('sources', []))

            if srcs:
                best = max(srcs, key=lambda t: t.get('activity', 0.0))
                if float(best.get('activity', 0.0)) > self.min_activity:
                    v = Vector3Stamped()
                    v.header.stamp = self.get_clock().now().to_msg()
                    v.header.frame_id = 'mic_array'
                    v.vector.x = float(best.get('x', 0.0))
                    v.vector.y = float(best.get('y', 0.0))
                    v.vector.z = float(best.get('z', 0.0))
                    self.pub_vec.publish(v)
                    self.pub_vact.publish(Bool(data=True))
                else:
                    self.pub_vact.publish(Bool(data=False))
            else:
                self.pub_vact.publish(Bool(data=False))
        except Exception as e:
            self.get_logger().warn(f'JSON parse error: {e}')
            self.pub_vact.publish(Bool(data=False))

def main():
    rclpy.init()
    n = TrackedTCPServer()
    try:
        rclpy.spin(n)
    finally:
        if n.conn: n.conn.close()
        n.srv.close()
        n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
