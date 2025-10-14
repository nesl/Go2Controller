#!/usr/bin/env python3
import rclpy, json, math, time, numpy as np
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray, Int32, Float32, String

def azimuth_deg(x, y) -> float:
    return math.degrees(math.atan2(y, x))  # [-180, 180)

class ActiveChannelSelector(Node):
    def __init__(self):
        super().__init__('active_channel_selector')
        # params
        self.declare_parameter('input_topic', '/mic/audio')
        self.declare_parameter('n_channels', 4)
        self.declare_parameter('hop_samples', 128)
        self.declare_parameter('switch_hysteresis_db', 3.0)
        self.declare_parameter('min_switch_ms', 400)
        self.declare_parameter('az_ttl_sec', 1.5)  # how long to keep last az before considering it stale

        self.N    = int(self.get_parameter('n_channels').value)
        self.hop  = int(self.get_parameter('hop_samples').value)
        self.hyst = float(self.get_parameter('switch_hysteresis_db').value)
        self.hold = float(self.get_parameter('min_switch_ms').value) / 1000.0
        self.az_ttl = float(self.get_parameter('az_ttl_sec').value)

        # subs & pubs
        self.sub_audio = self.create_subscription(
            UInt8MultiArray, self.get_parameter('input_topic').value, self.on_audio, 200
        )
        self.sub_json  = self.create_subscription(
            String, '/audio/odas/tracked_json', self.on_json, 50
        )

        self.pub_active_bytes = self.create_publisher(UInt8MultiArray, '/mic/audio/active', 50)
        self.pub_active_idx   = self.create_publisher(Int32, '/audio/active_channel', 10)
        self.pub_active_az    = self.create_publisher(Float32, '/audio/active_azimuth_deg', 10)

        # state
        self.current_idx = 0
        self.last_switch = 0.0
        self.latest_az   = None   # degrees
        self.latest_az_ts= 0.0    # seconds epoch

        self.get_logger().info(
            f'ActiveChannelSelector: N={self.N}, hysteresis={self.hyst} dB, min_hold={self.hold}s, az_ttl={self.az_ttl}s'
        )

    # --- DoA from ODAS tracked JSON ---
    def on_json(self, msg: String):
        try:
            j = json.loads(msg.data)
            # ODAS variants: "src" or "sources"
            srcs = j.get('src', j.get('sources', []))
            if not isinstance(srcs, list) or len(srcs) == 0:
                # no update; keep last az (sticky)
                return

            # pick the entry with max activity (even if it's 0.0)
            best = max(srcs, key=lambda s: float(s.get('activity', 0.0)))
            x = float(best.get('x', 0.0))
            y = float(best.get('y', 0.0))
            az = azimuth_deg(x, y)

            # update + publish
            self.latest_az = az
            self.latest_az_ts = time.time()
            self.pub_active_az.publish(Float32(data=float(self.latest_az)))
        except Exception as e:
            # ignore malformed chunks
            pass

    # --- audio: pick active separated channel and publish mono bytes ---
    def on_audio(self, msg: UInt8MultiArray):
        x = np.frombuffer(bytes(msg.data), dtype=np.int16)
        if len(x) < self.N:
            return
        # truncate to full interleaved frames
        rem = len(x) % self.N
        if rem != 0:
            x = x[:len(x) - rem]
        frames = x.reshape(-1, self.N)  # [hop, ch]

        # RMS (dBFS) per channel
        rms = np.sqrt(np.mean(frames.astype(np.float32)**2, axis=0) + 1e-12)
        db  = 20.0 * np.log10(np.maximum(rms / 32768.0, 1e-9))
        loudest = int(np.argmax(db))

        # hysteresis + min-hold to avoid rapid flipping
        now = time.time()
        if loudest != self.current_idx:
            if (db[loudest] - db[self.current_idx]) >= self.hyst and (now - self.last_switch) >= self.hold:
                self.current_idx = loudest
                self.last_switch = now
                self.pub_active_idx.publish(Int32(data=self.current_idx))

        # publish selected mono bytes
        active_bytes = frames[:, self.current_idx].astype(np.int16).tobytes()
        out = UInt8MultiArray(); out.data = list(active_bytes)
        self.pub_active_bytes.publish(out)

        # expire very stale az (optional)
        if self.latest_az is not None and (now - self.latest_az_ts) > self.az_ttl:
            self.latest_az = None  # consumer may treat None as "unknown"
            # (we don't publish anything here; /audio/active_azimuth_deg is event-driven on updates)

def main():
    rclpy.init()
    n = ActiveChannelSelector()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
