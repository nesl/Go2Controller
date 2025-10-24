#!/usr/bin/env python3
import os
import re
import math
import json
import sqlite3
from dataclasses import dataclass
from typing import Dict

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_srvs.srv import Trigger

from bt_msgs.msg import BtReading


@dataclass
class BestSite:
    x: float
    y: float
    rssi: int
    ts: float  # epoch seconds


class BtRssiMapper(Node):
    """
    Aggregates BtReading messages into a persistent SQLite DB and keeps
    a time-decayed "best" location per device_id. Publishes MarkerArray
    for RViz and exposes simple Trigger services.
    """

    def __init__(self):
        super().__init__('bt_rssi_mapper')

        # ---------------- Parameters ----------------
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('db_path', os.path.expanduser('~/.bt_rssi_map.sqlite'))
        self.declare_parameter('decay_half_life_sec', 300.0)        # 5 min
        self.declare_parameter('min_rssi', -120)                    # filter ultra-weak
        self.declare_parameter('name_pattern', r'CNode\d+')         # keep only CNode# by default
        self.declare_parameter('device_allow_regex', '')            # optional allowlist by MAC
        self.declare_parameter('device_deny_regex', '')             # optional denylist by MAC
        self.declare_parameter('marker_ns', 'bt_best')
        self.declare_parameter('marker_scale', 0.35)

        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.db_path = self.get_parameter('db_path').get_parameter_value().string_value
        self.half_life = float(self.get_parameter('decay_half_life_sec').get_parameter_value().double_value)
        self.tau = self.half_life / math.log(2.0) if self.half_life > 0 else 1e9
        self.min_rssi = int(self.get_parameter('min_rssi').get_parameter_value().integer_value)
        self.marker_ns = self.get_parameter('marker_ns').get_parameter_value().string_value
        self.marker_scale = float(self.get_parameter('marker_scale').get_parameter_value().double_value)

        # Filters
        name_pat = self.get_parameter('name_pattern').get_parameter_value().string_value
        self.name_pat = re.compile(name_pat) if name_pat else None
        allow_pat = self.get_parameter('device_allow_regex').get_parameter_value().string_value
        deny_pat = self.get_parameter('device_deny_regex').get_parameter_value().string_value
        self.allow_pat = re.compile(allow_pat) if allow_pat else None
        self.deny_pat  = re.compile(deny_pat) if deny_pat else None

        # ---------------- TF ----------------
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------- DB ----------------
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, isolation_level=None)  # autocommit
        self._ensure_schema()

        # ---------------- State ----------------
        self.best: Dict[str, BestSite] = {}

        # ---------------- ROS I/O ----------------
        self.sub = self.create_subscription(BtReading, '/bt/readings', self.on_reading, 1000)
        self.marker_pub = self.create_publisher(MarkerArray, '/bt/best_sites_markers', 10)

        # Timers and services
        self.timer = self.create_timer(1.0, self.publish_markers)
        self.srv_dump_dbpath = self.create_service(Trigger, '/bt/dump_db_path', self.on_dump_path)
        self.srv_best_sites  = self.create_service(Trigger, '/bt/best_sites', self.on_best_sites)

        self.get_logger().info(
            f"BtRssiMapper: target_frame={self.target_frame}, db={self.db_path}, half_life={self.half_life}s, "
            f"name_pattern={name_pat or '(none)'}"
        )

    # ---------------- Schema ----------------
    def _ensure_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY,
                device_id   TEXT,
                device_name TEXT,
                scanner_id  TEXT,
                rssi        INTEGER,
                x           REAL,
                y           REAL,
                ts          REAL
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dev_ts ON detections(device_id, ts);")

    # ---------------- Reading handler ----------------
    def on_reading(self, msg: BtReading):
        # Basic filters (RSSI, name pattern, allow/deny lists)
        if msg.rssi < self.min_rssi:
            return
        name = msg.device_name or ""
        mac  = msg.device_id or ""
        if self.name_pat and not self.name_pat.fullmatch(name):
            return
        if self.allow_pat and not (self.allow_pat.search(mac)):
            return
        if self.deny_pat and self.deny_pat.search(mac):
            return

        # Time and TF lookup
        try:
            t = Time.from_msg(msg.stamp)
        except Exception:
            t = self.get_clock().now()

        try:
            tf = self.tf_buffer.lookup_transform(self.target_frame, msg.frame_id, t, timeout=Duration(seconds=0.2))
            x = tf.transform.translation.x
            y = tf.transform.translation.y
        except (LookupException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed {msg.frame_id}->{self.target_frame} at {msg.stamp.sec}.{msg.stamp.nanosec}: {e}")
            return

        # Persist
        ts_epoch = t.nanoseconds / 1e9
        self.conn.execute(
            "INSERT INTO detections(device_id, device_name, scanner_id, rssi, x, y, ts) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mac, name, msg.scanner_id, int(msg.rssi), float(x), float(y), float(ts_epoch))
        )

        # Update best (time-decayed)
        now_epoch = self.get_clock().now().nanoseconds / 1e9
        new_score = self._decayed_score(msg.rssi, now_epoch - ts_epoch)

        b = self.best.get(mac)
        if b is None:
            self.best[mac] = BestSite(x=x, y=y, rssi=msg.rssi, ts=ts_epoch)
        else:
            old_score = self._decayed_score(b.rssi, now_epoch - b.ts)
            if new_score >= old_score:
                self.best[mac] = BestSite(x=x, y=y, rssi=msg.rssi, ts=ts_epoch)

    def _decayed_score(self, rssi: int, age_sec: float) -> float:
        # exponential decay over time; higher (less negative) RSSI is better
        return float(rssi) * math.exp(-age_sec / self.tau)

    # ---------------- Visualization ----------------
    def publish_markers(self):
        if not self.best:
            return
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        i = 0
        for dev, b in self.best.items():
            m = Marker()
            m.header.frame_id = self.target_frame
            m.header.stamp = now
            m.ns = self.marker_ns
            m.id = i; i += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = b.x
            m.pose.position.y = b.y
            m.pose.position.z = 0.2
            m.scale.x = m.scale.y = m.scale.z = self.marker_scale
            # Map RSSI [-100..-40] to gradient green->red
            val = max(0.0, min(1.0, (b.rssi + 100.0) / 60.0))
            m.color.r = 1.0 - val
            m.color.g = val
            m.color.b = 0.0
            m.color.a = 0.9
            # Put device id in text (optional visualization in RViz)
            m.text = f"{dev} ({b.rssi} dBm)"
            m.lifetime = Duration(seconds=2.0).to_msg()
            ma.markers.append(m)
        self.marker_pub.publish(ma)

    # ---------------- Services ----------------
    def on_dump_path(self, req, res):
        res.success = True
        res.message = self.db_path
        return res

    def on_best_sites(self, req, res):
        # Return JSON: { device_id: {x,y,rssi,ts} }
        payload = {
            dev: dict(x=b.x, y=b.y, rssi=b.rssi, ts=b.ts)
            for dev, b in self.best.items()
        }
        res.success = True
        res.message = json.dumps(payload)
        return res

    # ---------------- Shutdown ----------------
    def destroy_node(self):
        try:
            self.conn.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = BtRssiMapper()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
