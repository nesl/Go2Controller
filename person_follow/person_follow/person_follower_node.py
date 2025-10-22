#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist, PointStamped
from vision_msgs.msg import Detection2DArray

from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point
import re
from std_msgs.msg import String
import json
from rclpy.duration import Duration  # (already imported above in your file)
from rclpy.time import Time

class PersonFollower(Node):
    """
    Subscribe: detected_objects  (vision_msgs/Detection2DArray) in map frame
    Publish:   /cmd_vel          (geometry_msgs/Twist)
    TF:        map -> base_link

    Behavior:
      - Track nearest 'person' with a 3D pose
      - Maintain ~desired_distance in front
      - Stop if too close, or if person lost for lost_timeout_s
    """

    def __init__(self):
        super().__init__("person_follower")

        # ---------------- Params ----------------
        self.declare_parameter("detected_topic", "detected_objects")
        self.declare_parameter("robot_cmd_vel", "/cmd_vel")
        self.declare_parameter("target_class", "person")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        # control
        self.declare_parameter("desired_distance_m", 1.0)     # target following distance
        self.declare_parameter("stop_distance_m", 0.6)        # immediate stop if closer than this
        self.declare_parameter("max_lin_speed", 0.6)          # m/s
        self.declare_parameter("max_ang_speed", 1.2)          # rad/s
        self.declare_parameter("k_lin", 0.8)                  # linear gain on distance error
        self.declare_parameter("k_ang", 1.5)                  # angular gain on bearing
        self.declare_parameter("yaw_deadband_rad", 0.05)      # avoid oscillations
        self.declare_parameter("lost_timeout_s", 0.8)         # stop if not seen within this time
        self.declare_parameter("smoothing_alpha", 0.4)        # low-pass on commands [0..1]

        # --- Name-orient parameters ---
        self.declare_parameter("stt_json_topic", "/audio/stt_doa_json")
        self.declare_parameter("trigger_words", ["bob"])   # case-insensitive, word-boundary
        self.declare_parameter("name_debounce_s", 2.0)     # ignore re-triggers within this
        self.declare_parameter("name_timeout_s", 3.0)      # give up if not aligned by then
        self.declare_parameter("name_k_ang", 1.5)          # P gain for name-orient
        self.declare_parameter("name_max_ang_speed", 1.0)  # rad/s clamp
        self.declare_parameter("name_min_turn_speed", 0.15)# rad/s minimum to overcome friction
        self.declare_parameter("name_stop_thresh_deg", 2.0)# done if |err| <= this



        # --------------- Read params ---------------
        self.detected_topic   = self.get_parameter("detected_topic").value
        self.cmd_vel_topic    = self.get_parameter("robot_cmd_vel").value
        self.target_class     = self.get_parameter("target_class").value
        self.map_frame        = self.get_parameter("map_frame").value
        self.base_frame       = self.get_parameter("base_frame").value

        self.desired_d        = float(self.get_parameter("desired_distance_m").value)
        self.stop_d           = float(self.get_parameter("stop_distance_m").value)
        self.max_v            = float(self.get_parameter("max_lin_speed").value)
        self.max_w            = float(self.get_parameter("max_ang_speed").value)
        self.k_lin            = float(self.get_parameter("k_lin").value)
        self.k_ang            = float(self.get_parameter("k_ang").value)
        self.db_yaw           = float(self.get_parameter("yaw_deadband_rad").value)
        self.lost_timeout_s   = float(self.get_parameter("lost_timeout_s").value)
        self.alpha            = float(self.get_parameter("smoothing_alpha").value)


        self.stt_json_topic    = self.get_parameter("stt_json_topic").value
        self.trigger_words     = [str(w).lower() for w in self.get_parameter("trigger_words").value]
        self.name_debounce_s   = float(self.get_parameter("name_debounce_s").value)
        self.name_timeout_s    = float(self.get_parameter("name_timeout_s").value)
        self.name_k_ang        = float(self.get_parameter("name_k_ang").value)
        self.name_max_w        = float(self.get_parameter("name_max_ang_speed").value)
        self.name_min_turn     = float(self.get_parameter("name_min_turn_speed").value)
        self.name_stop_deg     = float(self.get_parameter("name_stop_thresh_deg").value)


        self._name_regex = re.compile(r"\b(" + "|".join(map(re.escape, self.trigger_words)) + r")\b", re.IGNORECASE)

        # Name-orient state
        self._orient_active = False
        self._orient_goal_delta = 0.0      # radians; how much we want to rotate from start yaw
        self._orient_start_yaw = 0.0       # radians; yaw when goal was armed
        self._orient_started_at = None     # rclpy Time
        self._last_name_trigger_at = self.get_clock().now() - Duration(seconds=999.0)

        stt_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
        self.sub_stt = self.create_subscription(String, self.stt_json_topic, self._on_stt_json, stt_qos)




        # --------------- QoS (sensor-ish) ---------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # --------------- Pub/Sub ---------------
        self.sub = self.create_subscription(Detection2DArray, self.detected_topic,
                                            self.dets_cb, qos)
        self.pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # --------------- TF ---------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --------------- State ---------------
        self.last_seen_time = None
        self.last_cmd = Twist()

        # 20 Hz control loop
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("person_follower started. Following nearest person.")

        # cache of the latest selected target in map frame
        self._latest_target_map_ps: Optional[PointStamped] = None
        
        self.declare_parameter("turn_ref_frame", "odom")  # continuous frame to measure yaw (odom recommended)
        self.turn_ref_frame = self.get_parameter("turn_ref_frame").value


    # ------------------ Helpers ------------------

    def _current_yaw_ref_base(self) -> float:
        """
        Return robot yaw (rad) of base_frame in the chosen reference frame (default: odom),
        wrapped to (-pi, pi]. This uses TF: target = turn_ref_frame, source = base_frame,
        i.e., the pose of base in the ref frame (no map involvement).
        """

        try:
            tf_base_to_ref = self.tf_buffer.lookup_transform(
                self.turn_ref_frame,   # target (ref)
                self.base_frame,       # source (base)
                Time(),                # latest
                Duration(seconds=0.2),
            )
            q = tf_base_to_ref.transform.rotation
            # Yaw from quaternion (Z-up)
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)  # (-pi, pi]
            return yaw
        except Exception as e:
            self.get_logger().warn(f"Yaw lookup (ref={self.turn_ref_frame}) failed: {e}")
            return 0.0


    def _wrap_pi(self, a: float) -> float:

        a = (a + math.pi) % (2.0*math.pi) - math.pi
        return a



    def _select_nearest_person(self, dets: Detection2DArray) -> Optional[PointStamped]:
        best_ps = None
        best_dist2 = float("inf")

        for d in dets.detections:
            # Should have at least one hypothesis with pose filled by your detector
            if not d.results:
                continue
            hyp = d.results[0]
            if not hyp.hypothesis.class_id:
                continue
            if self.target_class not in hyp.hypothesis.class_id.lower():
                continue

            # pose is expressed in dets.header.frame_id (expected: map)
            p = hyp.pose.pose.position
            ps = PointStamped()
            ps.header = dets.header
            ps.point.x, ps.point.y, ps.point.z = float(p.x), float(p.y), float(p.z)

            # distance in map from origin isn't meaningful—compare in base frame after TF,
            # but to choose "nearest", we can temporarily use map distance to (0,0) if multiple persons.
            # We'll transform later anyway.
            # Use planar distance wrt map origin—fine if multiple persons aren’t very close together.
            dist2 = p.x * p.x + p.y * p.y
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_ps = ps

        return best_ps

    def dets_cb(self, msg: Detection2DArray):
        target = self._select_nearest_person(msg)
        if target is None:
            return

        # Cache + timestamp
        self._latest_target_map_ps = target
        self.last_seen_time = self.get_clock().now()

    def _transform_to_base(self, ps_map: PointStamped) -> Optional[PointStamped]:
        try:
            # Ensure we have a transform for the target stamp; fall back to latest if needed
            tf_map_to_base = self.tf_buffer.lookup_transform(
                self.base_frame,
                ps_map.header.frame_id,
                rclpy.time.Time.from_msg(ps_map.header.stamp),
                timeout=Duration(seconds=0.2),
            )
        except TransformException:
            try:
                tf_map_to_base = self.tf_buffer.lookup_transform(
                    self.base_frame, ps_map.header.frame_id, rclpy.time.Time(), timeout=Duration(seconds=0.2)
                )
            except TransformException as e2:
                self.get_logger().warn(f"TF map->base failed: {e2}")
                return None

        try:
            ps_base = do_transform_point(ps_map, tf_map_to_base)
            return ps_base
        except Exception as e:
            self.get_logger().warn(f"Transform point failed: {e}")
            return None

    def _lp(self, prev: float, new: float) -> float:
        return (1.0 - self.alpha) * prev + self.alpha * new

    def _on_stt_json(self, msg: String):
        """
        Expect JSON like: {"text": "...","azimuth_deg": <float>, ...}
        azimuth_deg is relative to robot base frame: 0°=+X (forward), +90°=left.
        We rotate by -azimuth so the speaker lands at ~0° (straight ahead).
        """

        # Debounce
        now = self.get_clock().now()
        if (now - self._last_name_trigger_at) < rclpy.duration.Duration(seconds=self.name_debounce_s):
            return

        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Bad STT JSON: {e}")
            return

        text = (data.get("text") or "").strip()
        if not text or not self._name_regex.search(text):
            return

        az = data.get("azimuth_deg", None)
        if az is None:
            self.get_logger().warn("Trigger heard but no azimuth_deg in STT JSON.")
            return

        # Arm orientation goal
        yaw_now = self._current_yaw_ref_base()   # was _current_yaw_map_base()
        self._orient_start_yaw = yaw_now
        
        # Normalize azimuth to (-180, 180], and avoid the ambiguous +180 exactly
        az_deg = float(az)
        if az_deg <= -180.0 or az_deg > 180.0:
            az_deg = ((az_deg + 180.0) % 360.0) - 180.0
        # If exactly +180, choose a direction; use the sign of the provided value if any.
        if az_deg == 180.0:
            az_deg = 179.9  # tiny nudge to break tie to CCW (left). If you prefer right, use -179.9.

        # Arm the relative rotation goal: +deg = CCW (left), -deg = CW (right)
        self._orient_goal_delta = math.radians(az_deg)
        
        self._orient_started_at = now
        self._orient_active = True
        self._last_name_trigger_at = now

        self.get_logger().info(f"Name trigger: '{text}' az={float(az):.1f}° → rotate {math.degrees(self._orient_goal_delta):.1f}°")



    # ------------------ Control Loop ------------------

    def control_loop(self):
        now = self.get_clock().now()

        # 0) If we’re in name-orient mode, rotate first; pause following.
        if self._orient_active:

            # Timeout
            if (now - self._orient_started_at) > Duration(seconds=self.name_timeout_s):
                self.get_logger().info("Name-orient timeout; resuming follow.")
                self._orient_active = False
                self._stop("name-timeout")
                return

            # How far we've turned since we started the name-orient
            yaw_now = self._current_yaw_ref_base()
            turned = self._wrap_pi(yaw_now - self._orient_start_yaw)       # (-pi, pi]

            # Shortest-angle error to remaining goal
            # err = wrap(goal - turned)
            raw = self._orient_goal_delta - turned
            err = math.atan2(math.sin(raw), math.cos(raw))                  # (-pi, pi]

            # Done?
            if abs(math.degrees(err)) <= self.name_stop_deg:
                self._orient_active = False
                self._stop("name-aligned")
                return

            # P-control on angular error
            w = self.name_k_ang * err

            # Minimum turning rate to overcome static friction
            if abs(w) < self.name_min_turn:
                w = self.name_min_turn if w >= 0.0 else -self.name_min_turn

            # Clamp
            w = max(-self.name_max_w, min(self.name_max_w, w))

            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = float(w)
            self._publish_smoothed(cmd, reason="name-orient")
            return



        # Lost target?
        if self.last_seen_time is None or (now - self.last_seen_time) > Duration(seconds=self.lost_timeout_s):
            self._stop("lost/no target")
            return

        if self._latest_target_map_ps is None:
            self._stop("no cached target")
            return

        # Transform target into base frame
        ps_base = self._transform_to_base(self._latest_target_map_ps)
        if ps_base is None:
            self._stop("tf error")
            return

        # Person in robot base frame
        x = ps_base.point.x  # forward
        y = ps_base.point.y  # left
        d = math.hypot(x, y)
        bearing = math.atan2(y, x)  # +left

        # Safety: if too close or behind (x<0), stop linear and just reorient
        if d < self.stop_d:
            cmd = Twist()
            # keep a tiny orienting turn if bearing big and person not centered
            if abs(bearing) > self.db_yaw:
                cmd.angular.z = max(-self.max_w, min(self.max_w, self.k_ang * bearing))
            self._publish_smoothed(cmd, reason="too close")
            return

        # If we don't see them ahead (x <= 0), rotate in place toward them
        if x <= 0.0:
            cmd = Twist()
            if abs(bearing) > self.db_yaw:
                cmd.angular.z = max(-self.max_w, min(self.max_w, self.k_ang * bearing))
            self._publish_smoothed(cmd, reason="target behind")
            return

        # Proportional control
        err_d = d - self.desired_d
        v = self.k_lin * err_d
        w = self.k_ang * bearing

        # Deadband for small headings
        if abs(bearing) < self.db_yaw:
            w = 0.0

        # Clamp
        v = max(-self.max_v, min(self.max_v, v))
        w = max(-self.max_w, min(self.max_w, w))

        # Don’t drive forward aggressively while we’re not facing the target
        if abs(bearing) > 0.6:  # ~34°
            v *= 0.3

        # Publish
        cmd = Twist()
        cmd.linear.x = max(0.0, v)  # don’t back up; just let person walk out
        cmd.angular.z = w
        self._publish_smoothed(cmd, reason="tracking")

    def _publish_smoothed(self, cmd: Twist, reason: str):
        # Low-pass filter the commands to avoid jerks
        sm = Twist()
        sm.linear.x  = self._lp(self.last_cmd.linear.x, cmd.linear.x)
        sm.angular.z = self._lp(self.last_cmd.angular.z, cmd.angular.z)
        self.pub.publish(sm)
        self.last_cmd = sm
        # Optional: debug
        # self.get_logger().debug(f"{reason}: v={sm.linear.x:.2f}, w={sm.angular.z:.2f}")

    def _stop(self, why: str):
        stop = Twist()
        self._publish_smoothed(stop, reason=f"stop:{why}")


def main():
    rclpy.init()
    node = PersonFollower()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
