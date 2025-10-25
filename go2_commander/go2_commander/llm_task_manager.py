#!/usr/bin/env python3
import json, math, re
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from std_msgs.msg import String
from std_srvs.srv import Trigger
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose

from tf2_ros import Buffer, TransformListener

DEIXIS = re.compile(r"\b(this|that|here|there)\b", re.IGNORECASE)

def yaw_to_q(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw/2.0)
    q.w = math.cos(yaw/2.0)
    return q

class TaskReasoner(Node):
    """
    Wires speech->LLM (external) and executes robot actions:
      - Sub:  /audio/stt_doa_json (String JSON: {text, azimuth_deg, ...})
      - Pub:  /llm/request (String)  --> consumed by OpenAICommandParser
      - Sub:  /robot/command_json (String JSON) <-- produced by OpenAICommandParser
      - Pub:  /tts/say, /webrtc/gesture_req, /vlm/prompt
      - Srv:  /vlm/run (Trigger)
      - Cli:  /bt/best_sites (Trigger)
      - Pub:  /task/status_struct (String JSON) for auto summaries

    Orientation matches your PersonFollower "name-orient" behavior.
    """

    def __init__(self):
        super().__init__("task_reasoner")

        # ---------- Params ----------
        self.declare_parameter("trigger_words", ["bob"])  # gate by name
        self.declare_parameter("turn_ref_frame", "odom")  # continuous yaw frame
        self.declare_parameter("name_debounce_s", 2.0)
        self.declare_parameter("name_timeout_s", 3.0)
        self.declare_parameter("name_k_ang", 1.5)
        self.declare_parameter("name_max_ang_speed", 1.0)
        self.declare_parameter("name_min_turn_speed", 0.15)
        self.declare_parameter("name_stop_thresh_deg", 2.0)
        self.declare_parameter("smoothing_alpha", 0.4)

        self.declare_parameter("rotate_topic", "/cmd_vel")
        self.declare_parameter("approach_dist_m", 1.0)
        self.declare_parameter("vlm_prompt_prefix",
            "What is the user indicating? Summarize visible devices/objects with brief locations.")

        # ---------- Read params ----------
        self.trigger_words = [w.lower() for w in self.get_parameter("trigger_words").get_parameter_value().string_array_value]
        self.turn_ref_frame  = self.get_parameter("turn_ref_frame").get_parameter_value().string_value
        self.name_debounce_s = float(self.get_parameter("name_debounce_s").value)
        self.name_timeout_s  = float(self.get_parameter("name_timeout_s").value)
        self.name_k_ang      = float(self.get_parameter("name_k_ang").value)
        self.name_max_w      = float(self.get_parameter("name_max_ang_speed").value)
        self.name_min_turn   = float(self.get_parameter("name_min_turn_speed").value)
        self.name_stop_deg   = float(self.get_parameter("name_stop_thresh_deg").value)
        self.alpha           = float(self.get_parameter("smoothing_alpha").value)

        self.rotate_topic    = self.get_parameter("rotate_topic").get_parameter_value().string_value
        self.approach_dist   = float(self.get_parameter("approach_dist_m").value)
        self.vlm_prompt_prefix = self.get_parameter("vlm_prompt_prefix").get_parameter_value().string_value

        self._name_regex = re.compile(r"\b(" + "|".join(map(re.escape, self.trigger_words)) + r")\b", re.IGNORECASE)

        # ---------- TF + orientation state (like PersonFollower) ----------
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._orient_active = False
        self._orient_goal_delta = 0.0
        self._orient_start_yaw = 0.0
        self._orient_started_at: Optional[Time] = None
        self._last_name_trigger_at = self.get_clock().now() - Duration(seconds=999.0)

        self.last_cmd = Twist()

        # Execution state
        self._pending_cmd = None       # holds LLM command while orienting
        self._pending_deixis = False   # whether to run VLM after align
        self.latest_az_deg = 0.0

        # ---------- ROS I/O ----------
        self.sub_stt = self.create_subscription(String, "/audio/stt_doa_json", self.on_stt_json, 20)
        self.pub_llm_req = self.create_publisher(String, "/llm/request", 10)
        self.sub_cmd = self.create_subscription(String, "/robot/command_json", self.on_cmd_json, 20)

        self.tts_pub = self.create_publisher(String, "/tts/say", 10)
        self.gesture_pub = self.create_publisher(String, "/webrtc/gesture_req", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.rotate_topic, 10)

        self.vlm_prompt_pub = self.create_publisher(String, "/vlm/prompt", 10)
        self.vlm_srv = self.create_client(Trigger, "/vlm/run")

        self.db_best_sites_cli = self.create_client(Trigger, "/bt/best_sites")

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Status feed for auto-summarizer in OpenAICommandParser
        self.status_pub = self.create_publisher(String, "/task/status_struct", 10)

        # 20 Hz control loop for orientation
        self.create_timer(0.05, self._control_loop)

        self.get_logger().info(f"TaskReasoner up. Triggers={self.trigger_words}, ref_frame={self.turn_ref_frame}")

    # ---------- Orientation helpers (ported) ----------
    def _current_yaw_ref_base(self) -> float:
        try:
            tf_base_to_ref = self.tf_buffer.lookup_transform(
                self.turn_ref_frame, "base_link", Time(), Duration(seconds=0.2)
            )
            q = tf_base_to_ref.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return math.atan2(siny_cosp, cosy_cosp)
        except Exception as e:
            self.get_logger().warn(f"Yaw lookup ({self.turn_ref_frame}) failed: {e}")
            return 0.0

    @staticmethod
    def _wrap_pi(a: float) -> float:
        return (a + math.pi) % (2.0*math.pi) - math.pi

    def _lp(self, prev: float, new: float) -> float:
        return (1.0 - self.alpha) * prev + self.alpha * new

    def _publish_smoothed(self, cmd: Twist, why: str = ""):
        sm = Twist()
        sm.linear.x  = self._lp(self.last_cmd.linear.x, cmd.linear.x)
        sm.angular.z = self._lp(self.last_cmd.angular.z, cmd.angular.z)
        self.cmd_vel_pub.publish(sm)
        self.last_cmd = sm

    def _stop_turn(self, why: str = ""):
        self._publish_smoothed(Twist(), why)

    # ---------- STT: gate by name, arm orientation, forward to LLM ----------
    def on_stt_json(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Bad STT JSON: {e}")
            return

        text = (data.get("text") or "").strip()
        if not text or not self._name_regex.search(text):
            return  # ignore if not addressed

        now = self.get_clock().now()
        if (now - self._last_name_trigger_at) < Duration(seconds=self.name_debounce_s):
            return

        az = data.get("azimuth_deg", None)
        if az is None:
            self.get_logger().warn("Trigger heard but no azimuth_deg in STT JSON.")
            return

        # ---- Arm name-orient goal (exactly like PersonFollower) ----
        yaw_now = self._current_yaw_ref_base()
        self._orient_start_yaw = yaw_now

        az_deg = float(az)
        if az_deg <= -180.0 or az_deg > 180.0:
            az_deg = ((az_deg + 180.0) % 360.0) - 180.0
        if az_deg == 180.0:
            az_deg = 179.9

        self._orient_goal_delta = math.radians(az_deg)  # +left/-right
        self._orient_started_at = now
        self._orient_active = True
        self._last_name_trigger_at = now
        self.latest_az_deg = float(az)

        self._pending_deixis = bool(DEIXIS.search(text))
        self.get_logger().info(f"Name trigger: '{text}' az={float(az):.1f}° -> rotate {math.degrees(self._orient_goal_delta):.1f}°")

        # ---- Kick the external LLM parser ----
        # (Let it handle gating & schema; we just forward the raw utterance)
        self.pub_llm_req.publish(String(data=text))

    # ---------- LLM result arrives here ----------
    def on_cmd_json(self, msg: String):
        try:
            cmd = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Bad /robot/command_json: {e}")
            return

        # If we're still orienting, queue it
        if self._orient_active:
            self._pending_cmd = cmd
            return

        # If not orienting, execute now
        self._execute_with_optional_vlm(cmd)

    # ---------- Control loop: do the name-orient P loop ----------
    def _control_loop(self):
        if not self._orient_active:
            return

        now = self.get_clock().now()

        # Timeout?
        if (now - self._orient_started_at) > Duration(seconds=self.name_timeout_s):
            self._orient_active = False
            self._stop_turn("name-timeout")
            if self._pending_cmd:
                self._execute_with_optional_vlm(self._pending_cmd)
                self._pending_cmd = None
            return

        # Turned since start
        yaw_now = self._current_yaw_ref_base()
        turned = self._wrap_pi(yaw_now - self._orient_start_yaw)

        # Remaining shortest error
        raw = self._orient_goal_delta - turned
        err = math.atan2(math.sin(raw), math.cos(raw))

        # Done?
        if abs(math.degrees(err)) <= self.name_stop_deg:
            self._orient_active = False
            self._stop_turn("name-aligned")
            if self._pending_cmd:
                self._execute_with_optional_vlm(self._pending_cmd)
                self._pending_cmd = None
            return

        # P-control w/ min turn and clamp
        w = self.name_k_ang * err
        if abs(w) < self.name_min_turn:
            w = self.name_min_turn if w >= 0.0 else -self.name_min_turn
        w = max(-self.name_max_w, min(self.name_max_w, w))

        cmd = Twist()
        cmd.angular.z = float(w)
        self._publish_smoothed(cmd, "name-orient")

    # ---------- Execute command (and VLM if deictic or HERE/THAT) ----------
    def _execute_with_optional_vlm(self, cmd: dict):
        # Fire VLM first if user used deixis or the command has HERE/THAT
        try:
            params = cmd.get("params", {}) or {}
            ref = params.get("ref", None)
            need_vlm = self._pending_deixis or (ref in ("HERE", "THAT"))
        except Exception:
            need_vlm = self._pending_deixis

        if need_vlm:
            prompt = f"{self.vlm_prompt_prefix}"
            self.vlm_prompt_pub.publish(String(data=prompt))
            if not self.vlm_srv.service_is_ready():
                self.vlm_srv.wait_for_service(timeout_sec=0.5)
            if self.vlm_srv.service_is_ready():
                fut = self.vlm_srv.call_async(Trigger.Request())
                rclpy.spin_until_future_complete(self, fut, timeout_sec=8.0)

        self._pending_deixis = False
        self.dispatch_command(cmd)

    # ---------- DB query ----------
    def _query_best_sites(self):
        if not self.db_best_sites_cli.service_is_ready():
            self.db_best_sites_cli.wait_for_service(timeout_sec=0.5)
        if not self.db_best_sites_cli.service_is_ready():
            return {}
        fut = self.db_best_sites_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=1.5)
        if not fut.result() or not fut.result().success:
            return {}
        try:
            return json.loads(fut.result().message) or {}
        except Exception:
            return {}

    # ---------- Command dispatcher ----------
    def dispatch_command(self, cmd: dict):
        intent = cmd.get("intent")
        params = cmd.get("params", {}) or {}

        if intent == "query_results":
            best = self._query_best_sites()
            if not best:
                self.say("I don’t have any beacons detected yet.")
                self._publish_status("query_results", {"count": 0})
                return
            top = sorted(best.items(), key=lambda kv: kv[1].get("rssi", -999), reverse=True)[:5]
            parts = [f"{dev} at x={d['x']:.1f}, y={d['y']:.1f} ({d['rssi']} dBm)" for dev,d in top]
            reply = "Here are the strongest signals: " + "; ".join(parts)
            self.say(reply)
            self._publish_status("query_results", {"top": top})
            return

        if intent == "sense_area":
            # Optional small approach toward the speaker if HERE/THAT
            ref = params.get("ref","SELF")
            if ref in ("HERE","THAT"):
                self.navigate_relative(self.latest_az_deg, 0.6)
            self.say("Scanning this area.")
            self._publish_status("sense_area", {"ref": ref})
            return

        if intent == "navigate":
            ref = params.get("ref")
            goal = params.get("goal")
            if isinstance(goal, dict):
                frame = goal.get("frame","map")
                x,y = float(goal.get("x",0.0)), float(goal.get("y",0.0))
                yaw = float(goal.get("yaw",0.0))
                self.navigate_absolute(frame,x,y,yaw)
                self._publish_status("navigate", {"frame":frame,"x":x,"y":y,"yaw":yaw})
            else:
                if ref in ("HERE","THAT"):
                    self.navigate_relative(self.latest_az_deg, self.approach_dist)
                    self._publish_status("navigate", {"ref":ref,"dist":self.approach_dist})
                elif ref in ("SELF", None):
                    self.say("I’m already here.")
                    self._publish_status("navigate", {"ref":"SELF"})
                else:
                    self.say("I need a destination.")
                    self._publish_status("navigate", {"error":"no_goal"})
            return

        if intent == "handoff":
            self.gesture_pub.publish(String(data="greet"))
            self.say("I am ready. Please place the item in my basket.")
            self.navigate_relative(self.latest_az_deg, 0.6)
            self._publish_status("handoff", {"mode":"SPEAKER"})
            return

        if intent == "scan_all":
            scan_pub = self.create_publisher(String, "/scan/contour_cmd", 10)
            scan_pub.publish(String(data="start"))
            self.say("Starting a full area scan.")
            self._publish_status("scan_all", {"cmd":"start"})
            return

        self.say("I did not understand the request.")
        self._publish_status("unknown_intent", {"raw": cmd})

    # ---------- Navigation ----------
    def navigate_absolute(self, frame: str, x: float, y: float, yaw: float):
        if not self.nav_client.wait_for_server(timeout_sec=0.5):
            self.say("Navigation is not available.")
            return
        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation = yaw_to_q(yaw)
        goal.pose = ps
        send = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send)
        if not send.result() or not send.result().accepted:
            self.say("Failed to send navigation goal.")
            return
        res = send.result().get_result_async()
        rclpy.spin_until_future_complete(self, res)
        self.say("Arrived at the destination." if res.result() else "Navigation failed.")

    def navigate_relative(self, az_deg: float, dist_m: float):
        # Turn using the same orient loop behavior (brief, coarse)
        # Quick “manual” turn nudge:
        target = math.radians(az_deg)
        if abs(target) > math.radians(5.0):
            w = 0.25 if target > 0 else -0.25
            tw = Twist(); tw.angular.z = w
            start = self.get_clock().now()
            while (self.get_clock().now() - start) < Duration(seconds=1.0):
                self.cmd_vel_pub.publish(tw)
                rclpy.spin_once(self, timeout_sec=0.05)
            self.cmd_vel_pub.publish(Twist())

        # Move forward with Nav2 if available; else cmd_vel nudge
        if not self.nav_client.wait_for_server(timeout_sec=0.3):
            v = 0.25
            t = dist_m / max(v, 0.05)
            tw = Twist(); tw.linear.x = v
            start = self.get_clock().now()
            while (self.get_clock().now() - start) < Duration(seconds=t):
                self.cmd_vel_pub.publish(tw); rclpy.spin_once(self, timeout_sec=0.05)
            self.cmd_vel_pub.publish(Twist())
            return

        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = "base_link"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = dist_m
        ps.pose.orientation = yaw_to_q(0.0)
        goal.pose = ps
        send = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send)
        if send.result() and send.result().accepted:
            res = send.result().get_result_async()
            rclpy.spin_until_future_complete(self, res, timeout_sec=20.0)
            self.say("Moved closer.")

    # ---------- VLM + TTS + status ----------
    def say(self, text: str):
        self.tts_pub.publish(String(data=text))

    def _publish_status(self, action: str, details: dict):
        payload = {
            "ts": self.get_clock().now().nanoseconds * 1e-9,
            "action": action,
            "details": details
        }
        self.status_pub.publish(String(data=json.dumps(payload)))

def main():
    rclpy.init()
    node = TaskReasoner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

