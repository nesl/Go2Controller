#!/usr/bin/env python3
import json, math, re
from typing import Optional, List, Tuple
from vision_msgs.msg import Detection2DArray

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from std_msgs.msg import String
from std_srvs.srv import Trigger
from geometry_msgs.msg import Twist, PoseStamped, Quaternion, PointStamped
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from functools import partial
from tf2_ros import Buffer, TransformListener
from action_msgs.msg import GoalStatus

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
        
        self.declare_parameter("search_ang_speed", 0.6)   # rad/s while searching
        self.declare_parameter("full_turn_margin_deg", 5.0)  # consider full turn at ~355°


        self.declare_parameter("rotate_topic", "/cmd_vel")
        self.declare_parameter("approach_dist_m", 1.0)
        self.declare_parameter("vlm_prompt_prefix",
            "What is the user indicating? Summarize visible devices/objects with brief locations.")
        self.declare_parameter("people_topic", "detected_objects")

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

        self.people_topic = self.get_parameter("people_topic").get_parameter_value().string_value
        # cache: last seen persons in MAP frame, as PointStamped list
        self._people_map: List[PointStamped] = []    
        self._last_matched_person: Optional[PointStamped] = None    
        
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
        
        # subscribe
        self.sub_people = self.create_subscription(
            Detection2DArray, self.people_topic, self.on_people, 10
        )

        self.tts_pub = self.create_publisher(String, "tts", 10)
        self.gesture_pub = self.create_publisher(String, "/webrtc/gesture_req", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.rotate_topic, 10)

        self.vlm_prompt_pub = self.create_publisher(String, "/vlm/prompt", 10)
        self.vlm_srv = self.create_client(Trigger, "/vlm/run")

        self.db_best_sites_cli = self.create_client(Trigger, "/bt/best_sites")
        self.db_best_sites_cli.wait_for_service(timeout_sec=5.0)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Status feed for auto-summarizer in OpenAICommandParser
        self.status_pub = self.create_publisher(String, "/task/status_struct", 10)

        self.search_w      = float(self.get_parameter("search_ang_speed").value)
        self.full_turn_eps = math.radians(max(0.0, 180.0 - float(self.get_parameter("full_turn_margin_deg").value)))
        # We'll compare turned_abs >= (2*pi - margin); full_turn_eps holds a half-margin for safety
        # Person-search state
        self._search_active = False
        self._search_dir    = 1.0        # +1 left / -1 right
        self._search_started_at: Optional[Time] = None
        self._search_prev_yaw = 0.0
        self._search_turned_abs = 0.0


        # 20 Hz control loop for orientation
        self.create_timer(0.05, self._control_loop)

        self.get_logger().info(f"TaskReasoner up. Triggers={self.trigger_words}, ref_frame={self.turn_ref_frame}")

    def _any_person_in_view(self) -> bool:
        """Treat 'in view' as 'detector currently reports at least one person'."""
        return bool(self._people_map)

    def _start_person_search(self, doa_deg: float):
        """
        Begin rotating to look for a person, using DoA only for turn *direction*.
        If a person is already in view, do nothing.
        """
        if self._any_person_in_view():
            # No need to spin if we already see a person
            return

        # Choose direction: left for positive DoA, right for negative (0 -> left by default)
        self._search_dir = 1.0 if float(doa_deg) >= 0.0 else -1.0

        # Initialize cumulative rotation tracking
        yaw_now = self._current_yaw_ref_base()
        self._search_prev_yaw   = yaw_now
        self._search_turned_abs = 0.0
        self._search_started_at = self.get_clock().now()
        self._search_active     = True

        # cancel any ongoing name-orient (we only want the search behavior now)
        self._orient_active = False
        self._stop_turn("start-person-search")

        self.get_logger().info(
            f"Person search started. dir={'left' if self._search_dir>0 else 'right'}, speed={self.search_w:.2f} rad/s"
        )

    def _stop_person_search(self, why: str = ""):
        self._search_active = False
        self._stop_turn(why or "stop-person-search")



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

    def on_people(self, msg: Detection2DArray):
        """Keep a fresh cache of persons in map frame from the detector."""
        people: List[PointStamped] = []
        for d in msg.detections:
            if not d.results:
                continue
            hyp = d.results[0].hypothesis
            if not hyp.class_id:
                continue
            if "person" not in hyp.class_id.lower():
                continue
            p = d.results[0].pose.pose.position
            ps = PointStamped()
            ps.header = msg.header  # should be map frame per your node
            ps.point.x, ps.point.y, ps.point.z = float(p.x), float(p.y), float(p.z)
            people.append(ps)
        self._people_map = people

    def _transform_to_base(self, ps_map: PointStamped) -> Optional[PointStamped]:
        """map PointStamped -> base_link PointStamped"""
        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link", ps_map.header.frame_id,  # target, source
                rclpy.time.Time.from_msg(ps_map.header.stamp),
                Duration(seconds=0.2),
            )
        except Exception:
            try:
                tf = self.tf_buffer.lookup_transform(
                    "base_link", ps_map.header.frame_id, rclpy.time.Time(), Duration(seconds=0.2)
                )
            except Exception as e2:
                self.get_logger().warn(f"TF map->base failed: {e2}")
                return None
        try:
            from tf2_geometry_msgs import do_transform_point
            return do_transform_point(ps_map, tf)
        except Exception as e:
            self.get_logger().warn(f"Transform point failed: {e}")
            return None

    def _nearest_person_bearing(self) -> Optional[float]:
        """
        Return bearing (rad, +left) to the nearest person in base_link, or None.
        """
        best = None
        best_d2 = float("inf")
        for ps in self._people_map:
            pb = self._transform_to_base(ps)
            if pb is None:
                continue
            x, y = pb.point.x, pb.point.y
            d2 = x*x + y*y
            if d2 < best_d2:
                best_d2 = d2
                best = math.atan2(y, x)
        return best

    def _match_speaker_to_person(self, doa_deg: float, max_diff_deg: float = 30.0) -> Optional[PointStamped]:
        """
        Pick the person whose bearing (base_link) is closest to the DoA azimuth.
        doa_deg: 0 = +X forward, +90 = left (matches your STT node).
        Returns the MAP-frame PointStamped of the match, or None if no good match.
        """
        if not self._people_map:
            return None
        doa_rad = math.radians(doa_deg)
        best_ps = None
        best_err = 1e9
        for ps in self._people_map:
            pb = self._transform_to_base(ps)
            if pb is None:
                continue
            bearing = math.atan2(pb.point.y, pb.point.x)
            # normalize angular diff
            raw = doa_rad - bearing
            err = abs(math.atan2(math.sin(raw), math.cos(raw)))
            if err < best_err:
                best_err = err
                best_ps = ps
        if best_ps is None:
            return None
        if math.degrees(best_err) <= max_diff_deg:
            return best_ps
        return None



    # ---------- STT: gate by name, arm orientation, forward to LLM ----------
    def on_stt_json(self, msg: String):
        self.get_logger().info("Hello.")
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Bad STT JSON: {e}")
            return

        text = (data.get("text") or "").strip()
        if not text or not self._name_regex.search(text):
            self.get_logger().warn("Not addressed.")
            return  # ignore if not addressed

        now = self.get_clock().now()
        if (now - self._last_name_trigger_at) < Duration(seconds=self.name_debounce_s):
            self.get_logger().warn("Few time.")
            return

        az = data.get("azimuth_deg", None)
        if az is None:
            self.get_logger().warn("Trigger heard but no azimuth_deg in STT JSON.")
            return

        # ---- Arm name-orient goal (exactly like PersonFollower) ----
        yaw_now = self._current_yaw_ref_base()
        self._orient_start_yaw = yaw_now

        az_deg = float(az) # - 45 #45 adjustment
        if az_deg <= -180.0 or az_deg > 180.0:
            az_deg = ((az_deg + 180.0) % 360.0) - 180.0
        if az_deg == 180.0:
            az_deg = 179.9

        # ---- New: start person-search instead of DoA-targeted turn ----
        self.latest_az_deg = float(az)
        self._start_person_search(self.latest_az_deg)
        self._pending_deixis = bool(DEIXIS.search(text))
        self.get_logger().info(f"Name trigger: '{text}' az={self.latest_az_deg:.1f}° (search mode)")


        # try to associate DoA with a visible person
        matched = self._match_speaker_to_person(self.latest_az_deg, max_diff_deg=35.0)
        if matched:
            # optional: store for immediate use (approach/handoff)
            self._last_matched_person = matched
        else:
            self._last_matched_person = None

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
        if self._orient_active or self._search_active:
            self._pending_cmd = cmd
            return


        # If not orienting, execute now
        self._execute_with_optional_vlm(cmd)

    # ---------- Control loop: do the name-orient P loop ----------
    def _control_loop(self):
    
        # -------- Person-search behavior --------
        if self._search_active:
            # If a person is now visible, stop immediately
            if self._any_person_in_view():
                self._stop_person_search("person-seen")
                self.say("I see you.")
                if self._pending_cmd:
                    self._execute_with_optional_vlm(self._pending_cmd)
                    self._pending_cmd = None
                return

            # Track cumulative rotation
            yaw_now = self._current_yaw_ref_base()
            dyaw    = self._wrap_pi(yaw_now - self._search_prev_yaw)
            self._search_prev_yaw = yaw_now
            self._search_turned_abs += abs(dyaw)

            # Check for full turn (~2*pi minus a small margin)
            if self._search_turned_abs >= (2.0*math.pi - math.radians(5.0)):
                self._stop_person_search("full-turn-complete")
                self.say("I didn’t find you.")
                if self._pending_cmd:
                    self._execute_with_optional_vlm(self._pending_cmd)
                    self._pending_cmd = None
                return

            # Publish constant angular velocity in chosen direction
            cmd = Twist()
            cmd.angular.z = self._search_dir * max(0.05, min(self.search_w, self.name_max_w))
            self._publish_smoothed(cmd, "person-search")
            return  # search has priority over name-orient

    
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


    def _execute_with_optional_vlm(self, cmd: dict):
        # Decide if we need the VLM
        try:
            params = cmd.get("params", {}) or {}
            ref = params.get("ref", None)
            need_vlm = self._pending_deixis or (ref in ("HERE", "THAT"))
        except Exception:
            need_vlm = self._pending_deixis

        self._pending_deixis = False  # clear flag either way

        if need_vlm:
            prompt = f"{self.vlm_prompt_prefix}"
            self.vlm_prompt_pub.publish(String(data=prompt))

            if not self.vlm_srv.service_is_ready():
                # Don’t block — if not ready, just proceed with the command
                self.get_logger().warn("VLM service not ready; executing command without VLM.")
                self.dispatch_command(cmd)
                return

            fut = self.vlm_srv.call_async(Trigger.Request())
            # When VLM finishes, continue with the command
            fut.add_done_callback(partial(self._on_vlm_done_then_dispatch, cmd))
            return

        # No VLM needed — execute now
        self.dispatch_command(cmd)

    def _on_vlm_done_then_dispatch(self, cmd, fut):
        try:
            _ = fut.result()  # we don’t use the response now; you can log it if you want
        except Exception as e:
            self.get_logger().warn(f"VLM call failed: {e}; continuing.")
        self.dispatch_command(cmd)
        
    # ---------- DB query ----------

    def _query_best_sites_async(self):
        if not self.db_best_sites_cli.service_is_ready():
            self.db_best_sites_cli.wait_for_service(timeout_sec=0.5)
        if not self.db_best_sites_cli.service_is_ready():
            self.say("The beacon map service is not available yet.")
            self._publish_status("query_results", {"error": "service_unavailable"})
            return
        fut = self.db_best_sites_cli.call_async(Trigger.Request())
        fut.add_done_callback(self._on_best_sites_reply)

    def _on_best_sites_reply(self, fut):
        # This runs on the executor when the reply arrives (non-blocking).
        try:
            resp = fut.result()
        except Exception as e:
            self.get_logger().error(f"best_sites: exception: {e}")
            self.say("I couldn’t retrieve beacon results.")
            self._publish_status("query_results", {"error": "exception", "detail": str(e)})
            return

        if not resp or not resp.success:
            msg = getattr(resp, "message", "")
            self.get_logger().warn(f"best_sites: failed; message={msg!r}")
            self.say("I couldn’t retrieve beacon results.")
            self._publish_status("query_results", {"error": "server_failed", "message": msg})
            return

        # Parse and speak top results
        try:
            best = json.loads(resp.message or "{}")
        except Exception as e:
            self.get_logger().error(f"best_sites: JSON parse error: {e}; raw={resp.message!r}")
            self.say("I got a reply, but it wasn’t readable.")
            self._publish_status("query_results", {"error": "bad_json"})
            return

        if not best:
            self.say("I don’t have any beacons detected yet.")
            self._publish_status("query_results", {"count": 0})
            return

        top = sorted(best.items(), key=lambda kv: kv[1].get("rssi", -999), reverse=True)[:5]
        parts = [f"{dev} with {d['rssi']} dBm, {d['contaminated']} with {d['probability']}% probability" for dev, d in top]
        self.say("Here are the strongest signals: " + "; ".join(parts))
        self._publish_status("query_results", {"top": top})

    def dispatch_command(self, cmd: dict):
        intent = cmd.get("intent")
        params = cmd.get("params", {}) or {}
        ref = params.get("ref", None)

        # Helper: convert a MAP point to a short approach via Nav2 or cmd_vel
        def _approach_map_point(ps_map: PointStamped, stop_dist=0.8):
            pb = self._transform_to_base(ps_map)
            if pb is None:
                return self.say("I cannot localize the person right now.")
            # bearing + distance in base
            x, y = pb.point.x, pb.point.y
            bearing = math.atan2(y, x)
            dist = math.hypot(x, y)
            # rotate toward, then move forward (bounded)
            self._orient_goal_delta = bearing
            self._orient_start_yaw = self._current_yaw_ref_base()
            self._orient_started_at = self.get_clock().now()
            self._orient_active = True
            self._pending_cmd = {"intent":"_internal_after_turn_go", "params":{"fwd":max(0.0, dist - stop_dist)}}
            return

        if intent == "_internal_after_turn_go":
            fwd = float(params.get("fwd", 0.0))
            if fwd > 0.05:
                # quick forward nudge in base_link (or Nav2 short goal)
                self.navigate_relative(0.0, fwd)
            return

        # ---------- query_results ----------
        if intent == "query_results":
            self.say("Let me check the beacon map.")
            self._publish_status("query_results", {"status": "requesting"})
            self._query_best_sites_async()
            return

        # ---------- sense_area ----------
        if intent == "sense_area":
            if ref in ("HERE","THAT"):
                # prefer matched speaker; else nearest person; else DoA-based relative approach
                if self._last_matched_person:
                    _approach_map_point(self._last_matched_person, stop_dist=1.0)
                else:
                    bearing = self._nearest_person_bearing()
                    if bearing is not None:
                        # do a small face+step
                        self._orient_goal_delta = bearing
                        self._orient_start_yaw = self._current_yaw_ref_base()
                        self._orient_started_at = self.get_clock().now()
                        self._orient_active = True
                        self._pending_cmd = {"intent":"_internal_after_turn_go", "params":{"fwd":0.6}}
                    else:
                        # fall back to DoA-based approach
                        self.navigate_relative(self.latest_az_deg, 0.6)
            self.say("Scanning this area.")
            self._publish_status("sense_area", {"ref": ref})
            return

        # ---------- navigate ----------
        if intent == "navigate":
            goal = params.get("goal")
            if isinstance(goal, dict):
                frame = goal.get("frame","map")
                x,y = float(goal.get("x",0.0)), float(goal.get("y",0.0))
                yaw = float(goal.get("yaw",0.0))
                self.navigate_absolute(frame,x,y,yaw)
                self._publish_status("navigate", {"frame":frame,"x":x,"y":y,"yaw":yaw})
            else:
                if ref in ("HERE","THAT"):
                    self.say("Moving towards location.")
                    if self._last_matched_person:
                        _approach_map_point(self._last_matched_person, stop_dist=self.approach_dist)
                    else:
                        self.navigate_relative(self.latest_az_deg, self.approach_dist)
                    self._publish_status("navigate", {"ref":ref})
                elif ref in ("SELF", None):
                    self.say("I’m already here.")
                    self._publish_status("navigate", {"ref":"SELF"})
                else:
                    self.say("I need a destination.")
                    self._publish_status("navigate", {"error":"no_goal"})
            return

        # ---------- handoff ----------
        if intent == "handoff":
            self.gesture_pub.publish(String(data="greet"))
            self.say("I am ready. Please place the item in my basket.")
            if self._last_matched_person:
                _approach_map_point(self._last_matched_person, stop_dist=0.7)
            else:
                self.navigate_relative(self.latest_az_deg, 0.6)
            self._publish_status("handoff", {"who":"SPEAKER"})
            return

        # ---------- scan_all ----------
        if intent == "scan_all":
            scan_pub = self.create_publisher(String, "/scan/contour_cmd", 10)
            scan_pub.publish(String(data="start"))
            self.say("Starting a full area scan.")
            self._publish_status("scan_all", {"cmd":"start"})
            return

        # default
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
        
        # Send goal async and chain callbacks
        send_future = self.nav_client.send_goal_async(goal)
        send_future.add_done_callback(self._on_nav_goal_response_abs)

    def navigate_relative(self, az_deg: float, dist_m: float):
        # Small “face” nudge with a short-lived timer rather than a busy loop
        target = math.radians(az_deg)
        if abs(target) > math.radians(5.0):
            # 1 second angular nudge at ~0.25 rad/s
            self._start_twist_timer(duration_s=1.0, linear_x=0.0, angular_z=0.25 if target > 0 else -0.25)

        # If Nav2 is available, do a base_link goal async; else cmd_vel forward nudge with a timer
        if self.nav_client.wait_for_server(timeout_sec=0.05):
            goal = NavigateToPose.Goal()
            ps = PoseStamped()
            ps.header.frame_id = "base_link"
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = dist_m
            ps.pose.orientation = yaw_to_q(0.0)
            goal.pose = ps

            send_future = self.nav_client.send_goal_async(goal)
            send_future.add_done_callback(self._on_nav_goal_response_rel)
        else:
            # Linear forward nudge using a timer, no blocking
            v = 0.25
            t = dist_m / max(v, 0.05)
            self._start_twist_timer(duration_s=float(t), linear_x=v, angular_z=0.0,
                                    on_complete=lambda: self.say("Moved closer."))

    def _on_nav_goal_response_rel(self, future):
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            # Fallback to a small forward timer nudge if the goal wasn’t accepted
            v = 0.25
            t = 0.6 / v
            self._start_twist_timer(duration_s=float(t), linear_x=v, angular_z=0.0,
                                    on_complete=lambda: self.say("Moved closer."))
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_nav_result_rel)

    def _on_nav_result_rel(self, future):
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f"Nav result error (relative): {e}")
            self.say("Navigation failed.")
            return

        # Treat any completion as “moved closer”; you can refine based on status
        if getattr(result, "status", 0) == GoalStatus.STATUS_SUCCEEDED:
            self.say("Moved closer.")
        else:
            self.say("Navigation failed.")

    def _on_nav_goal_response_abs(self, future):
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.say("Failed to send navigation goal.")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_nav_result_abs)

    def _on_nav_result_abs(self, future):
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f"Nav result error (absolute): {e}")
            self.say("Navigation failed.")
            return

        if getattr(result, "status", 0) == GoalStatus.STATUS_SUCCEEDED:
            self.say("Arrived at the destination.")
        else:
            self.say("Navigation failed.")


    # ---------- VLM + TTS + status ----------
    def say(self, text: str):
        
        self.get_logger().info(f"Saying {text}")
        self.tts_pub.publish(String(data=text))

    def _publish_status(self, action: str, details: dict):
        payload = {
            "ts": self.get_clock().now().nanoseconds * 1e-9,
            "action": action,
            "details": details
        }
        #self.status_pub.publish(String(data=json.dumps(payload)))

    def _start_twist_timer(self, duration_s: float, *, linear_x: float = 0.0, angular_z: float = 0.0, on_complete=None):
        """
        Publishes a Twist at 20 Hz for duration_s seconds using a Timer.
        Non-blocking; cleans itself up; calls on_complete() at the end if provided.
        """
        period = 0.05  # 20 Hz
        remaining = duration_s
        tw = Twist()
        tw.linear.x = float(linear_x)
        tw.angular.z = float(angular_z)

        # Keep references on self so they don't get GC'd
        state = {"remaining": remaining}
        def _tick():
            state["remaining"] -= period
            if state["remaining"] > 0.0:
                self.cmd_vel_pub.publish(tw)
            else:
                # stop and cleanup
                self.cmd_vel_pub.publish(Twist())
                try:
                    self._twist_timer.cancel()
                except Exception:
                    pass
                self._twist_timer = None
                if callable(on_complete):
                    try:
                        on_complete()
                    except Exception as e:
                        self.get_logger().warn(f"on_complete error: {e}")

        # cancel any existing twist timer
        if getattr(self, "_twist_timer", None):
            try:
                self._twist_timer.cancel()
            except Exception:
                pass
            self._twist_timer = None

        self._twist_timer = self.create_timer(period, _tick)
        

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

