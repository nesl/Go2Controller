#!/usr/bin/env python3
# skills_node.py
#
# Single-file skills runtime + ROS agent implementing basic actions and
# executing composite skills based on EventLayer rule hits — with:
#   • hot-reloadable skills library (YAML via skills_path)
#   • planning API (what skills are eligible right now)
#   • execution API (run a skill by name or by mapped rule id)
#
from __future__ import annotations
import json, math, os, re, sqlite3, time, inspect
from typing import Any, Dict, List, Optional

import yaml
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from std_msgs.msg import String as StringMsg, Bool
from std_srvs.srv import Trigger
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from tf2_ros import Buffer, TransformListener
from go2_interfaces.msg import WebRtcReq
import requests
from nav2_msgs.srv import GetCostmap
import threading
import random
from dataclasses import dataclass, field
import time

from typing import Tuple

from .skills_engine_v2 import (  # adjust to ".skills_engine_v2" if inside a package
    SkillEngineV2,
    RulesViewROS,
    StepHandle,
    DEFAULT_SKILLS_V2,
    _normalize_tts_text,
    _num_to_words,
    _box_id_from_node_id,
    canonical_box_action_from_execute,
    same_canonical_box_action,
)


YELLOW = "\033[93m"
RESET = "\033[0m"

# ───────────────────────────────────────────────────────────────────────────────
#                            Low-level helpers (quaternion)
# ───────────────────────────────────────────────────────────────────────────────
def yaw_to_q(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw/2.0)
    q.w = math.cos(yaw/2.0)
    return q

# ───────────────────────────────────────────────────────────────────────────────
#                              The Agent Node
# ───────────────────────────────────────────────────────────────────────────────
class SkillsAgent(Node):
    """
    Implements:
      - Basic actions (tts, gesture, navigate abs/rel, turn search, beacon DB)
      - Rules ingestion (/events/*) and hot-reload of skills library
      - Planning + Execution APIs (services + topics)
    """
    def __init__(self):
        super().__init__("skills_agent")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("rotate_topic", "/cmd_vel")
        self.declare_parameter("approach_dist_m", 1.0)
        self.declare_parameter("bt_db_path", os.path.expanduser("~/.bt_rssi_map.sqlite"))
        self.declare_parameter("search_ang_speed", 0.6)
        self.declare_parameter("full_turn_margin_deg", 5.0)
        self.declare_parameter("turn_ref_frame", "odom")
        self.declare_parameter("smoothing_alpha", 0.4)
        self.declare_parameter("name_max_ang_speed", 1.0)

        # skills library config
        self.declare_parameter("skills_base_path", "")
        self.declare_parameter("skills_composite_path", "")
        self.declare_parameter("skills_rescan_period_s", 1.0)

        self.declare_parameter("turn_speed_rad_s", 0.6)  # was 0.25
        self.declare_parameter("fwd_speed_m_s", 0.25)
        
        self.declare_parameter("announce_box_ops", True)


        # --- Simulation movement ---
        self.declare_parameter("sim_move_enable", True)   # movement simulation on/off (only used if sim_mode)
        self.declare_parameter("sim_lin_speed_mps", 0.35) # how fast we "move" in sim
        self.declare_parameter("sim_ang_speed_rps", 0.8)  # how fast we "turn" in sim
        self.declare_parameter("sim_move_min_s", 0.15)    # minimum movement duration
        self.declare_parameter("sim_move_jitter_s", 0.05) # optional jitter

        
        # Box server (for calling /sense directly from skills)
        self.declare_parameter("box_server_url", "http://172.17.40.64:8080")
        self.declare_parameter("box_req_timeout", 200.0)
        self.declare_parameter("agent_id", "robot")  # logical agent id for /sense

        # --- Simulated speech mode (TTS -> STT loopback) ---
        self.declare_parameter("sim_mode", False)
        self.declare_parameter("sim_tts_publish_topic", "/audio/stt_text")
        self.declare_parameter("sim_tts_speaker_id", "robot")   # IMPORTANT: mark as robot
        self.declare_parameter("sim_tts_delay_s", 0.0)          # optional latency


        self.declare_parameter("box_cancel_timeout", 2.0)

        self.box_cancel_timeout = float(self.get_parameter("box_cancel_timeout").value)

        self.box_server_url = self.get_parameter("box_server_url").get_parameter_value().string_value
        self.box_req_timeout = float(self.get_parameter("box_req_timeout").value)
        self.agent_id = self.get_parameter("agent_id").get_parameter_value().string_value or "robot"

        self._current_box_action = None
        
        # internal simulated pose (used only in sim mode)
        self._sim_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self._sim_pose_lock = threading.Lock()


        self.boxop_status_pub = self.create_publisher(StringMsg, "/box/op_status", 10)

        
        self.turn_speed = float(self.get_parameter("turn_speed_rad_s").value)
        self.fwd_speed  = float(self.get_parameter("fwd_speed_m_s").value)

        self.rotate_topic    = self.get_parameter("rotate_topic").get_parameter_value().string_value
        self.approach_dist   = float(self.get_parameter("approach_dist_m").value)
        self.bt_db_path      = self.get_parameter("bt_db_path").get_parameter_value().string_value
        self.search_w        = float(self.get_parameter("search_ang_speed").value)
        self.full_turn_eps   = math.radians(max(0.0, 180.0 - float(self.get_parameter("full_turn_margin_deg").value)))
        self.turn_ref_frame  = self.get_parameter("turn_ref_frame").get_parameter_value().string_value
        self.alpha           = float(self.get_parameter("smoothing_alpha").value)
        self.name_max_w      = float(self.get_parameter("name_max_ang_speed").value)

        # NEW: paths + mtimes
        self.skills_base_path      = self.get_parameter("skills_base_path").get_parameter_value().string_value
        self.skills_composite_path = self.get_parameter("skills_composite_path").get_parameter_value().string_value
        self._skills_rescan  = float(self.get_parameter("skills_rescan_period_s").value)
        self._skills_base_mtime: Optional[float] = None
        self._skills_comp_mtime: Optional[float] = None

        # ── ROS I/O for basic actions ─────────────────────────────────────────
        self.tts_pub     = self.create_publisher(StringMsg, "tts", 10)
        self.webrtc_req_pub = self.create_publisher(WebRtcReq, "webrtc_req", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.rotate_topic, 10)
        self.nav_client  = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.coverage_req_pub = self.create_publisher(StringMsg, "/coverage/req", 10)
        self.coverage_status_sub = self.create_subscription(
            StringMsg, "/coverage/status", self._cb_coverage_status, 10
        )
        self._coverage_pending = {}   # id -> {"handle": StepHandle, "ctx": dict}
        self._coverage_next_id = 1

        self._sim_tts_pub = self.create_publisher(
            StringMsg,
            self.get_parameter("sim_tts_publish_topic").value,
            10
        )

        # LLM speech_check req/resp
        self.llm_speech_req_pub = self.create_publisher(StringMsg, "/llm/speech_check_req", 10)
        self.llm_speech_resp_sub = self.create_subscription(
            StringMsg,
            "/llm/speech_check_resp",
            self._cb_llm_speech_resp,
            10,
        )
        
        self.sub_tts_immediate = self.create_subscription(
            StringMsg,
            "/skills/tts_immediate",
            self._on_tts_immediate,
            10,
        )
        
        self._llm_speech_pending = {}   # req_id -> {"handle": StepHandle, "ctx": dict}
        self._llm_speech_next_id = 1

        # VLM request/response (generic, like llm_speech_check)
        self.vlm_req_pub = self.create_publisher(
            StringMsg, "/vlm/req", 10
        )
        self.vlm_resp_sub = self.create_subscription(
            StringMsg,
            "/vlm/answer",
            self._cb_vlm_resp,
            10,
        )
        self._vlm_pending = {}   # req_id -> {"handle": StepHandle, "ctx": dict}
        self._vlm_next_id = 1


        # TF buffer for turning reference
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Turn/Spin state
        self.last_cmd = Twist()
        self._search_active = False
        self._search_dir    = 1.0
        self._search_prev_yaw = 0.0
        self._search_turned_abs = 0.0
        self._twist_timer = None
        
        self._live_timers = set()  # keep strong refs to timers so GC can't kill them

        # Control loop (20 Hz) for turn_search
        #self.create_timer(0.05, self._control_loop)

        # ── Rules + Engine + Orchestrator ─────────────────────────────────────
        self.rules_view = RulesViewROS(self, window_ms=3000)
        self.skill_engine = SkillEngineV2(self._make_bindings_for_self(), self.rules_view, logger=self.get_logger(), event_cb=self._on_skill_event)

        self._load_skills_initial()


        # Nav snapping (goal projection onto nearest free cell)
        self.declare_parameter("nav_snap_enable", True)
        self.declare_parameter("nav_snap_radius_m", 0.8)          # search radius
        self.declare_parameter("nav_snap_cost_threshold", 0)     # <= is acceptable (0 free, higher = closer to obstacles)
        self.declare_parameter("nav_snap_use_local_costmap", True)
        self.declare_parameter("nav_snap_costmap_timeout_s", 0.25)

        self._local_costmap_cli = self.create_client(GetCostmap, "/local_costmap/get_costmap")
        self._global_costmap_cli = self.create_client(GetCostmap, "/global_costmap/get_costmap")


        # ── Planning & Execution APIs ─────────────────────────────────────────
        # Topics:
        #   /skills/execute  (String JSON): {"skill":"sense.here","ctx":{...}} OR {"rule":"greet_with_presence","ctx":{...}}
        #   /skills/plan_req (String JSON or empty) -> reply on /skills/plan_result
        self.sub_execute = self.create_subscription(StringMsg, "/skills/execute", self._on_execute_msg, 20)
        self.sub_planreq = self.create_subscription(StringMsg, "/skills/plan_req", self._on_plan_req_msg, 10)
        self.pub_planres = self.create_publisher(StringMsg, "/skills/plan_result", 10)

        # Services:
        #   /skills/reload (Trigger)     : reload library from file or fallback default
        #   /skills/plan   (Trigger)     : return eligible skills as JSON in response.message
        #   /skills/run_all_eligible (Trigger) : run every eligible skill once (use with care)
        self.create_service(Trigger, "/skills/reload", self._srv_reload_skills)
        self.create_service(Trigger, "/skills/plan", self._srv_plan)
        self.create_service(Trigger, "/skills/run_all_eligible", self._srv_run_all_eligible)

        self.create_service(Trigger, "/skills/cancel_all", self._srv_cancel_all)


        # Hot-reload timer
        #self.create_timer(self._skills_rescan, self._maybe_reload_skills)

        self.skill_status_pub = self.create_publisher(StringMsg, "/skills/status", 10)

        # Track TTS playback state from TTSPlayerNode (/tts_busy)
        self._tts_busy = False          # current busy flag
        self._tts_has_busy = False      # did we ever see /tts_busy?
        self._tts_waiting: List[StepHandle] = []  # handles waiting for speech to finish

        self.create_subscription(
            Bool,
            "/tts_busy",
            self._cb_tts_busy,
            10,
        )
        
        self.get_logger().info("SkillsAgent ready.")

        # call engine.tick() ~10–20 Hz, lightweight
        self.create_timer(0.05, self._tick_engine)


    def _sim_is_on(self) -> bool:
        return bool(self.get_parameter("sim_mode").value) and bool(self.get_parameter("sim_move_enable").value)

    def _publish_boxop(self, *, phase: str, op: str, box_id: int, prop: str,
                       req_id: str = "", status: str = "", success=None, detected=None,
                       probability=None, why: str = "", extra: dict | None = None):
        evt = {
            "phase": phase,          # start|finish|cancel|skip
            "op": op,                # sense|dispose
            "agent_id": self.agent_id,
            "box_id": int(box_id),
            "prop": str(prop),
            "req_id": str(req_id),
            "status": status,
            "success": success,
            "detected": detected,
            "probability": probability,
            "why": why,
            "ts": float(self.get_clock().now().nanoseconds * 1e-9),
        }
        if extra:
            evt["extra"] = extra
        self.boxop_status_pub.publish(StringMsg(data=json.dumps(evt)))



    def _send_nav_goal_handle(self, frame: str, x: float, y: float, yaw: float,
                              h: StepHandle, ctx: dict) -> StepHandle:
        """
        Send a Nav2 NavigateToPose goal and complete StepHandle when done.
        Uses h._cancel if caller sets it, but also installs a nav cancel hook.
        """
        if self._sim_is_on():
            # duration based on distance and yaw delta
            with self._sim_pose_lock:
                x0 = float(self._sim_pose["x"])
                y0 = float(self._sim_pose["y"])
                yaw0 = float(self._sim_pose["yaw"])

            dx = float(x) - x0
            dy = float(y) - y0
            dist = math.hypot(dx, dy)

            # yaw delta (shortest)
            yaw1 = float(yaw)
            dyaw = (yaw1 - yaw0 + math.pi) % (2.0 * math.pi) - math.pi

            lin = max(0.05, float(self.get_parameter("sim_lin_speed_mps").value))
            ang = max(0.05, float(self.get_parameter("sim_ang_speed_rps").value))
            min_s = max(0.0, float(self.get_parameter("sim_move_min_s").value))
            jitter = max(0.0, float(self.get_parameter("sim_move_jitter_s").value))

            total = max(min_s, dist / lin + abs(dyaw) / ang)
            if jitter > 0.0:
                total += (random.random() * 2.0 - 1.0) * jitter
                total = max(min_s, total)

            canceled = {"v": False}
            prev_cancel = getattr(h, "_cancel", None)

            def _cancel():
                canceled["v"] = True
                try:
                    if callable(prev_cancel):
                        prev_cancel()
                finally:
                    if not h.done():
                        h.outcome = "canceled"
                        h.mark_done()

            h._cancel = _cancel

            tbox = {"t": None}
            def _finish():
                t = tbox["t"]
                try:
                    if t: t.cancel()
                except Exception:
                    pass
                self._live_timers.discard(t)

                if canceled["v"] or h.done():
                    return

                with self._sim_pose_lock:
                    self._sim_pose["x"] = float(x)
                    self._sim_pose["y"] = float(y)
                    self._sim_pose["yaw"] = float(yaw)

                ctx.setdefault("nav", {})
                ctx["nav"]["status"] = int(GoalStatus.STATUS_SUCCEEDED)
                ctx["nav"]["sim"] = {"dist_m": dist, "dyaw_rad": dyaw, "duration_s": total}
                h.outcome = "ok"
                h.mark_done()

            t = self.create_timer(max(0.01, total), _finish)
            tbox["t"] = t
            self._live_timers.add(t)
            return h

        
        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = str(frame or "map")
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.orientation = yaw_to_q(float(yaw))
        goal.pose = ps

        canceled = {"v": False}
        goal_handle_box = {"gh": None}

        # If caller already set a cancel, keep it, but also cancel nav goal if possible.
        prev_cancel = getattr(h, "_cancel", None)

        def _cancel():
            canceled["v"] = True
            try:
                gh = goal_handle_box["gh"]
                if gh is not None:
                    gh.cancel_goal_async()
            except Exception:
                pass
            try:
                if callable(prev_cancel):
                    prev_cancel()
            finally:
                if not h.done():
                    h.outcome = "canceled"
                    h.mark_done()

        h._cancel = _cancel

        def _goal_done(fut):
            if canceled["v"] or h.done():
                return
            try:
                goal_handle = fut.result()
            except Exception as e:
                self.get_logger().error(f"Nav goal error: {e}")
                self.say("Navigation failed.")
                h.outcome = "error"
                h.mark_done()
                return

            if not goal_handle or not goal_handle.accepted:
                self.say("Navigation goal was rejected.")
                h.outcome = "error"
                h.mark_done()
                return

            goal_handle_box["gh"] = goal_handle

            def _result_done(res_fut):
                if canceled["v"] or h.done():
                    return
                try:
                    res = res_fut.result()
                except Exception as e:
                    self.get_logger().error(f"Nav result error: {e}")
                    self.say("Navigation failed.")
                    h.outcome = "error"
                    h.mark_done()
                    return

                status = getattr(res, "status", 0)
                ctx.setdefault("nav", {})
                ctx["nav"]["status"] = int(status)

                if status == GoalStatus.STATUS_SUCCEEDED:
                    self.say("Arrived.")
                    h.outcome = "ok"
                else:
                    self.say("Navigation failed.")
                    h.outcome = "error"
                h.mark_done()

            goal_handle.get_result_async().add_done_callback(_result_done)

        self.nav_client.send_goal_async(goal).add_done_callback(_goal_done)
        return h


    def _get_costmap_async(self, use_local: bool, timeout_s: float, on_done):
        """
        Request costmap asynchronously. Calls on_done(costmap_msg_or_None).
        Returns StepHandle that completes when response (or timeout) happens.
        """
        h = StepHandle()

        cli = self._local_costmap_cli if use_local else self._global_costmap_cli
        which = "local" if use_local else "global"

        # Ensure service is available
        if not cli.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn(f"[costmap] {which} get_costmap service not ready")
            on_done(None)
            h.outcome = "error"
            h.mark_done()
            return h

        req = GetCostmap.Request()
        fut = cli.call_async(req)

        done = {"v": False}

        def _finish(cm, outcome):
            if done["v"]:
                return
            done["v"] = True
            try:
                on_done(cm)
            finally:
                h.outcome = outcome
                h.mark_done()

        # Timeout guard
        def _timeout():
        
            try: t.cancel()
            except Exception: pass
            self._live_timers.discard(t)
            self.get_logger().warn(f"[costmap] {which} get_costmap timeout after {timeout_s:.2f}s")
            _finish(None, "timeout")

        t = self.create_timer(float(timeout_s), _timeout)
        self._live_timers.add(t)

        def _cb(f):
            # cancel timeout timer
            try:
                t.cancel()
            except Exception:
                pass
            self._live_timers.discard(t)

            try:
                resp = f.result()
                cm = resp.map if resp else None
                _finish(cm, "ok" if cm is not None else "error")
            except Exception as e:
                self.get_logger().warn(f"[costmap] {which} get_costmap call failed: {e}")
                _finish(None, "error")

        fut.add_done_callback(_cb)
        return h



    def _snap_goal_to_free_costmap_cell_from_msg(
        self,
        cm,                     # nav2_msgs/msg/Costmap
        x: float,
        y: float,
        radius_m: float,
        cost_threshold: int,
        used_costmap: str = "local",   # just for info/debug
    ):
        info = {
            "used_costmap": str(used_costmap),
            "radius_m": float(radius_m),
            "cost_threshold": int(cost_threshold),
            "snapped": False,
            "reason": "",
        }

        if cm is None:
            info["reason"] = "costmap_unavailable"
            return None, None, info

        meta = cm.metadata
        res = float(meta.resolution)
        ox = float(meta.origin.position.x)
        oy = float(meta.origin.position.y)
        sx = int(meta.size_x)
        sy = int(meta.size_y)
        data = cm.data  # already a sequence of uint8

        def world_to_map(wx, wy):
            mx = int((wx - ox) / res)
            my = int((wy - oy) / res)
            return mx, my

        def map_to_world(mx, my):
            wx = ox + (mx + 0.5) * res
            wy = oy + (my + 0.5) * res
            return wx, wy

        def in_bounds(mx, my):
            return 0 <= mx < sx and 0 <= my < sy

        def cost_at(mx, my):
            return int(data[my * sx + mx])

        mx0, my0 = world_to_map(float(x), float(y))
        if not in_bounds(mx0, my0):
            info["reason"] = "target_out_of_costmap_bounds"
            return None, None, info

        c0 = cost_at(mx0, my0)
        info["target_cost"] = c0

        # already ok?
        if c0 != 255 and c0 < 254 and c0 <= int(cost_threshold):
            info["reason"] = "target_already_free_enough"
            return float(x), float(y), info

        max_r_cells = max(1, int(float(radius_m) / res))
        best = None  # (dist2, mx, my, cost)

        for r in range(1, max_r_cells + 1):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    mx = mx0 + dx
                    my = my0 + dy
                    if not in_bounds(mx, my):
                        continue
                    c = cost_at(mx, my)
                    if c == 255 or c >= 254:
                        continue
                    if c > int(cost_threshold):
                        continue
                    dist2 = dx * dx + dy * dy
                    if best is None or dist2 < best[0]:
                        best = (dist2, mx, my, c)

            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    mx = mx0 + dx
                    my = my0 + dy
                    if not in_bounds(mx, my):
                        continue
                    c = cost_at(mx, my)
                    if c == 255 or c >= 254:
                        continue
                    if c > int(cost_threshold):
                        continue
                    dist2 = dx * dx + dy * dy
                    if best is None or dist2 < best[0]:
                        best = (dist2, mx, my, c)

            if best is not None:
                break

        if best is None:
            info["reason"] = "no_free_cell_found_within_radius"
            return None, None, info

        _, mx_best, my_best, c_best = best
        sxw, syw = map_to_world(mx_best, my_best)

        info["snapped"] = True
        info["snapped_cost"] = int(c_best)
        info["snapped_cell"] = [int(mx_best), int(my_best)]
        info["snapped_xy"] = [float(sxw), float(syw)]
        info["reason"] = "snapped_to_nearest_free_cell"
        return float(sxw), float(syw), info



    def _srv_cancel_all(self, req, resp):
        """
        Cancel all active skills: stop their current primitive (if any)
        and mark them done / prune them from the active list.
        """
        canceled = 0
        for inst in list(self.skill_engine._active):
            if not inst.done:
                # cancel running primitive if there is one
                if inst.handle is not None and not inst.handle.done():
                    try:
                        inst.handle.cancel()
                    except Exception as e:
                        self.get_logger().warn(f"cancel_all: handle cancel error: {e}")
                inst.done = True
                canceled += 1

        # prune finished instances
        self.skill_engine._active = [i for i in self.skill_engine._active if not i.done]

        resp.success = True
        resp.message = f"Canceled {canceled} active skills."
        self.get_logger().info(resp.message)
        return resp


    def _cancel_all_active(self, why: str = "") -> int:
        canceled = 0
        for inst in list(self.skill_engine._active):
            if inst.done:
                continue
            # cancel running primitive if there is one
            if inst.handle is not None and not inst.handle.done():
                try:
                    inst.handle.cancel()
                except Exception as e:
                    self.get_logger().warn(f"cancel_all({why}): handle cancel error: {e}")
            inst.done = True
            canceled += 1

        # prune finished instances
        self.skill_engine._active = [i for i in self.skill_engine._active if not i.done]
        if canceled:
            self.get_logger().info(f"[SkillsAgent] canceled {canceled} active skills ({why})")
        return canceled


    # ───────────────────────────── Skills loading (base + composite) ─────────
    
    def _on_skill_event(self, event: dict):
        """
        Event from SkillEngineV2: publish to /skills/status as JSON.
        Typical payload:
          {
            "kind": "skill_started"|"skill_finished"|"step_started",
            "skill": "sense.here",
            "step_idx": 0,
            "reason": "all_steps" | "composite_until" | ...,
            "ctx": {...},
            "started_ms": ...,
            "activated": true,
            "done": true/false
          }
        """
        # publish status as before
        try:
            msg = StringMsg()
            msg.data = json.dumps(event, ensure_ascii=False)
            self.skill_status_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish skill status: {e}")

        # NEW: say something when a high-level skill finishes
        try:
            if event.get("kind") == "skill_finished" and event.get("is_root", True):
                reason = event.get("reason", "")
                self.get_logger().info(
                    f"[SkillsAgent] High-level skill '{event.get('skill')}' finished; announcing."
                )
                #self.say("Execution ended.")
        except Exception as e:
            self.get_logger().warn(f"Failed to emit final TTS on skill_finished: {e}")



    def _cb_tts_busy(self, msg: Bool):
        """
        Track speaking state from TTSPlayerNode.

        We treat each falling edge (True -> False) as 'one utterance finished',
        and complete one pending TTS StepHandle for it.
        """
        prev = self._tts_busy
        self._tts_busy = bool(msg.data)
        self._tts_has_busy = True

        if prev != self._tts_busy:
            self.get_logger().info(f"[TTS busy] {prev} -> {self._tts_busy}")

        # On falling edge (done speaking): complete one waiting handle
        if prev and not self._tts_busy:
            if self._tts_waiting:
                h = self._tts_waiting.pop(0)
                if not h.done():
                    self.get_logger().info("[TTS] marking one waiting handle done (speech finished).")
                    h.mark_done()

    
    def _read_yaml_if_exists(self, path: str) -> Optional[dict]:
        if not path:
            return None
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().error(f"Failed to read skills YAML '{path}': {e}")
            return None

    def _merged_skills_doc(self) -> dict:
        """
        Merge base + composite libraries into a single doc for SkillEngineV2.
        Base is treated as immutable; composite adds more skills on top.
        """
        base_doc = self._read_yaml_if_exists(self.skills_base_path) or {}
        comp_doc = self._read_yaml_if_exists(self.skills_composite_path) or {}

        merged = {
            "version": base_doc.get("version", 2),
            "defaults": base_doc.get("defaults", {"window_ms": 3000}),
            "skills": []
        }

        # merge base skills
        base_skills = base_doc.get("skills") or []
        if isinstance(base_skills, list):
            merged["skills"].extend(base_skills)

        # merge composite skills (append)
        comp_skills = comp_doc.get("skills") or []
        if isinstance(comp_skills, list):
            merged["skills"].extend(comp_skills)

        return merged

    def _load_skills_merged(self):
        """
        Load merged skills into the engine (base + composite).
        If base path is missing or invalid, falls back to DEFAULT_SKILLS_V2.
        """
        if self.skills_base_path:
            try:
                merged = self._merged_skills_doc()
                yaml_text = yaml.safe_dump(merged)
                self.skill_engine.load_from_string(yaml_text)

                # NEW: clear quarantines on reload (new definitions)
                if hasattr(self.skill_engine, "bad_skills"):
                    self.skill_engine.bad_skills.clear()
                    self.get_logger().info("[SkillEngine] Cleared quarantined skills on reload.")


                # update mtimes
                self._skills_base_mtime = os.path.getmtime(self.skills_base_path) if os.path.isfile(self.skills_base_path) else None
                self._skills_comp_mtime = os.path.getmtime(self.skills_composite_path) if os.path.isfile(self.skills_composite_path) else None

                self.get_logger().info(
                    f"Loaded merged skills from base='{self.skills_base_path}', composite='{self.skills_composite_path}'"
                )
                return
            except Exception as e:
                self.get_logger().error(f"Failed to load merged skills: {e}")

        # Fallback: no base file → inline default
        self.skill_engine.load_from_string(DEFAULT_SKILLS_V2)
        self._skills_base_mtime = None
        self._skills_comp_mtime = None
        self.get_logger().info("Loaded inline DEFAULT_SKILLS_V2 (no base skills file)")



    def _sum_rule_field_since(self, rule_id: str, field: str, since_ms: int) -> float:
        """Sum numeric payload[field] for hits of rule_id with ts >= since_ms."""
        total = 0.0
        try:
            for e in self.rules_view._events:
                if e.get('id') == str(rule_id) and e.get('ts_ms', 0) >= int(since_ms):
                    v = e.get('payload', {}).get(field)
                    if isinstance(v, (int, float)):
                        total += float(v)
        except Exception:
            pass
        return total


    def _call_soon(self, fn):
        """
        Schedule fn() to run on the ROS thread ASAP using a one-shot timer.
        Keeps a strong ref so it won't be GC'd before firing.
        """
        tbox = {"t": None}

        def _run_once():
            t = tbox["t"]
            try:
                if t:
                    t.cancel()
            except Exception:
                pass
            self._live_timers.discard(t)

            try:
                fn()
            except Exception as e:
                self.get_logger().warn(f"_call_soon callback error: {e}")

        t = self.create_timer(0.001, _run_once)  # 1ms one-shot
        tbox["t"] = t
        self._live_timers.add(t)
        return t


    def _tick_engine(self):
        try:
            self.skill_engine.tick()
        except Exception as e:
            self.get_logger().error(f"tick error: {e}")

    # ───────────────────────────── Planning / Execution ────────────────────────
    def _eligible_report(self) -> List[dict]:
        """
        Returns a list of {"name": <composite>, "passing_steps": [indices...]}
        """
        return self.skill_engine.plan_eligible()

    def _run_skill_name(self, name: str, ctx: dict):
        self.skill_engine.run(name, ctx or {})


    def _on_tts_immediate(self, msg: StringMsg):
        """
        One-shot immediate TTS. Does NOT touch the SkillEngine at all.
        Expected payload:
          - either plain text (msg.data is the text)
          - or JSON: {"text": "..."}
        """
        text = msg.data or ""
        try:
            # Try to parse JSON first
            obj = json.loads(msg.data)
            if isinstance(obj, dict) and "text" in obj:
                text = str(obj["text"])
        except Exception:
            # Not JSON, treat as raw text
            pass

        if not text.strip():
            self.get_logger().warn("[tts_immediate] empty text, ignoring.")
            return

        self.get_logger().info(f"{YELLOW}[tts_immediate] saying: {text!r}{RESET}")
        # This uses the existing normalization + publish to /tts
        self.say(text, True)


    def _announce_box_op(self, op: str, phase: str, box_id: int, prop: str, *,
                         detected: Optional[bool] = None,
                         success: Optional[bool] = None,
                         status: Optional[str] = None):
        """
        op: "sensing" | "disposal"
        phase: "start" | "finish" | "cancel"
        """
        if not bool(self.get_parameter("announce_box_ops").value):
            return

        # Keep it short; numbers get normalized by _normalize_tts_text in say()
        if phase == "start":
            self.say(f"Starting {op} for box {box_id}, property {prop}.")
            return

        if phase == "cancel":
            self.say(f"Canceled {op} for box {box_id}, property {prop}.")
            return

        # finish
        if op == "sensing":
            if detected is True:
                self.say(f"Finished sensing box {box_id}, property {prop}. Detected.")
            elif detected is False:
                self.say(f"Finished sensing box {box_id}, property {prop}. Not detected.")
            else:
                self.say(f"Finished sensing box {box_id}, property {prop}.")
            return

        if op == "disposal":
            if success is True:
                self.say(f"Finished disposal for box {box_id}, property {prop}. Success.")
            elif success is False:
                self.say(f"Finished disposal for box {box_id}, property {prop}. Failed.")
            else:
                self.say(f"Finished disposal for box {box_id}, property {prop}.")
            return


    # Topic: /skills/execute
    def _on_execute_msg(self, msg: StringMsg):
        try:
            obj = json.loads(msg.data) if msg.data else {}
        except Exception as e:
            self.get_logger().warn(f"/skills/execute bad JSON: {e}")
            return

        try:
            name = str(obj["skill"])
            ctx  = obj.get("ctx") or {}

            # ------------------------------------------------------------
            # B) CANCEL-GATING LOGIC (PUTS HERE)
            # ------------------------------------------------------------
            new_sig = canonical_box_action_from_execute(self.skill_engine, name, ctx)
            cur_sig = self._current_box_action

            # Only skip cancel if:
            #  - new action is sense/dispose (new_sig != None implies that), AND
            #  - it matches the currently running box action
            if same_canonical_box_action(new_sig, cur_sig):
                self.get_logger().info(
                    f"[SkillsAgent] /skills/execute '{name}': NOT canceling; "
                    f"new matches current box op {cur_sig}"
                )
                return
            else:
                self._cancel_all_active(why=f"before_execute:{name}")

            self._current_box_action = new_sig

            # orchestrator explicitly requested this skill → reset quarantine for it
            if hasattr(self.skill_engine, "bad_skills"):
                if name in self.skill_engine.bad_skills:
                    self.get_logger().info(
                        f"/skills/execute: clearing quarantine for skill '{name}' "
                        f"due to explicit orchestrator request."
                    )
                    self.skill_engine.bad_skills.discard(name)

            self.skill_engine.arm(name, ctx)   # ← arm, don’t run once
            self.get_logger().info(
                f"/skills/execute armed '{name}' "
                f"(active={self.skill_engine.active_count()})"
            )
        except KeyError:
            self.get_logger().warn("execute: expected key 'skill'")
        except Exception as e:
            self.get_logger().error(f"execute error: {e}")


    # Topic: /skills/plan_req -> reply on /skills/plan_result
    def _on_plan_req_msg(self, msg: StringMsg):
        try:
            rep = {"ts": time.time(), "eligible": self._eligible_report()}
            self.pub_planres.publish(StringMsg(data=json.dumps(rep)))
        except Exception as e:
            self.get_logger().error(f"plan_req error: {e}")

    # Service: /skills/plan (Trigger)
    def _srv_plan(self, req, resp):
        try:
            eligible = self._eligible_report()
            resp.success = True
            resp.message = json.dumps(eligible)
        except Exception as e:
            resp.success = False
            resp.message = str(e)
        return resp

    # Service: /skills/run_all_eligible (Trigger)
    def _srv_run_all_eligible(self, req, resp):
        try:
            count = 0
            for entry in self._eligible_report():
                self._run_skill_name(entry["name"], ctx={})
                count += 1
            resp.success = True
            resp.message = f"Ran {count} eligible skills."
        except Exception as e:
            resp.success = False
            resp.message = f"run_all_eligible error: {e}"
        return resp

    # Service: /skills/reload (Trigger)
    def _srv_reload_skills(self, req, resp):
        try:
            self._load_skills_merged()
            resp.success = True
            resp.message = "Reloaded merged skills (base + composite)."
            self.get_logger().info(resp.message)
        except Exception as e:
            resp.success = False
            resp.message = f"Reload failed: {e}"
            self.get_logger().error(resp.message)
        return resp


    # ───────────────────────────── Basic Actions ──────────────────────────────

    def _cb_coverage_status(self, msg: StringMsg):
        try:
            obj = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"coverage/status bad JSON: {e}")
            return

        if obj.get("client") not in (None, "skills"):
            return

        req_id = obj.get("id")
        if not req_id:
            return

        pending = self._coverage_pending.get(req_id)
        if not pending:
            return

        ctx = pending["ctx"]
        ctx["coverage"] = obj  # store latest status in ctx for debugging/introspection

        state = obj.get("state", "")
        if state in ("done", "canceled", "error"):
            handle = pending["handle"]
            self._coverage_pending.pop(req_id, None)
            handle.mark_done()


    def _cb_llm_speech_resp(self, msg: StringMsg):
        try:
            obj = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"llm_speech_check_resp bad JSON: {e}")
            return

        req_id = obj.get("id")
        if req_id is None:
            return

        pending = self._llm_speech_pending.pop(req_id, None)
        if not pending:
            return

        handle: StepHandle = pending["handle"]
        ctx = pending["ctx"]

        ctx["speech_check"] = {
            "success": bool(obj.get("success", False)),
            "raw_text": obj.get("raw_text", ""),
            "json_text": obj.get("json_text", ""),
            "model_id": obj.get("model_id", ""),
            "lat_ms": float(obj.get("lat_ms", 0.0)),
            "tag": obj.get("tag", ""),
        }

        self.get_logger().info(
            f"[llm_speech_check] id={req_id} tag={ctx['speech_check']['tag']} "
            f"success={ctx['speech_check']['success']} "
            f"lat={ctx['speech_check']['lat_ms']:.1f} ms"
        )

        handle.mark_done()

    def _cb_vlm_resp(self, msg: StringMsg):
        try:
            obj = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"vlm_resp bad JSON: {e}")
            return

        client = obj.get("client")
        if client not in (None, "skills"):
            # Ignore responses from other callers (e.g., eventlayer)
            return


        req_id = obj.get("id")
        if not req_id:
            return

        key = str(req_id)
        pending = self._vlm_pending.pop(key, None)

        if not pending:
            self.get_logger().warn(f"[VLM] response for unknown id={key!r} (client={client!r}); pending={list(self._vlm_pending.keys())[:5]}")
            return



        handle: StepHandle = pending["handle"]
        ctx = pending["ctx"]

        # Parse json_text into a Python object, if present
        raw_json = obj.get("json_text", "") or ""
        parsed = None
        if raw_json:
            try:
                parsed = json.loads(raw_json)
            except Exception as e:
                self.get_logger().warn(f"[VLM] failed to parse json_text: {e} (snippet={raw_json[:160]!r})")

        # Mirror llm_speech_check, but with parsed structure
        ctx["vlm"] = {
            "success":  bool(obj.get("success", False)),
            "raw_text": obj.get("raw_text", ""),
            "json_text": raw_json,
            "parsed":   parsed,                   # ← NEW: structured dict/list
            "model_id": obj.get("model_id", ""),
            "lat_ms":   float(obj.get("lat_ms", 0.0)),
            "tag":      obj.get("tag", ""),
        }

        self.get_logger().info(
            f"[VLM] id={key} tag={ctx['vlm']['tag']} "
            f"success={ctx['vlm']['success']} "
            f"lat={ctx['vlm']['lat_ms']:.1f} ms "
            f"has_parsed={parsed is not None}"
        )

        handle.mark_done()



    def say(self, text, mediator = False):
        raw = str(text)
        spoken = _normalize_tts_text(raw)

        # read param live so you can toggle at runtime
        sim = bool(self.get_parameter("sim_mode").value)

        if sim:
            if mediator:
                self._publish_simulated_stt(spoken)
            return

        self.get_logger().info(f"[TTS] {spoken}")
        self.tts_pub.publish(StringMsg(data=spoken))


    def _publish_simulated_stt(self, text: str):
        if not text.strip():
            return

        speaker_id = str(self.get_parameter("sim_tts_speaker_id").value or "robot")
        delay_s = float(self.get_parameter("sim_tts_delay_s").value)

        payload = {
            "text": text,
            "speaker_id": speaker_id,
            "ts": float(self.get_clock().now().nanoseconds * 1e-9),
            "simulated": True,
            "source": "skills_agent",
        }

        def _do_pub():
            try:
                self._sim_tts_pub.publish(StringMsg(data=json.dumps(payload)))
                self.get_logger().info(f"[SIM_TTS->STT] {payload['text']!r} (speaker_id={speaker_id})")
            except Exception as e:
                self.get_logger().warn(f"[SIM_TTS->STT] publish failed: {e}")

        if delay_s > 0.0:
            # one-shot timer; keep strong ref to avoid GC
            tbox = {"t": None}
            def _fire_once():
                t = tbox["t"]
                try:
                    if t: t.cancel()
                except Exception:
                    pass
                self._live_timers.discard(t)
                _do_pub()

            t = self.create_timer(delay_s, _fire_once)
            tbox["t"] = t
            self._live_timers.add(t)
        else:
            _do_pub()



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

    def _start_person_search(self, doa_deg: float):
        if self._search_active:
            return
        self._search_dir = 1.0 if float(doa_deg) >= 0.0 else -1.0
        yaw_now = self._current_yaw_ref_base()
        self._search_prev_yaw   = yaw_now
        self._search_turned_abs = 0.0
        self._search_active     = True
        self._stop_turn("start-person-search")
        self.get_logger().info(
            f"Person search started. dir={'left' if self._search_dir>0 else 'right'}, speed={self.search_w:.2f} rad/s"
        )

    def _stop_person_search(self, why: str = ""):
        self._search_active = False
        self._stop_turn(why or "stop-person-search")

    def _control_loop(self):
        if not self._search_active:
            return
        yaw_now = self._current_yaw_ref_base()
        dyaw = self._wrap_pi(yaw_now - self._search_prev_yaw)
        self._search_prev_yaw = yaw_now
        self._search_turned_abs += abs(dyaw)
        if self._search_turned_abs >= (2.0*math.pi - math.radians(5.0)):
            self._stop_person_search("full-turn-complete")
            self.say("I didn’t find you.")
            return
        cmd = Twist()
        cmd.angular.z = self._search_dir * max(0.05, min(self.search_w, self.name_max_w))
        self._publish_smoothed(cmd, "person-search")

    # Nav2 absolute
    def navigate_absolute(self, frame: str, x: float, y: float, yaw: float):
        if not self.nav_client.wait_for_server(timeout_sec=0.5):
            self.say("Navigation is not available.")
            return
        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.orientation = yaw_to_q(float(yaw))
        goal.pose = ps
        send_future = self.nav_client.send_goal_async(goal)
        send_future.add_done_callback(self._on_nav_goal_response_abs)

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

    # Nav2 relative
    def navigate_relative(self, az_deg: float, dist_m: float):
        """
        Relative move composed of:
          1) rotate-in-place by az_deg (cmd_vel)
          2) forward nudge by dist_m (cmd_vel)
        No Nav2 goals are sent here.
        """
        # --- params / clamps ---
        az_deg = float(az_deg)
        dist_m = float(dist_m)
        turn_speed = 0.25  # rad/s (tunable: expose as a ROS param if you want)
        fwd_speed  = 0.25  # m/s   (tunable)
        min_turn_deg = 3.0
        eps_dist = 0.02

        # --- 1) rotate ---
        if abs(az_deg) >= min_turn_deg:
            target = math.radians(az_deg)
            w = turn_speed if target >= 0 else -turn_speed
            # duration = angle / speed; clamp to reasonable bounds
            rot_duration = max(0.1, min(abs(target) / max(abs(w), 1e-3), 4.0))
            self._start_twist_timer(duration_s=float(rot_duration),
                                    linear_x=0.0, angular_z=w)

            # Let the rotation finish before the forward nudge
            # Chain the forward phase by scheduling it after rot_duration
            def _after_turn():
                if dist_m > eps_dist:
                    t = dist_m / max(fwd_speed, 0.05)
                    self._start_twist_timer(duration_s=float(t),
                                            linear_x=fwd_speed, angular_z=0.0,
                                            on_complete=lambda: self.say("Moved closer."))
            # one-shot timer to start forward after turn completes
            self.create_timer(rot_duration, _after_turn)
        else:
            # No meaningful rotation; just forward if requested
            if dist_m > eps_dist:
                t = dist_m / max(fwd_speed, 0.05)
                self._start_twist_timer(duration_s=float(t),
                                        linear_x=fwd_speed, angular_z=0.0,
                                        on_complete=lambda: self.say("Moved closer."))




    # Timed Twist helper
    def _start_twist_timer(self, duration_s: float, *, linear_x=0.0, angular_z=0.0, on_complete=None):
        """
        Publishes a Twist at 20 Hz for duration_s seconds using a Timer.
        Keeps a strong reference to the timer to prevent GC.
        Returns the timer so callers can cancel it.
        """
        period = 0.05  # 20 Hz
        remaining = float(duration_s)

        tw = Twist()
        tw.linear.x = float(linear_x)
        tw.angular.z = float(angular_z)

        # publish once immediately so you don't wait for the first tick
        self.cmd_vel_pub.publish(tw)
        self.get_logger().info(f"[twist] start: vx={tw.linear.x:.3f} wz={tw.angular.z:.3f} for {remaining:.2f}s")

        state = {"remaining": remaining}

        def _tick():
            state["remaining"] -= period
            if state["remaining"] > 0.0:
                self.cmd_vel_pub.publish(tw)
            else:
                # stop and cleanup
                self.cmd_vel_pub.publish(Twist())
                self.get_logger().info("[twist] stop")
                try:
                    timer.cancel()
                except Exception:
                    pass
                # drop strong ref so GC can reclaim
                self._live_timers.discard(timer)
                if callable(on_complete):
                    try:
                        on_complete()
                    except Exception as e:
                        self.get_logger().warn(f"on_complete error: {e}")

        timer = self.create_timer(period, _tick)
        # keep the timer alive
        self._live_timers.add(timer)
        return timer


    # Beacon DB speech
    def _say_top3_beacons_from_db(self, top_n: int = 3):
        db = os.path.expanduser(self.bt_db_path)
        if not os.path.exists(db):
            self.say("I don’t have any beacon data yet.")
            return
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro&cache=shared", uri=True, timeout=0.5)
            cur = conn.cursor()
            cur.execute("""
                WITH best AS (
                  SELECT
                    om.object_id,
                    COALESCE(
                      (SELECT rssi FROM obj_measurements WHERE object_id=om.object_id AND slot='current' LIMIT 1),
                      (SELECT rssi FROM obj_measurements WHERE object_id=om.object_id AND slot='min' LIMIT 1)
                    ) AS rssi,
                    COALESCE(
                      (SELECT contaminated FROM obj_measurements WHERE object_id=om.object_id AND slot='current' LIMIT 1),
                      (SELECT contaminated FROM obj_measurements WHERE object_id=om.object_id AND slot='min' LIMIT 1)
                    ) AS contaminated_local,
                    COALESCE(
                      (SELECT probability FROM obj_measurements WHERE object_id=om.object_id AND slot='current' LIMIT 1),
                      (SELECT probability FROM obj_measurements WHERE object_id=om.object_id AND slot='min' LIMIT 1)
                    ) AS probability_local,
                    COALESCE(
                      (SELECT phone_id FROM obj_measurements WHERE object_id=om.object_id AND slot='current' LIMIT 1),
                      (SELECT phone_id FROM obj_measurements WHERE object_id=om.object_id AND slot='min' LIMIT 1)
                    ) AS phone_id
                  FROM obj_measurements om
                  GROUP BY om.object_id
                ),
                merged AS (
                  SELECT
                    b.object_id,
                    b.rssi,
                    COALESCE(b.contaminated_local, cr.contaminated) AS contaminated,
                    COALESCE(b.probability_local,  cr.probability)  AS probability
                  FROM best b
                  LEFT JOIN contamination_records cr
                    ON cr.object_id = b.object_id AND cr.phone_id = b.phone_id
                )
                SELECT object_id, rssi, contaminated, probability
                FROM merged
                ORDER BY rssi DESC
                LIMIT ?;
            """, (int(top_n),))
            rows = cur.fetchall()
            conn.close()
        except Exception as e:
            self.get_logger().warn(f"DB read failed: {e}")
            self.say("I couldn’t read the beacon map.")
            return

        if not rows:
            self.say("I don’t have any beacons detected yet.")
            return



        items_spoken = []
        for object_id, rssi, contaminated, probability in rows:
            try:
                rssi_i = int(rssi)
            except Exception:
                rssi_i = -999
            rssi_words = f"{_num_to_words(rssi_i)} decibels"
            contam_words = ""
            if contaminated is not None:
                contam_words = " contaminated" if int(contaminated) == 1 else " clean"
                if probability is not None:
                    try:
                        p = float(probability)
                        if p <= 1.0: p *= 100.0
                        p_str = f"{p:.1f}".rstrip("0").rstrip(".")
                        contam_words += f" at {p_str} percent"
                    except Exception:
                        pass
            node_id_spoken = str(object_id)
            try:
                nid = int(str(object_id).split("CNode")[1])
                node_id_spoken = f"node {_num_to_words(nid)}"
            except Exception:
                pass
            phrase = f"{node_id_spoken}: {rssi_words}"
            if contam_words:
                phrase += f", {contam_words.strip()}"
            items_spoken.append(phrase)

        if len(items_spoken) == 1:
            self.say(f"The strongest signal is {items_spoken[0]}.")
        else:
            spoken = ", ".join(items_spoken[:-1]) + ", and " + items_spoken[-1] if len(items_spoken) > 2 else " and ".join(items_spoken)
            self.say(f"The top {min(top_n, len(items_spoken))} signals are {spoken}.")

    # Bindings factory (so actions call our methods)
    def _make_bindings_for_self(self):
    
        def sense_box(node_id: str, property: str = "X", ctx: dict = None):
            """
            Sensing primitive.

            Synchronously calls the FastAPI /sense endpoint:

              POST /sense
              {
                "agent_id": <self.agent_id>,
                "box_id":   <int>,
                "property": "X" | "Y"
              }

            and records the result into ctx["box"]["sense_result"].

            On cancel(), we best-effort call /sense/cancel for the same
            (agent_id, box_id, property).
            """
            h = StepHandle()
            if ctx is None:
                ctx = {}

            node_id = str(node_id or "").strip()
            if not node_id:
                self.get_logger().warn("[box.sense] empty node_id; skipping.")
                h.mark_done()
                return h

            if not self.box_server_url:
                self.get_logger().warn("[box.sense] box_server_url is empty; skipping /sense call.")
                h.mark_done()
                return h

            # CNode1## → int box_id
            def _box_id_from_node_id(nid: str) -> int | None:
                s = str(nid).strip()
                if s.lower().startswith("cnode"):
                    s = s[5:]  # remove 'CNode'

                # expect format: '1##'
                if len(s) >= 3 and s[0] == "1":
                    s = s[1:]  # drop the fixed '1' prefix

                try:
                    return int(s)
                except Exception:
                    return None


            box_id = _box_id_from_node_id(node_id)
            if box_id is None:
                self.get_logger().warn(f"[box.sense] could not map node_id={node_id!r} to box_id; skipping.")
                h.mark_done()
                return h

            prop = str(property or "X").upper()
            if prop not in ("X", "Y"):
                self.get_logger().warn(f"[box.sense] invalid property={property!r}; forcing 'X'.")
                prop = "X"

            # Track "current action" for cancel gating
            #self._current_box_action = {"kind": "sense", "node_id": node_id, "property": prop}


            base_url   = self.box_server_url.rstrip("/")
            url        = base_url + "/sense"
            cancel_url = base_url + "/sense/cancel"

            payload = {
                "agent_id": self.agent_id,
                "box_id":   box_id,
                "property": prop,
            }

            req_id = f"sense:{self.agent_id}:{box_id}:{prop}:{int(time.time()*1000)}"

            self._publish_boxop(
                phase="start", op="sense", box_id=box_id, prop=prop,
                req_id=req_id, status="starting"
            )


            self.get_logger().info(f"[box.sense] POST {url} {payload}")

            result = {
                "status": None,          # "completed" | "cached" | "cancelled"
                "detected": None,        # bool | None
                "probability": None,     # float | None
                "deadline": None,
                "x": None,
                "y": None,
                "requested_at": None,
                "completed_at": None,
                "error": None,
            }

            done = {"v": False}

            def _finish():
                if done["v"] or h.done():
                    return
                done["v"] = True

                self._announce_box_op(
                    "sensing",
                    "finish",
                    box_id,
                    prop,
                    detected=result.get("detected"),
                    status=result.get("status"),
                )

                ctx.setdefault("box", {})
                ctx["box"].update({
                    "node_id": node_id,
                    "box_id": box_id,
                    "property": prop,
                    "sense_result": result,
                })

                self.get_logger().info(
                    f"[box.sense] box_id={box_id}, prop={prop}, "
                    f"status={result.get('status')}, detected={result.get('detected')}, "
                    f"prob={result.get('probability')}"
                )

                if same_canonical_box_action(self._current_box_action, {"kind":"sense","node_id":node_id,"property":prop}):
                    self._current_box_action = None

                self._publish_boxop(
                    phase="finish", op="sense", box_id=box_id, prop=prop,
                    req_id=req_id,
                    status=result.get("status") or "",
                    detected=result.get("detected"),
                    probability=result.get("probability"),
                    why="completed"
                )


                h.mark_done()


            # --- define cancel hook before the blocking call ---
            def _cancel():
                """
                Best-effort cancellation via /sense/cancel.
                """
                if done["v"] or h.done():
                    return

                try:
                    cancel_payload = dict(payload)
                    self.get_logger().info(f"[box.sense] POST {cancel_url} {cancel_payload} (cancel)")
                    resp_c = requests.post(cancel_url, json=cancel_payload, timeout=self.box_cancel_timeout)
                    self.get_logger().info(
                        f"[box.sense] cancel response code={resp_c.status_code} "
                        f"body={resp_c.text[:160]!r}"
                    )
                    result["status"] = "cancelled"
                except Exception as e:
                    self.get_logger().warn(f"[box.sense] /sense/cancel failed: {e}")
                finally:
                    self._publish_boxop(
                        phase="cancel", op="sense", box_id=box_id, prop=prop,
                        req_id=req_id, status="cancelled", why="explicit_cancel"
                    )

                    self._announce_box_op("sensing", "cancel", box_id, prop, status="cancel")
                    self._call_soon(_finish)

            h._cancel = _cancel

            self._announce_box_op("sensing", "start", box_id, prop, status="starting")

            def _worker():
                try:
                    resp = requests.post(url, json=payload, timeout=self.box_req_timeout)
                    if resp.status_code != 200:
                        msg = f"non-200 status {resp.status_code}"
                        self.get_logger().warn(f"[box.sense] /sense {payload} -> {msg}")
                        result["error"] = msg
                    else:
                        data = resp.json()
                        result.update(
                            status=data.get("status"),
                            detected=data.get("detected"),
                            probability=data.get("probability"),
                            deadline=data.get("deadline"),
                            x=data.get("x"),
                            y=data.get("y"),
                            requested_at=data.get("requested_at"),
                            completed_at=data.get("completed_at"),
                        )
                except Exception as e:
                    msg = f"/sense failed: {e}"
                    self.get_logger().warn(f"[box.sense] {msg}")
                    result["error"] = msg

                self._call_soon(_finish)

            threading.Thread(target=_worker, daemon=True).start()
            return h



        def dispose_box(node_id: str, property: str = "X", ctx: dict = None):
            """
            Disposal primitive.

            Synchronously calls the FastAPI /dispose endpoint:

              POST /dispose
              {
                "agent_id": <self.agent_id>,
                "box_id":   <int>,
                "property": "X" | "Y"
              }

            and records the result into ctx["box"]["dispose_result"].

            On cancel(), we best-effort call /dispose/cancel for the same
            (agent_id, box_id, property).
            """
            h = StepHandle()
            if ctx is None:
                ctx = {}

            node_id = str(node_id or "").strip()
            if not node_id:
                self.get_logger().warn("[box.dispose] empty node_id; skipping.")
                h.mark_done()
                return h

            if not self.box_server_url:
                self.get_logger().warn("[box.dispose] box_server_url is empty; skipping /dispose call.")
                h.mark_done()
                return h

            box_id = _box_id_from_node_id(node_id)
            if box_id is None:
                self.get_logger().warn(f"[box.dispose] could not map node_id={node_id!r} to box_id; skipping.")
                h.mark_done()
                return h

            prop = str(property or "X").upper()
            if prop not in ("X", "Y"):
                self.get_logger().warn(f"[box.dispose] invalid property={property!r}; forcing 'X'.")
                prop = "X"
            #self._current_box_action = {"kind": "dispose", "node_id": node_id, "property": prop}


            base_url   = self.box_server_url.rstrip("/")
            url        = base_url + "/dispose"
            cancel_url = base_url + "/dispose/cancel"

            payload = {
                "agent_id": self.agent_id,
                "box_id":   box_id,
                "property": prop,
            }

            req_id = f"dispose:{self.agent_id}:{box_id}:{prop}:{int(time.time()*1000)}"
            self._publish_boxop(phase="start", op="dispose", box_id=box_id, prop=prop,
                                req_id=req_id, status="starting")


            self.get_logger().info(f"[box.dispose] POST {url} {payload}")

            result = {
                "status": None,          # "completed" | "cancelled"
                "success": None,         # bool | None
                "deadline": None,
                "x": None,
                "y": None,
                "requested_at": None,
                "completed_at": None,
                "error": None,
            }

            done = {"v": False}

            def _finish():
                if done["v"] or h.done():
                    return
                done["v"] = True

                self._announce_box_op(
                    "disposal",
                    "finish",
                    node_id,
                    prop,
                    success=result.get("success"),
                    status=result.get("status"),
                )

                ctx.setdefault("box", {})
                ctx["box"].update({
                    "node_id": node_id,
                    "box_id": box_id,
                    "property": prop,
                    "dispose_result": result,
                })

                self.get_logger().info(
                    f"[box.dispose] box_id={box_id}, prop={prop}, "
                    f"status={result.get('status')}, success={result.get('success')}"
                )

                if same_canonical_box_action(self._current_box_action, {"kind":"dispose","node_id":node_id,"property":prop}):
                    self._current_box_action = None

                self._publish_boxop(phase="finish", op="dispose", box_id=box_id, prop=prop,
                                    req_id=req_id,
                                    status=result.get("status") or "",
                                    success=result.get("success"),
                                    why="completed")


                h.mark_done()


            # --- define cancel hook *before* the blocking request ---
            def _cancel():
                """
                Best-effort cancellation via /dispose/cancel.
                """
                if done["v"] or h.done():
                    return

                try:
                    cancel_payload = dict(payload)
                    self.get_logger().info(f"[box.dispose] POST {cancel_url} {cancel_payload} (cancel)")
                    resp_c = requests.post(cancel_url, json=cancel_payload, timeout=self.box_cancel_timeout)
                    
     
                    self.get_logger().info(
                        f"[box.dispose] cancel response code={resp_c.status_code} body={resp_c.text[:160]!r}"
                    )
                    result["status"] = "cancelled"
                except Exception as e:
                    self.get_logger().warn(f"[box.dispose] /dispose/cancel failed: {e}")
                finally:
                    self._publish_boxop(phase="cancel", op="dispose", box_id=box_id, prop=prop,
                                        req_id=req_id, status="cancelled", why="explicit_cancel")

                    self._announce_box_op("disposal", "cancel", node_id, prop)
                    self._call_soon(_finish)


            h._cancel = _cancel

            self._announce_box_op("disposal", "start", node_id, prop)

            def _worker():
                try:
                    resp = requests.post(url, json=payload, timeout=self.box_req_timeout)
                    if resp.status_code != 200:
                        msg = f"non-200 status {resp.status_code}"
                        self.get_logger().warn(f"[box.dispose] /dispose {payload} -> {msg}")
                        result["error"] = msg
                    else:
                        data = resp.json()
                        result.update(
                            status=data.get("status"),
                            success=data.get("success"),
                            deadline=data.get("deadline"),
                            x=data.get("x"),
                            y=data.get("y"),
                            requested_at=data.get("requested_at"),
                            completed_at=data.get("completed_at"),
                        )
                except Exception as e:
                    msg = f"/dispose failed: {e}"
                    self.get_logger().warn(f"[box.dispose] {msg}")
                    result["error"] = msg

                self._call_soon(_finish)

            threading.Thread(target=_worker, daemon=True).start()
            return h



    
        def wait_box_nearby(target_node_id: str, timeout_s: float = 10.0, ctx: dict = None):
            """
            Wait until bt_rssi_seen fires for the specified node.

            We compare the normalized node name (CNode###) from the rule payload
            with the given target_node_id.
            """
            h = StepHandle()
            if ctx is None:
                ctx = {}

            target_node_id = str(target_node_id or "").strip()
            if not target_node_id:
                self.get_logger().warn("[wait_box_nearby] empty target_node_id; finishing immediately.")
                h.mark_done()
                return h

            if bool(self.get_parameter("sim_mode").value):
                # assume always detected
                now_ms = int(self.get_clock().now().nanoseconds * 1e-6)
                norm_target = target_node_id if target_node_id.lower().startswith("cnode") else f"CNode{target_node_id}"

                ctx.setdefault("box", {})
                ctx["box"].update({
                    "node_id": norm_target,
                    "box_id": _box_id_from_node_id(norm_target),
                    "seen_nearby_ms": now_ms,
                    "last_bt_payload": {
                        "object_id": norm_target,
                        "rssi": -40,
                        "simulated": True,
                    },
                })

                h.mark_done()
                return h


            start_ms = int(self.get_clock().now().nanoseconds * 1e-6)
            timeout_ms = max(0.5, float(timeout_s)) * 1000.0
            period = 0.1  # 10 Hz
            canceled = {"v": False}
            timers = {"t": None}

            def _cancel():
                canceled["v"] = True
                t = timers["t"]
                if t:
                    try:
                        t.cancel()
                    except Exception:
                        pass
                    self._live_timers.discard(t)
                h.mark_done()

            h._cancel = _cancel

            def _tick():
                if canceled["v"]:
                    return

                now_ms = int(self.get_clock().now().nanoseconds * 1e-6)
                elapsed = now_ms - start_ms

                # Look at latest bt_rssi_seen payload in recent window
                payload = self.rules_view.latest_payload("bt_rssi_seen", within_ms=3000)
                if payload is not None:
                    obj_id_raw = str(payload.get("object_id", "")).strip()

                    # Normalize both to a canonical "CNode###" form for comparison
                    norm_seen = obj_id_raw
                    if not norm_seen.lower().startswith("cnode"):
                        norm_seen = f"CNode{norm_seen}"

                    norm_target = target_node_id
                    if not norm_target.lower().startswith("cnode"):
                        norm_target = f"CNode{norm_target}"

                    if norm_seen == norm_target:
                        self.get_logger().info(
                            f"[wait_box_nearby] target {norm_target} is nearby (bt_rssi_seen.object_id={obj_id_raw!r})"
                        )
                        try:
                            timers["t"].cancel()
                        except Exception:
                            pass
                        self._live_timers.discard(timers["t"])

                        # Record in ctx for downstream states
                        ctx.setdefault("box", {})
                        ctx["box"].update({
                            "node_id": norm_target,
                            "box_id": _box_id_from_node_id(norm_target),
                            "seen_nearby_ms": now_ms,
                            "last_bt_payload": payload,
                        })

                        h.mark_done()
                        return

                if elapsed >= timeout_ms:
                    self.get_logger().warn(
                        f"[wait_box_nearby] timeout waiting for {target_node_id} (~{timeout_s:.1f}s)."
                    )
                    try:
                        timers["t"].cancel()
                    except Exception:
                        pass
                    self._live_timers.discard(timers["t"])
                    
                    h.outcome = "timeout"
                    h.mark_done()
                    return

            timers["t"] = self.create_timer(period, _tick)
            self._live_timers.add(timers["t"])
            return h


    
        def tts(text: str, ctx: dict = None):
            """
            TTS primitive that waits for speech playback to finish.

            Flow:
              - publish text via self.say() (downstream turns it into /tts_wav)
              - if we have a /tts_busy signal, queue a StepHandle that will
                be completed on the next 'speaking=False' edge
              - if we never see /tts_busy, fall back to immediate completion
                so the state machine doesn't deadlock
            """
            
            raw_text = "" if text is None else str(text)

            # Strip common “template produced quotes” artifacts
            candidate = raw_text.strip()
            if (candidate == '""') or (candidate == "''"):
                candidate = ""

            # If empty after cleanup: DO NOT publish, just advance state
            if candidate.strip() == "":
                self.get_logger().warn("[TTS] empty text; skipping publish and completing immediately.")
                h = StepHandle()
                h.mark_done()
                return h
            
            h = StepHandle()
            self.say(str(text))

            # If we've ever seen /tts_busy, treat it as authoritative
            if getattr(self, "_tts_has_busy", False):
                # Queue this handle; _cb_tts_busy will mark it done
                self._tts_waiting.append(h)
                self.get_logger().info(
                    f"[TTS] queued handle waiting for /tts_busy to go False "
                    f"(waiting={len(self._tts_waiting)})"
                )
            else:
                # No TTS player feedback available → don't block
                self.get_logger().warn(
                    "[TTS] /tts_busy has not been observed; "
                    "completing TTS handle immediately."
                )
                h.mark_done()

            return h

        def gesture(kind: str):
            """
            Map a logical gesture 'kind' to a Go2 sport API (api_id)
            and publish a WebRtcReq on /webrtc_req, equivalent to:

              ros2 topic pub /webrtc_req go2_interfaces/msg/WebRtcReq "
                topic: 'rt/api/sport/request'
                api_id: 1016
                parameter: ''
                id: 1" --once
            """
            h = StepHandle()
            try:
                # Map high-level gesture names to the table’s API IDs
                gesture_api_map = {
                    "greet":        1016,  # Hello
                    "hello":        1016,
                }

                api_id = gesture_api_map.get(str(kind), 1016)  # default to Hello

                msg = WebRtcReq()
                msg.topic     = "rt/api/sport/request"
                msg.api_id    = int(api_id)
                msg.parameter = ""   # same as data: '' in the JS / CLI example

                # Use a simple unique-ish ID; or hard-code 1 if you prefer
                msg.id = int(time.time() * 1000) & 0x7FFFFFFF

                self.get_logger().info(
                    f"[gesture] kind='{kind}' -> api_id={msg.api_id}, topic='{msg.topic}', id={msg.id}"
                )
                self.webrtc_req_pub.publish(msg)
            finally:
                h.mark_done()
            return h


        def move_relative(azimuth_deg: float, dist_m: float):
            # return a handle that completes when the chained timers finish
            return self._move_relative_handle(float(azimuth_deg), float(dist_m))

        def move_absolute(frame: str, x: float, y: float, yaw: float, ctx: dict = None):
            h = StepHandle()
            if ctx is None:
                ctx = {}

            if not self.nav_client.wait_for_server(timeout_sec=0.5):
                self.say("Navigation is not available.")
                h.outcome = "error"
                h.mark_done()
                return h

            frame = str(frame or "map")
            x = float(x); y = float(y); yaw = float(yaw)

            snap_enable = bool(self.get_parameter("nav_snap_enable").value)
            radius_m    = float(self.get_parameter("nav_snap_radius_m").value)
            thr         = int(self.get_parameter("nav_snap_cost_threshold").value)
            cm_timeout  = float(self.get_parameter("nav_snap_costmap_timeout_s").value)

            if self._sim_is_on():
                snap_enable = False


            nav_info = {"requested": {"frame": frame, "x": x, "y": y, "yaw": yaw}}
            ctx["nav"] = nav_info

            if not (snap_enable and frame == "map"):
                # no snapping: just send goal directly
                return self._send_nav_goal_handle(frame, x, y, yaw, h, ctx)

            canceled = {"v": False}
            def _cancel():
                canceled["v"] = True
                h.outcome = "timeout"
                h.mark_done()
            h._cancel = _cancel

            # ---- Step 1: request local costmap; if OOB then request global ----
            def after_local(cm_local):
                if canceled["v"] or h.done():
                    return

                snapped_x = snapped_y = None
                info_local = {"reason": "costmap_unavailable"} if cm_local is None else None

                if cm_local is not None:
                    snapped_x, snapped_y, info_local = self._snap_goal_to_free_costmap_cell_from_msg(
                        cm_local, x, y, radius_m=radius_m, cost_threshold=thr
                    )
                nav_info["snap_local"] = info_local

                # if local succeeded, proceed
                if snapped_x is not None and snapped_y is not None:
                    nav_info["final_goal"] = {"frame": frame, "x": snapped_x, "y": snapped_y, "yaw": yaw}
                    return self._send_nav_goal_handle(frame, snapped_x, snapped_y, yaw, h, ctx)

                # If local couldn’t help because goal outside local bounds -> try global
                if info_local and info_local.get("reason") == "target_out_of_costmap_bounds":
                    def after_global(cm_global):
                        if canceled["v"] or h.done():
                            return

                        if cm_global is None:
                            nav_info["snap_global_fallback"] = {"reason": "costmap_unavailable"}
                            self.say("That location is blocked.")
                            h.outcome = "error"
                            h.mark_done()
                            return

                        gx, gy, info_g = self._snap_goal_to_free_costmap_cell_from_msg(
                            cm_global, x, y, radius_m=max(1.5, radius_m * 2.0), cost_threshold=thr
                        )
                        nav_info["snap_global_fallback"] = info_g

                        if gx is None or gy is None:
                            self.say("That location is blocked.")
                            h.outcome = "error"
                            h.mark_done()
                            return

                        nav_info["final_goal"] = {"frame": frame, "x": gx, "y": gy, "yaw": yaw}
                        return self._send_nav_goal_handle(frame, gx, gy, yaw, h, ctx)

                    self._get_costmap_async(use_local=False, timeout_s=max(0.6, cm_timeout * 2.0), on_done=after_global)
                    return

                # Other local failures: treat as blocked
                self.say("That location is blocked.")
                h.outcome = "error"
                h.mark_done()

            self._get_costmap_async(use_local=True, timeout_s=max(0.6, cm_timeout * 2.0), on_done=after_local)
            return h




        def query_beacons(top_n: int, ctx: dict):
            h = StepHandle()
            self._say_top3_beacons_from_db(int(top_n))
            ctx['last_query_beacons_speech'] = "Beacon report complete."
            h.mark_done()
            return h

        def llm_speech_check(prompt: str = "",
                             output_schema: str = "",
                             text: str = "",
                             tag: str = "",
                             ctx: dict = None):
            """
            Generic LLM JSON worker.
            Caller specifies prompt + output_schema; we just relay and await response.
            Result goes into ctx["speech_check"].
            """
            h = StepHandle()
            if ctx is None:
                ctx = {}

            req_id = int(time.time() * 1000) ^ self._llm_speech_next_id
            self._llm_speech_next_id += 1

            self._llm_speech_pending[req_id] = {
                "handle": h,
                "ctx": ctx,
            }

            payload = {
                "id": req_id,
                "prompt": prompt or "",
                "output_schema": output_schema or "",
                "text": text or "",
                "tag": tag or "",
            }
            self.llm_speech_req_pub.publish(StringMsg(data=json.dumps(payload)))
            return h

        def coverage_wait(
            spacing_m: float = 1.5,
            visited_radius_m: float = 0.9,
            dwell_sec: float = 2.0,
            persist_path: str = "/tmp/coverage_wait_visited.json",
            ctx: dict = None,
        ):
            h = StepHandle()
            if ctx is None:
                ctx = {}

            ts_ms = int(time.time() * 1000)
            req_id = f"skills:coverage:{ts_ms}:{self._coverage_next_id}"
            self._coverage_next_id += 1

            self._coverage_pending[req_id] = {"handle": h, "ctx": ctx}

            payload = {
                "id": req_id,
                "client": "skills",
                "cmd": "start",
                "params": {
                    "spacing_m": float(spacing_m),
                    "visited_radius_m": float(visited_radius_m),
                    "dwell_sec": float(dwell_sec),
                    "persist_path": str(persist_path),
                }
            }
            self.coverage_req_pub.publish(StringMsg(data=json.dumps(payload)))

            # cancel hook sends cancel command
            def _cancel():
                try:
                    cancel_payload = {"id": req_id, "client": "skills", "cmd": "cancel"}
                    self.coverage_req_pub.publish(StringMsg(data=json.dumps(cancel_payload)))
                finally:
                    h.mark_done()

            h._cancel = _cancel
            return h



        def vlm_inference(prompt: str = "",
                          output_schema: str = "",
                          tag: str = "",
                          mode: str = "generic",
                          ctx: dict = None):
            """
            Generic VLM micro-service, symmetric with llm_speech_check.
            We just relay request; answer comes back on /vlm/answer.
            """
            h = StepHandle()
            if ctx is None:
                ctx = {}

            ts_ms = int(time.time() * 1000)
            tag = tag or "vlm"
            #Unique + traceable id
            req_id = f"skills:{tag}:{ts_ms}:{self._vlm_next_id}"
            self._vlm_next_id += 1

            self._vlm_pending[req_id] = {"handle": h, "ctx": ctx}

            self.get_logger().info(f"[VLM primitive] id={req_id} prompt={prompt!r}, tag={tag}, mode={mode}")

            payload = {
                "id": req_id,
                "client": "skills",                 # NEW
                "prompt": prompt or "",
                "output_schema": output_schema or "",
                "tag": tag,
                "mode": mode or "generic",
            }
            # Send to VLM node; it should look at the latest frame and respond.
            self.vlm_req_pub.publish(StringMsg(data=json.dumps(payload)))
            return h


        return {
            'tts': tts,
            'gesture': gesture,
            'move_relative': move_relative,
            'move_absolute': move_absolute,
            'query_beacons': query_beacons,
            'llm_speech_check': llm_speech_check,
            'vlm_inference': vlm_inference,
            'coverage_wait': coverage_wait,
            'wait_box_nearby': wait_box_nearby,
            'sense_box': sense_box,
            'dispose_box': dispose_box,
        }

    def _move_relative_handle(self, az_deg: float, dist_m: float) -> StepHandle:
        """
        Rotate-in-place by |az_deg| and then move forward dist_m,
        using *odometry events* to decide when to stop each phase:
          - rotation: accumulate sum(dyaw_deg) from rule 'odom_rot_delta'
          - forward : accumulate sum(dxy)      from rule 'odom_dist_delta'
        Falls back on timeouts to avoid runaway if events are missing.
        """
        
        if self._sim_is_on():
            h = StepHandle()
            az = float(az_deg)
            dist = float(dist_m)

            lin = max(0.05, float(self.get_parameter("sim_lin_speed_mps").value))
            ang = max(0.05, float(self.get_parameter("sim_ang_speed_rps").value))
            min_s = max(0.0, float(self.get_parameter("sim_move_min_s").value))
            jitter = max(0.0, float(self.get_parameter("sim_move_jitter_s").value))

            t_turn = abs(math.radians(az)) / ang if abs(az) > 0.5 else 0.0
            t_fwd  = abs(dist) / lin if abs(dist) > 1e-3 else 0.0
            total = max(min_s, t_turn + t_fwd)

            if jitter > 0.0:
                total += (random.random() * 2.0 - 1.0) * jitter
                total = max(min_s, total)

            canceled = {"v": False}

            def _cancel():
                canceled["v"] = True
                if not h.done():
                    h.outcome = "canceled"
                    h.mark_done()

            h._cancel = _cancel

            tbox = {"t": None}
            def _finish():
                t = tbox["t"]
                try:
                    if t: t.cancel()
                except Exception:
                    pass
                self._live_timers.discard(t)

                if canceled["v"] or h.done():
                    return

                # update simulated pose
                with self._sim_pose_lock:
                    yaw0 = float(self._sim_pose["yaw"])
                    yaw1 = yaw0 + math.radians(az)
                    # wrap
                    yaw1 = (yaw1 + math.pi) % (2.0 * math.pi) - math.pi
                    self._sim_pose["yaw"] = yaw1

                    # move forward in the new heading
                    dx = dist * math.cos(yaw1)
                    dy = dist * math.sin(yaw1)
                    self._sim_pose["x"] += dx
                    self._sim_pose["y"] += dy

                h.outcome = "ok"
                h.mark_done()

            t = self.create_timer(max(0.01, total), _finish)
            tbox["t"] = t
            self._live_timers.add(t)
            return h

        
        h = StepHandle()
        turn_speed = self.turn_speed
        fwd_speed  = self.fwd_speed
        min_turn_deg = 3.0
        eps_dist = 0.02

        # which rules/fields to read from RulesViewROS
        ROT_RULE, ROT_FIELD = "odom_rot_delta", "dyaw_deg"
        DIST_RULE, DIST_FIELD = "odom_dist_delta", "dxy"

        timers = {"rot": None, "fwd": None}
        canceled = {"v": False}

        def stop_all():
            # hard-stop base and cancel timers
            self.cmd_vel_pub.publish(Twist())
            for k, t in list(timers.items()):
                if t:
                    try: t.cancel()
                    except Exception: pass
                    self._live_timers.discard(t)
                    timers[k] = None

        def cancel_fn():
            canceled["v"] = True
            stop_all()

        h._cancel = cancel_fn

        # ------------- Forward phase (distance via odom events) -------------
        def start_forward_phase():
            if canceled["v"]:
                h.mark_done(); return

            if dist_m <= eps_dist:
                stop_all()
                h.mark_done()
                return

            period = 0.05  # 20 Hz
            start_ms = int(self.get_clock().now().nanoseconds * 1e-6)
            target_m = max(0.0, float(dist_m))
            elapsed = 0.0

            # conservative timeout: expected time @ fwd_speed + margin
            exp_t = (target_m / max(fwd_speed, 1e-3))
            timeout_s = max(2.0, min(20.0, exp_t * 2.0))

            tw = Twist()
            tw.linear.x = max(0.05, fwd_speed)  # ensure nonzero
            self.get_logger().info(f"[move_relative] forward start target={target_m:.2f} m, vx={tw.linear.x:.2f} m/s")

            # publish once upfront
            self.cmd_vel_pub.publish(tw)

            def _tick():
                nonlocal elapsed
                if canceled["v"]:
                    stop_all(); h.mark_done(); return

                # keep moving
                self.cmd_vel_pub.publish(tw)

                # accumulate distance from events since start
                acc_m = self._sum_rule_field_since(DIST_RULE, DIST_FIELD, start_ms)

                if acc_m >= target_m:
                    stop_all()
                    h.mark_done()
                    self.get_logger().info(f"[move_relative] forward complete acc={acc_m:.3f} m")
                    return

                elapsed += period
                if elapsed >= timeout_s:
                    stop_all()
                    h.mark_done()
                    self.get_logger().warn(f"[move_relative] forward timeout acc={acc_m:.3f}/{target_m:.3f} m")
                    return

            timers["fwd"] = self.create_timer(period, _tick)
            self._live_timers.add(timers["fwd"])

        # ------------- Rotation phase (degrees via odom events) -------------
        def start_rotation_phase():
            if canceled["v"]:
                h.mark_done(); return

            if abs(az_deg) < min_turn_deg:
                # no meaningful rotation → go forward directly
                start_forward_phase()
                return

            period = 0.05  # 20 Hz
            start_ms = int(self.get_clock().now().nanoseconds * 1e-6)
            target_deg = abs(float(az_deg))
            direction = 1.0 if float(az_deg) >= 0.0 else -1.0
            elapsed = 0.0

            # conservative timeout: angle/speed + margin
            exp_t = (math.radians(target_deg) / max(abs(turn_speed), 1e-3))
            timeout_s = max(2.0, min(20.0, exp_t * 2.0))

            tw = Twist()
            tw.angular.z = direction * max(0.05, min(self.search_w, self.name_max_w))
            self.get_logger().info(f"[move_relative] turn start target={target_deg:.1f}°, wz={tw.angular.z:.2f} rad/s")

            # publish once upfront
            self.cmd_vel_pub.publish(tw)

            def _tick():
                nonlocal elapsed
                if canceled["v"]:
                    stop_all(); h.mark_done(); return

                # keep turning
                self.cmd_vel_pub.publish(tw)

                # accumulate rotation (degrees) from events since start
                acc_deg = self._sum_rule_field_since(ROT_RULE, ROT_FIELD, start_ms)

                if acc_deg >= target_deg:
                    # rotation done → stop and start forward
                    self.cmd_vel_pub.publish(Twist())
                    try:
                        timers["rot"].cancel()
                    except Exception:
                        pass
                    self._live_timers.discard(timers["rot"])
                    timers["rot"] = None
                    self.get_logger().info(f"[move_relative] turn complete acc={acc_deg:.1f}°")
                    start_forward_phase()
                    return

                elapsed += period
                if elapsed >= timeout_s:
                    # timeout but still proceed to forward to avoid deadlock
                    self.get_logger().warn(f"[move_relative] turn timeout acc={acc_deg:.1f}/{target_deg:.1f}°")
                    try:
                        timers["rot"].cancel()
                    except Exception:
                        pass
                    self._live_timers.discard(timers["rot"])
                    timers["rot"] = None
                    self.cmd_vel_pub.publish(Twist())
                    start_forward_phase()
                    return

            timers["rot"] = self.create_timer(period, _tick)
            self._live_timers.add(timers["rot"])

        # Kick it off
        start_rotation_phase()
        return h



    # ───────────────────────────── Hot Reload ─────────────────────────────────
    def _load_skills_initial(self):
        self._load_skills_merged()
        
    def _reload_skills_if_changed(self) -> bool:
        """
        Hot reload if either base or composite file changed on disk.
        Base is *expected* to be immutable at runtime, but we still
        allow reload if it did change for convenience.
        """
        changed = False
        try:
            if self.skills_base_path and os.path.isfile(self.skills_base_path):
                m = os.path.getmtime(self.skills_base_path)
                if self._skills_base_mtime is None or m > self._skills_base_mtime:
                    changed = True

            if self.skills_composite_path and os.path.isfile(self.skills_composite_path):
                m = os.path.getmtime(self.skills_composite_path)
                if self._skills_comp_mtime is None or m > self._skills_comp_mtime:
                    changed = True
        except Exception as e:
            self.get_logger().warn(f"skills reload check failed: {e}")
            return False

        if changed:
            self._load_skills_merged()
            return True
        return False

    def _maybe_reload_skills(self):
        self._reload_skills_if_changed()

# ───────────────────────────────────────────────────────────────────────────────
#                                       main
# ───────────────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = SkillsAgent()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

