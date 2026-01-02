#!/usr/bin/env python3
"""
coverage_wait_node.py

Coverage worker node (RPC-style) that you can drive from SkillsAgent like llm/vlm:

- Subscribes:
    /map               (nav_msgs/OccupancyGrid)
    /coverage/req      (std_msgs/String, JSON)

- Publishes:
    /coverage/status   (std_msgs/String, JSON)

Behavior (when active session started via /coverage/req):
  - Uses TF map->base_link to get robot pose
  - Generates candidate "stop" locations in free space on a grid spacing
  - Always navigates to the closest unvisited location (Nav2 NavigateToPose)
  - On arrival, dwells for dwell_sec, then marks visited and continues
  - Persists visited grid to disk so it can resume after interruption

RPC messages:

/coverage/req JSON:
  {
    "id": "skills:coverage:173...:12",
    "client": "skills",
    "cmd": "start" | "cancel",
    "params": { ... optional overrides ... }
  }

/coverage/status JSON:
  {
    "id": "...",
    "client": "skills",
    "state": "running"|"done"|"canceled"|"error"|"idle",
    "ts": 123.45,
    "msg": "...",
    "progress": {
      "candidates": 123,
      "visited_centers": 45
    }
  }

Notes:
- Single-session model: a new "start" replaces any existing session.
- No rclpy.spin_until_future_complete inside timers (Nav2 fully async).
"""

import os
import math
import time
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Quaternion

from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus

import tf2_ros
from tf_transformations import euler_from_quaternion, quaternion_from_euler


@dataclass
class MapMeta:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float


class CoverageWaitNode(Node):
    def __init__(self):
        super().__init__('coverage_wait_node')

        # ---------------- Params ----------------
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        self.declare_parameter('spacing_m', 1.0)
        self.declare_parameter('visited_radius_m', 0.6)
        self.declare_parameter('dwell_sec', 3.0)
        self.declare_parameter('arrival_radius_m', 0.35)

        # NEW: margin from map borders in meters
        self.declare_parameter('border_margin_m', 0.5)

        self.declare_parameter('occ_free_max', 0)
        self.declare_parameter('occ_occupied_min', 50)
        self.declare_parameter('treat_unknown_as_blocked', True)

        self.declare_parameter('persist_path', '/tmp/coverage_wait_visited.json')
        self.declare_parameter('persist_every_n', 5)

        self.declare_parameter('max_candidates', 40000)
        self.declare_parameter('rebuild_candidates_sec', 5.0)

        # RPC / status
        self.declare_parameter('status_period_sec', 0.5)

        self.map_topic = self.get_parameter('map_topic').value
        self.global_frame = self.get_parameter('global_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.spacing_m = float(self.get_parameter('spacing_m').value)
        self.visited_radius_m = float(self.get_parameter('visited_radius_m').value)
        self.dwell_sec = float(self.get_parameter('dwell_sec').value)
        self.arrival_radius_m = float(self.get_parameter('arrival_radius_m').value)

        self.border_margin_m = float(self.get_parameter('border_margin_m').value)

        self.occ_free_max = int(self.get_parameter('occ_free_max').value)
        self.occ_occupied_min = int(self.get_parameter('occ_occupied_min').value)
        self.treat_unknown_as_blocked = bool(self.get_parameter('treat_unknown_as_blocked').value)

        self.persist_path = str(self.get_parameter('persist_path').value)
        self.persist_every_n = int(self.get_parameter('persist_every_n').value)

        self.max_candidates = int(self.get_parameter('max_candidates').value)
        self.rebuild_candidates_sec = float(self.get_parameter('rebuild_candidates_sec').value)
        self.status_period_sec = float(self.get_parameter('status_period_sec').value)

        # ---------------- Map / visited state ----------------
        self.map_msg: Optional[OccupancyGrid] = None
        self.map_meta: Optional[MapMeta] = None
        self.visited: Optional[List[bool]] = None  # flattened bool list aligned to map cells

        # "visited_centers" = how many dwell targets we've completed since session start
        self.visited_centers = 0
        self.visit_count_since_persist = 0

        self.candidates: List[Tuple[int, int]] = []
        self._last_candidates_build = 0.0

        # persistence load guard (per map meta)
        self._persist_loaded_for_meta = False

        # ---------------- TF ----------------
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------------- Nav2 Action ----------------
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.nav_goal_handle = None
        self.nav_result_future = None
        self.nav_active_target: Optional[Tuple[int, int]] = None

        # states: WAIT_MAP, IDLE, NAVIGATING, DWELLING
        self.state = 'WAIT_MAP'

        # dwelling bookkeeping
        self._dwell_start = 0.0
        self._dwell_target: Optional[Tuple[int, int]] = None

        # ---------------- RPC session state ----------------
        self.active = False
        self.session_id: Optional[str] = None
        self.client: str = "skills"
        self._last_status_pub = 0.0

        # ---------------- Subs / pubs / timers ----------------
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, qos)

        self.req_sub = self.create_subscription(String, "/coverage/req", self._on_req, 10)
        self.status_pub = self.create_publisher(String, "/coverage/status", 10)

        self.loop_timer = self.create_timer(0.25, self._loop)

        self.get_logger().info(
            f"coverage_wait_node ready. map_topic={self.map_topic}, global_frame={self.global_frame}, "
            f"base_frame={self.base_frame}, persist_path={self.persist_path}"
        )
        self._pub_status(state="idle", msg="ready")

    # ---------------- RPC ----------------
    def _on_req(self, msg: String):
        try:
            obj = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"/coverage/req bad JSON: {e}")
            return

        cmd = str(obj.get("cmd", "")).strip().lower()
        req_id = obj.get("id", None)
        client = obj.get("client", "skills")

        if cmd == "start":
            params = obj.get("params") or {}
            # replace any running session
            self._cancel_session(reason="replaced_by_new_start", publish=False)

            self.session_id = str(req_id) if req_id else f"coverage:{int(time.time()*1000)}"
            self.client = str(client)

            # apply per-request overrides (optional)
            self._apply_overrides(params)

            self.active = True
            self.visited_centers = 0
            self.visit_count_since_persist = 0

            # do NOT clear visited by default — resume behavior
            # if user wants fresh, they can set params.reset_visited = true (optional)
            if bool(params.get("reset_visited", False)):
                if self.map_meta is not None:
                    self.get_logger().warn("[coverage] reset_visited requested; clearing visited grid")
                    self._init_visited_grid(load_persisted=False)
                    self.candidates = []
                    self._last_candidates_build = 0.0

            self.state = "WAIT_MAP" if self.map_msg is None or self.map_meta is None or self.visited is None else "IDLE"
            self._pub_status(state="running", msg="started")
            self.get_logger().info(f"[coverage] session started id={self.session_id}")
            return

        if cmd == "cancel":
            # If id is present, only cancel matching session; else cancel unconditionally.
            if req_id and self.session_id and str(req_id) != self.session_id:
                return
            self._cancel_session(reason="canceled_by_request", publish=True)
            return

        self.get_logger().warn(f"/coverage/req unknown cmd={cmd!r}")

    def _apply_overrides(self, params: Dict[str, Any]):
        # numeric overrides
        if "spacing_m" in params:
            self.spacing_m = float(params["spacing_m"])
        if "visited_radius_m" in params:
            self.visited_radius_m = float(params["visited_radius_m"])
        if "dwell_sec" in params:
            self.dwell_sec = float(params["dwell_sec"])
        if "arrival_radius_m" in params:
            self.arrival_radius_m = float(params["arrival_radius_m"])

        if "treat_unknown_as_blocked" in params:
            self.treat_unknown_as_blocked = bool(params["treat_unknown_as_blocked"])

        if "border_margin_m" in params:
            self.border_margin_m = float(params["border_margin_m"])
            
        if "persist_path" in params:
            self.persist_path = str(params["persist_path"])
            # allow re-load for same meta if persist file changed
            self._persist_loaded_for_meta = False

        if "persist_every_n" in params:
            self.persist_every_n = int(params["persist_every_n"])

        if "max_candidates" in params:
            self.max_candidates = int(params["max_candidates"])

        if "rebuild_candidates_sec" in params:
            self.rebuild_candidates_sec = float(params["rebuild_candidates_sec"])

    def _cancel_session(self, reason: str, publish: bool = True):
        if not self.active and self.session_id is None:
            return

        self.get_logger().info(f"[coverage] cancel session reason={reason}")
        self.active = False

        # cancel Nav2 goal if any
        try:
            if self.nav_goal_handle is not None:
                self.nav_goal_handle.cancel_goal_async()
        except Exception:
            pass

        self.nav_goal_handle = None
        self.nav_result_future = None
        self.nav_active_target = None

        # persist what we have
        if self.map_meta is not None and self.visited is not None:
            self._persist_visited()

        # reset behavior state
        self.state = "IDLE" if (self.map_meta is not None and self.visited is not None) else "WAIT_MAP"

        if publish:
            self._pub_status(state="canceled", msg=reason)

        self.session_id = None
        self.client = "skills"

    def _pub_status(self, state: str, msg: str = "", extra: Dict[str, Any] | None = None):
        payload: Dict[str, Any] = {
            "id": self.session_id,
            "client": self.client,
            "state": state,  # running|done|canceled|error|idle
            "ts": time.time(),
            "msg": msg,
        }
        if extra:
            payload.update(extra)
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    # ---------------- Map handling ----------------
    def _on_map(self, msg: OccupancyGrid):
        meta = MapMeta(
            width=int(msg.info.width),
            height=int(msg.info.height),
            resolution=float(msg.info.resolution),
            origin_x=float(msg.info.origin.position.x),
            origin_y=float(msg.info.origin.position.y),
        )
        changed = (self.map_meta is None) or (
            self.map_meta.width != meta.width or
            self.map_meta.height != meta.height or
            abs(self.map_meta.resolution - meta.resolution) > 1e-9 or
            abs(self.map_meta.origin_x - meta.origin_x) > 1e-6 or
            abs(self.map_meta.origin_y - meta.origin_y) > 1e-6
        )

        self.map_msg = msg
        self.map_meta = meta

        if changed:
            self.get_logger().warn("[coverage] Map metadata changed (or first map). Initializing visited grid/candidates.")
            self._persist_loaded_for_meta = False
            self._init_visited_grid(load_persisted=True)
            self.candidates = []
            self._last_candidates_build = 0.0

        # if we were waiting on map, we can proceed
        if self.state == 'WAIT_MAP' and self.map_meta is not None and self.visited is not None:
            self.state = 'IDLE'

    def _init_visited_grid(self, load_persisted: bool = True):
        assert self.map_meta is not None
        n = self.map_meta.width * self.map_meta.height
        self.visited = [False] * n

        if load_persisted and (not self._persist_loaded_for_meta):
            self._load_persisted_visited()
            self._persist_loaded_for_meta = True

    def _idx(self, i: int, j: int) -> int:
        return j * self.map_meta.width + i

    def _in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.map_meta.width and 0 <= j < self.map_meta.height

    def _cell_to_world_center(self, i: int, j: int) -> Tuple[float, float]:
        x = self.map_meta.origin_x + (i + 0.5) * self.map_meta.resolution
        y = self.map_meta.origin_y + (j + 0.5) * self.map_meta.resolution
        return x, y

    def _cell_is_free(self, i: int, j: int) -> bool:
        if not self._in_bounds(i, j) or self.map_msg is None:
            return False
        v = self.map_msg.data[self._idx(i, j)]
        if v == -1:
            return not self.treat_unknown_as_blocked
        if v >= self.occ_occupied_min:
            return False
        return v <= self.occ_free_max

    # ---------------- Persistence ----------------
    def _load_persisted_visited(self):
        if not os.path.exists(self.persist_path):
            self.get_logger().info("[coverage] No persisted visited file found (starting fresh).")
            return
        try:
            with open(self.persist_path, 'r') as f:
                payload = json.load(f)
            pm = payload.get('map_meta', {})

            if (pm.get('width') != self.map_meta.width or
                pm.get('height') != self.map_meta.height or
                abs(float(pm.get('resolution', -1)) - self.map_meta.resolution) > 1e-9 or
                abs(float(pm.get('origin_x', 1e9)) - self.map_meta.origin_x) > 1e-6 or
                abs(float(pm.get('origin_y', 1e9)) - self.map_meta.origin_y) > 1e-6):
                self.get_logger().warn("[coverage] Persisted visited exists but map meta differs; ignoring.")
                return

            bits = payload.get('visited', [])
            if len(bits) != self.map_meta.width * self.map_meta.height:
                self.get_logger().warn("[coverage] Persisted visited length mismatch; ignoring.")
                return

            self.visited = [bool(b) for b in bits]
            self.get_logger().info(f"[coverage] Loaded persisted visited grid from {self.persist_path}")
        except Exception as e:
            self.get_logger().warn(f"[coverage] Failed to load persisted visited: {e}")

    def _persist_visited(self):
        if self.map_meta is None or self.visited is None:
            return
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            payload = {
                "map_meta": {
                    "width": self.map_meta.width,
                    "height": self.map_meta.height,
                    "resolution": self.map_meta.resolution,
                    "origin_x": self.map_meta.origin_x,
                    "origin_y": self.map_meta.origin_y,
                },
                "ts": time.time(),
                "visited": [1 if v else 0 for v in self.visited],
            }
            tmp = self.persist_path + ".tmp"
            with open(tmp, 'w') as f:
                json.dump(payload, f)
            os.replace(tmp, self.persist_path)
            self.visit_count_since_persist = 0
            self.get_logger().info(f"[coverage] Persisted visited -> {self.persist_path}")
        except Exception as e:
            self.get_logger().warn(f"[coverage] Failed to persist visited: {e}")

    # ---------------- Candidate generation ----------------
    def _build_candidates_if_needed(self):
        if self.map_msg is None or self.map_meta is None or self.visited is None:
            return

        now = time.time()
        if self.candidates and (now - self._last_candidates_build) < self.rebuild_candidates_sec:
            return

        stride = max(1, int(round(self.spacing_m / self.map_meta.resolution)))
        self.get_logger().info(
            f"[coverage] Building candidates stride={stride} cells (~{stride*self.map_meta.resolution:.2f}m)"
        )

        cand: List[Tuple[int, int]] = []
        w, h = self.map_meta.width, self.map_meta.height

        margin_cells = 0
        if self.border_margin_m > 0.0:
            margin_cells = max(1, int(round(self.border_margin_m / self.map_meta.resolution)))


        for j in range(0, h, stride):
            row = range(0, w, stride) if (j // stride) % 2 == 0 else range(w - 1 - ((w - 1) % stride), -1, -stride)
            for i in row:
                if len(cand) >= self.max_candidates:
                    break
                    
                # NEW: skip cells too close to borders
                if margin_cells > 0:
                    if (
                        i < margin_cells or
                        j < margin_cells or
                        i >= (w - margin_cells) or
                        j >= (h - margin_cells)
                    ):
                        continue
                    
                if self._cell_is_free(i, j):
                    cand.append((i, j))
            if len(cand) >= self.max_candidates:
                break

        self.candidates = cand
        self._last_candidates_build = now
        self.get_logger().info(f"[coverage] Candidates built: {len(self.candidates)}")

    # ---------------- Visited marking ----------------
    def _is_visited(self, i: int, j: int) -> bool:
        return bool(self.visited[self._idx(i, j)])

    def _mark_visited_radius(self, i: int, j: int):
        r_cells = max(0, int(round(self.visited_radius_m / self.map_meta.resolution)))
        for dj in range(-r_cells, r_cells + 1):
            for di in range(-r_cells, r_cells + 1):
                ii, jj = i + di, j + dj
                if not self._in_bounds(ii, jj):
                    continue
                if (di * di + dj * dj) <= (r_cells * r_cells):
                    self.visited[self._idx(ii, jj)] = True

    # ---------------- Robot pose ----------------
    def _get_robot_pose_map(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2)
            )
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            q = tf.transform.rotation
            yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
            return x, y, yaw
        except Exception:
            return None

    # ---------------- Target selection ----------------
    def _pick_closest_unvisited(self, robot_x: float, robot_y: float) -> Optional[Tuple[int, int]]:
        best = None
        best_d2 = None
        for (i, j) in self.candidates:
            if self._is_visited(i, j):
                continue
            wx, wy = self._cell_to_world_center(i, j)
            dx, dy = wx - robot_x, wy - robot_y
            d2 = dx * dx + dy * dy
            if best is None or d2 < best_d2:
                best = (i, j)
                best_d2 = d2
        return best

    # ---------------- Nav2 (fully async) ----------------
    def _send_nav_goal(self, target_cell: Tuple[int, int], yaw: Optional[float] = None) -> bool:
        if not self.nav_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warn("[coverage] Nav2 action server not available (navigate_to_pose).")
            return False

        i, j = target_cell
        x, y = self._cell_to_world_center(i, j)

        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0

        yaw = float(yaw) if yaw is not None else 0.0
        q = quaternion_from_euler(0.0, 0.0, yaw)
        pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        goal = NavigateToPose.Goal()
        goal.pose = pose

        self.get_logger().info(f"[coverage] Navigating to cell=({i},{j}) world=({x:.2f},{y:.2f})")
        self.state = 'NAVIGATING'
        self.nav_active_target = target_cell

        send_future = self.nav_client.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_response)
        return True

    def _on_goal_response(self, fut):
        if not self.active:
            return
        try:
            gh = fut.result()
        except Exception as e:
            self.get_logger().warn(f"[coverage] goal send error: {e}")
            self.state = 'IDLE'
            self.nav_goal_handle = None
            self.nav_active_target = None
            return

        if not gh or not gh.accepted:
            self.get_logger().warn("[coverage] Nav goal rejected.")
            self.state = 'IDLE'
            self.nav_goal_handle = None
            self.nav_active_target = None
            return

        self.nav_goal_handle = gh
        self.nav_result_future = gh.get_result_async()
        self.nav_result_future.add_done_callback(self._on_nav_result)

    def _on_nav_result(self, fut):
        if not self.active:
            return

        target = self.nav_active_target
        self.nav_active_target = None
        self.nav_goal_handle = None
        self.nav_result_future = None

        try:
            res = fut.result()
            status = getattr(res, "status", 0)
        except Exception as e:
            self.get_logger().warn(f"[coverage] nav result exception: {e}")
            self.state = 'IDLE'
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("[coverage] Arrived. Dwelling...")
            self._dwell_start = time.time()
            self._dwell_target = target
            self.state = 'DWELLING'
        else:
            self.get_logger().warn(f"[coverage] Navigation failed status={status}.")
            self.state = 'IDLE'

    # ---------------- Main loop ----------------
    def _loop(self):
        # Only run coverage when session is active
        if not self.active:
            # still allow building map/visited via /map callback
            return

        if self.map_msg is None or self.map_meta is None or self.visited is None:
            # wait for map
            self.state = 'WAIT_MAP'
            return

        # periodic status heartbeat
        now = time.time()
        if (now - self._last_status_pub) >= max(0.1, self.status_period_sec):
            self._pub_status(
                state="running",
                msg=f"state={self.state}",
                extra={"progress": {"candidates": len(self.candidates), "visited_centers": self.visited_centers}},
            )
            self._last_status_pub = now

        self._build_candidates_if_needed()

        # NAVIGATING handled by callbacks; nothing to do here
        if self.state == 'NAVIGATING':
            return

        # DWELLING: wait and then mark visited
        if self.state == 'DWELLING':
            if (time.time() - self._dwell_start) < self.dwell_sec:
                return

            if self._dwell_target is not None:
                i, j = self._dwell_target
                self._mark_visited_radius(i, j)
                self.visited_centers += 1
                self.visit_count_since_persist += 1
                self.get_logger().info(f"[coverage] Marked visited around cell=({i},{j}).")

                if self.visit_count_since_persist >= self.persist_every_n:
                    self._persist_visited()

            self._dwell_target = None
            self.state = 'IDLE'
            return

        # IDLE: pick next closest unvisited and go
        if self.state in ('WAIT_MAP', 'IDLE'):
            pose = self._get_robot_pose_map()
            if pose is None:
                self.get_logger().warn(2000, "[coverage] No TF map->base_link yet; waiting.")
                return
            rx, ry, ryaw = pose

            next_cell = self._pick_closest_unvisited(rx, ry)
            if next_cell is None:
                self.get_logger().info("[coverage] No remaining unvisited candidates. Done.")
                self._persist_visited()
                self._pub_status(state="done", msg="completed")
                # End session
                self.active = False
                self.session_id = None
                self.client = "skills"
                self.state = 'IDLE'
                return

            # already close -> dwell without nav
            wx, wy = self._cell_to_world_center(*next_cell)
            if math.hypot(wx - rx, wy - ry) <= self.arrival_radius_m:
                self.get_logger().info("[coverage] Already near target. Dwelling without navigation.")
                self._dwell_start = time.time()
                self._dwell_target = next_cell
                self.state = 'DWELLING'
                return

            # send nav goal async
            ok = self._send_nav_goal(next_cell, yaw=ryaw)
            if not ok:
                # if nav not available, just idle and try next tick
                self.state = 'IDLE'
            return

    # ---------------- Shutdown ----------------
    def shutdown_cleanup(self):
        # Persist on shutdown if we have data
        try:
            if self.map_meta is not None and self.visited is not None:
                self._persist_visited()
        except Exception:
            pass



def main():
    rclpy.init()
    node = CoverageWaitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

