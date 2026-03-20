#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import requests
import rclpy
from rclpy.node import Node
from std_msgs.msg import String as StringMsg

Property = Literal["X", "Y"]


@dataclass
class Pose2D:
    x: float
    y: float


@dataclass
class Step:
    plan_id: str
    step_idx: int
    box_id: int
    prop: Property
    kind: Literal["sense", "dispose"]


class PlanFollowerAgentNode(Node):
    """
    Spawn N instances with different agent_id.
    This node:
      - subscribes /central_plan
      - takes only its steps
      - executes travel + POST /sense or POST /dispose
      - publishes /agent_result
    """

    def __init__(self):
        super().__init__("plan_follower_agent")

        self.declare_parameter("agent_id", "robot")
        self.declare_parameter("server_base_url", "http://URL:8080")
        self.declare_parameter("request_timeout_sec", 120.0)

        self.declare_parameter("speed_mps", 1.0)
        self.declare_parameter("tick_period_sec", 0.25)

        self.declare_parameter("execute_one_action_per_tick", True)
        self.declare_parameter("preempt_on_new_plan", True)

        # optional cancel endpoints
        self.declare_parameter("enable_cancel", True)

        self._executing_step: Optional[Step] = None


        self.agent_id = str(self.get_parameter("agent_id").value)
        self.base_url = str(self.get_parameter("server_base_url").value).rstrip("/")
        self.timeout = float(self.get_parameter("request_timeout_sec").value)

        self.speed_mps = float(self.get_parameter("speed_mps").value)
        self.tick_period = float(self.get_parameter("tick_period_sec").value)

        self.execute_one_action_per_tick = bool(self.get_parameter("execute_one_action_per_tick").value)
        self.preempt_on_new_plan = bool(self.get_parameter("preempt_on_new_plan").value)
        self.enable_cancel = bool(self.get_parameter("enable_cancel").value)

        self.sub_plan = self.create_subscription(StringMsg, "/central_plan", self._on_plan, 10)
        self.pub_res = self.create_publisher(StringMsg, "/agent_result", 50)

        self.pose = Pose2D(0.0, 0.0)

        self._plan_lock = threading.Lock()
        self._active_plan_id: Optional[str] = None
        self._queue: List[Step] = []

        self._busy_lock = threading.Lock()
        self._busy = False

        self._op_lock = threading.Lock()
        self._current_op: Optional[Dict[str, Any]] = None

        self._cancel_lock = threading.Lock()
        self._cancel_evt: Optional[threading.Event] = None

        self._work_lock = threading.Lock()
        self._work_thread: Optional[threading.Thread] = None

        self._timer = self.create_timer(self.tick_period, self._tick)

        self.pub_pose = self.create_publisher(StringMsg, "/agent_pose", 50)

        # publish pose periodically so central has something even while idle
        self.declare_parameter("pose_pub_period_sec", 0.5)
        self.pose_pub_period = float(self.get_parameter("pose_pub_period_sec").value)
        self._pose_timer = self.create_timer(self.pose_pub_period, self._publish_pose)


        self.get_logger().info(f"PlanFollowerAgentNode up agent_id={self.agent_id} server={self.base_url}")

    def _publish_pose(self) -> None:
        try:
            # If you want sim time from server, do it here; otherwise omit.
            # Keep it lightweight: no server calls needed.
            payload = {
                "agent_id": self.agent_id,
                "x": float(self.pose.x),
                "y": float(self.pose.y),
                "t_wall": float(time.time()),
            }
            self.pub_pose.publish(StringMsg(data=json.dumps(payload)))
        except Exception:
            pass


    def _log_server_result(self, kind: str, box_id: int, prop: str, js: dict) -> None:
        if kind == "sense":
            self.get_logger().info(
                f"[SENSE] box={box_id} prop={prop} "
                f"status={js.get('status')} "
                f"detected={js.get('detected')} "
                f"prob={js.get('probability')}"
            )
        elif kind == "dispose":
            self.get_logger().info(
                f"[DISPOSE] box={box_id} prop={prop} "
                f"status={js.get('status')} "
                f"success={js.get('success')}"
            )


    # ---------------------------
    # HTTP
    # ---------------------------

    def _http(self, method: str, path: str, *, json_body: Optional[dict] = None) -> requests.Response:
        url = self.base_url + path
        if method == "GET":
            return requests.get(url, timeout=self.timeout)
        if method == "POST":
            return requests.post(url, json=json_body, timeout=self.timeout)
        raise ValueError(method)

    def _time(self) -> Dict[str, Any]:
        r = self._http("GET", "/time")
        r.raise_for_status()
        return r.json()

    def _boxes_state(self) -> List[Dict[str, Any]]:
        r = self._http("GET", "/boxes/state")
        r.raise_for_status()
        return list(r.json())

    def _sense(self, box_id: int, prop: Property) -> Dict[str, Any]:
        # ✅ server call
        r = self._http("POST", "/sense", json_body={"agent_id": self.agent_id, "box_id": int(box_id), "property": prop})
        r.raise_for_status()
        return r.json()

    def _dispose(self, box_id: int, prop: Property) -> Dict[str, Any]:
        # ✅ server call
        r = self._http("POST", "/dispose", json_body={"agent_id": self.agent_id, "box_id": int(box_id), "property": prop})
        r.raise_for_status()
        return r.json()

    def _cancel_current_server_op(self) -> bool:
        if not self.enable_cancel:
            return False
        with self._op_lock:
            op = dict(self._current_op) if self._current_op else None
        if not op:
            return False

        kind = str(op["kind"])
        box_id = int(op["box_id"])
        prop = str(op["property"])

        try:
            if kind == "sense":
                r = self._http("POST", "/sense/cancel", json_body={"agent_id": self.agent_id, "box_id": box_id, "property": prop})
            elif kind == "dispose":
                r = self._http("POST", "/dispose/cancel", json_body={"agent_id": self.agent_id, "box_id": box_id, "property": prop})
            else:
                return False

            return (r.status_code == 200)
        except Exception:
            return False

    def _action_key(self, step: Step) -> tuple:
        return (int(step.box_id), str(step.prop), str(step.kind))


    # ---------------------------
    # Plan receive
    # ---------------------------

    def _on_plan(self, msg: StringMsg) -> None:
        # Always log receipt early (even if we later ignore)
        self.get_logger().info(f"[PLAN] rx raw_len={len(msg.data)}")

        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"[PLAN] bad json: {e}")
            return

        plan_id = str(data.get("plan_id", ""))
        agents = data.get("agents", {})
        if not plan_id or not isinstance(agents, dict):
            return

        my_steps_raw = agents.get(self.agent_id, [])
        if not isinstance(my_steps_raw, list):
            return

        steps: List[Step] = []
        for s in my_steps_raw:
            try:
                step = Step(
                    plan_id=plan_id,
                    step_idx=int(s["step_idx"]),
                    box_id=int(s["box_id"]),
                    prop=str(s["property"]),
                    kind=str(s["kind"]),
                )
            except Exception:
                continue
            if step.prop not in ("X", "Y"):
                continue
            if step.kind not in ("sense", "dispose"):
                continue
            steps.append(step)

        incoming_first = steps[0] if steps else None
        incoming_first_key = self._action_key(incoming_first) if incoming_first else None

        with self._plan_lock:
            # If we're executing something:
            if self._executing_step is not None:
                cur_key = self._action_key(self._executing_step)

                # If incoming wants us to keep doing the same current action -> do nothing
                if incoming_first_key == cur_key:
                    self.get_logger().info(
                        f"[PLAN] recv plan={plan_id} steps={len(steps)} (ignore: same current action)"
                    )
                    return

                # Different first action -> preempt ONLY if allowed
                if self.preempt_on_new_plan:
                    self.get_logger().warn(
                        f"[PREEMPT] plan={plan_id} switch {cur_key} -> {incoming_first_key}"
                    )
                    self._cancel_current_server_op()
                    self._set_cancel_evt()
                    self._active_plan_id = plan_id
                    self._queue = steps
                    self.get_logger().info(f"[PLAN] recv plan={plan_id} steps={len(steps)} (preempted)")
                    return
                else:
                    # Non-preemptive: don't touch queue while executing
                    self.get_logger().info(
                        f"[PLAN] recv plan={plan_id} steps={len(steps)} (ignore: executing, preempt off)"
                    )
                    return

            # If idle: ignore if identical queue
            def queue_fingerprint(q: List[Step]) -> str:
                canonical = [{"box_id": s.box_id, "property": s.prop, "kind": s.kind} for s in q]
                return json.dumps(canonical, sort_keys=True, separators=(",", ":"))

            if queue_fingerprint(self._queue) == queue_fingerprint(steps):
                self.get_logger().info(
                    f"[PLAN] recv plan={plan_id} steps={len(steps)} (ignore: same queue)"
                )
                return

            # Accept new queue
            self._active_plan_id = plan_id
            self._queue = steps
            self.get_logger().info(f"[PLAN] recv plan={plan_id} steps={len(steps)} (accepted)")


    # ---------------------------
    # Concurrency helpers
    # ---------------------------

    def _set_busy(self, v: bool) -> None:
        with self._busy_lock:
            self._busy = bool(v)

    def _is_busy(self) -> bool:
        with self._busy_lock:
            return bool(self._busy)

    def _new_cancel_evt(self) -> threading.Event:
        with self._cancel_lock:
            self._cancel_evt = threading.Event()
            return self._cancel_evt

    def _set_cancel_evt(self) -> None:
        with self._cancel_lock:
            if self._cancel_evt is not None:
                self._cancel_evt.set()

    def _get_cancel_evt(self) -> Optional[threading.Event]:
        with self._cancel_lock:
            return self._cancel_evt

    def _set_current_op(self, kind: str, box_id: int, prop: str) -> None:
        with self._op_lock:
            self._current_op = {"kind": kind, "box_id": int(box_id), "property": str(prop)}

    def _clear_current_op(self) -> None:
        with self._op_lock:
            self._current_op = None

    def _pop_next(self) -> Optional[Step]:
        with self._plan_lock:
            if not self._queue:
                return None
            return self._queue.pop(0)

    # ---------------------------
    # Travel
    # ---------------------------

    def _travel_to_box(self, box: Dict[str, Any], cancel_evt: Optional[threading.Event]) -> bool:
        x, y = float(box["x"]), float(box["y"])
        dist = math.hypot(x - self.pose.x, y - self.pose.y)
        dt = dist / max(1e-6, self.speed_mps)

        self.get_logger().info(
            f"[TRAVEL] start box={box['box_id']} from=({self.pose.x:.2f},{self.pose.y:.2f}) "
            f"to=({x:.2f},{y:.2f}) t={dt:.2f}s"
        )

        end = time.time() + dt
        while time.time() < end:
            if cancel_evt is not None and cancel_evt.is_set():
                self.get_logger().warn("[TRAVEL] cancelled")
                return False
            time.sleep(0.05)

        self.pose = Pose2D(x, y)
        self.get_logger().info(f"[TRAVEL] done box={box['box_id']} now=({self.pose.x:.2f},{self.pose.y:.2f})")
        return True

    # ---------------------------
    # Result publish
    # ---------------------------

    def _publish_result(self, step: Step, success: bool, extra: Dict[str, Any]) -> None:
        try:
            t = self._time()
            now_sim = float(t["server_time"])
        except Exception:
            now_sim = 0.0

        payload = {
            "plan_id": step.plan_id,
            "agent_id": self.agent_id,
            "step_idx": step.step_idx,
            "box_id": step.box_id,
            "property": step.prop,
            "kind": step.kind,
            "success": bool(success),
            "finished_at": now_sim,
            "extra": extra or {},
        }
        self.pub_res.publish(StringMsg(data=json.dumps(payload)))

    # ---------------------------
    # Execute
    # ---------------------------

    def _execute_step(self, step: Step) -> None:
        self._set_busy(True)
        cancel_evt = self._new_cancel_evt()
        self._executing_step = step


        try:
            boxes = self._boxes_state()
            box = next((b for b in boxes if int(b["box_id"]) == int(step.box_id)), None)
            if box is None:
                self.get_logger().warn(f"[EXEC] unknown box={step.box_id}")
                self._publish_result(step, False, {"error": "unknown_box"})
                return

            # treat disposed-any like your example
            if bool(box.get("disposed_X", False)) or bool(box.get("disposed_Y", False)):
                self.get_logger().info(f"[EXEC] skip (already disposed-any) box={step.box_id}")
                self._publish_result(step, True, {"skipped": "already_disposed_any"})
                return

            ok = self._travel_to_box(box, cancel_evt)
            if not ok:
                self._publish_result(step, False, {"cancelled": True})
                return

            if step.kind == "sense":
                self._set_current_op("sense", step.box_id, step.prop)
                js = self._sense(step.box_id, step.prop)  # ✅ real server call

                # ALWAYS log sensing outcome
                self._log_server_result("sense", step.box_id, step.prop, js)
                self._publish_result(step, True, {"server": js})

            else:
                self._set_current_op("dispose", step.box_id, step.prop)
                js = self._dispose(step.box_id, step.prop)  # ✅ real server call
                # ALWAYS log disposal outcome
                self._log_server_result("dispose", step.box_id, step.prop, js)
                self.get_logger().info(f"[DISPOSE] box={step.box_id} prop={step.prop} status={js.get('status')}")
                self._publish_result(step, True, {"server": js})

        except Exception as e:
            self.get_logger().warn(f"[FAIL] execute failed: {e}")
            self._publish_result(step, False, {"exception": repr(e)})

        finally:
            self._clear_current_op()
            self._set_busy(False)
            self._executing_step = None


    # ---------------------------
    # Tick loop (threaded)
    # ---------------------------

    def _tick(self) -> None:
        with self._work_lock:
            if self._work_thread is not None and self._work_thread.is_alive():
                return
            self._work_thread = threading.Thread(target=self._worker_main, daemon=True)
            self._work_thread.start()

    def _worker_main(self) -> None:
        try:
            if self._is_busy():
                return

            step = self._pop_next()
            if step is None:
                return

            self._execute_step(step)

            if not self.execute_one_action_per_tick:
                while True:
                    if self._is_busy():
                        break
                    step2 = self._pop_next()
                    if step2 is None:
                        break
                    self._execute_step(step2)

        finally:
            with self._work_lock:
                self._work_thread = None


def main():
    rclpy.init()
    node = PlanFollowerAgentNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

