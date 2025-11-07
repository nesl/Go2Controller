#!/usr/bin/env python3
import json, time, yaml, re, ast
from pathlib import Path
from collections import deque, defaultdict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import String as StringMsg
from vision_msgs.msg import Detection2DArray
from std_srvs.srv import Trigger, SetBool

from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from std_srvs.srv import SetBool

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import math

# --- add this helper near the top ---
def _norm_ros_type(s: str) -> str:
    if not s: return s
    return s.replace('::msg::', '/').replace('/msg/', '/')

# expand so both forms map to the class
MSG_CLASS = {
    "std_msgs/String": StringMsg,
    "std_msgs/msg/String": StringMsg,
    "vision_msgs/Detection2DArray": Detection2DArray,
    "vision_msgs/msg/Detection2DArray": Detection2DArray,
    "nav_msgs/Odometry": Odometry,
    "nav_msgs/msg/Odometry": Odometry,
}

def quat_to_yaw(q: Quaternion) -> float:
    # simple yaw extraction
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def load_yaml(p: str): return yaml.safe_load(Path(p).read_text())

# ---------- safe expression evaluator ----------


class SafeEval:
    def __init__(self, extra_funcs=None, builtins=None):
        base_funcs = {
            "len": len, "any": any, "all": all,
            "min": min, "max": max, "abs": abs, "round": round,
        }
        if extra_funcs:
            base_funcs.update(extra_funcs)
        self.funcs = base_funcs
        self.builtins = builtins if builtins is not None else __builtins__

    def eval(self, expr: str, names: dict):
        if not expr:
            return False
        env = {"__builtins__": self.builtins}
        env.update(self.funcs)   # helpers
        env.update(names)        # context: text, doa, persons, ...
        # IMPORTANT: use the same dict for globals and locals
        return eval(expr, env, env)


# helpers exposed to expressions
def _re_search(pattern: str, text: str):
    if text is None: return False
    return bool(re.search(pattern, text, flags=re.IGNORECASE))

class EventLayerNode(Node):
    """
    Rules-driven (expression) event layer.
      - Loads static registry + dynamic rules.
      - Subscribes only to topics required by ENABLED basic rules.
      - For each message, builds a context and evaluates rule.expr (bool).
      - Keeps per-rule recent 'hits' for composite expressions via exists(rule_id, ms).
    """

    def __init__(self):
        super().__init__('event_layer_node')

        # ----- params -----
        self.declare_parameter('registry_path', '')
        self.declare_parameter('rules_path', '')
        self.declare_parameter('rescan_period_s', 2.0)
        self.declare_parameter('enabled', True)

        self.registry_path = self.get_parameter('registry_path').get_parameter_value().string_value
        self.rules_path    = self.get_parameter('rules_path').get_parameter_value().string_value
        self.rescan_period = float(self.get_parameter('rescan_period_s').value)
        self.enabled       = bool(self.get_parameter('enabled').value)

        if not self.registry_path or not self.rules_path:
            self.get_logger().fatal("Set both registry_path and rules_path.")
            raise SystemExit(2)

        self.registry = load_yaml(self.registry_path)
        self.tasks_doc = self.registry.get("tasks", {})

        # publishers
        self.pub_basic = self.create_publisher(StringMsg, '/events/basic', 100)
        self.pub_comp  = self.create_publisher(StringMsg, '/events/composite', 100)

        # runtime
        self._subs = {}
        self.rules_all = []
        self.rules_enabled = []
        self._rule_state = {}     # rule_id -> bool (last satisfied)
        self._last_emit_ts = {}   # rule_id -> float (optional diag)
        
        self.default_window_ms = 3000
        self.default_comp_ms = 2000

        # per-rule recent hits (for exists())
        self.rule_hits = defaultdict(lambda: deque())

        # buffers (optional diagnostics)
        self.buf_text = deque(maxlen=1000)

        # safe eval with extra helpers that we bind per-call:
        self.safe = SafeEval(extra_funcs={"re_search": _re_search})

        # services
        self.create_service(Trigger, '/event_layer/reload_rules', self._srv_reload_rules)
        self.create_service(SetBool, '/event_layer/enable', self._srv_enable)

        # timers
        #self.create_timer(self.rescan_period, self._resubscribe_if_needed)
        self.create_timer(0.1, self._tick)

        self._load_rules()
        self._resubscribe_if_needed()
        self.get_logger().info("event_layer_node (expr) up")

    # ---------- utils ----------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _exists(self, rule_id: str, ms: int) -> bool:
        dq = self.rule_hits.get(rule_id)
        if not dq: return False
        thr = self._now() - ms * 1e-3
        # prune and check
        while dq and dq[0][0] < thr:
            dq.popleft()
        return len(dq) > 0

    def _call_set_bool(self, service_name: str, value: bool, timeout=1.5):
        try:
            cli = self.create_client(SetBool, service_name)
            if not cli.wait_for_service(timeout_sec=timeout):
                self.get_logger().warn(f"{service_name} not available")
                return
            req = SetBool.Request()
            req.data = bool(value)
            fut = cli.call_async(req)
            # don't block spin; attach a done-callback
            fut.add_done_callback(lambda f: None)
        except Exception as e:
            self.get_logger().warn(f"SetBool {service_name} error: {e}")

    def _call_set_param(self, service_name: str, param_name: str, value: str, timeout=1.5):
        try:
            cli = self.create_client(SetParameters, service_name)
            if not cli.wait_for_service(timeout_sec=timeout):
                self.get_logger().warn(f"{service_name} not available")
                return
            p = Parameter()
            p.name = str(param_name)
            pv = ParameterValue()
            pv.type = ParameterType.PARAMETER_STRING
            pv.string_value = str(value)
            p.value = pv
            req = SetParameters.Request(parameters=[p])
            fut = cli.call_async(req)
            fut.add_done_callback(lambda f: None)
        except Exception as e:
            self.get_logger().warn(f"SetParameters {service_name} error: {e}")


    # ---------- rules ----------
    def _load_rules(self):
        doc = load_yaml(self.rules_path)
        defs = doc.get("defaults", {})
        self.default_window_ms = int(defs.get("window_ms", 3000))
        self.default_comp_ms   = int(defs.get("composite_window_ms", 2000))
        self.rules_all = doc.get("rules", [])
        self.rules_enabled = [r for r in self.rules_all if r.get("enabled", True)]

        # Build desired topics from basic rules (type not composite)
        self.desired_topics = {}
        for r in self.rules_enabled:
            if r.get("type") == "composite":  # composites don't bind to topics
                continue
            task = r.get("task"); out_id = r.get("output")
            tdoc = self.tasks_doc.get(task, {})
            for o in tdoc.get("outputs", []):
                if o.get("id") == out_id:
                    ros = o.get("ros", {})
                    topic = ros.get("topic"); msg = ros.get("msg")
                    if topic and msg:
                        self.desired_topics[topic] = msg

        self.get_logger().info(f"Enabled rules: {[r['id'] for r in self.rules_enabled]}")
        self.get_logger().info(f"Desired topics: {list(self.desired_topics.keys())}")
        self._reconcile_node_services()

    def _collect_required_nodes(self):
        """
        From enabled BASIC rules, figure:
          - which tasks/outputs are needed (already used for subscriptions),
          - which nodes must be enabled,
          - desired model_id per task (majority-wins if conflicts).
        Returns:
          need_nodes: set of node names
          desired_models: dict task -> desired model_id (or None)
        """
        desired_models = {}
        task_model_votes = {}
        need_nodes = set()

        # Walk enabled basic rules
        for r in self.rules_enabled:
            if r.get("type") == "composite":
                continue
            task = r.get("task"); out_id = r.get("output")
            tdoc = self.tasks_doc.get(task, {})
            node = tdoc.get("node")
            if node:
                need_nodes.add(node)
            # model votes
            mid = r.get("model_id")
            if mid:
                task_model_votes.setdefault(task, {}).setdefault(mid, 0)
                task_model_votes[task][mid] += 1

        # majority-wins per task
        for task, votes in task_model_votes.items():
            best = max(votes.items(), key=lambda kv: (kv[1], -len(desired_models)))  # count desc
            desired_models[task] = best[0]

        return need_nodes, desired_models

    def _find_model_cfg(self, task: str, model_id: str):
        tdoc = self.tasks_doc.get(task, {})
        for m in tdoc.get("models", []):
            if m.get("id") == model_id:
                return m
        return None

    def _typed_value(self, v):
        # Keep original types if already typed; otherwise try to coerce
        return v

    def _push_params_for_model(self, task: str, model_id: str):
        tdoc = self.tasks_doc.get(task, {})
        node = tdoc.get("node")
        if not node:
            return

        svc_set = tdoc.get("services", {}).get("set_model", {})
        psvc    = svc_set.get("param_service")
        pname   = svc_set.get("param_name")  # e.g., "yolo_weights" or "yolo_pose_weights"
        stype   = svc_set.get("type")
        if not (psvc and stype == "rcl_interfaces/SetParameters"):
            return

        m = self._find_model_cfg(task, model_id)
        if not m:
            self.get_logger().warn(f"No model cfg for {task}:{model_id}")
            return

        # Primary mapping: if param_name points to weights, set weights filename
        params_to_set = []
        model_params = (m.get("parameters") or {})  # from registry models.parameters
        # if user used a "weights" field, apply it when the target param is weights
        if pname and "weights" in model_params:
            params_to_set.append((pname, self._typed_value(model_params["weights"])))

        # Optionally push other parameters declared under this model (thresholds, device, etc.)
        # Only push ones that the target node actually declares.
        # Example: detection_threshold, device, etc.
        for k, v in model_params.items():
            if k == "weights":
                continue
            # If you want, you can gate which task maps to which node param names.
            # Here, we assume model parameters share the same names as the node params.
            params_to_set.append((k, self._typed_value(v.get("default", v))))

        # Send them (with your improved _call_set_param that infers types)
        for name, val in params_to_set:
            self._call_set_param(psvc, name, val)


    def _reconcile_node_services(self):
        """
        Enable/disable nodes based on current need, and push desired model_id via SetParameters.
        """
        need_nodes, desired_models = self._collect_required_nodes()

        # Build reverse mapping: node -> {services,...}
        nodes_cfg = {}
        for task, tdoc in self.tasks_doc.items():
            node = tdoc.get("node")
            if not node: continue
            svc = tdoc.get("services", {})
            nodes_cfg.setdefault(node, {"tasks": set(), "services": svc})
            nodes_cfg[node]["tasks"].add(task)

        # Enable/disable
        for node, info in nodes_cfg.items():
            svc_enable = info["services"].get("enable", {})
            name = svc_enable.get("name")
            typ  = svc_enable.get("type")
            if not name or typ != "std_srvs/SetBool":
                continue
            want = (node in need_nodes)
            self._call_set_bool(name, want)

        for task, model in desired_models.items():
            self._push_params_for_model(task, model)


    def _srv_reload_rules(self, req, resp):
        try:
            self._drop_all_subs()
            self._load_rules()
            self._resubscribe_if_needed()
            self._reconcile_node_services()
            resp.success = True
            resp.message = "Rules reloaded"
        except Exception as e:
            resp.success = False
            resp.message = f"Reload failed: {e}"
        return resp

    def _srv_enable(self, req, resp):
        self.enabled = bool(req.data)
        resp.success = True
        resp.message = f"Event layer {'ENABLED' if self.enabled else 'DISABLED'}"
        self.get_logger().info(resp.message)
        return resp

    # ---------- subscriptions ----------
    def _drop_all_subs(self):
        self._subs.clear()

    def _resubscribe_if_needed(self):
        live = dict(self.get_topic_names_and_types())

        # debug: show what ROS sees (already printing this, but with normalized types is clearer)
        dbg = {t: [_norm_ros_type(tp) for tp in types] for t, types in live.items()}
        #self.get_logger().info(f"Live topics/types: {dbg}")
        self.get_logger().info(f"Desired topics: {self.desired_topics}")

        for topic, msgstr in self.desired_topics.items():
            if topic in self._subs:
                self.get_logger().info(f"Already subscribed to {topic} [{msgstr}]")
                continue

            want = _norm_ros_type(msgstr)
            have = [_norm_ros_type(tp) for tp in live.get(topic, [])]

            # OK to subscribe before a publisher exists; only warn if a publisher exists with a different type
            type_ok = (not have) or (want in have)

            cls = MSG_CLASS.get(msgstr) or MSG_CLASS.get(want)
            if not cls:
                self.get_logger().warn(f"No Python class for {msgstr} (norm={want}) on {topic}")
                continue

            if not type_ok:
                self.get_logger().warn(f"Type mismatch for {topic}: want {want}, live {have} — subscribing anyway")

            qos = QoSProfile(
                depth=50,
                reliability=QoSReliabilityPolicy.RELIABLE,  # match default pub QoS
                history=QoSHistoryPolicy.KEEP_LAST,
            )
            cb = self._make_cb(topic, msgstr)
            self._subs[topic] = self.create_subscription(cls, topic, cb, qos)
            self.get_logger().info(f"Subscribed {topic} [{want}]")

        # prune subs we no longer desire
        for topic in list(self._subs.keys()):
            if topic not in self.desired_topics:
                self.destroy_subscription(self._subs.pop(topic))
                self.get_logger().info(f"Dropped {topic} (no longer desired)")


    def _make_cb(self, topic: str, msgstr: str):
        if msgstr == "vision_msgs/Detection2DArray":
            return self._cb_detection
        if msgstr == "nav_msgs/Odometry":
            return self._cb_odom
        if topic == "/yolo_pose_json":
            return self._cb_pose_json
        if topic in ("/audio/stt_partial_json", "/audio/stt_doa_json"):
            return self._cb_text_partial
        if topic in ("/audio/stt_text",):
            return self._cb_text_final
        # default String
        return self._cb_text_partial

    # ---------- per-output handlers: build ctx → eval expr ----------
    def _emit_hit(self, rule_id: str, data: dict):
        ts = self._now()
        self.rule_hits[rule_id].append((ts, data))
        # prune (use default window as generic retention)
        thr = ts - self.default_window_ms * 1e-3
        dq = self.rule_hits[rule_id]
        while dq and dq[0][0] < thr:
            dq.popleft()

    def _publish_basic(self, rule_id: str, payload: dict):
        self.pub_basic.publish(StringMsg(data=json.dumps({
            "ts": self._now(),
            "rule": rule_id,
            "data": payload
        })))

    def _eval_for_rules(self, task: str, output: str, ctx: dict):
        """Evaluate all enabled basic rules that target (task, output)."""
        if not self.enabled:
            return
        # helpers exposed to expr
        local_funcs = {
            "exists": lambda rid, ms: self._exists(str(rid), int(ms)),
            "now": self._now,
        }
        safe = SafeEval(extra_funcs={**local_funcs, "re_search": _re_search})
        for r in self.rules_enabled:
            if r.get("type") == "composite":
                continue
            if r.get("task") != task or r.get("output") != output:
                continue
            expr = r.get("expr", "")
            try:
                ok = bool(safe.eval(expr, ctx))
            except Exception as e:
                self.get_logger().warn(f"Rule {r.get('id')} expr error: {e}, context {ctx}, task {task}, output {output}")
                continue
                
            prev = self._rule_state.get(r["id"], False)
            if ok and not prev:
                payload = dict(ctx)  # include evaluated context (small)
                payload["rule_id"] = r["id"]
                self._emit_hit(r["id"], payload)
                self._publish_basic(r["id"], payload)
                self._last_emit_ts[r["id"]] = self._now()
            self._rule_state[r["id"]] = ok
            
    # --- detection.2d
    def _cb_detection(self, msg: Detection2DArray):
        ts = self._now()
        for d in msg.detections:
            if not d.results: continue
            hyp = d.results[0].hypothesis
            cls = str(hyp.class_id)
            score = float(hyp.score)
            bbox = {
                "cx": float(d.bbox.center.position.x),
                "cy": float(d.bbox.center.position.y),
                "w":  float(d.bbox.size_x),
                "h":  float(d.bbox.size_y),
            }
            ctx = {"cls": cls, "score": score, "bbox": bbox, "ts": ts}
            self._eval_for_rules("object_detection", "detection.2d", ctx)

    def _cb_odom(self, msg: Odometry):
        ts = self._now()
        pos = msg.pose.pose.position
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        ctx = {"x": pos.x, "y": pos.y, "yaw": yaw, "ts": ts}

        # Compute deltas if we have a previous pose
        if hasattr(self, "_last_odom"):
            dx = pos.x - self._last_odom["x"]
            dy = pos.y - self._last_odom["y"]
            dyaw = yaw - self._last_odom["yaw"]

            # Normalize rotation to [-pi, pi]
            while dyaw > math.pi: dyaw -= 2*math.pi
            while dyaw < -math.pi: dyaw += 2*math.pi

            ctx["dxy"] = math.sqrt(dx*dx + dy*dy)
            ctx["dyaw_deg"] = abs(math.degrees(dyaw))

            # Evaluate distance/turn rules
            self._eval_for_rules("odometry_tracking", "odom", ctx)

        self._last_odom = ctx


    # --- pose.json
    def _cb_pose_json(self, msg: StringMsg):
        ts = self._now()
        try:
            obj = json.loads(msg.data)
        except Exception:
            return
        persons = obj.get("persons", []) or obj.get("people", [])
        # avg confidence if present
        num, s = 0, 0.0
        for p in persons:
            for kp in p.get("keypoints", []):
                if len(kp) >= 3 and isinstance(kp[2], (int, float)):
                    s += float(kp[2]); num += 1
        kconf_avg = (s / num) if num else 0.0
        ctx = {"persons": int(len(persons)), "kpts": persons, "kconf_avg": kconf_avg, "ts": ts}
        self._eval_for_rules("pose_detection", "pose.json", ctx)

    # --- text.partial
    def _cb_text_partial(self, msg: StringMsg):
        ts = self._now()
        text, doa = "", {}
        try:
            obj = json.loads(msg.data)
            text = (obj.get("text") or "").strip()
            doa = obj.get("doa", {}) or {}
        except Exception:
            text = (msg.data or "").strip()
            self.get_logger().warn(f"Partial text error: {msg.data}")
        ctx = {"text": text, "doa": doa, "ts": ts}
        self._eval_for_rules("audio_gating", "text.partial", ctx)

    # --- text.final
    def _cb_text_final(self, msg: StringMsg):
        ts = self._now()
        text = ""
        try:
            obj = json.loads(msg.data)
            text = (obj.get("text") or "").strip()
        except Exception:
            text = (msg.data or "").strip()
        ctx = {"text": text, "ts": ts}
        self._eval_for_rules("audio_asr", "text.final", ctx)

    # ---------- composites ----------
    def _tick(self):
        if not self.enabled: return
        now = self._now()
        # Evaluate composite rules each tick
        local_funcs = {
            "exists": lambda rid, ms: self._exists(str(rid), int(ms)),
            "now": self._now,
        }
        safe = SafeEval(extra_funcs={**local_funcs, "re_search": _re_search})
        for r in self.rules_enabled:
            if r.get("type") != "composite":
                continue
            expr = r.get("expr", "")
            try:
                ok = bool(safe.eval(expr, {}))
            except Exception as e:
                self.get_logger().warn(f"Composite {r.get('id')} expr error: {e}")
                continue
            if ok:
                evt = {"type": "composite", "rule": r["id"], "expr": expr, "ts": now}
                self.pub_comp.publish(StringMsg(data=json.dumps(evt)))
                # light hysteresis: clear the oldest hit of each referenced rule id in expr
                # (optional — left simple to avoid regex parsing)
                
                
def main():
    rclpy.init()
    node = EventLayerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

