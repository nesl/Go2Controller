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
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from bt_msgs.msg import BtReading  # NEW

from geometry_msgs.msg import Quaternion, TransformStamped
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


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
    "bt_msgs/BtReading": BtReading,                     # NEW
    "bt_msgs/msg/BtReading": BtReading,                 # NEW
}

def quat_to_yaw(q: Quaternion) -> float:
    # simple yaw extraction
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def load_yaml(p: str): return yaml.safe_load(Path(p).read_text())

@dataclass
class EdgeState:
    active: bool = False            # currently considered "on"
    last_true_ts: Optional[float] = None  # seconds

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
        self.declare_parameter('rules_init_path', '')

        # NEW: treat rules with this prefix as planner one-shot triggers
        self.declare_parameter('planner_trigger_prefix', 'trigger_')
        
        # NEW: trigger rule ids that must NEVER be one-shot
        # default includes trigger_speech_final
        self.declare_parameter(
            'always_on_trigger_ids_json',
            '["trigger_speech_final", "trigger_idle"]'
        )

        # NEW: paths to skills YAML (same idea as SkillsAgent)
        self.declare_parameter('skills_base_path', '')
        self.declare_parameter('skills_composite_path', '')

        self.declare_parameter('zone_split_x', 0.0)   # simple A/B split on x
        self.zone_split_x = float(self.get_parameter('zone_split_x').value)


        self.registry_path = self.get_parameter('registry_path').get_parameter_value().string_value
        self.rules_path    = self.get_parameter('rules_path').get_parameter_value().string_value
        self.rescan_period = float(self.get_parameter('rescan_period_s').value)
        self.enabled       = bool(self.get_parameter('enabled').value)
        self.rules_init_path = self.get_parameter('rules_init_path').get_parameter_value().string_value

        # NEW
        self.planner_trigger_prefix = (
            self.get_parameter('planner_trigger_prefix')
            .get_parameter_value()
            .string_value
        )

        # NEW: parse always-on trigger ids
        try:
            ids_raw = self.get_parameter('always_on_trigger_ids_json').get_parameter_value().string_value
            self.always_on_triggers = set(json.loads(ids_raw) or [])
        except Exception:
            self.always_on_triggers = {"trigger_speech_final", "trigger_idle"}

        # NEW: skills paths
        self.skills_base_path = self.get_parameter('skills_base_path').get_parameter_value().string_value
        self.skills_composite_path = self.get_parameter('skills_composite_path').get_parameter_value().string_value


        if not self.registry_path or not self.rules_path:
            self.get_logger().fatal("Set both registry_path and rules_path.")
            raise SystemExit(2)

        self.registry = load_yaml(self.registry_path)
        self.tasks_doc = self.registry.get("tasks", {})

        # publishers
        self.pub_basic = self.create_publisher(StringMsg, '/events/basic', 100)
        self.pub_comp  = self.create_publisher(StringMsg, '/events/composite', 100)

        self.rules_status_pub = self.create_publisher(StringMsg, '/rules/status', 10)

        # runtime
        self._subs = {}
        self.rules_all = []
        self.rules_enabled = []
        self._rule_state = {}     # rule_id -> bool (last satisfied)
        self._last_emit_ts = {}   # rule_id -> float (optional diag)
        self._edge_states = {}     # <-- ADD THIS
        self.bad_rules = set()     # <-- ADD THIS
        self._one_shot_fired = set()
        
        # NEW: maps for skill → rule deps etc.
        self._skill_rule_deps = {}      # composite_name -> set(rule_ids)
        self._primitive_rule_deps = {}  # primitive_name -> set(rule_ids)
        self._rules_by_id = {}          # rule_id -> rule dict (for mode lookup)
        
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

        # --- TF2 for map->base transforms (robot pose in map frame) ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # configurable frames if you want (or keep literals in helper)
        self.map_frame = "map"
        self.base_frame = "base_link"   # or "base_link" depending on your tree


        # timers
        #self.create_timer(self.rescan_period, self._resubscribe_if_needed)
        self.create_timer(0.1, self._tick)

        self._load_rules()
        self._load_skills_deps()          # ← NEW
        self._resubscribe_if_needed()

        # Always watch skill execution bus so we can reset edge rules per-skill
        self.sub_skill_status = self.create_subscription(
            StringMsg,
            "/skills/status",
            self._cb_skill_status,
            10,
        )
      
        # --- LLM/VLM call machinery ---
        # rules that own an llm_call block
        self.llm_call_rules = []
        # per-rule last call time (sec)
        self._last_llm_call_ts: Dict[str, float] = {}
        # lazily-created publishers for request topics (e.g. /vlm/req, /llm/speech_check_req)
        self._llm_req_pubs: Dict[str, Any] = {}


        self.get_logger().info("event_layer_node (expr) up")

    def _publish_llm_request(self, llm_cfg: dict, envelope: dict):
        """
        Publish the envelope as JSON to the appropriate request topic.

        Defaults:
          task == 'vlm_inference'   -> /vlm/req
          task == 'llm_speech_check'-> /llm/speech_check_req
        You can override with llm_call.request_topic.
        """
        if not envelope:
            return

        task = llm_cfg.get("task", "")
        topic = llm_cfg.get("request_topic")

        if not topic:
            if task == "vlm_inference":
                topic = "/vlm/req"
            elif task == "llm_speech_check":
                topic = "/llm/speech_check_req"
            else:
                # fallback generic channel if you add one later
                topic = "/llm/req"

        # lazily create publisher per topic
        pub = self._llm_req_pubs.get(topic)
        if pub is None:
            pub = self.create_publisher(StringMsg, topic, 10)
            self._llm_req_pubs[topic] = pub
            self.get_logger().info(f"EventLayer: created LLM/VLM req publisher on {topic}")

        try:
            pub.publish(StringMsg(data=json.dumps(envelope)))
        except Exception as e:
            self.get_logger().warn(f"EventLayer: failed to publish LLM/VLM request on {topic}: {e}")


    def _build_llm_request_envelope(self, rule: dict, llm_cfg: dict) -> Optional[dict]:
        """
        Construct a JSON-serializable request envelope for an llm_call rule.

        For now:
          - task 'vlm_inference' -> /vlm/req (no text field required)
          - task 'llm_speech_check' -> /llm/speech_check_req (needs 'text')
        """
        rid = str(rule.get("id") or "")
        if not rid:
            return None

        now_ms = int(self._now() * 1000)
        req_id = f"{rid}:{now_ms}"

        prompt_template = llm_cfg.get("prompt_template", "") or ""
        output_schema = llm_cfg.get("output_schema", "") or ""
        tag = llm_cfg.get("tag", "") or rid

        env = {
            "id": req_id,
            "prompt": prompt_template,
            "output_schema": output_schema,
            "tag": tag,
            # 'mode' is optional; VLM ignores it, but it's handy for future routing
            "mode": llm_cfg.get("mode", "generic"),
            # also include the originating rule id for debugging
            "rule_id": rid,
        }

        task = llm_cfg.get("task", "")

        # For llm_speech_check, we MUST provide 'text'
        if task == "llm_speech_check":
            # Allow override; default to 'speech_final_any'
            src_rule = llm_cfg.get("text_from_rule_id", "speech_final_any")
            last_evt = self._latest_payload_for_rule(src_rule)
            text = ""
            if last_evt and isinstance(last_evt, dict):
                # last_evt is the payload we gave to _publish_basic
                data = last_evt.get("data") or last_evt
                text = (data.get("text") or "").strip()
            if not text:
                # nothing to check, skip this call
                self.get_logger().info(
                    f"LLM call for rule '{rid}' skipped: no text from '{src_rule}'."
                )
                return None
            env["text"] = text

        # For VLM, just use prompt/metadata; no extra fields required by the server
        return env


    def _latest_payload_for_rule(self, rule_id: str) -> Optional[dict]:
        """
        Return the most recent payload for a given basic rule id, or None.
        Uses self.rule_hits[rule_id], where entries are (ts, payload).
        """
        dq = self.rule_hits.get(str(rule_id))
        if not dq:
            return None
        return dq[-1][1]  # (ts, payload) -> payload


    # ---------- utils ----------
    
    def _zone_from_xy(self, x: float | None, y: float | None) -> str:
        """
        Simple A/B zone split on x coordinate.
        If x is None, default to 'B' (or whatever you like).
        """
        if x is None:
            return "B"
        return "A" if x < self.zone_split_x else "B"


    def _lookup_robot_pose_map(self):
        """
        Return (x, y, yaw, ts) of the robot in the map frame, or None on failure.
        yaw is in radians.
        """
        try:
            # time=0 → latest available transform
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # optional: comment out if too chatty
            # self.get_logger().debug(f"TF lookup failed: {e}")
            return None

        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = quat_to_yaw(q)

        # stamp to seconds
        stamp = tf.header.stamp
        tf_ts = stamp.sec + stamp.nanosec * 1e-9

        return float(t.x), float(t.y), float(yaw), tf_ts


    # ---------- one-shot (planner trigger) helpers ----------

    def _is_one_shot_candidate(self, rule: dict) -> bool:
        """
        A rule is 'one-shot' if:
          - YAML says one_shot: true, OR
          - its id starts with planner_trigger_prefix.
        """
        rid = str(rule.get("id") or "")
        if not rid:
            return False

        # NEW: some triggers must never be one-shot
        if rid in getattr(self, "always_on_triggers", set()):
            return False

        if bool(rule.get("one_shot", False)):
            return True

        if self.planner_trigger_prefix and rid.startswith(self.planner_trigger_prefix):
            return True

        return False

    def _has_one_shot_fired(self, rid: str) -> bool:
        return rid in self._one_shot_fired

    def _mark_one_shot_fired(self, rid: str):
        """
        Remember that this one-shot rule has fired; all future evaluations
        (basic & composite) will ignore it.
        """
        self._one_shot_fired.add(str(rid))
        self.get_logger().info(f"EventLayer: one-shot rule '{rid}' fired; disabling further events.")



    def _reset_rules_for_ids(self, rule_ids: set[str] | list[str]):
        """
        Reset edge-related state (rule_hits, edge state, last emit) for the given rule ids,
        but only if the rule is edge-based.
        """
        if not rule_ids:
            return
        now = self._now()
        for rid in rule_ids:
            sid = str(rid)
            r = self._rules_by_id.get(sid)
            if not r:
                continue
            mode = r.get("mode", "edge")
            if mode != "edge":
                # no need to reset level rules
                continue

            # Clear temporal state so the next true evaluation will fire a fresh edge
            if sid in self.rule_hits:
                self.rule_hits.pop(sid, None)
            if sid in self._rule_state:
                self._rule_state[sid] = False
            if sid in self._edge_states:
                st = self._edge_states[sid]
                st.active = False
                st.last_true_ts = None
            if sid in self._last_emit_ts:
                self._last_emit_ts.pop(sid, None)

            # optional: debug log (comment out if too chatty)
            # self.get_logger().info(f"EventLayer: reset edge state for rule '{sid}' at {now:.3f}")

    
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


    # ---------- skills introspection (for edge resets) ----------

    def _read_yaml_if_exists_local(self, path: str) -> Optional[dict]:
        if not path:
            return None
        p = Path(path)
        if not p.is_file():
            return None
        try:
            return yaml.safe_load(p.read_text()) or {}
        except Exception as e:
            self.get_logger().warn(f"EventLayer: failed to read skills YAML '{path}': {e}")
            return None

    def _merged_skills_doc(self) -> dict:
        """
        Merge base + composite skills into a single doc:
          {version, defaults, skills: [...]}
        """
        base_doc = self._read_yaml_if_exists_local(self.skills_base_path) or {}
        comp_doc = self._read_yaml_if_exists_local(self.skills_composite_path) or {}

        merged = {
            "version": base_doc.get("version", 2),
            "defaults": base_doc.get("defaults", {"window_ms": 3000}),
            "skills": []
        }

        base_skills = base_doc.get("skills") or []
        if isinstance(base_skills, list):
            merged["skills"].extend(base_skills)

        comp_skills = comp_doc.get("skills") or []
        if isinstance(comp_skills, list):
            merged["skills"].extend(comp_skills)

        return merged

    def _collect_exists_from_cond(self, cond: Any, acc: set):
        """
        Recursively collect rule_ids from 'exists' conditions:
          - {exists: <rule_id>, within_ms: ...}
          - {any: [...]} / {all: [...]}
        """
        if not cond:
            return
        if isinstance(cond, dict):
            if "exists" in cond:
                rid = cond.get("exists")
                if rid:
                    acc.add(str(rid))
            if "any" in cond and isinstance(cond["any"], list):
                for c in cond["any"]:
                    self._collect_exists_from_cond(c, acc)
            if "all" in cond and isinstance(cond["all"], list):
                for c in cond["all"]:
                    self._collect_exists_from_cond(c, acc)
        elif isinstance(cond, list):
            for c in cond:
                self._collect_exists_from_cond(c, acc)

    def _load_skills_deps(self):
        """
        Read skills YAMLs, compute:
          - skill_name -> rule_ids it depends on
          - primitive_name -> rule_ids of steps that use that primitive
        using 'exists: rule_id' in when/until conditions.
        """
        self._skill_rule_deps = {}
        self._primitive_rule_deps = {}

        if not (self.skills_base_path or self.skills_composite_path):
            self.get_logger().info("EventLayer: no skills paths provided; skipping skill deps.")
            return

        try:
            merged = self._merged_skills_doc()
        except Exception as e:
            self.get_logger().warn(f"EventLayer: failed to merge skills docs: {e}")
            return

        skills = merged.get("skills") or []
        if not isinstance(skills, list):
            return

        for s in skills:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name", ""))
            kind = s.get("kind", "")
            if not name:
                continue

            # Collect rule ids from composite-level when/until
            rule_ids = set()
            self._collect_exists_from_cond(s.get("when"), rule_ids)
            self._collect_exists_from_cond(s.get("until"), rule_ids)

            steps = s.get("steps") or []
            if isinstance(steps, list):
                for step in steps:
                    self._collect_exists_from_cond(step.get("when"), rule_ids)
                    self._collect_exists_from_cond(step.get("until"), rule_ids)

                    # Map primitive -> rule_ids (for per-primitive reset)
                    ref = step.get("use")
                    if ref and isinstance(ref, str):
                        # step-specific rule ids
                        step_rule_ids = set()
                        self._collect_exists_from_cond(step.get("when"), step_rule_ids)
                        self._collect_exists_from_cond(step.get("until"), step_rule_ids)
                        if step_rule_ids:
                            d = self._primitive_rule_deps.setdefault(ref, set())
                            d.update(step_rule_ids)

            if rule_ids:
                self._skill_rule_deps[name] = rule_ids

        self.get_logger().info(
            f"EventLayer: loaded skills deps for {len(self._skill_rule_deps)} skills, "
            f"{len(self._primitive_rule_deps)} primitives."
        )


    # ---------- rules ----------
    def _load_rules(self):
        """
        Load base rules from rules_init_path (read-only) and dynamic rules
        from rules_path (writable). Both are merged into a single rules list.
        """
        base_doc = {}
        if self.rules_init_path:
            try:
                base_doc = load_yaml(self.rules_init_path) or {}
            except Exception as e:
                self.get_logger().error(f"Failed to load rules_init: {e}")
                base_doc = {}

        dyn_doc = {}
        try:
            dyn_doc = load_yaml(self.rules_path) or {}
        except Exception as e:
            self.get_logger().warn(f"Failed to load dynamic rules: {e}")
            dyn_doc = {}

        # defaults: prefer base, but allow dynamic to override if present
        defs_base = base_doc.get("defaults", {})
        defs_dyn  = dyn_doc.get("defaults", {})
        defs = dict(defs_base)
        defs.update(defs_dyn)

        self.default_window_ms = int(defs.get("window_ms", 3000))
        self.default_comp_ms   = int(defs.get("composite_window_ms", 2000))

        base_rules = base_doc.get("rules", []) or []
        dyn_rules  = dyn_doc.get("rules", []) or []

        self.rules_all = list(base_rules) + list(dyn_rules)
        self.rules_enabled = [r for r in self.rules_all if r.get("enabled", True)]

        # NEW: fast lookup by id
        self._rules_by_id = {str(r.get("id")): r for r in self.rules_all if "id" in r}
        self.bad_rules.clear()

        self._one_shot_fired = set()    

        # Build desired topics from basic rules (type not composite)
        self.desired_topics = {}
        for r in self.rules_enabled:
            if r.get("type") == "composite":
                continue
            task = r.get("task"); out_id = r.get("output")
            tdoc = self.tasks_doc.get(task, {})
            for o in tdoc.get("outputs", []):
                if o.get("id") == out_id:
                    ros = o.get("ros", {})
                    topic = ros.get("topic"); msg = ros.get("msg")
                    if topic and msg:
                        self.desired_topics[topic] = msg

        # --- LLM/VLM call rules (those that declare llm_call: {...}) ---
        self.llm_call_rules = [
            r for r in self.rules_enabled
            if isinstance(r.get("llm_call"), dict)
        ]
        # reset last-call timestamps when rules reload
        self._last_llm_call_ts = {}


        self.get_logger().info(f"Enabled rules: {[r['id'] for r in self.rules_enabled]}")
        self.get_logger().info(f"Desired topics: {list(self.desired_topics.keys())}")
        self._reconcile_node_services()
        

    def _quarantine_rule(self, rid: str, reason: str, exc: Exception | None, rtype: str):
        """
        Mark a rule as bad in this process and emit a status event so the
        orchestrator (or any watcher) can remove it from rules.yaml.
        rtype: "basic" | "composite"
        """
        rid = str(rid)
        self.bad_rules.add(rid)
        err_str = str(exc) if exc else ""
        self.get_logger().error(
            f"Quarantining {rtype} rule '{rid}' due to {reason}: {err_str}"
        )

        evt = {
            "kind": "rule_error",
            "rule_id": rid,
            "rule_type": rtype,
            "reason": reason,
            "error": err_str,
            "ts": self._now(),
        }
        try:
            self.rules_status_pub.publish(StringMsg(data=json.dumps(evt)))
        except Exception as e2:
            self.get_logger().warn(
                f"Failed to publish rule_error for '{rid}': {e2}"
            )



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
            self.get_logger().info(f"Rules reloaded")
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
        for topic, sub in self._subs.items():
            try:
                self.destroy_subscription(sub)
                self.get_logger().info(f"Dropped subscription on {topic}")
            except Exception as e:
                self.get_logger().warn(f"Error destroying subscription {topic}: {e}")
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


    def _cb_vlm_answer(self, msg: StringMsg):
        """
        Handle VLM JSON envelope from /vlm/answer and evaluate vlm_inference rules.

        Expected payload (from QwenVLMServer):
          {
            "id": <int or str>,
            "success": <bool>,
            "raw_text": <str>,
            "json_text": <str>,
            "model_id": <str>,
            "lat_ms": <int>,
            "tag": <str>,
            "ts": <float>   # when VLM node recorded it
          }
        """
        ts_event = self._now()
        try:
            env = json.loads(msg.data or "{}")
        except Exception:
            self.get_logger().warn(f"Bad JSON on /vlm/answer: {msg.data}")
            return

        # Choose a canonical 'text' field for rules; keep raw_text for compatibility
        raw = (env.get("raw_text") or env.get("text") or "").strip()
        json_text = env.get("json_text") or ""

        ctx = {
            # what most simple rules will look at
            "text": raw,
            # extra fields for richer rules
            "raw_text": raw,
            "json_text": json_text,
            "success": bool(env.get("success", False)),
            "model_id": env.get("model_id") or "",
            "lat_ms": float(env.get("lat_ms", 0.0)),
            "tag": env.get("tag") or "",
            "id": env.get("id"),
            # timestamps: both the EventLayer time and the VLM node's own ts if present
            "ts": ts_event,
            "vlm_ts": float(env.get("ts", ts_event)),
        }

        self._eval_for_rules("vlm_inference", "answer.text", ctx)


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
        if msgstr == "bt_msgs/BtReading":                   # NEW
            return self._cb_bt_reading                      # NEW
        if topic == "/vlm/answer":
            return self._cb_vlm_answer
        if topic == "/llm/speech_check":
            return self._cb_llm_speech_check
        #if topic == "/skills/status":              # NEW
        #    return self._cb_skill_status          # NEW
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
        ts_event = self._now()

        # Derive zone:
        # 1) if payload already has a robot_zone, use that
        # 2) otherwise, try to infer from TF
        zone = payload.get("robot_zone")
        if zone is None:
            try:
                pose_map = self._lookup_robot_pose_map()
                if pose_map is not None:
                    rx, ry, ryaw, rts = pose_map
                    zone = self._zone_from_xy(rx, ry)
                else:
                    zone = "unknown"
            except Exception:
                zone = "unknown"

        evt = {
            "ts": ts_event,
            "rule": rule_id,
            "data": payload,
            "zone": zone,          # ← new top-level field
        }

        self.pub_basic.publish(StringMsg(data=json.dumps(evt)))

    def _eval_for_rules(self, task: str, output: str, ctx: dict):
        if not self.enabled:
            return

        now = self._now()

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

            rid = str(r.get("id") or "")
            if not rid:
                continue

            # NEW: skip quarantined rules
            if rid in self.bad_rules:
                continue

            expr = r.get("expr", "")
            try:
                ok = bool(safe.eval(expr, ctx))
            except Exception as e:
                # NEW: quarantine on expression error
                self._quarantine_rule(rid, "expr_error_basic", e, "basic")
                continue
            
            # NEW: if this rule is one-shot and has already fired, never evaluate again
            if self._has_one_shot_fired(rid):
                continue
                
            prev = self._rule_state.get(rid, False)
            state = self._edge_states.setdefault(rid, EdgeState(active=prev))

            mode = r.get("mode", "edge")          # "edge" or "level"
            emit_off = bool(r.get("emit_off", False))

            # how long we must stay false before emitting OFF (ms)
            off_delay_ms = int(r.get("edge_off_ms", self.default_window_ms))

            fire = False
            active_flag = True   # True = ON event, False = OFF event
            new_state = prev

            # Track last time the expr was true
            if ok:
                state.last_true_ts = now

            if mode == "edge":
                # Rising edge: became true
                if ok and not prev:
                    fire = True
                    active_flag = True
                    new_state = True

                # Falling edge: became false, but only after persistence
                elif emit_off and (not ok and prev):
                    # Only allow OFF if we've been false long enough
                    # i.e., time since last_true_ts >= off_delay_ms
                    # If we never saw a true, treat as immediately off.
                    if state.last_true_ts is None or (now - state.last_true_ts) * 1000.0 >= off_delay_ms:
                        fire = True
                        active_flag = False
                        new_state = False
                    else:
                        # Too soon to declare OFF: keep state ON and skip emit
                        new_state = True
                        self._rule_state[rid] = new_state
                        state.active = new_state
                        continue

            elif mode == "level":
                # Level mode: fire whenever expr is true
                if ok:
                    fire = True
                    active_flag = True
                new_state = ok

            else:
                # unknown mode → default to simple rising edge
                if ok and not prev:
                    fire = True
                    active_flag = True
                    new_state = True

            # Commit debounced state
            self._rule_state[rid] = new_state
            state.active = new_state

            if fire:
                payload = dict(ctx)
                payload["rule_id"] = rid
                payload["active"] = active_flag   # ON or OFF

                # Only ON edges count as "hits" for exists()
                if active_flag:
                    self._emit_hit(rid, payload)

                self._publish_basic(rid, payload)
                self._last_emit_ts[rid] = now


                # NEW: if this is a one-shot rule and we just fired an ON event,
                # mark it so it never fires again.
                if active_flag and self._is_one_shot_candidate(r):
                    self._mark_one_shot_fired(rid)


            
    # --- detection.2d
    def _cb_detection(self, msg: Detection2DArray):
        ts = self._now()
        for d in msg.detections:
            if not d.results: 
                continue
            hyp = d.results[0].hypothesis
            cls = str(hyp.class_id)
            score = float(hyp.score)
            bbox = {
                "cx": float(d.bbox.center.position.x),
                "cy": float(d.bbox.center.position.y),
                "w":  float(d.bbox.size_x),
                "h":  float(d.bbox.size_y),
            }

            # NEW: extract 3D map coords if detector provided them
            map_x = map_y = map_z = None
            try:
                pose = d.results[0].pose.pose.position
                map_x, map_y, map_z = float(pose.x), float(pose.y), float(pose.z)
            except Exception:
                pass

            frame_id = getattr(d, "header", None).frame_id if hasattr(d, "header") and d.header else ""

            ctx = {
                "cls": cls,
                "score": score,
                "bbox": bbox,
                "map_x": map_x, "map_y": map_y, "map_z": map_z,  # may be None
                "frame_id": frame_id,                             # typically your target_frame (map)
                "ts": ts
            }

            # Existing rule path for generic detections:
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

    def _cb_llm_speech_check(self, msg: StringMsg):
        """
        Handle LLM speech check envelope from /llm/speech_check and
        evaluate rules for task 'llm_speech_check', output 'check.json'.

        Expected outer envelope (from llm_speech_check node):
          {
            "id": str,
            "success": bool,
            "raw_text": str,
            "json_text": str,
            "model_id": str,
            "lat_ms": float,
            "tag": str,
            "ts": float
          }

        Optionally, json_text is a STRICT JSON string with:
          {"kind":str,"intent":str,"ok":bool,"confidence":float,"text":str,"ts":float}
        which we will parse and surface as extra fields.
        """
        ts_event = self._now()
        try:
            env = json.loads(msg.data or "{}")
        except Exception:
            self.get_logger().warn(f"Bad JSON on /llm/speech_check: {msg.data}")
            return

        # Outer envelope fields
        raw_text  = (env.get("raw_text") or "").strip()
        json_text = env.get("json_text") or ""

        # Try to parse inner structured JSON (if present)
        inner = {}
        if json_text:
            try:
                inner = json.loads(json_text)
            except Exception:
                self.get_logger().warn(f"Bad inner json_text in /llm/speech_check: {json_text!r}")

        # Prefer the normalized 'text' from inner JSON if available
        inner_text = (inner.get("text") or "").strip() if isinstance(inner, dict) else ""
        canonical_text = inner_text or raw_text

        ctx = {
            # what most rules will look at
            "text": canonical_text,

            # outer envelope
            "raw_text": raw_text,
            "json_text": json_text,
            "success": bool(env.get("success", False)),
            "model_id": env.get("model_id") or "",
            "lat_ms": float(env.get("lat_ms", 0.0)),
            "tag": env.get("tag") or "",
            "id": env.get("id"),
            "ts": ts_event,
            "llm_ts": float(env.get("ts", ts_event)),

            # inner structured fields (if any)
            "kind": inner.get("kind", "") if isinstance(inner, dict) else "",
            "intent": inner.get("intent", "") if isinstance(inner, dict) else "",
            "ok": bool(inner.get("ok", False)) if isinstance(inner, dict) else False,
            "confidence": float(inner.get("confidence", 0.0)) if isinstance(inner, dict) else 0.0,
            "inner_ts": float(inner.get("ts", ts_event)) if isinstance(inner, dict) else ts_event,
        }

        self._eval_for_rules("llm_speech_check", "check.json", ctx)




    def _cb_bt_reading(self, msg: BtReading):
        """
        Build a lightweight context from BtReading and evaluate basic rules.
        Attach robot pose in the map frame (via TF2) if available.
        """
        ts = self._now()

        # Start with the raw BT fields
        ctx = {
            "rssi": int(msg.rssi),
            "device_id": (msg.device_id or "").strip(),
            "object_id": (msg.device_name or "").strip(),     # often your CNode###
            "phone_id": (msg.scanner_id or "Robot").strip(),
            "frame_id": (msg.frame_id or "").strip(),
            "ts": ts,
        }

        # Optional: simple strength bucket
        try:
            ctx["strength_bucket"] = (
                "strong" if msg.rssi >= -65
                else ("medium" if msg.rssi >= -85 else "weak")
            )
        except Exception:
            ctx["strength_bucket"] = "unknown"

        # --- NEW: attach robot pose in map frame via TF ---
        pose_map = self._lookup_robot_pose_map()
        if pose_map is not None:
            rx, ry, ryaw, rts = pose_map
            ctx.update({
                "robot_map_x": rx,
                "robot_map_y": ry,
                "robot_map_yaw": ryaw,    # radians
                "robot_map_ts": rts,
            })
            # pose age relative to BT reading
            ctx["robot_pose_age_ms"] = (ts - rts) * 1000.0

        # Evaluate rules for this task/output
        self._eval_for_rules("bt_proximity", "bt.reading", ctx)


    def _cb_skill_status(self, msg: StringMsg):
        ts = self._now()
        try:
            obj = json.loads(msg.data)
        except Exception:
            self.get_logger().warn(f"Bad JSON on /skills/status: {msg.data}")
            return

        kind = obj.get("kind", "")
        skill_name = obj.get("skill", "") or ""
        step_idx = int(obj.get("step_idx", 0))
        primitive_name = obj.get("primitive")  # present on step_started for primitives
        nested_composite = obj.get("composite")  # present on step_started for nested composites

        # --- NEW: reset edge-based rules based on skill events ---
        # Per composite skill reset when it starts
        if kind == "skill_started" and skill_name:
            deps = self._skill_rule_deps.get(skill_name, set())
            if deps:
                self._reset_rules_for_ids(deps)

        # Per primitive step reset when that primitive starts
        if kind == "step_started":
            # primitive dependency reset
            if primitive_name:
                deps_p = self._primitive_rule_deps.get(primitive_name, set())
                # also include composite-level deps for this skill
                deps_s = self._skill_rule_deps.get(skill_name, set())
                deps = set(deps_p) | set(deps_s)
                if deps:
                    self._reset_rules_for_ids(deps)

            # if the step launches a nested composite, you may also reset its deps
            if nested_composite:
                deps_c = self._skill_rule_deps.get(nested_composite, set())
                if deps_c:
                    self._reset_rules_for_ids(deps_c)

        # --- Existing path: expose skill_status as a rule context if you want ---
        ctx = {
            "kind": kind,
            "skill": skill_name,
            "step_idx": step_idx,
            "reason": obj.get("reason", ""),
            "activated": bool(obj.get("activated", False)),
            "done": bool(obj.get("done", False)),
            "ts": ts,
        }
        inner_ctx = obj.get("ctx") or {}
        if isinstance(inner_ctx, dict):
            ctx["inner_ctx"] = inner_ctx

        self._eval_for_rules("skill_status", "skill.status", ctx)



    # ---------- composites ----------
    def _tick(self):
        if not self.enabled:
            return
        now = self._now()

        # 1) Composite rules (existing code)
        local_funcs = {
            "exists": lambda rid, ms: self._exists(str(rid), int(ms)),
            "now": self._now,
        }
        safe = SafeEval(extra_funcs={**local_funcs, "re_search": _re_search})
        for r in self.rules_enabled:
            if r.get("type") != "composite":
                continue

            rid = str(r.get("id") or "")
            if not rid:
                continue

            # NEW: skip quarantined rules
            if rid in self.bad_rules:
                continue

            # NEW: skip composites that are one-shot and already fired
            if self._has_one_shot_fired(rid):
                continue

            expr = r.get("expr", "")
            try:
                ok = bool(safe.eval(expr, {}))
            except Exception as e:
                # NEW: quarantine this composite rule on error
                self._quarantine_rule(rid, "expr_error_composite", e, "composite")
                continue

            if ok:
                # existing zone / composite event publishing logic...
                zone = "unknown"
                try:
                    pose_map = self._lookup_robot_pose_map()
                    if pose_map is not None:
                        rx, ry, ryaw, rts = pose_map
                        zone = self._zone_from_xy(rx, ry)
                except Exception as e:
                    self.get_logger().debug(f"Composite zone lookup failed: {e}")

                evt = {
                    "type": "composite",
                    "rule": r["id"],
                    "expr": expr,
                    "ts": now,
                    "zone": zone,
                }
                self.pub_comp.publish(StringMsg(data=json.dumps(evt)))

                # NEW: mark one-shot composite triggers as fired
                if self._is_one_shot_candidate(r):
                    self._mark_one_shot_fired(rid)

        # 2) NEW: LLM/VLM call rules (llm_call: {...})
        for r in self.llm_call_rules:
            rid = str(r.get("id") or "")
            if not rid:
                continue

            if rid in self.bad_rules:
                continue

            # if rule is one-shot and already fired, do not call again
            if self._has_one_shot_fired(rid):
                continue

            llm_cfg = r.get("llm_call") or {}
            min_period_ms = int(llm_cfg.get("min_period_ms", 0))
            last_ts = self._last_llm_call_ts.get(rid, 0.0)
            if min_period_ms > 0 and (now - last_ts) * 1000.0 < min_period_ms:
                # still in cooldown
                continue

            expr = r.get("expr", "")
            try:
                ok = bool(safe.eval(expr, {}))
            except Exception as e:
                self._quarantine_rule(rid, "expr_error_llm_call", e, "composite")
                continue

            if not ok:
                continue

            # Build request envelope (may return None if missing text, etc.)
            env = self._build_llm_request_envelope(r, llm_cfg)
            if not env:
                continue

            self._publish_llm_request(llm_cfg, env)
            self._last_llm_call_ts[rid] = now

            # Optionally treat llm_call rules as one-shot if marked
            if self._is_one_shot_candidate(r):
                self._mark_one_shot_fired(rid)

        # 3) Edge rule timeouts → synthesize OFF when no hit for window
        for r in self.rules_enabled:
            if r.get("type") == "composite":
                continue
            if r.get("mode", "edge") != "edge":
                continue
            if not r.get("emit_off", False):
                continue

            rid = str(r["id"])

            # NEW: for one-shot rules, once they fired, don't bother creating OFF events
            if self._has_one_shot_fired(rid):
                continue

            state = self._edge_states.get(rid)
            if not state or not state.active:
                continue

            last_true = state.last_true_ts
            if last_true is None:
                continue

            off_delay_ms = int(r.get("edge_off_ms", self.default_window_ms))
            if (now - last_true) * 1000.0 < off_delay_ms:
                continue

            # Time-based OFF
            state.active = False
            self._rule_state[rid] = False
            self._last_emit_ts[rid] = now

            payload = {
                "rule_id": rid,
                "active": False,
                "ts": now,
            }
            self._publish_basic(rid, payload)

 
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

