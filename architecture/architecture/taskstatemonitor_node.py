#!/usr/bin/env python3
import os, json, math, sqlite3, threading, time, re, hashlib
from typing import Optional, Tuple, Dict, Set, List
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from std_msgs.msg import String as StringMsg
from std_srvs.srv import Trigger

from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException
import requests

from openai import OpenAI
from jsonschema import validate, ValidationError

import yaml  # NEW
from rclpy.parameter import Parameter          # NEW
from rcl_interfaces.msg import SetParametersResult  # NEW
from groq import Groq

from .optimizer_client import AgentState, BoxInfo, plan_assignments_gurobi

# LLM reply must be STRICT JSON like: {"sql":"SELECT ...", "params": {...}, "purpose":"..."}
LLM_SQL_SCHEMA = {
    "type": "object",
    "required": ["sql"],
    "properties": {
        "sql": {
            "type": "string",
        },
        "params": {
            "type": "object",
            # OpenAI's json_schema mode requires object schemas to have `properties`.
            # We don't constrain param names, so we leave it empty and allow additional props.
            "properties": {},
            "additionalProperties": True,
        },
        "purpose": {
            "type": "string",
        },
    },
    "additionalProperties": False,
}


EVENT_SUMMARY_SCHEMA = {
    "type": "object",
    "required": ["summary"],
    "properties": {
        "summary": {"type": "string"},
    },
    "additionalProperties": False,
}


# ------------------------------ Broker Node ------------------------------

class BrokerNode(Node):
    """
    State-owning broker with LLM-driven SQL:
      • Creates/owns SQLite DB
      • Subscribes to /events/basic and ingests bt_proximity/bt.reading events
      • Owns context: trigger, event trace, mini world snapshot, human profiles
      • Builds SchemaCard + ContextCapsule
      • Calls an LLM to synthesize a single read-only SQL, validates and executes it
      • Proactive: on trigger, publishes /broker/facts
      • Reactive: consumes /planner/needs and publishes /broker/facts_delta
      • Keeps a working-set (ws) per planning session to avoid duplicate info
    """

    # ------------------------------ Init ------------------------------
    def __init__(self):
        super().__init__('broker_node')

        # ------------ Params ------------
        self.declare_parameter('db_path', os.path.expanduser('~/.broker_world.sqlite'))
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('zone_split_x', 0.0)               # simple A/B: x < split → 'A' else 'B'
        self.declare_parameter('subscribe_topic', '/events/basic') # event-layer basic bus
        self.declare_parameter('bt_rule_id', 'bt_rssi_seen')       # rule id to ingest
        self.declare_parameter('human3d_rule_id', 'human_detected_3d')

        # Event → trigger mapping (basic/composite rule id → semantic trigger)
        self.declare_parameter('trigger_map_json', json.dumps({
            "trigger_speech_final": "human_command",
            "trigger_idle": "idle"
        }))

        self.declare_parameter('planner_trigger_prefix', 'trigger_')

        # Contamination fetch policy (broker owns it)
        self.declare_parameter('contam_enable_server_calls', True)
        self.declare_parameter('contam_server_url', 'http://172.17.40.64:8080/check')
        self.declare_parameter('contam_request_timeout_sec', 0.6)
        self.declare_parameter('contam_min_refresh_sec', 120.0)    # throttle per (agent_id,node_id)

        # ---------- Optimizer / planner integration ----------
        self.declare_parameter("optimizer_enabled", True)
        self.declare_parameter("optimizer_base_url", "http://172.17.40.64:8080")
        self.declare_parameter("optimizer_horizon_sec", 60.0)

        # Time budgets per agent for this planning horizon (seconds)
        self.declare_parameter("optimizer_time_robot", 60.0)
        self.declare_parameter("optimizer_time_human_a", 60.0)
        self.declare_parameter("optimizer_time_human_b", 60.0)

        # Nominal walking speeds (m/s) used to turn distances into travel times
        self.declare_parameter("optimizer_speed_robot_mps", 0.5)
        self.declare_parameter("optimizer_speed_human_a_mps", 1.0)
        self.declare_parameter("optimizer_speed_human_b_mps", 1.0)


        # When to consider a new best/current as “meaningful change” for refresh
        self.declare_parameter('contam_best_delta_db', 5)          # recheck if best improved by ≥5 dB
        self.declare_parameter('contam_stale_sec', 900.0)          # or if label older than 15 min

        # LLM + SQL budgets
        self.declare_parameter('sql_max_rows', 64)
        self.declare_parameter('sql_max_bytes', 20000)
        self.declare_parameter('sql_timeout_ms', 120)
        self.declare_parameter('iteration_limit', 2)
        self.declare_parameter('pull_limit', 2)

        # Allowed SQL objects (read-only)
        self.declare_parameter('allowed_objects_json', json.dumps([
            "bt_nodes","nodes_state","bt_measurements",
            "agent_node_labels",
            "vw_bt_nodes_summary","vw_agent_node_labels",
            "vw_backlog_counts","vw_object_sheet", "box_env_state","vw_box_env"
        ]))

        # Optional: mock LLM for offline dev (pass JSON {"sql": "...", "params": {...}, "purpose": "..."} in param)
        self.declare_parameter('llm_mock_json', '')
        
        self.declare_parameter("model", "gpt-5-nano")

        self.model = self.get_parameter("model").get_parameter_value().string_value

        self.declare_parameter("llm_perf_topic", "/llm/broker_perf")
        self.llm_perf_topic = (
            self.get_parameter("llm_perf_topic")
            .get_parameter_value()
            .string_value
        )
        
        self.optimizer_enabled = bool(self.get_parameter("optimizer_enabled").value)
        self.optimizer_base_url = (
            self.get_parameter("optimizer_base_url").get_parameter_value().string_value
        )
        self.optimizer_horizon_sec = float(
            self.get_parameter("optimizer_horizon_sec").value
        )

        self.optimizer_time_robot = float(
            self.get_parameter("optimizer_time_robot").value
        )
        self.optimizer_time_human_a = float(
            self.get_parameter("optimizer_time_human_a").value
        )
        self.optimizer_time_human_b = float(
            self.get_parameter("optimizer_time_human_b").value
        )

        self.optimizer_speed_robot = float(
            self.get_parameter("optimizer_speed_robot_mps").value
        )
        self.optimizer_speed_human_a = float(
            self.get_parameter("optimizer_speed_human_a_mps").value
        )
        self.optimizer_speed_human_b = float(
            self.get_parameter("optimizer_speed_human_b_mps").value
        )

        self._agent_det_agents, self._agent_det_default = self._load_agent_detection_params()

        # Last plan and “fingerprint” of box server state
        self._last_plan = None
        self._last_boxes_fp = None
        self._optimizer_running = False

        # Publish plan as JSON so planner / reactive node can consume it
        self.pub_opt_plan = self.create_publisher(
            StringMsg, "/optimizer/plan", 10
        )

        
        # NEW: enable/disable use of LLM (everything else still works)
        self.declare_parameter("llm_enabled", False)
        self.llm_enabled = (
            self.get_parameter("llm_enabled")
            .get_parameter_value()
            .bool_value
        )
        
        # NEW: task registry path so we can subscribe to perf topics
        self.declare_parameter(
            "task_registry_path",
            ""
        )
        
        self.declare_parameter("human_agent_id", "human_a")  # or whatever label you use
        self.human_agent_id = (
            self.get_parameter("human_agent_id")
            .get_parameter_value()
            .string_value
        )
        
        # --- Event-trace summarizer parameters ---
        self.declare_parameter("event_summary_enabled", True)
        self.declare_parameter("event_summary_model", "gpt-4o-mini")  # fast, cheap, small context
        self.declare_parameter("event_summary_batch_size", 8)         # run summary after N events

        self.event_summary_enabled = (
            self.get_parameter("event_summary_enabled")
            .get_parameter_value()
            .bool_value
        )

        self.event_summary_model = (
            self.get_parameter("event_summary_model")
            .get_parameter_value()
            .string_value
        )

        self.event_summary_batch_size = int(
            self.get_parameter("event_summary_batch_size").value
        )

        # Async running event summary state
        self._event_summary_text: Optional[str] = None      # last full running summary
        self._event_summary_ts: Optional[float] = None      # ts of last event included
        self._unsummarized_events: List[dict] = []          # events since last summary
        self._events_since_summary: int = 0                 # counter since last summary
        self._event_summary_running: bool = False           # background worker in flight?
        self._event_summary_lock = threading.Lock()

        
        self.task_registry_path = (
            self.get_parameter("task_registry_path")
            .get_parameter_value()
            .string_value
        )

        # simple EMA of broker LLM latency (ms)
        self._llm_lat_ema_ms: Optional[float] = None
        self._llm_lat_alpha: float = 0.3

        self.pub_llm_perf = self.create_publisher(
            StringMsg, self.llm_perf_topic, 10
        )


        # Parameters → members
        self.db_path       = self.get_parameter('db_path').get_parameter_value().string_value
        self.target_frame  = self.get_parameter('target_frame').get_parameter_value().string_value
        self.zone_split_x  = float(self.get_parameter('zone_split_x').value)
        self.bus_topic     = self.get_parameter('subscribe_topic').get_parameter_value().string_value
        self.bt_rule_id    = self.get_parameter('bt_rule_id').get_parameter_value().string_value
        self.human3d_rule_id = self.get_parameter('human3d_rule_id').get_parameter_value().string_value

        self.trigger_map   = json.loads(self.get_parameter('trigger_map_json').get_parameter_value().string_value)

        self.planner_trigger_prefix = (
            self.get_parameter('planner_trigger_prefix')
            .get_parameter_value()
            .string_value
        )

        self.enable_server = bool(self.get_parameter('contam_enable_server_calls').value)
        self.server_url    = self.get_parameter('contam_server_url').get_parameter_value().string_value
        self.req_timeout   = float(self.get_parameter('contam_request_timeout_sec').value)
        self.min_refresh   = float(self.get_parameter('contam_min_refresh_sec').value)
        self.best_delta_db = int(self.get_parameter('contam_best_delta_db').value)
        self.stale_sec     = float(self.get_parameter('contam_stale_sec').value)

        self.sql_max_rows  = int(self.get_parameter('sql_max_rows').value)
        self.sql_max_bytes = int(self.get_parameter('sql_max_bytes').value)
        self.sql_timeout_ms= int(self.get_parameter('sql_timeout_ms').value)
        self.iteration_limit = int(self.get_parameter('iteration_limit').value)
        self.pull_limit    = int(self.get_parameter('pull_limit').value)
        self.allowed       = set(json.loads(self.get_parameter('allowed_objects_json').get_parameter_value().string_value))

        mock_s = self.get_parameter('llm_mock_json').get_parameter_value().string_value.strip()
        self._llm_mock = json.loads(mock_s) if mock_s else None

        # ------------ TF ------------
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ------------ DB ------------
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)  # autocommit
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

        # ------------ Runtime caches / state ------------
        # contamination
        self._contam_cache: Dict[Tuple[str, str], Dict] = {}
        self._pending_refresh: Set[Tuple[str, str]] = set()
        self._lock = threading.Lock()

        # context capsule (owned by broker)
        self._profiles = {"H1": None, "H2": None}  # if you publish HDT, wire subs below
        self._event_trace = deque(maxlen=40)       # compact recent events
        self._current_trigger = None               # {"type": "...", "hints": {...}}
        self._ws = {}                               # ws_id -> {"hashes": set(), "iters": int}


        self._last_published_summary_fp = None


        # NEW: robot objective & timing for task state
        self._robot_objective: Optional[str] = None
        self._start_time = time.time()

        # NEW: /task_state publisher
        self.pub_task_state = self.create_publisher(StringMsg, "/task_state", 10)

        # NEW: periodic TaskState tick (e.g., 2 Hz)
        self.create_timer(0.5, self._tick_task_state)

        # NEW: last known robot zone (from event-layer)
        self._last_robot_zone: Optional[str] = None


        # NEW: runtime perf database (EMA per task/model)
        # key: (task_id, model_id or "default")
        # val: {"lat_ms_ema": float, "n": int, "last_ts": float}
        self._perf_ema = {}
        self._perf_lock = threading.Lock()

        # Subscribe to all perf topics from task_registry
        self._perf_subscriptions = []
        self._load_task_registry_and_subscribe_perf()

        # ------------ ROS I/O ------------
        # Events
        self.sub_basic = self.create_subscription(StringMsg, self.bus_topic, self._on_basic_event, 1000)
        self.sub_comp  = self.create_subscription(StringMsg, "/events/composite", self._on_comp_event, 500)

        # Allow changing llm_model via /broker_node/set_parameters
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # Planner needs (reactive loop)
        self.sub_needs = self.create_subscription(StringMsg, "/planner/needs", self._on_planner_needs, 20)

        # Optional DT profiles (if available)
        # self.create_subscription(StringMsg, "/digital_twin/profile/H1", self._on_profile_h1, 10)
        # self.create_subscription(StringMsg, "/digital_twin/profile/H2", self._on_profile_h2, 10)

        # Publications for facts
        self.pub_facts = self.create_publisher(StringMsg, "/broker/facts", 10)
        self.pub_delta = self.create_publisher(StringMsg, "/broker/facts_delta", 10)
        self.pub_sql_debug = self.create_publisher(StringMsg, "/broker/sql_plan_debug", 10)

        self.pub_capsule = self.create_publisher(StringMsg, "/broker/context_capsule", 10)


        # Services
        self.srv_dump_db = self.create_service(Trigger, '/broker/dump_db_path', self._srv_dump_db)
        self.srv_nodes_summary = self.create_service(Trigger, '/broker/query_nodes_summary', self._srv_query_nodes_summary)
        self.srv_agent_labels  = self.create_service(Trigger, '/broker/query_agent_labels', self._srv_query_agent_labels)

        # LLM-driven runs (no context args; broker owns context)
        self.srv_run_initial = self.create_service(Trigger, '/broker/run_initial', self._srv_run_initial)
        self.srv_run_more    = self.create_service(Trigger, '/broker/run_more',    self._srv_run_more)

        # Background contamination worker
        self.create_timer(0.25, self._process_pending_refresh)
        
        self._save_registry_srv = self.create_service(
            Trigger,
            "/broker/save_task_registry_with_perf",
            self._on_save_registry_with_perf,
        )

        if self.optimizer_enabled:
            # Small polling period; can tune (e.g. 0.5–2.0 s)
            self.create_timer(1.0, self._optimizer_tick)


        self.get_logger().info(
            f"broker_node up | db={self.db_path} bus={self.bus_topic} rule={self.bt_rule_id} "
            f"target_frame={self.target_frame} zone_split_x={self.zone_split_x} server={self.server_url} "
            f"enable_server={self.enable_server}"
        )


    def _load_agent_detection_params(self):
        """
        Call the box server /agents/params endpoint once at startup
        to get per-agent detection probabilities.

        Returns:
            (agents_cfg, default_cfg) where:
              agents_cfg: dict[agent_id] -> {"X": {"present", "absent"}, "Y": {...}}
              default_cfg: same shape, used as fallback.
        """
        base = self.optimizer_base_url.rstrip("/")
        url = base + "/agents/params"

        # Reasonable default if server is down or older version without this route
        default_cfg = {
            "X": {"present": 0.8, "absent": 0.2},
            "Y": {"present": 0.8, "absent": 0.2},
        }

        try:
            r = requests.get(url, timeout=self.req_timeout)
            if r.status_code != 200:
                self.get_logger().warn(
                    f"[optimizer] /agents/params returned {r.status_code}; "
                    f"using default detection params"
                )
                return {}, default_cfg

            data = r.json()
            agents_cfg = data.get("agents", {})
            default_raw = data.get("default") or default_cfg

            # Minimal sanity check
            if "X" not in default_raw or "Y" not in default_raw:
                self.get_logger().warn(
                    "[optimizer] /agents/params default missing X/Y; "
                    "falling back to hardcoded defaults"
                )
                default_raw = default_cfg

            self.get_logger().info(
                f"[optimizer] loaded agent detection params for agents={list(agents_cfg.keys())}"
            )
            return agents_cfg, default_raw

        except Exception as e:
            self.get_logger().warn(
                f"[optimizer] failed to load /agents/params: {e}; "
                f"using default detection params"
            )
            return {}, default_cfg


    # ------------------------------ Perf topic discovery ------------------------------
    
    def _on_save_registry_with_perf(self, req, resp):
        out_path = None  # or e.g. self.task_registry_path.replace(".yaml","_runtime.yaml")
        self.write_task_registry_with_perf(self.task_registry_path, out_path=out_path)
        resp.success = True
        resp.message = "task_registry updated with runtime perf EMA"
        return resp
    
    def _load_task_registry_and_subscribe_perf(self):
        """
        Read task_registry.yaml, find all outputs with kind: perf,
        and subscribe to their topics as std_msgs/String.

        Expected structure (per your registry):
          tasks:
            task_id:
              outputs:
                - id: perf.json
                  kind: perf
                  ros:
                    topic: "/yolo_perf"
                    msg: "std_msgs/String"
        """
        path = self.task_registry_path
        if not path or not os.path.isfile(path):
            self.get_logger().warn(f"[broker] task_registry_path not found: {path}")
            return

        try:
            with open(path, "r") as f:
                doc = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().warn(f"[broker] failed to read task registry '{path}': {e}")
            return

        tasks = doc.get("tasks") or {}
        seen_topics = set()

        for task_id, task in tasks.items():
            outputs = task.get("outputs") or []
            for out in outputs:
                # Perf outputs are marked with kind: perf in your registry
                if str(out.get("kind", "")).lower() != "perf":
                    continue
                ros_cfg = out.get("ros") or {}
                topic = ros_cfg.get("topic")
                if not topic or topic in seen_topics:
                    continue
                seen_topics.add(topic)

                # Subscribe as std_msgs/String; we parse JSON ourselves
                self.get_logger().info(
                    f"[broker] subscribing to perf topic '{topic}' for task '{task_id}'"
                )
                sub = self.create_subscription(
                    StringMsg,
                    topic,
                    # capture task_id & topic in closure
                    lambda msg, t_id=task_id, t_topic=topic: self._on_perf_msg(t_id, t_topic, msg),
                    50,
                )
                self._perf_subscriptions.append(sub)

    # ------------------------------ Perf ingestion ------------------------------
    
    def _perf_summary(self) -> dict:
        """
        Return a nested summary:
        {
          "task_id": {
            "model_id": {
              "lat_ms_ema": float,
              "fps_ema": float | None,
              "samples": int,
              "last_ts": float
            },
            ...
          },
          ...
        }
        """
        out: Dict[str, Dict[str, dict]] = {}
        with self._perf_lock:
            for (task_id, model_id), ent in self._perf_ema.items():
                task_entry = out.setdefault(task_id, {})
                task_entry[model_id] = {
                    "lat_ms_ema": ent.get("lat_ms_ema"),
                    "fps_ema": ent.get("fps_ema"),
                    "samples": ent.get("n", 0),
                    "last_ts": ent.get("last_ts"),
                }
        return out

    def merge_perf_into_task_registry(self, registry: dict) -> dict:
        """
        Mutate a loaded task_registry dict to include current perf EMAs.

        For each (task_id, model_id) in the perf summary, if there is a matching
        task + model in registry["tasks"], we augment:

          tasks[task_id].models[k].metrics.latency_ms:
            ema_runtime_ms: <lat_ms_ema>
            runtime_samples: <samples>
            runtime_last_ts: <last_ts>

          tasks[task_id].models[k].metrics.throughput_fps:
            runtime_fps_ema: <fps_ema>  (if available)
        """
        summary = self._perf_summary()
        tasks_cfg = registry.get("tasks")
        if not isinstance(tasks_cfg, dict):
            return registry

        for task_id, models_perf in summary.items():
            task_entry = tasks_cfg.get(task_id)
            if not isinstance(task_entry, dict):
                continue

            models_list = task_entry.get("models")
            if not isinstance(models_list, list):
                continue

            for model_cfg in models_list:
                if not isinstance(model_cfg, dict):
                    continue

                # We use the 'id' field from YAML as the key to match perf.model_id.
                model_key = str(model_cfg.get("id") or model_cfg.get("version") or "")
                if not model_key:
                    continue

                perf_ent = models_perf.get(model_key)
                if not perf_ent:
                    continue

                lat_ms_ema = perf_ent.get("lat_ms_ema")
                fps_ema = perf_ent.get("fps_ema")
                samples = perf_ent.get("samples", 0)
                last_ts = perf_ent.get("last_ts")

                metrics = model_cfg.setdefault("metrics", {})
                lat_cfg = metrics.setdefault("latency_ms", {})

                if isinstance(lat_ms_ema, (int, float)):
                    lat_cfg["ema_runtime_ms"] = round(float(lat_ms_ema), 2)
                lat_cfg["runtime_samples"] = int(samples)
                if isinstance(last_ts, (int, float)):
                    lat_cfg["runtime_last_ts"] = float(last_ts)

                if isinstance(fps_ema, (int, float)):
                    thr_cfg = metrics.setdefault("throughput_fps", {})
                    thr_cfg["runtime_fps_ema"] = round(float(fps_ema), 2)

        return registry

    def write_task_registry_with_perf(self,
                                      in_path: str,
                                      out_path: Optional[str] = None) -> None:
        """
        Load task_registry YAML, merge in runtime perf EMAs, and write it back.

        If out_path is None, overwrite in_path; otherwise write to out_path.
        """
        try:
            with open(in_path, "r") as f:
                registry = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().error(f"[broker] failed to load task_registry from {in_path}: {e}")
            return

        registry = self.merge_perf_into_task_registry(registry)

        target_path = out_path or in_path
        tmp_path = target_path + ".tmp"

        try:
            with open(tmp_path, "w") as f:
                yaml.safe_dump(registry, f, sort_keys=False)
            os.replace(tmp_path, target_path)
            self.get_logger().info(f"[broker] wrote updated task registry with perf EMA to {target_path}")
        except Exception as e:
            self.get_logger().error(f"[broker] failed to write updated task_registry to {target_path}: {e}")


    
    def _on_perf_msg(self, task_id: str, topic: str, msg: StringMsg):
        """
        Generic perf handler.

        We try to extract:
          - model_id (if present, e.g., payload["model"])
          - a latency value in ms (lat_ms or latency_ms{...})
          - optional fps_ema (if present)
        and maintain an EMA per (task_id, model_id).
        """
        try:
            payload = json.loads(msg.data) if msg.data else {}
        except Exception as e:
            self.get_logger().warn(f"[broker] perf JSON parse error from {topic}: {e}")
            return

        ts = time.time()
        model_id = None
        lat_ms = None
        fps_ema = None

        if isinstance(payload, dict):
            # LLM / TTS / VLM perf:
            # {"node":"broker","model":str,"lat_ms":num,"ok":bool,"phase":str,...}
            if "model" in payload and isinstance(payload.get("lat_ms"), (int, float)):
                model_id = str(payload.get("model"))
                lat_ms = float(payload.get("lat_ms"))

            # Vision / audio perf: {"latency_ms": {...}, "fps_ema": ...}
            if lat_ms is None and isinstance(payload.get("latency_ms"), dict):
                lm = payload["latency_ms"]
                for key in ("total", "det", "pose", "utter_infer_mean", "window_infer_mean"):
                    v = lm.get(key)
                    if isinstance(v, (int, float)):
                        lat_ms = float(v)
                        break

            # Fallback: direct numeric latency_ms
            if lat_ms is None and isinstance(payload.get("latency_ms"), (int, float)):
                lat_ms = float(payload["latency_ms"])

            # Optional throughput info
            v_fps = payload.get("fps_ema")
            if isinstance(v_fps, (int, float)):
                fps_ema = float(v_fps)

        if lat_ms is None:
            # Nothing we can aggregate
            return

        if not model_id:
            model_id = "default"

        key = (task_id, model_id)
        alpha = 0.3  # EMA smoothing factor

        with self._perf_lock:
            ent = self._perf_ema.get(key)
            if ent is None:
                ent = {
                    "lat_ms_ema": lat_ms,
                    "fps_ema": fps_ema,
                    "n": 1,
                    "last_ts": ts,
                }
            else:
                # Update EMA for latency
                ent["lat_ms_ema"] = (1.0 - alpha) * ent["lat_ms_ema"] + alpha * lat_ms
                # Update fps EMA if we have it
                if fps_ema is not None:
                    old_fps = ent.get("fps_ema", fps_ema)
                    ent["fps_ema"] = (1.0 - alpha) * old_fps + alpha * fps_ema
                ent["n"] = ent.get("n", 0) + 1
                ent["last_ts"] = ts

            self._perf_ema[key] = ent


    # ------------------------------ Dynamic param handling ------------------------------
    def _on_set_parameters(self, params):
        """
        React to dynamic parameter updates, in particular llm_model and llm_enabled.
        ...
        """
        for p in params:
            if p.name == "model" and p.type_ == Parameter.Type.STRING:
                self.model = p.value
                self.get_logger().info(f"[broker] model changed to: {self.model}")

            # NEW: enable / disable broker LLM
            if p.name == "llm_enabled" and p.type_ == Parameter.Type.BOOL:
                self.llm_enabled = bool(p.value)
                self.get_logger().info(f"[broker] llm_enabled changed to: {self.llm_enabled}")

            if p.name == "event_summary_enabled" and p.type_ == Parameter.Type.BOOL:
                self.event_summary_enabled = bool(p.value)
                self.get_logger().info(f"[broker] event_summary_enabled = {self.event_summary_enabled}")

            if p.name == "event_summary_model" and p.type_ == Parameter.Type.STRING:
                self.event_summary_model = p.value
                self.get_logger().info(f"[broker] event_summary_model = {self.event_summary_model}")

            if p.name == "event_summary_batch_size" and p.type_ in (
                Parameter.Type.INTEGER,
                Parameter.Type.DOUBLE,
            ):
                self.event_summary_batch_size = int(p.value)
                self.get_logger().info(
                    f"[broker] event_summary_batch_size = {self.event_summary_batch_size}"
                )

        return SetParametersResult(successful=True, reason="ok")




    def _publish_llm_perf(self, lat_ms: int, ok: bool, phase: str):
        """
        Publish LLM latency as a small JSON perf record and keep a local EMA.
        phase: e.g. 'proactive_sql' or 'reactive_sql'
        """
        try:
            lat_ms = int(lat_ms)
        except Exception:
            lat_ms = -1

        # Update EMA (if latency valid)
        if lat_ms >= 0:
            if self._llm_lat_ema_ms is None:
                self._llm_lat_ema_ms = float(lat_ms)
            else:
                a = self._llm_lat_alpha
                self._llm_lat_ema_ms = (1.0 - a) * self._llm_lat_ema_ms + a * float(lat_ms)

        payload = {
            "node": "broker",
            "model": self.model,
            "lat_ms": lat_ms,
            "ok": bool(ok),
            "phase": phase,
            "ts": time.time(),
        }
        try:
            self.pub_llm_perf.publish(StringMsg(data=json.dumps(payload)))
        except Exception as e:
            self.get_logger().warn(f"broker: failed to publish llm perf: {e}")


    def _publish_context_capsule(self, summary_only: bool = False):
        cap = self._context_capsule(summary_only=summary_only)
        self.pub_capsule.publish(StringMsg(data=json.dumps(cap)))



    # ------------------------------ Schema ------------------------------
    def _ensure_schema(self):
        cur = self.conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")

        # Drop-and-create (for clean dev boots)
        for obj in ["vw_bt_nodes_summary", "vw_agent_node_labels", "vw_backlog_counts", "vw_object_sheet"]:
            cur.execute(f"DROP VIEW IF EXISTS {obj};")
        for trg in [
            "trg_best_on_current_insert_init",
            "trg_best_on_current_insert_if_better"
        ]:
            cur.execute(f"DROP TRIGGER IF EXISTS {trg};")
        for tbl in [
            "contamination_records", "obj_measurements",
            "bt_measurements", "nodes_state", "bt_nodes",
            "agent_status", "agent_locations", "agent_node_labels"
        ]:
            cur.execute(f"DROP TABLE IF EXISTS {tbl};")

        # Canonical nodes
        cur.execute("""
            CREATE TABLE bt_nodes (
                node_id     TEXT PRIMARY KEY,
                created_ts  REAL NOT NULL DEFAULT (strftime('%s','now'))
            );
        """)

        # Node lifecycle state
        cur.execute("""
            CREATE TABLE nodes_state (
                node_id     TEXT PRIMARY KEY
                            REFERENCES bt_nodes(node_id) ON DELETE CASCADE,
                in_basket   INTEGER NOT NULL DEFAULT 0 CHECK(in_basket IN (0,1)),
                disposed_to TEXT NOT NULL DEFAULT 'none'
                            CHECK(disposed_to IN ('none','clean_bin','contaminated_bin')),
                updated_ts  REAL NOT NULL DEFAULT (strftime('%s','now'))
            );
        """)

        # Two slots per node: current / best
        cur.execute("""
            CREATE TABLE bt_measurements (
                node_id     TEXT NOT NULL
                            REFERENCES bt_nodes(node_id) ON DELETE CASCADE,
                slot        TEXT NOT NULL CHECK(slot IN ('current','best')),
                rssi        INTEGER NOT NULL,
                ts          REAL    NOT NULL,
                x           REAL,
                y           REAL,
                zone        TEXT NOT NULL CHECK(zone IN ('A','B')),
                sensed_by   TEXT NOT NULL
                            CHECK(sensed_by IN ('robot','human_a','human_b')),
                PRIMARY KEY (node_id, slot)
            );
        """)
        cur.execute("CREATE INDEX idx_bt_meas_slot   ON bt_measurements(slot);")
        cur.execute("CREATE INDEX idx_bt_meas_ts     ON bt_measurements(ts);")
        cur.execute("CREATE INDEX idx_bt_meas_zone   ON bt_measurements(zone);")
        cur.execute("CREATE INDEX idx_bt_meas_sensed ON bt_measurements(sensed_by);")

        # Per-agent per-node contamination label
        cur.execute("""
            CREATE TABLE agent_node_labels (
                agent_id    TEXT NOT NULL
                            CHECK(agent_id IN ('robot','human_a','human_b')),
                node_id     TEXT NOT NULL
                            REFERENCES bt_nodes(node_id) ON DELETE CASCADE,
                contaminated INTEGER NOT NULL CHECK(contaminated IN (0,1)),
                probability  REAL NOT NULL CHECK(probability BETWEEN 0.0 AND 1.0),
                updated_ts   REAL NOT NULL DEFAULT (strftime('%s','now')),
                PRIMARY KEY (agent_id, node_id)
            );
        """)
        cur.execute("CREATE INDEX idx_anl_node ON agent_node_labels(node_id);")

        # Agents: last known status + optional history
        cur.execute("""
            CREATE TABLE agent_status (
                agent_id  TEXT PRIMARY KEY
                          CHECK(agent_id IN ('robot','human_a','human_b')),
                zone      TEXT NOT NULL CHECK(zone IN ('A','B')),
                x         REAL,
                y         REAL,
                ts        REAL NOT NULL
            );
        """)
        cur.execute("""
            CREATE TABLE agent_locations (
                id       INTEGER PRIMARY KEY,
                agent_id TEXT NOT NULL
                         CHECK(agent_id IN ('robot','human_a','human_b')),
                zone     TEXT NOT NULL CHECK(zone IN ('A','B')),
                x        REAL,
                y        REAL,
                ts       REAL NOT NULL
            );
        """)
        
        # Box world state (mirrors info learned from the box server)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS box_env_state (
                node_id                TEXT PRIMARY KEY
                                        REFERENCES bt_nodes(node_id) ON DELETE CASCADE,
                box_id                 INTEGER NOT NULL,
                deadline               REAL,      -- sim-time seconds from box server
                x                      REAL,      -- box position from server
                y                      REAL,
                last_sense_status      TEXT,      -- completed / cached / cancelled
                last_sense_detected    INTEGER,   -- 0/1 or NULL
                last_sense_probability REAL,      -- sensor probability used
                last_sense_agent       TEXT,      -- 'robot','human_a','human_b',...
                last_sense_completed_at REAL      -- sim-time from server
            );
        """)

        
        cur.execute("CREATE INDEX idx_agent_loc_agent_ts ON agent_locations(agent_id, ts);")

        # Views
        cur.execute("""
            CREATE VIEW vw_bt_nodes_summary AS
            SELECT
                n.node_id,
                s.in_basket,
                s.disposed_to,
                s.updated_ts AS node_updated_ts,

                c.rssi   AS current_rssi,
                c.ts     AS current_ts,
                c.x      AS current_x,
                c.y      AS current_y,
                c.zone   AS current_zone,
                c.sensed_by AS current_sensed_by,

                b.rssi   AS best_rssi,
                b.ts     AS best_ts,
                b.x      AS best_x,
                b.y      AS best_y,
                b.zone   AS best_zone,
                b.sensed_by AS best_sensed_by

            FROM bt_nodes n
            LEFT JOIN nodes_state s ON s.node_id = n.node_id
            LEFT JOIN bt_measurements c ON c.node_id = n.node_id AND c.slot = 'current'
            LEFT JOIN bt_measurements b ON b.node_id = n.node_id AND b.slot = 'best';
        """)
        cur.execute("""
            CREATE VIEW vw_agent_node_labels AS
            SELECT
              n.node_id,
              r.contaminated AS robot_contaminated,
              r.probability  AS robot_probability,
              a.contaminated AS human_a_contaminated,
              a.probability  AS human_a_probability,
              b.contaminated AS human_b_contaminated,
              b.probability  AS human_b_probability
            FROM bt_nodes n
            LEFT JOIN agent_node_labels r ON r.node_id=n.node_id AND r.agent_id='robot'
            LEFT JOIN agent_node_labels a ON a.node_id=n.node_id AND a.agent_id='human_a'
            LEFT JOIN agent_node_labels b ON b.node_id=n.node_id AND b.agent_id='human_b';
        """)
        cur.execute("""
            CREATE VIEW vw_backlog_counts AS
            SELECT
              SUM(CASE WHEN disposed_to='none' AND in_basket=0 THEN 1 ELSE 0 END) AS to_pick,
              SUM(CASE WHEN disposed_to='none' AND in_basket=1 THEN 1 ELSE 0 END) AS in_basket,
              SUM(CASE WHEN disposed_to='clean_bin' THEN 1 ELSE 0 END) AS delivered_clean,
              SUM(CASE WHEN disposed_to='contaminated_bin' THEN 1 ELSE 0 END) AS delivered_contaminated
            FROM nodes_state;
        """)
        cur.execute("""
            CREATE VIEW vw_object_sheet AS
            SELECT
              s.node_id,
              s.in_basket,
              s.disposed_to,
              c.rssi   AS current_rssi,  c.zone   AS current_zone,  c.ts AS current_ts,
              b.rssi   AS best_rssi,     b.zone   AS best_zone,     b.ts AS best_ts,
              alr.contaminated AS robot_contaminated,  alr.probability AS robot_probability,
              ala.contaminated AS human_a_contaminated, ala.probability AS human_a_probability,
              alb.contaminated AS human_b_contaminated, alb.probability AS human_b_probability
            FROM nodes_state s
            LEFT JOIN bt_measurements c ON c.node_id=s.node_id AND c.slot='current'
            LEFT JOIN bt_measurements b ON b.node_id=s.node_id AND b.slot='best'
            LEFT JOIN agent_node_labels alr ON alr.node_id=s.node_id AND alr.agent_id='robot'
            LEFT JOIN agent_node_labels ala ON ala.node_id=s.node_id AND ala.agent_id='human_a'
            LEFT JOIN agent_node_labels alb ON alb.node_id=s.node_id AND alb.agent_id='human_b';
        """)
        
        cur.execute("""
            CREATE VIEW IF NOT EXISTS vw_box_env AS
            SELECT
              b.node_id,
              s.box_id,
              s.deadline,
              s.x,
              s.y,
              s.last_sense_status,
              s.last_sense_detected,
              s.last_sense_probability,
              s.last_sense_agent,
              s.last_sense_completed_at,
              os.in_basket,
              os.disposed_to,
              os.disposed_to <> 'none' AS is_delivered
            FROM bt_nodes b
            LEFT JOIN box_env_state s ON s.node_id = b.node_id
            LEFT JOIN nodes_state os ON os.node_id = b.node_id;
        """)

        
        # ----- "best" maintenance via triggers -----
        # 1) Initialize best from first current
        cur.execute("""
            CREATE TRIGGER trg_best_on_current_insert_init
            AFTER INSERT ON bt_measurements
            WHEN NEW.slot='current'
                 AND NOT EXISTS (
                     SELECT 1 FROM bt_measurements b
                     WHERE b.node_id=NEW.node_id AND b.slot='best'
                 )
            BEGIN
                INSERT INTO bt_measurements(node_id, slot, rssi, ts, x, y, zone, sensed_by)
                VALUES (NEW.node_id, 'best', NEW.rssi, NEW.ts, NEW.x, NEW.y, NEW.zone, NEW.sensed_by);
            END;
        """)

        # 2) If a new current is "better" than best, overwrite best
        # NOTE: "best" here = LOWEST RSSI (more negative); if you want HIGHEST to be best, flip the comparator.
        cur.execute("""
            CREATE TRIGGER trg_best_on_current_insert_if_better
            AFTER INSERT ON bt_measurements
            WHEN NEW.slot='current'
                 AND EXISTS (
                     SELECT 1 FROM bt_measurements b
                     WHERE b.node_id=NEW.node_id AND b.slot='best' AND NEW.rssi < b.rssi
                 )
            BEGIN
                UPDATE bt_measurements
                SET rssi=NEW.rssi, ts=NEW.ts, x=NEW.x, y=NEW.y, zone=NEW.zone, sensed_by=NEW.sensed_by
                WHERE node_id=NEW.node_id AND slot='best';
            END;
        """)

        # 3) Same logic when the current row is UPDATED via UPSERT
        cur.execute("""
            CREATE TRIGGER trg_best_on_current_update_if_better
            AFTER UPDATE OF rssi, ts, x, y, zone, sensed_by ON bt_measurements
            WHEN NEW.slot='current'
                 AND EXISTS (
                     SELECT 1 FROM bt_measurements b
                     WHERE b.node_id=NEW.node_id AND b.slot='best' AND NEW.rssi < b.rssi
                 )
            BEGIN
                UPDATE bt_measurements
                SET rssi=NEW.rssi, ts=NEW.ts, x=NEW.x, y=NEW.y, zone=NEW.zone, sensed_by=NEW.sensed_by
                WHERE node_id=NEW.node_id AND slot='best';
            END;
        """)

        '''
        # 4) Guard: no one should write directly to slot='best'
        cur.execute("""
            CREATE TRIGGER trg_best_guard_manual
            BEFORE INSERT ON bt_measurements
            WHEN NEW.slot='best'
            BEGIN
                SELECT RAISE(ABORT, 'best slot is managed by triggers; write to slot=current only');
            END;
        """)
        '''
        
        self.conn.commit()
        cur.close()
        self.get_logger().info(f"Broker schema ready at {self.db_path}")

    # ------------------------------ Event ingestion ------------------------------
    def _on_basic_event(self, msg: StringMsg):
        try:
            o = json.loads(msg.data)
        except Exception:
            self.get_logger().warn("broker: invalid JSON on /events/basic")
            return

        rule = str(o.get("rule") or "")
        data = o.get("data") or {}
        ts   = float(o.get("ts") or time.time())
        zone = o.get("zone")  # NEW: top-level zone from EventLayer
        
        # compact trace entry with json_text support
        trace_entry = {
            "rule": rule,
            "ts": ts,
        }

        if zone is not None:
            trace_entry["zone"] = zone

        if isinstance(data, dict):
            trace_entry["data"] = {}  # keep data nested, not flattened

            for k, v in data.items():
                # 1. If this is json_text → keep both raw and parsed
                if k == "json_text" and isinstance(v, str):
                    trace_entry["data"]["json_text"] = v

                    # Try parsing to structured JSON
                    try:
                        parsed = json.loads(v)
                        trace_entry["data"]["json"] = parsed   # structured JSON
                    except Exception:
                        pass  # keep raw only if parsing fails

                # 2. Keep small scalar fields
                elif isinstance(v, (str, int, float, bool)):
                    trace_entry["data"][k] = v

        self._event_trace.append(trace_entry)
        
        # Feed running event summary (async)
        self._record_event_for_summary(trace_entry, ts)


        # remember last robot zone for capsule
        if zone is not None:
            self._last_robot_zone = zone        # NEW
            
        # trigger state (used by LLM prompt)
        trig_type = self.trigger_map.get(rule)

        # NEW: any basic rule whose id starts with the trigger prefix is a planner trigger
        if rule.startswith(self.planner_trigger_prefix):
            if not trig_type:
                trig_type = "planner_trigger"

        if trig_type:
            self._current_trigger = {
                "type": trig_type,
                "trigger_event": o,
                "ts": ts,
            }
            if zone is not None:
                self._current_trigger["zone"] = zone

        # ingestion
        if rule == self.bt_rule_id:
            self._ingest_bt_reading(data, o)
        elif rule == self.human3d_rule_id:
            self._ingest_human3d(data, o)

        # --- NEW: track robot_objective from skill status events (via EventLayer) ---
        # Expect EventLayer skill_status payload to include fields like:
        #   { "kind": "skill_started" | "skill_finished", "skill": "<name>", "done": bool, ... }
        try:
            if rule == "skill_started_any":
                skill_name = (data.get("skill") or "").strip()
                if skill_name:
                    self._robot_objective = f"execute:{skill_name}"
            elif rule == "skill_done_any":
                # when a skill finishes, clear objective
                if data.get("done", False):
                    self._robot_objective = None
        except Exception as e:
            self.get_logger().debug(f"robot_objective update failed: {e}")

        '''
        # Proactive: if this event is a planning trigger, immediately run initial LLM-SQL
        if trig_type in ("new_object", "finish_or_fail", "human_command", "idle", "presence", "planner_trigger"):
            try:
                self._publish_context_capsule()
                pack = self._llm_sql_to_facts(proactive=True)
                self.pub_facts.publish(StringMsg(data=json.dumps(pack)))
                self._emit_sql_debug(pack)
            except Exception as e:
                self.get_logger().warn(f"proactive run failed: {e}")
        '''

    def _on_comp_event(self, msg: StringMsg):
        try:
            o = json.loads(msg.data)
        except Exception:
            self.get_logger().warn("broker: invalid JSON on /events/composite")
            return

        rid  = str(o.get("rule") or "")
        ts   = float(o.get("ts") or time.time())
        expr = o.get("expr") or ""
        zone = o.get("zone")  # zone stamped by EventLayer composite
        data = o.get("data") or {}

        # trace entry (keep expr + mark composite; add zone if present)
        trace_entry = {
            "rule": rid,
            "ts": ts,
            "composite": True,
            "expr": expr[:160],
        }
        if zone is not None:
            trace_entry["zone"] = zone

        # --- 1) If composite message already has data, treat it like a basic event ---
        if isinstance(data, dict):
            trace_entry["data"] = {}
            for k, v in data.items():
                # Special handling for json_text: keep both raw + parsed
                if k == "json_text" and isinstance(v, str):
                    trace_entry["data"]["json_text"] = v
                    try:
                        parsed = json.loads(v)
                        trace_entry["data"]["json"] = parsed
                    except Exception:
                        # if parsing fails, we still keep raw json_text
                        pass
                # Keep small scalar fields
                elif isinstance(v, (str, int, float, bool)):
                    trace_entry["data"][k] = v

            # If the data dict ended up empty, remove it to keep the trace clean
            if not trace_entry["data"]:
                trace_entry.pop("data", None)

        # --- 2) If this is an LLM/VLM composite, try to copy json_text from recent basic events ---
        needs_json = rid.startswith("llm_") or rid.startswith("vlm_")
        if needs_json:
            has_json_already = (
                isinstance(trace_entry.get("data"), dict)
                and "json_text" in trace_entry["data"]
            )
            if not has_json_already:
                # Walk backward through the existing trace looking for the last event
                # with the same rule id that has json_text (from /events/basic).
                for e in reversed(self._event_trace):
                    if e.get("rule") != rid:
                        continue
                    edata = e.get("data")
                    if not isinstance(edata, dict):
                        continue
                    if "json_text" in edata:
                        trace_entry.setdefault("data", {})
                        trace_entry["data"]["json_text"] = edata["json_text"]
                        if "json" in edata:
                            trace_entry["data"]["json"] = edata["json"]
                        break  # stop at the first match

        # Now store + feed summary
        self._event_trace.append(trace_entry)
        self._record_event_for_summary(trace_entry, ts)

        # remember last robot zone for capsule
        if zone is not None:
            self._last_robot_zone = zone

        # map composite rule id → trigger (use the same trigger_map param)
        trig_type = self.trigger_map.get(rid, "composite_hit")

        # any composite rule whose id starts with the trigger prefix is a planner trigger
        if rid.startswith(self.planner_trigger_prefix):
            if trig_type == "composite_hit":
                trig_type = "planner_trigger"

        self._current_trigger = {
            "type": trig_type,
            "ts": ts,
            "composite": True,
            "rid": rid,
        }
        if zone is not None:
            self._current_trigger["zone"] = zone  # optional but handy

        # (proactive runs are still commented out)


    def _ingest_human3d(self, data: dict, envelope: dict):
        # Expect: cls='person', map_x,map_y,map_z, frame_id, ts
        if data.get("cls") != "person":
            return
        mx, my = data.get("map_x"), data.get("map_y")
        if mx is None or my is None:
            return
        ts_epoch = float(data.get("ts") or envelope.get("ts") or time.time())
        zone = self._zone_from_xy(mx, my)
        self._upsert_agent_status(self.human_agent_id, zone, float(mx), float(my), ts_epoch)

    def _upsert_agent_status(self, agent_id: str, zone: str,
                             x: Optional[float], y: Optional[float], ts: float):
        self.conn.execute("""
            INSERT INTO agent_status(agent_id, zone, x, y, ts)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                zone=excluded.zone, x=excluded.x, y=excluded.y, ts=excluded.ts
        """, (agent_id, zone, x, y, ts))
        self.conn.execute("""
            INSERT INTO agent_locations(agent_id, zone, x, y, ts)
            VALUES (?, ?, ?, ?, ?)
        """, (agent_id, zone, x, y, ts))

    def _ingest_bt_reading(self, data: dict, envelope: dict):
        """
        Expect from event layer:
          { "node_id":"CNode103", "rssi":-72, "sensed_by":"robot|human_a|human_b",
            "frame_id":"base_link", "ts": 1762470000.12 }
        """
        node_id = (data.get("node_id") or data.get("object_id") or "").strip()
        if not node_id:
            return
        rssi      = int(data.get("rssi"))
        sensed_by = (data.get("sensed_by") or data.get("phone_id") or "robot").strip()
        frame_id  = (data.get("frame_id") or "").strip()
        ts_epoch  = float(data.get("ts") or envelope.get("ts") or time.time())
        agent_id  = sensed_by if sensed_by in ('robot','human_a','human_b') else 'robot'

        self._ensure_node(node_id)
        self._ensure_node_state(node_id)

        x, y = self._tf_to_map(frame_id)
        zone = self._zone_from_xy(x, y)
        self._upsert_current(node_id, rssi, ts_epoch, x, y, zone, agent_id)

        # Only fetch contamination if (agent,node) new
        #self._maybe_queue_contamination_refresh(agent_id, node_id, rssi, ts_epoch)

    # ------------------------------ DB helpers ------------------------------
    def _ensure_node(self, node_id: str):
        self.conn.execute("INSERT OR IGNORE INTO bt_nodes(node_id) VALUES (?)", (node_id,))

    def _ensure_node_state(self, node_id: str):
        self.conn.execute(
            "INSERT OR IGNORE INTO nodes_state(node_id, in_basket, disposed_to) VALUES (?, 0, 'none')",
            (node_id,)
        )

    def _upsert_current(self, node_id: str, rssi: int, ts: float,
                        x: Optional[float], y: Optional[float],
                        zone: str, sensed_by: str):
        self.conn.execute("""
            INSERT INTO bt_measurements(node_id, slot, rssi, ts, x, y, zone, sensed_by)
            VALUES (?, 'current', ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id, slot) DO UPDATE SET
                rssi=excluded.rssi, ts=excluded.ts, x=excluded.x, y=excluded.y,
                zone=excluded.zone, sensed_by=excluded.sensed_by
        """, (node_id, int(rssi), float(ts), x, y, zone, sensed_by))

    # ------------------------------ TF & Zone ------------------------------
    def _tf_to_map(self, frame_id: str) -> Tuple[Optional[float], Optional[float]]:
        if not frame_id:
            return (None, None)
        try:
            tf = self.tf_buffer.lookup_transform(self.target_frame, frame_id, rclpy.time.Time(),
                                                 timeout=Duration(seconds=0.2))
            return (float(tf.transform.translation.x), float(tf.transform.translation.y))
        except (LookupException, ExtrapolationException) as e:
            self.get_logger().debug(f"TF {frame_id}->{self.target_frame} failed: {e}")
            return (None, None)

    def _zone_from_xy(self, x: Optional[float], y: Optional[float]) -> str:
        if x is None:
            return 'B'
        return 'A' if (x < self.zone_split_x) else 'B'

    # ------------------------------ Contamination pipeline ------------------------------
    def _maybe_queue_contamination_refresh(self, agent_id: str, node_id: str, current_rssi: int, ts: float):
        row = self.conn.execute("""
            SELECT 1 FROM agent_node_labels WHERE agent_id=? AND node_id=? LIMIT 1
        """, (agent_id, node_id)).fetchone()
        if row is not None:
            return
        now = time.time()
        key = (agent_id, node_id)
        with self._lock:
            ent = self._contam_cache.get(key)
            if ent and (now - ent["ts"] < self.min_refresh):
                return
            self._pending_refresh.add(key)

    def _process_pending_refresh(self):
        batch = []
        with self._lock:
            while self._pending_refresh and len(batch) < 8:
                batch.append(self._pending_refresh.pop())
        for agent_id, node_id in batch:
            self._refresh_one_label(agent_id, node_id)

    def _box_id_from_node(self, node_id: str) -> Optional[int]:
        """
        Map a bt node_id like 'CNode12' or 'Box_7' to an integer box_id
        used by the FastAPI box server.

        Returns None if we can't parse a positive integer.
        """
        m = re.search(r'(\d+)', node_id)
        if not m:
            self.get_logger().warn(f"[broker] cannot map node_id='{node_id}' to box_id")
            return None
        try:
            val = int(m.group(1))
            return val if val > 0 else None
        except ValueError:
            self.get_logger().warn(f"[broker] invalid numeric portion in node_id='{node_id}'")
            return None

    def _node_id_from_box(self, box_id: int) -> str:
        """
        Map an integer box_id (from the FastAPI box server) to a canonical node_id
        in the broker DB. Adjust the format if you use a different naming scheme.
        """
        return f"CNode{box_id}"


    def _refresh_one_label(self, agent_id: str, node_id: str):
        """
        NEW VERSION:

        Instead of calling the old contamination '/check' endpoint, we call the
        FastAPI box server's /sense endpoint for PROPERTY X and interpret its
        detection outcome as this agent's label for the node.

        - node_id is mapped to an integer box_id via _box_id_from_node().
        - We do a blocking POST /sense (the box server simulates sensing time).
        - We write the result into:
            * agent_node_labels   (contaminated + probability)
            * box_env_state       (deadline, x,y, and last sense info)
        """
        now = time.time()
        key = (agent_id, node_id)
        with self._lock:
            ent = self._contam_cache.get(key)
            if ent and (now - ent["ts"] < self.min_refresh):
                return

        if not self.enable_server or not self.server_url:
            return

        box_id = self._box_id_from_node(node_id)
        if box_id is None:
            return

        url = self.server_url.rstrip("/") + "/sense"
        payload = {
            "agent_id": agent_id,
            "box_id": box_id,
            "property": "X",  # we treat X as the contamination-like property
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.req_timeout)
            if resp.status_code != 200:
                self.get_logger().warn(
                    f"[broker] box server /sense non-200 for {(agent_id,node_id)}: {resp.status_code}"
                )
                return

            data = resp.json()
        except Exception as e:
            self.get_logger().warn(f"[broker] box server /sense failed for {(agent_id,node_id)}: {e}")
            return

        # SenseResponse fields from the new server:
        # {
        #   "agent_id": str,
        #   "box_id": int,
        #   "property": "X"|"Y",
        #   "status": "completed"|"cached"|"cancelled",
        #   "detected": bool | null,
        #   "probability": float | null,
        #   "deadline": float,
        #   "x": float,
        #   "y": float,
        #   "requested_at": float,
        #   "completed_at": float | null
        # }
        status = data.get("status")
        detected = data.get("detected")
        probability = data.get("probability")
        deadline = data.get("deadline")
        bx = data.get("x")
        by = data.get("y")
        completed_at = data.get("completed_at")

        # We only treat 'completed' or 'cached' with a boolean detected value as a usable label
        if status not in ("completed", "cached") or detected is None or probability is None:
            self.get_logger().info(
                f"[broker] box /sense returned unusable status='{status}' for {(agent_id,node_id)}"
            )
            return

        contaminated = bool(detected)  # our label semantics: detected X == contaminated

        # --- Update agent_node_labels (same schema as before) ---
        self.conn.execute(
            """
            INSERT INTO agent_node_labels(agent_id, node_id, contaminated, probability, updated_ts)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, node_id) DO UPDATE SET
                contaminated = excluded.contaminated,
                probability  = excluded.probability,
                updated_ts   = excluded.updated_ts
            """,
            (agent_id, node_id, int(1 if contaminated else 0), float(probability), now),
        )

        # --- Update box_env_state with richer info from the server ---
        self.conn.execute(
            """
            INSERT INTO box_env_state(
                node_id, box_id, deadline, x, y,
                last_sense_status, last_sense_detected,
                last_sense_probability, last_sense_agent,
                last_sense_completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                box_id                 = excluded.box_id,
                deadline               = excluded.deadline,
                x                      = excluded.x,
                y                      = excluded.y,
                last_sense_status      = excluded.last_sense_status,
                last_sense_detected    = excluded.last_sense_detected,
                last_sense_probability = excluded.last_sense_probability,
                last_sense_agent       = excluded.last_sense_agent,
                last_sense_completed_at= excluded.last_sense_completed_at
            """,
            (
                node_id,
                int(box_id),
                float(deadline) if deadline is not None else None,
                float(bx) if bx is not None else None,
                float(by) if by is not None else None,
                status,
                int(1 if contaminated else 0),
                float(probability),
                agent_id,
                float(completed_at) if completed_at is not None else None,
            ),
        )

        with self._lock:
            self._contam_cache[key] = {
                "ts": now,
                "contaminated": contaminated,
                "probability": probability,
            }


    def _sync_box_state_from_server(self, boxes_state: list):
        """
        Mirror the FastAPI box server state into the broker DB.

        For each box:
          - ensure bt_nodes/nodes_state rows exist
          - update box_env_state (deadline, x, y, last sense info for X)
          - update nodes_state.disposed_to based on disposed_X / disposed_Y
          - update agent_node_labels for property X from the latest completed
            sense result (we interpret X as the 'contamination-like' property).
        """
        now = time.time()
        cur = self.conn.cursor()
        try:
            for b in boxes_state:
                try:
                    box_id = int(b["box_id"])
                except Exception:
                    continue

                node_id = self._node_id_from_box(box_id)
                self._ensure_node(node_id)
                self._ensure_node_state(node_id)

                deadline = float(b.get("deadline", 1e9))
                x = float(b.get("x", 0.0))
                y = float(b.get("y", 0.0))

                # --- pick latest completed sense result for property X ---
                sense_results = b.get("sense_results") or []
                last_x = None
                for sr in sense_results:
                    if sr.get("property") != "X":
                        continue
                    if sr.get("status") != "completed":
                        continue
                    # assuming /boxes/state gives sense_results in chronological order,
                    # this leaves us with the latest completed X result
                    last_x = sr

                if last_x:
                    status_x = last_x.get("status")
                    detected_x = last_x.get("detected")
                    prob_x = last_x.get("probability")
                    agent_x = str(last_x.get("agent_id") or "")
                    completed_at_x = last_x.get("completed_at")
                else:
                    status_x = None
                    detected_x = None
                    prob_x = None
                    agent_x = None
                    completed_at_x = None

                # --- Update box_env_state (we treat last_sense_* as property X) ---
                cur.execute(
                    """
                    INSERT INTO box_env_state(
                        node_id, box_id, deadline, x, y,
                        last_sense_status, last_sense_detected,
                        last_sense_probability, last_sense_agent,
                        last_sense_completed_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        box_id                 = excluded.box_id,
                        deadline               = excluded.deadline,
                        x                      = excluded.x,
                        y                      = excluded.y,
                        last_sense_status      = excluded.last_sense_status,
                        last_sense_detected    = excluded.last_sense_detected,
                        last_sense_probability = excluded.last_sense_probability,
                        last_sense_agent       = excluded.last_sense_agent,
                        last_sense_completed_at= excluded.last_sense_completed_at
                    """,
                    (
                        node_id,
                        box_id,
                        deadline,
                        x,
                        y,
                        status_x,
                        int(1 if detected_x else 0) if detected_x is not None else None,
                        float(prob_x) if prob_x is not None else None,
                        agent_x,
                        float(completed_at_x) if completed_at_x is not None else None,
                    ),
                )

                # --- Map dispose flags into nodes_state.disposed_to ---
                # Assumption (edit if your semantics differ):
                #   disposed_X  -> contaminated_bin
                #   disposed_Y  -> clean_bin (if X not already disposed)
                disposed_X = bool(b.get("disposed_X", False))
                disposed_Y = bool(b.get("disposed_Y", False))

                if disposed_X:
                    disposed_to = "contaminated_bin"
                elif disposed_Y:
                    disposed_to = "clean_bin"
                else:
                    disposed_to = "none"

                cur.execute(
                    """
                    UPDATE nodes_state
                    SET disposed_to = ?
                    WHERE node_id = ?
                    """,
                    (disposed_to, node_id),
                )

                # --- agent_node_labels: property X → contaminated label ---
                if (
                    agent_x
                    and status_x == "completed"
                    and detected_x is not None
                    and prob_x is not None
                ):
                    cur.execute(
                        """
                        INSERT INTO agent_node_labels(
                            agent_id, node_id, contaminated, probability, updated_ts
                        )
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(agent_id, node_id) DO UPDATE SET
                            contaminated = excluded.contaminated,
                            probability  = excluded.probability,
                            updated_ts   = excluded.updated_ts
                        """,
                        (
                            agent_x,
                            node_id,
                            int(1 if detected_x else 0),
                            float(prob_x),
                            now,
                        ),
                    )

            self.conn.commit()
        except Exception as e:
            self.get_logger().warn(f"[optimizer] sync box state to DB failed: {e}")
            self.conn.rollback()
        finally:
            cur.close()


    # ---------- Async running event summary ----------

    def _record_event_for_summary(self, trace_entry: dict, ts: float):
        """
        Track events for the running summary and schedule an async LLM update
        every `event_summary_batch_size` events.

        This is cheap and called from the event callbacks.
        """
        if not self.event_summary_enabled:
            return

        batch_to_run: Optional[List[dict]] = None
        last_ts_for_batch: Optional[float] = None

        with self._event_summary_lock:
            self._unsummarized_events.append(trace_entry)
            self._events_since_summary += 1

            # Trigger a batch when we hit the threshold and no worker is running.
            if (
                not self._event_summary_running
                and self._events_since_summary >= self.event_summary_batch_size
                and self._unsummarized_events
            ):
                batch_to_run = self._unsummarized_events
                last_ts_for_batch = batch_to_run[-1].get("ts", ts)

                # Reset buffer & counter for future events
                self._unsummarized_events = []
                self._events_since_summary = 0
                self._event_summary_running = True

        # Run LLM outside the lock in a background thread
        if batch_to_run:
            threading.Thread(
                target=self._event_summary_worker,
                args=(batch_to_run, last_ts_for_batch),
                daemon=True,
            ).start()

    def _event_summary_worker(self, batch_events: List[dict], last_ts: Optional[float]):
        """
        Background worker that updates the running event summary by combining
        the previous summary with a batch of new events.
        """
        try:
            with self._event_summary_lock:
                prev_summary = self._event_summary_text

            new_summary = self._build_running_event_summary(prev_summary, batch_events)
            if not new_summary:
                return

            with self._event_summary_lock:
                self._event_summary_text = new_summary
                if last_ts is not None:
                    self._event_summary_ts = float(last_ts)

        except Exception as e:
            self.get_logger().warn(f"[broker] async event summary worker failed: {e}")
        finally:
            with self._event_summary_lock:
                self._event_summary_running = False
                
        try:
            fp = hashlib.sha256((new_summary or "").encode("utf-8")).hexdigest()
            if fp != self._last_published_summary_fp:
                self._last_published_summary_fp = fp
                self._publish_context_capsule(summary_only=True)
        except Exception as e:
            self.get_logger().warn(f"[broker] publish-on-change failed: {e}")


    def _build_running_event_summary(
        self,
        previous_summary: Optional[str],
        new_events: List[dict],
    ) -> Optional[str]:
        """
        Build/refresh a *running* summary.

        LLM sees:
          - previous_summary: the last global summary (or null)
          - new_events: the latest batch (chronological list)

        Returns updated single-sentence summary (or None).
        """
        if not new_events:
            return previous_summary

        # Keep only a tail of the batch to avoid prompt blow-up
        tail = new_events[-10:]
        try:
            tail_json = json.dumps(tail, ensure_ascii=False)
        except Exception as e:
            self.get_logger().warn(f"[broker] failed to encode new_events for running summary: {e}")
            return previous_summary

        if not self.event_summary_enabled:
            return previous_summary

        try:
            system_msg = (
                "You maintain a RUNNING SUMMARY of a mobile robot's recent events. The name of the robot is Bob.\n"
                "- You receive the previous summary (may be null) and a batch of NEW events.\n"
                "- Events are JSON objects with fields like rule, ts, text, skill, zone, etc.\n\n"
                "Your job:\n"
                "- Produce ONE SHORT SENTENCE that summarizes the overall recent situation,\n"
                "  updating or refining the previous summary using the new events.\n"
                "- Prioritize the NEW events but keep any still-relevant context from the old summary.\n"
                "- If nothing important changed, you may return a very similar summary.\n"
                "- Output will be wrapped in JSON with key 'summary'.\n"
            )

            user_payload = {
                "previous_summary": previous_summary,
                "new_events": tail,
            }

            obj = self._chat_json(
                messages=[
                    {"role": "system", "content": system_msg},
                    {
                        "role": "user",
                        "content": (
                            "Here is the previous summary and the latest batch of events. "
                            'Return a JSON object: {"summary": "one short sentence"}.'
                        ),
                    },
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                temperature=0.2,
                max_tokens=80,
                retries=1,
                schema=EVENT_SUMMARY_SCHEMA,
                schema_name="EventRunningSummary",
                model=self.event_summary_model,
                perf_phase="event_summary",
            )

            summary = (obj.get("summary") or "").strip()
            return summary or previous_summary

        except Exception as e:
            self.get_logger().warn(f"[broker] running event summary LLM failed: {e}")
            return previous_summary



    def _summarize_event_trace(self, events: List[dict]) -> Optional[str]:
        """
        Summarize a chronological event list (oldest -> newest) into
        a short one-sentence summary using the existing _chat_json helper.
        """
        if not events:
            return None

        # Keep only the most recent N events in the prompt
        tail = events[-10:]
        try:
            tail_json = json.dumps(tail, ensure_ascii=False)
        except Exception as e:
            self.get_logger().warn(f"broker: failed to encode event_trace for summary: {e}")
            return None

        # If LLM summarization is disabled, do a trivial heuristic
        if not getattr(self, "event_summary_enabled", True):
            last = tail[-1]
            text = last.get("text")
            skill = last.get("skill")
            zone = last.get("zone")
            if text:
                if zone:
                    return f'Latest human utterance in zone {zone}: "{text}"'
                else:
                    return f'Latest human utterance: "{text}"'
            if skill:
                return f"Latest skill event: {skill} ({last.get('kind', '')})"
            return None

        # LLM-based summary via _chat_json
        try:
            system_msg = (
                "You summarize event traces for a mobile robot whose name is Bob.\n"
                "- You receive an \"event_trace\" which is a JSON array of events.\n"
                "- Each event may include fields like: rule, rule_id, ts, text, skill, kind, zone, etc.\n\n"
                "Your task:\n"
                "- Produce ONE SHORT SENTENCE summarizing the most important facts.\n"
                "- Treat the LAST events as more important (they are more recent).\n"
                "- Output should be natural language, but we will wrap it in JSON with key 'summary'.\n"
                "- If you mention human speech, briefly paraphrase it but keep the intent and key phrases.\n"
            )


            user_msg = (
                'Here is the event_trace (chronological, oldest first):\n'
                f'{tail_json}\n\n'
                "Remember: the last element is the most recent event.\n"
                "Return a JSON object: {\"summary\": \"one short sentence\"}."
            )

            obj = self._chat_json(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=80,
                retries=1,
                schema=EVENT_SUMMARY_SCHEMA,
                schema_name="EventSummary",
                model=self.event_summary_model,
                perf_phase="event_summary",
            )

            summary = (obj.get("summary") or "").strip()
            return summary or None

        except Exception as e:
            self.get_logger().warn(f"broker: event summary LLM failed: {e}")
            return None



    # ------------------------------ LLM SQL layer ------------------------------
    # Strict validators
    _SQL_BAD = re.compile(r'(--|/\*|\*/|;|\b(ATTACH|DETACH|ALTER|DROP|CREATE|INSERT|UPDATE|DELETE|REPLACE|VACUUM|PRAGMA|BEGIN|END|COMMIT|ROLLBACK)\b)', re.I)
    _SQL_TABLES = re.compile(r'\b(from|join)\s+([a-zA-Z0-9_\.]+)', re.I)

    def _validate_sql_readonly(self, sql: str) -> Optional[str]:
        # 1) Existing checks: forbid DDL/DML/etc.
        if self._SQL_BAD.search(sql):
            return "sql_contains_prohibited_tokens"

        # 2) Existing checks: only allowed objects in FROM/JOIN
        used = [m.group(2) for m in self._SQL_TABLES.finditer(sql)]
        for name in used:
            base = name.split('.')[-1]
            if base not in self.allowed:
                return f"object_not_allowed:{base}"

        # 3) NEW: have SQLite parse the query and fail early on unknown columns.
        try:
            # Collect named parameters like :zone, :object_id, etc.
            param_names = set(re.findall(r":([a-zA-Z0-9_]+)", sql))
            dummy_params = {name: 0 for name in param_names}

            # Let SQLite parse the query (no actual rows read).
            # Any "no such column" or binding errors will raise here.
            self.conn.execute("EXPLAIN " + sql, dummy_params)
        except sqlite3.Error as e:
            # Treat *any* EXPLAIN error as validation failure.
            return f"sql_invalid:{e}"

        return None


    def _exec_sql_safely(self, sql: str, params: dict,
                         max_rows: int, max_bytes: int, timeout_ms: int):
        con = self.conn

        aborted = {"v": False}
        start = time.time()

        def _progress():
            if (time.time() - start) * 1000.0 > timeout_ms:
                aborted["v"] = True
                return 1
            return 0

        con.set_progress_handler(_progress, 1000)

        try:
            cur = con.execute(sql, params or {})
        except sqlite3.Error as e:
            # Make sure we clear the progress handler even on error
            con.set_progress_handler(None, 0)
            self.get_logger().warn(f"[broker] SQL runtime error: {e} | sql={sql}")
            # Re-raise so caller can decide whether to fall back
            raise

        colnames = [d[0] for d in cur.description] if cur.description else []
        rows, size, i = [], 0, -1
        for i, row in enumerate(cur):
            if i >= max_rows:
                break
            vals = []
            for v in row:
                if isinstance(v, (bytes, bytearray)):
                    v = "<blob>"
                vals.append(v)
            size += sum(len(str(x)) for x in vals) + 2 * len(vals)
            if size > max_bytes:
                break
            rows.append(vals)

        con.set_progress_handler(None, 0)
        truncated = aborted["v"] or (i + 1 >= max_rows) or (size > max_bytes)
        return colnames, rows, truncated, int((time.time() - start) * 1000)


    def _schema_card(self) -> dict:
        # object → columns
        objects = []
        for name in sorted(self.allowed):
            cols = []
            try:
                cur = self.conn.execute(f"SELECT * FROM {name} LIMIT 0")
                cols = [d[0] for d in cur.description] if cur.description else []
            except Exception:
                pass
            objects.append(f"{name}({', '.join(cols)})")
        samples = {}
        for name in ("vw_object_sheet","vw_bt_nodes_summary","vw_agent_node_labels","vw_backlog_counts"):
            try:
                cur = self.conn.execute(f"SELECT * FROM {name} LIMIT 2")
                colnames = [d[0] for d in cur.description] if cur.description else []
                rows = [dict(zip(colnames, r)) for r in cur.fetchall()]
                samples[name] = rows
            except Exception:
                pass
        return {"objects": objects, "samples": samples}

    def _context_capsule(self, summary_only: bool = True) -> dict:
        with self._event_summary_lock:
            summary_text = self._event_summary_text
            summary_ts = self._event_summary_ts

        event_summary = None
        if summary_text is not None:
            event_summary = {
                "summary": summary_text,
                "last_event_ts": summary_ts,
            }

        if summary_only:
            # capsule is ONLY the summary (plus a ts so downstream can reason about freshness)
            return {
                "ts": time.time(),
                "event_summary": event_summary,
            }



        return {
            "event_trace": summary_text,

        }



    def _ws_id(self) -> str:
        # one session per trigger time (coarse); override with planner-provided ws later if needed
        if not self._current_trigger:
            return "ws-default"
        t = int(self._current_trigger.get("ts", time.time()) * 1000)
        return f"ws-{t}"

    def _ws_add(self, ws_id: str, sql: str, rows: List[List]):
        h = hashlib.sha256(json.dumps({"sql": sql, "rows": rows}, sort_keys=True).encode("utf-8")).hexdigest()
        ent = self._ws.setdefault(ws_id, {"hashes": set(), "iters": 0})
        ent["hashes"].add(h)

    def _ws_changed(self, ws_id: str, sql: str, rows: List[List]) -> bool:
        h = hashlib.sha256(json.dumps({"sql": sql, "rows": rows}, sort_keys=True).encode("utf-8")).hexdigest()
        ent = self._ws.setdefault(ws_id, {"hashes": set(), "iters": 0})
        return h not in ent["hashes"]

    def _emit_sql_debug(self, pack: dict):
        meta = pack.get("sql_meta") or {}
        dbg = {
            "sql": meta.get("sql"),
            "params": meta.get("params"),
            "ms": meta.get("ms"),
            "truncated": (pack.get("table") or {}).get("truncated", False),
            "rationale": pack.get("rationale"),
            "mode": pack.get("mode")
        }
        self.pub_sql_debug.publish(StringMsg(data=json.dumps(dbg)))

    # ---- LLM call (replace with your real endpoint) ----
    
    def _build_llm_messages_proactive(self, schema_card: dict, context_capsule: dict) -> list:
        system = (
            "You are a SQLite query planner for a mobile robot. "
            "Return ONE read-only SQL SELECT (no semicolons/DDL/DML/PRAGMA), using only allowed objects (prefer vw_*). "
            "Keep results compact within the provided budgets. If the trigger hints an object_id, prioritize it. "
            'Output STRICT JSON: {"sql":"... :named_params ...","params":{...},"purpose":"<=20 words"}'
        )
        fewshot_user = {"SchemaCard":{"objects":["vw_object_sheet(...)", "vw_backlog_counts(...)"]},
                        "ContextCapsule":{"trigger":{"type":"new_object","hints":{"object_id":"CNode12"}}}}
        fewshot_assistant = {"sql":"SELECT * FROM vw_object_sheet WHERE node_id=:object_id LIMIT 1",
                             "params":{"object_id":"CNode12"},
                             "purpose":"object sheet for hinted node"}
        user = {"SchemaCard": schema_card, "ContextCapsule": context_capsule}
        return [
            {"role":"system","content":system},
            {"role":"user","content":json.dumps(fewshot_user)},
            {"role":"assistant","content":json.dumps(fewshot_assistant)},
            {"role":"user","content":json.dumps(user)},
        ]

    def _build_llm_messages_reactive(self, schema_card: dict, context_capsule: dict,
                                     planner_needs: dict, already_returned: dict) -> list:
        system = (
            "You extend prior facts. Produce ONE read-only SQL SELECT to resolve the most blocking OPEN need. "
            "Do NOT repeat already returned facts. Use only allowed objects, prefer vw_*, respect budgets. "
            'Output STRICT JSON: {"sql":"... :named_params ...","params":{...},"purpose":"<=20 words"}'
        )
        fewshot_user = {"PlannerNeeds":{"needs":[{"why":"confirm label","focus":"object","object_id":"CNode37"}]}}
        fewshot_assistant = {"sql":"SELECT * FROM vw_object_sheet WHERE node_id=:object_id LIMIT 1",
                             "params":{"object_id":"CNode37"},
                             "purpose":"resolve object label gap"}
        user = {"SchemaCard": schema_card, "ContextCapsule": context_capsule,
                "PlannerNeeds": planner_needs or {}, "AlreadyReturned": already_returned or {}}
        return [
            {"role":"system","content":system},
            {"role":"user","content":json.dumps(fewshot_user)},
            {"role":"assistant","content":json.dumps(fewshot_assistant)},
            {"role":"user","content":json.dumps(user)},
        ]


    def _chat_json(
        self,
        messages,
        temperature: float = 0.2,
        max_tokens: int = 300,
        retries: int = 1,
        schema: Optional[dict] = None,
        schema_name: str = "BrokerSQL",
        model: Optional[str] = None,
        perf_phase: str = "sql_plan",
    ):
        """
        Call chat.completions with JSON-SCHEMA response_format and validate output.

        - `schema`: JSON schema to enforce (defaults to LLM_SQL_SCHEMA).
        - `schema_name`: name used in response_format.
        - `model`: override model (defaults to self.model).
        - `perf_phase`: label for latency telemetry (e.g. 'sql_plan', 'event_summary').
        """
        last_exc = None
        used_schema = schema or LLM_SQL_SCHEMA
        used_model = model or self.model

        for attempt in range(retries + 1):
            t0 = time.time()
            ok_api = False
            self.get_logger().info("\n=== LLM PROMPT ===\n" + json.dumps(messages, indent=2))

            try:
                if "gpt-oss" in used_model:
                    client = Groq()
                    resp = client.chat.completions.create(
                        model="openai/" + used_model,
                        messages=messages,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": schema_name,
                                "schema": used_schema,
                            },
                        },
                        reasoning_effort="medium",
                    )
                else:
                    client = OpenAI()
                    resp = client.chat.completions.create(
                        model=used_model,
                        messages=messages,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": schema_name,
                                "schema": used_schema,
                            },
                        },
                    )

                dt_ms = int((time.time() - t0) * 1000)
                ok_api = True

                content = resp.choices[0].message.content
                obj = json.loads(content)
                validate(instance=obj, schema=used_schema)

                self.get_logger().info(
                    f"\n=== LLM RAW RESPONSE ({schema_name}) ===\n{content}\nLatency: {str(dt_ms)}\n"
                )

                # publish perf (using the broker model, not used_model, for now)
                # If you want per-task metrics, you could add a separate publisher.
                self._publish_llm_perf(dt_ms, ok=True, phase=perf_phase)
                return obj

            except (json.JSONDecodeError, ValidationError) as e:
                # schema/json error
                dt_ms = int((time.time() - t0) * 1000)
                self._publish_llm_perf(dt_ms, ok=False, phase=perf_phase + "_schema")
                last_exc = e
                messages = messages + [{
                    "role": "system",
                    "content": "Return ONLY valid JSON per the given schema. No prose.",
                }]
                continue

            except Exception as e:
                dt_ms = int((time.time() - t0) * 1000)
                self._publish_llm_perf(dt_ms, ok=False, phase=perf_phase + "_api")
                last_exc = e
                continue

        raise ValueError(f"LLM did not return valid JSON ({schema_name}): {last_exc}")


    

    def _call_openai_chat(self, messages: list, model: str = "gpt-4o-mini", timeout_s: float = 8.0) -> str:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY env var not set")
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return content

    def _current_ws_summary(self) -> dict:
        ws_id = self._ws_id()
        ent = self._ws.get(ws_id, {"hashes": set(), "iters": 0})
        return {"returned_sets": len(ent["hashes"]), "iters": ent["iters"]}

    def _llm_plan_sql(self, proactive: bool, schema_card: dict, context_capsule: dict,
                      planner_needs: Optional[dict]) -> tuple[str, dict, str]:
        if self._llm_mock and "sql" in self._llm_mock:
            return self._llm_mock["sql"], self._llm_mock.get("params", {}), self._llm_mock.get("purpose", "mock")

        if proactive:
            msgs = self._build_llm_messages_proactive(schema_card, context_capsule)
        else:
            msgs = self._build_llm_messages_reactive(schema_card, context_capsule,
                                                     planner_needs or {}, self._current_ws_summary())
        try:
            obj = self._chat_json(msgs, temperature=0.2, max_tokens=300, retries=1)
            sql    = (obj.get("sql") or "").strip()
            params = obj.get("params") or {}
            purpose= (obj.get("purpose") or "")[:80]
            if not sql.lower().startswith("select"):
                raise ValueError("LLM did not return a SELECT")
            return sql, params, purpose

        except Exception:
            # NEW: reuse the same fallback for both errors and disabled LLM
            self.get_logger().info(f"\nERROR BROKER\n")
            return self._fallback_sql_from_capsule(context_capsule)


    # NEW: shared fallback logic used on errors *and* when LLM is disabled
    def _fallback_sql_from_capsule(self, context_capsule: dict) -> tuple[str, dict, str]:
        """
        Produce a default SQL query when the LLM is disabled or fails.

        Prefer a single hinted object, otherwise return a small shortlist.
        """
        hints = (context_capsule.get("trigger") or {}).get("hints") or {}
        oid = hints.get("object_id")
        if oid:
            return (
                "SELECT * FROM vw_object_sheet WHERE node_id=:object_id LIMIT 1",
                {"object_id": oid},
                "fallback object sheet"
            )

        return (
            "SELECT node_id, best_zone, best_rssi, in_basket, disposed_to "
            "FROM vw_object_sheet WHERE disposed_to='none' "
            "ORDER BY best_ts DESC LIMIT 5",
            {},
            "fallback shortlist"
        )


    # ---- Turn LLM SQL into facts ----
    def _llm_sql_to_facts(self, *, proactive: bool, needs: Optional[dict] = None) -> dict:
        ws_id = self._ws_id()
        ent = self._ws.setdefault(ws_id, {"hashes": set(), "iters": 0})
        if not proactive and ent["iters"] >= self.iteration_limit:
            return {"mode": "reactive", "done": True, "reason": "iteration_limit"}

        schema = self._schema_card()
        capsule = self._context_capsule()

        # If LLM disabled, directly use fallback SQL
        if not getattr(self, "llm_enabled", True):
            sql, params, purpose = self._fallback_sql_from_capsule(capsule)
            purpose = (purpose + " (llm_disabled)").strip()
        else:
            sql, params, purpose = self._llm_plan_sql(
                proactive=proactive,
                schema_card=schema,
                context_capsule=capsule,
                planner_needs=needs,
            )

        err = self._validate_sql_readonly(sql)
        if err:
            self.get_logger().warn(f"[broker] SQL validation failed ({err}); falling back.")
            sql, params, purpose = self._fallback_sql_from_capsule(capsule)

        try:
            cols, rows, truncated, ms = self._exec_sql_safely(
                sql, params, self.sql_max_rows, self.sql_max_bytes, self.sql_timeout_ms
            )
        except sqlite3.Error as e:
            # Last line of defence: even if validation missed something, don't die.
            self.get_logger().warn(f"[broker] SQL exec failed ({e}); using fallback.")
            sql, params, purpose = self._fallback_sql_from_capsule(capsule)
            cols, rows, truncated, ms = self._exec_sql_safely(
                sql, params, self.sql_max_rows, self.sql_max_bytes, self.sql_timeout_ms
            )

        changed = self._ws_changed(ws_id, sql, rows)
        if changed:
            self._ws_add(ws_id, sql, rows)
        if not proactive:
            ent["iters"] += 1

        pack = {
            "mode": ("proactive" if proactive else "reactive"),
            "ws_id": ws_id,
            "rationale": purpose,
            "table": {
                "columns": cols,
                "rows": rows,
                "truncated": truncated,
                "changed": changed,
            },
            "sql_meta": {"sql": sql, "params": params, "ms": ms},
        }
        return pack


    # ------------------------------ Planner needs (reactive loop) ------------------------------
    def _on_planner_needs(self, msg: StringMsg):
        # Store last needs (structured or unstructured). We don't trust schemas here; just keep JSON.
        self.get_logger().info(f"got needs message: {msg}")
        try:
            self._last_needs = json.loads(msg.data) if msg.data else {}
        except Exception:
            self._last_needs = {"open": [msg.data]}
        # Optionally: auto-run a reactive turn on needs arrival
        try:
        
            self._publish_context_capsule()
        
            pack = self._llm_sql_to_facts(proactive=False, needs=self._last_needs)
            self.get_logger().info(f"reactive pack: {json.dumps(pack)[:500]}")
            if pack.get("table"):
                self.get_logger().info(f"publishing delta: {pack}")
                self.pub_delta.publish(StringMsg(data=json.dumps(pack)))
                self._emit_sql_debug(pack)
        except Exception as e:
            self.get_logger().warn(f"reactive run failed: {e}")

    # ------------------------------ Services: run_initial / run_more ------------------------------
    def _srv_run_initial(self, req, res):
        try:
        
            self._publish_context_capsule()
            pack = self._llm_sql_to_facts(proactive=True)
            self.pub_facts.publish(StringMsg(data=json.dumps(pack)))
            self._emit_sql_debug(pack)
            res.success, res.message = True, "ok"
        except Exception as e:
            res.success, res.message = False, str(e)
        return res

    def _srv_run_more(self, req, res):
        try:
            needs = getattr(self, "_last_needs", None)
            
            self._publish_context_capsule()
            
            pack = self._llm_sql_to_facts(proactive=False, needs=needs)
            if pack.get("table", {}).get("changed", False):
                self.pub_delta.publish(StringMsg(data=json.dumps(pack)))
                self._emit_sql_debug(pack)
            res.success, res.message = True, "ok"
        except Exception as e:
            res.success, res.message = False, str(e)
        return res

    # ------------------------------ Legacy simple services ------------------------------
    def _srv_dump_db(self, req, res):
        res.success = True
        res.message = self.db_path
        return res

    def _srv_query_nodes_summary(self, req, res):
        rows = self.conn.execute("SELECT * FROM vw_bt_nodes_summary").fetchall()
        cur = self.conn.execute("SELECT * FROM vw_bt_nodes_summary LIMIT 1")
        colnames = [d[0] for d in cur.description] if cur.description else []
        cur.close()
        payload = []
        for row in rows:
            obj = {}
            for i, k in enumerate(colnames):
                obj[k] = row[i]
            payload.append(obj)
        res.success = True
        res.message = json.dumps(payload)
        return res

    def _srv_query_agent_labels(self, req, res):
        rows = self.conn.execute("""
            SELECT agent_id, node_id, contaminated, probability, updated_ts
            FROM agent_node_labels
        """).fetchall()
        payload = [
            dict(agent_id=r[0], node_id=r[1], contaminated=bool(r[2]),
                 probability=float(r[3]), updated_ts=float(r[4]))
            for r in rows
        ]
        res.success = True
        res.message = json.dumps(payload)
        return res

    # ------------------------------ TaskState aggregation ------------------------------

    def _compute_task_state(self) -> dict:
        """
        Build a compact TaskState snapshot from the DB + agent status + recent events.

        Shape (approx):

          {
            "ts": ...,
            "robot": { ... },
            "agents": { "robot": {...}, "human_a": {...}, "human_b": {...} },
            "zones": {
              "A": { "total":..., "pending":..., "in_basket":..., "delivered_clean":..., "delivered_contaminated":..., "progress": ... },
              "B": {...},
              "unknown": {...}
            },
            "bottlenecks": [...],
            "robot_objective": "execute:<skill>" or null,
            "subgoals": { "collect_zone_A": {...}, ... },
            "time": { "elapsed_sec": ..., "remaining_estimate_sec": null },
            "bt_summary": {
              "by_zone": { "A": {...}, "B": {...}, "unknown": {...} },
              "objects": { "CNode001": {...}, ... }
            },
            "high_level_objective": {
              "kind": "unknown",
              "note": "planner may override / refine this"
            }
          }
        """
        now = time.time()

        # --- Agents (robot + humans) from agent_status ---
        agents: Dict[str, dict] = {}
        try:
            rows = self.conn.execute(
                "SELECT agent_id, zone, x, y, ts FROM agent_status"
            ).fetchall()
            for agent_id, zone, x, y, ts_row in rows:
                agents[agent_id] = {
                    "zone": zone,
                    "x": x,
                    "y": y,
                    "ts": ts_row,
                }
        except Exception as e:
            self.get_logger().warn(f"[task_state] failed to read agent_status: {e}")

        robot_agent = agents.get("robot", {})

        # --- Zone-level progress from vw_object_sheet ---
        zones: Dict[str, dict] = {
            "A": {
                "total": 0,
                "pending": 0,
                "in_basket": 0,
                "delivered_clean": 0,
                "delivered_contaminated": 0,
                "progress": None,
            },
            "B": {
                "total": 0,
                "pending": 0,
                "in_basket": 0,
                "delivered_clean": 0,
                "delivered_contaminated": 0,
                "progress": None,
            },
            "unknown": {
                "total": 0,
                "pending": 0,
                "in_basket": 0,
                "delivered_clean": 0,
                "delivered_contaminated": 0,
                "progress": None,
            },
        }

        try:
            cur = self.conn.execute(
                """
                SELECT
                  COALESCE(best_zone, current_zone, 'unknown') AS zone,
                  disposed_to,
                  in_basket
                FROM vw_object_sheet
                """
            )
            for zone, disposed_to, in_basket in cur:
                z = zone if zone in zones else "unknown"
                st = zones[z]
                st["total"] += 1

                if disposed_to == "none":
                    # Still somewhere in the environment
                    st["pending"] += 1
                    if in_basket:
                        st["in_basket"] += 1
                elif disposed_to == "clean_bin":
                    st["delivered_clean"] += 1
                elif disposed_to == "contaminated_bin":
                    st["delivered_contaminated"] += 1
        except Exception as e:
            self.get_logger().warn(f"[task_state] failed to read vw_object_sheet: {e}")

        # Progress fraction per zone
        for z, st in zones.items():
            total = st["total"]
            done = st["delivered_clean"] + st["delivered_contaminated"]
            st["progress"] = (float(done) / float(total)) if total > 0 else None

        # --- BT summary: by_zone + per-object from vw_object_sheet ---
        bt_by_zone: Dict[str, dict] = {
            "A": {"count": 0},
            "B": {"count": 0},
            "unknown": {"count": 0},
        }
        objects: Dict[str, dict] = {}

        try:
            cur2 = self.conn.execute(
                """
                SELECT
                  node_id,
                  in_basket,
                  disposed_to,
                  current_rssi,
                  current_zone,
                  best_rssi,
                  best_zone
                FROM vw_object_sheet
                """
            )
            for (
                node_id,
                in_basket,
                disposed_to,
                current_rssi,
                current_zone,
                best_rssi,
                best_zone,
            ) in cur2:
                zone = best_zone or current_zone or "unknown"
                if zone not in bt_by_zone:
                    zone = "unknown"
                bt_by_zone[zone]["count"] = bt_by_zone[zone].get("count", 0) + 1

                objects[node_id] = {
                    "zone": zone,
                    "in_basket": bool(in_basket),
                    "disposed_to": disposed_to,
                    "current_rssi": current_rssi,
                    "best_rssi": best_rssi,
                }
        except Exception as e:
            self.get_logger().warn(f"[task_state] failed to build bt_summary: {e}")

        # --- Bottlenecks (heuristic) ---
        bottlenecks: List[str] = []
        for z, st in zones.items():
            # backlog: lots of pending items and no completed ones
            if st["pending"] > 0 and (st["delivered_clean"] + st["delivered_contaminated"] == 0):
                if st["pending"] >= 3:
                    bottlenecks.append(f"zone_{z}_backlog")

            # "bins" almost full ≈ many in_basket items in this zone
            if st["in_basket"] >= 5:
                bottlenecks.append(f"bin_{z}_almost_full")

        # --- Subgoal status per zone ---
        subgoals: Dict[str, dict] = {}
        for z, st in zones.items():
            label = f"collect_zone_{z}"
            if st["total"] == 0:
                status = "pending"  # nothing there yet, but conceptually not done
            elif st["pending"] == 0:
                status = "done"
            elif st["pending"] == st["total"]:
                status = "pending"
            else:
                status = "in_progress"
            subgoals[label] = {
                "zone": z,
                "status": status,
                "progress": st["progress"],
            }

        # --- Time info ---
        elapsed = now - self._start_time
        time_info = {
            "elapsed_sec": elapsed,
            "remaining_estimate_sec": None,  # planner can override
        }

        # --- Robot objective (from skill events) ---
        robot_objective = self._robot_objective

        task_state = {
            "ts": now,
            "robot": robot_agent,
            "agents": agents,
            "zones": zones,
            "bottlenecks": bottlenecks,
            "robot_objective": robot_objective,
            "subgoals": subgoals,
            "time": time_info,
            "bt_summary": {
                "by_zone": bt_by_zone,
                "objects": objects,
            },
            "high_level_objective": {
                "kind": "unknown",
                "note": "planner may override / refine this",
            },
        }
        return task_state

    def _tick_task_state(self):
        """
        Periodic publisher for /task_state.
        """
        try:
            state = self._compute_task_state()
            self.pub_task_state.publish(StringMsg(data=json.dumps(state)))
        except Exception as e:
            self.get_logger().warn(f"[task_state] tick failed: {e}")


    def _optimizer_tick(self):
        """
        Periodic tick: pull /boxes/state and /time from the box server.

        If the world state fingerprint changed since last tick, recompute plan.
        """
        if self._optimizer_running or not self.optimizer_enabled:
            return

        try:
            base = self.optimizer_base_url.rstrip("/")
            url_state = base + "/boxes/state"
            url_time  = base + "/time"

            r_state = requests.get(url_state, timeout=self.req_timeout)
            r_time  = requests.get(url_time, timeout=self.req_timeout)

            if r_state.status_code != 200 or r_time.status_code != 200:
                self.get_logger().warn(
                    f"[optimizer] box server unavailable: "
                    f"state={r_state.status_code}, time={r_time.status_code}"
                )
                return

            boxes_state = r_state.json()   # list[BoxState-like dicts]
            time_resp   = r_time.json()    # {"server_time": float}
            current_time = float(time_resp.get("server_time", 0.0))

        except Exception as e:
            self.get_logger().warn(f"[optimizer] failed to contact box server: {e}")
            return

        # Compute a cheap fingerprint of the *world state* that is
        # insensitive to time (server_time, deadlines).
        try:
            canonical_boxes = []
            for b in boxes_state:
                canonical_boxes.append(
                    {
                        # identity / geometry
                        "box_id": b.get("box_id"),
                        "x": b.get("x"),
                        "y": b.get("y"),

                        # sensing + disposal state
                        # (we keep sense_results as-is so new completed senses
                        #  or detections will trigger a replan)
                        "sense_results": b.get("sense_results") or [],
                        "disposed_X": bool(b.get("disposed_X", False)),
                        "disposed_Y": bool(b.get("disposed_Y", False)),
                    }
                )

            # Sort to make hashing order-independent
            canonical_boxes.sort(key=lambda bb: bb["box_id"])

            fp_str = json.dumps(canonical_boxes, sort_keys=True)
            fp = hashlib.sha256(fp_str.encode("utf-8")).hexdigest()
        except Exception as e:
            self.get_logger().warn(f"[optimizer] fingerprint failed: {e}")
            return



        if fp == self._last_boxes_fp:
            # No relevant change in box world since last plan
            return
            
        # Optional: much smaller log
        self.get_logger().info(f"[optimizer] Plan fingerprint={fp}")

        self.get_logger().info(f"[optimizer] Running optimizer")
        # Mark and run optimizer in background to avoid blocking callbacks
        self._last_boxes_fp = fp
        self._optimizer_running = True

        threading.Thread(
            target=self._run_optimizer_thread,
            args=(boxes_state, current_time),
            daemon=True,
        ).start()

    def _run_optimizer_thread(self, boxes_state: list, current_time: float):
        """
        Background worker that:
          1) syncs box-server world into broker DB
          2) builds optimizer inputs
          3) runs Gurobi and publishes a plan
        """
        try:
            # 1) sync DB with latest box server state
            self._sync_box_state_from_server(boxes_state)

            # 2) build optimizer inputs
            agents = self._build_agents_for_optimizer()
            boxes, box_positions = self._build_boxes_for_optimizer(boxes_state)
            if not agents or not boxes:
                self.get_logger().info(
                    "[optimizer] no agents or boxes available; skipping"
                )
                return

            horizon = self.optimizer_horizon_sec
            agent_positions = self._snapshot_agent_positions()

            def travel_time_fn(agent_id: str, box_id: int) -> float:
                ax, ay = agent_positions.get(agent_id, (None, None))
                bx, by = box_positions.get(box_id, (None, None))
                if ax is None or ay is None or bx is None or by is None:
                    return 0.0

                dx = ax - bx
                dy = ay - by
                dist = (dx * dx + dy * dy) ** 0.5

                if agent_id == "robot":
                    speed = self.optimizer_speed_robot
                elif agent_id == "human_a":
                    speed = self.optimizer_speed_human_a
                elif agent_id == "human_b":
                    speed = self.optimizer_speed_human_b
                else:
                    speed = 1.0

                if speed <= 0.0:
                    return 0.0
                return dist / speed

            # 3) run Gurobi
            plan = plan_assignments_gurobi(
                agents=agents,
                boxes=boxes,
                current_time=current_time,
                horizon=horizon,
                travel_time_fn=travel_time_fn,
            )

            self._last_plan = plan
            self._publish_optimizer_plan(plan, current_time, box_positions)

        except Exception as e:
            self.get_logger().warn(f"[optimizer] optimization failed: {e}")
        finally:
            self._optimizer_running = False


    def _build_agents_for_optimizer(self) -> List[AgentState]:
        """
        Build AgentState list for Gurobi.
        - human_a can only sense X
        - human_b can only sense Y
        - robot can sense both
        Detection quality (present/absent) is pulled from box server /agents/params.
        """
        agents_cfg = self._agent_det_agents or {}
        default_cfg = self._agent_det_default or {
            "X": {"present": 0.8, "absent": 0.2},
            "Y": {"present": 0.8, "absent": 0.2},
        }

        def get_det(agent_id: str):
            cfg = agents_cfg.get(agent_id, default_cfg)
            x_cfg = cfg["X"]
            y_cfg = cfg["Y"]
            return (
                float(x_cfg["present"]),
                float(x_cfg["absent"]),
                float(y_cfg["present"]),
                float(y_cfg["absent"]),
            )

        # human_a
        hA_pX, hA_aX, hA_pY, hA_aY = get_det("human_a")
        # human_b
        hB_pX, hB_aX, hB_pY, hB_aY = get_det("human_b")
        # robot
        r_pX, r_aX, r_pY, r_aY = get_det("robot")

        agents = [
            AgentState(
                agent_id="human_a",
                max_time=self.optimizer_time_human_a,
                can_sense_X=True,
                can_sense_Y=False,
                detect_present_X=hA_pX,
                detect_absent_X=hA_aX,
                detect_present_Y=hA_pY,
                detect_absent_Y=hA_aY,
            ),
            AgentState(
                agent_id="human_b",
                max_time=self.optimizer_time_human_b,
                can_sense_X=False,
                can_sense_Y=True,
                detect_present_X=hB_pX,
                detect_absent_X=hB_aX,
                detect_present_Y=hB_pY,
                detect_absent_Y=hB_aY,
            ),
            AgentState(
                agent_id="robot",
                max_time=self.optimizer_time_robot,
                can_sense_X=True,
                can_sense_Y=True,
                detect_present_X=r_pX,
                detect_absent_X=r_aX,
                detect_present_Y=r_pY,
                detect_absent_Y=r_aY,
            ),
        ]
        return agents


    def _snapshot_agent_positions(self) -> Dict[str, Tuple[float, float]]:
        """
        Read agent_status and return {agent_id: (x,y)} for travel_time_fn.
        """
        positions: Dict[str, Tuple[float, float]] = {}
        try:
            rows = self.conn.execute(
                "SELECT agent_id, x, y FROM agent_status"
            ).fetchall()
            for agent_id, x, y in rows:
                if x is None or y is None:
                    continue
                positions[str(agent_id)] = (float(x), float(y))
        except Exception as e:
            self.get_logger().warn(f"[optimizer] failed to read agent_status: {e}")
        return positions

    def _build_boxes_for_optimizer(
        self,
        boxes_state: list,
    ) -> Tuple[List[BoxInfo], Dict[int, Tuple[float, float]]]:
        """
        Convert /boxes/state payload into List[BoxInfo] + box positions.

        Heuristics:
        - p_true_X/Y: based on last detection(s) per property (very simple).
        - info_X/Y: grows with #completed senses for that property (0..1).
        - already_sensed: True if an agent has a completed sense for (box, prop).
        """
        boxes: List[BoxInfo] = []
        positions: Dict[int, Tuple[float, float]] = {}

        for b in boxes_state:
            try:
                box_id = int(b["box_id"])
            except Exception:
                continue

            deadline = float(b.get("deadline", 1e9))
            x = float(b.get("x", 0.0))
            y = float(b.get("y", 0.0))
            positions[box_id] = (x, y)

            # If your /boxes/state does NOT yet include these,
            # you can fall back to constants or broker params.
            sense_time_X = float(b.get("sense_time_X", 3.0))
            sense_time_Y = float(b.get("sense_time_Y", 3.0))
            dispose_time_X = float(b.get("dispose_time_X", 4.0))
            dispose_time_Y = float(b.get("dispose_time_Y", 4.0))

            disposed_X = bool(b.get("disposed_X", False))
            disposed_Y = bool(b.get("disposed_Y", False))

            # --- Build already_sensed + crude beliefs from sense_results ---
            already_sensed: Dict[str, Dict[str, bool]] = {}
            sense_results = b.get("sense_results") or []

            # Track last detection for each (prop)
            last_det_X = None
            last_det_Y = None
            count_X = 0
            count_Y = 0

            for sr in sense_results:
                agent_id = str(sr.get("agent_id") or "")
                prop = sr.get("property")
                status = sr.get("status")
                detected = sr.get("detected")

                if prop not in ("X", "Y") or not agent_id:
                    continue

                if status == "completed":
                    # mark already_sensed[agent_id][prop] = True
                    amap = already_sensed.setdefault(agent_id, {})
                    amap[prop] = True

                    if prop == "X":
                        count_X += 1
                        last_det_X = detected
                    else:
                        count_Y += 1
                        last_det_Y = detected

            # Very crude belief heuristics:
            #   - prior 0.5
            #   - if we've seen a completed detection True, bump to 0.8
            #   - if only detections False, drop to 0.2
            def belief_from_counts(last_det, count):
                if count == 0:
                    return 0.5
                if last_det is True:
                    return 0.8
                if last_det is False:
                    return 0.2
                return 0.5

            p_true_X = belief_from_counts(last_det_X, count_X)
            p_true_Y = belief_from_counts(last_det_Y, count_Y)

            # info_X/Y: just saturating with #completed senses
            info_X = min(1.0, 0.3 * count_X)
            info_Y = min(1.0, 0.3 * count_Y)

            box_info = BoxInfo(
                box_id=box_id,
                deadline=deadline,
                sense_time_X=sense_time_X,
                sense_time_Y=sense_time_Y,
                dispose_time_X=dispose_time_X,
                dispose_time_Y=dispose_time_Y,
                p_true_X=p_true_X,
                p_true_Y=p_true_Y,
                disposed_X=disposed_X,
                disposed_Y=disposed_Y,
                info_X=info_X,
                info_Y=info_Y,
                already_sensed=already_sensed,
            )
            boxes.append(box_info)

        return boxes, positions

    def _publish_optimizer_plan(
        self,
        plan: dict,
        current_time: float,
        box_positions: Dict[int, Tuple[float, float]],
    ):
        """
        Publish the latest plan to /optimizer/plan as JSON.

        Shape:
          {
            "ts": ...,
            "current_time": ...,
            "agents": {
              "<agent_id>": [
                {"box_id": int, "property": "X"|"Y", "kind": "sense"|"dispose"},
                ...
              ],
              ...
            },
            "nodes": {
              "<node_id>": {"box_id": int, "x": float, "y": float},
              ...
            },
            "agent_positions": {
              "<agent_id>": {"x": float, "y": float},
              ...
            }
          }
        """
        try:
            # Per-agent action lists (already built by plan_assignments_gurobi wrapper)
            agents_block = {
                aid: [
                    {"box_id": box_id, "property": prop, "kind": kind}
                    for (box_id, prop, kind) in actions
                ]
                for aid, actions in plan.items()
            }

            # Map box_id -> node_id + location
            nodes_block = {
                f"CNode{int(box_id)}": {
                    "box_id": int(box_id),
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                }
                for box_id, pos in box_positions.items()
            }


            payload = {
                "ts": time.time(),
                "current_time": current_time,
                "agents": agents_block,
                "nodes": nodes_block,
            }

            msg = StringMsg(data=json.dumps(payload))
            self.pub_opt_plan.publish(msg)
            self.get_logger().info(
                f"[optimizer] published plan with "
                f"{sum(len(v) for v in agents_block.values())} actions across "
                f"{len(agents_block)} agents."
            )
        except Exception as e:
            self.get_logger().warn(f"[optimizer] failed to publish plan: {e}")



    # ------------------------------ Shutdown ------------------------------
    def destroy_node(self):
        try:
            self.conn.close()
        except Exception:
            pass
        super().destroy_node()


# ------------------------------ Main ------------------------------

def main():
    rclpy.init()
    node = BrokerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

