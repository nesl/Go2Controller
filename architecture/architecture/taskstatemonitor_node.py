#!/usr/bin/env python3
import os, json, math, sqlite3, threading, time, re, hashlib
from typing import Optional, Tuple, Dict, Set, List, Any
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from std_msgs.msg import String as StringMsg
from std_srvs.srv import Trigger

from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException
import requests

from openai import OpenAI, APITimeoutError
from jsonschema import validate, ValidationError

import yaml  # NEW
from rclpy.parameter import Parameter          # NEW
from rcl_interfaces.msg import SetParametersResult  # NEW
from groq import Groq

from .optimizer_client import AgentState, BoxInfo, plan_assignments_gurobi

from .optimizer_client import (
    build_plan_from_llm_agents_plan,
    evaluate_candidate_plan,
    Plan,
    PlannerWeights,
    extend_plan_with_prefix,
)

from .broker_mediation import BrokerMediationMixin

from .plan_mediator import (
    PlanMediator,
    MediationLLMConfig,
    MediationState,
    MediationObjectiveMetrics,
    MediationSocialContext,
    MediationInteractionContext,
    MediationTurn,
)


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

CHAT_REPLY_SCHEMA = {
    "type": "object",
    "required": ["robot_utterance", "should_reply"],
    "properties": {
        "robot_utterance": {"type": "string"},
        "should_reply": {"type": "boolean"},
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

class BrokerNode(Node, BrokerMediationMixin):
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
        self.declare_parameter("optimizer_speed_robot_mps", 0.2)
        self.declare_parameter("optimizer_speed_human_a_mps", 0.2)
        self.declare_parameter("optimizer_speed_human_b_mps", 0.2)


        # When to consider a new best/current as “meaningful change” for refresh
        self.declare_parameter('contam_best_delta_db', 5)          # recheck if best improved by ≥5 dB
        self.declare_parameter('contam_stale_sec', 900.0)          # or if label older than 15 min

        # LLM + SQL budgets
        self.declare_parameter('sql_max_rows', 64)
        self.declare_parameter('sql_max_bytes', 20000)
        self.declare_parameter('sql_timeout_ms', 120)
        self.declare_parameter('iteration_limit', 2)
        self.declare_parameter('pull_limit', 2)

        self.declare_parameter("plan_accept_policy", "normal")

        self.declare_parameter('sim_mode', False)

        # Allowed SQL objects (read-only)
        self.declare_parameter('allowed_objects_json', json.dumps([
            "bt_nodes","nodes_state",
            "agent_node_labels",
             "box_env_state"
        ]))


        self.declare_parameter("human_a_name", "Sam")
        self.declare_parameter("human_b_name", "Jacob")


        # Optional: mock LLM for offline dev (pass JSON {"sql": "...", "params": {...}, "purpose": "..."} in param)
        self.declare_parameter('llm_mock_json', '')
        
        self.declare_parameter("model", "gpt-4.1-mini")

        self.model = self.get_parameter("model").get_parameter_value().string_value

        
        human_a_name = self.get_parameter("human_a_name").get_parameter_value().string_value
        human_b_name = self.get_parameter("human_b_name").get_parameter_value().string_value

        self.agent_id_to_human_name = {
            "robot": "robot",
            "human_a": human_a_name,
            "human_b": human_b_name,
        }
        self.human_name_to_agent_id = {v: k for k, v in self.agent_id_to_human_name.items()}
        
        
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

        # valid: "normal", "always_accept"
        self.plan_accept_policy = (
            self.get_parameter("plan_accept_policy").get_parameter_value().string_value
        )

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
        
        self.declare_parameter("human_agent_id", "human_a")  # or whatever label you use
        self.human_agent_id = (
            self.get_parameter("human_agent_id")
            .get_parameter_value()
            .string_value
        )
        
        # --- Event-trace summarizer parameters ---
        self.declare_parameter("event_summary_enabled", False)
        self.declare_parameter("event_summary_model", "gpt-5-mini")  # fast, cheap, small context
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

        self.sim_mode = int(
            self.get_parameter("sim_mode")
            .get_parameter_value()
            .bool_value
        )

        # Async running event summary state
        self._event_summary_text: Optional[str] = None      # last full running summary
        self._event_summary_ts: Optional[float] = None      # ts of last event included
        self._unsummarized_events: List[dict] = []          # events since last summary
        self._events_since_summary: int = 0                 # counter since last summary
        self._event_summary_running: bool = False           # background worker in flight?
        self._event_summary_lock = threading.Lock()

        self.human_agent_ids = ["human_a", "human_b"]

        # simple EMA of broker LLM latency (ms)
        self._llm_lat_ema_ms: Optional[float] = None
        self._llm_lat_alpha: float = 0.3


        # --- NEW: nested current action tracking (robot skill stack) ---
        self._action_stack: List[dict] = []   # stack of {"skill", "ts", "step_idx", "inner_ctx", "kind"}
        self._action_stack_lock = threading.Lock()


        # In BrokerNode.__init__:
        self._plan_mediator = PlanMediator(
            MediationLLMConfig(
                model_name="gpt-5-mini",
                llm_call=self._mediate_llm_call,  # small wrapper around self._chat_json
            )
        )
        self._mediation_sessions = {}  # req_id -> MediationState
        self._active_mediation_id: Optional[str] = None

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
        self._profiles = {hid: None for hid in self.human_agent_ids}  # if you publish HDT, wire subs below
        self._event_trace = deque(maxlen=40)       # compact recent events
        self._current_trigger = None               # {"type": "...", "hints": {...}}
        self._ws = {}                               # ws_id -> {"hashes": set(), "iters": int}

        self._init_mediation_tts()
        # in BrokerNode.__init__ after self._init_mediation_tts()
        self._mediation_watchdog_timer = self.create_timer(0.5, self._mediation_watchdog_tick)


        self._last_published_summary_fp = None

        # In BrokerMediationMixin.__init__ or some init method:
        self._profile_msg_counts = {hid: 0 for hid in self.human_agent_ids}

        self.profile_reflection_every = 5     # e.g. every 10 utterances per human


        # NEW: robot objective & timing for task state
        self._robot_objective: Optional[str] = None
        self._start_time = time.time()

        self._last_server_time: Optional[float] = None



        # NEW: last known robot zone (from event-layer)
        self._last_robot_zone: Optional[str] = None

        self._agent_det_agents, self._agent_det_default = self._load_agent_detection_params()

        # ------------ ROS I/O ------------
        # Events
        self.sub_basic = self.create_subscription(StringMsg, self.bus_topic, self._on_basic_event, 1000)
        self.sub_comp  = self.create_subscription(StringMsg, "/events/composite", self._on_comp_event, 500)

        # Allow changing llm_model via /broker_node/set_parameters
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # Planner needs (reactive loop)

        # Optional DT profiles (if available)
        # self.create_subscription(StringMsg, "/digital_twin/profile/H1", self._on_profile_h1, 10)
        # self.create_subscription(StringMsg, "/digital_twin/profile/H2", self._on_profile_h2, 10)

        self.pub_capsule = self.create_publisher(StringMsg, "/broker/context_capsule", 10)


        # Services

        # Background contamination worker

        # --- Mediation control publisher (for EventLayer) ---
        self.pub_mediation_ctrl = self.create_publisher(
            StringMsg, "/mediation/control", 10
        )


        if self.optimizer_enabled:
            # Small polling period; can tune (e.g. 0.5–2.0 s)
            self.create_timer(1.0, self._optimizer_tick)


        self.get_logger().info(
            f"broker_node up | db={self.db_path} bus={self.bus_topic} rule={self.bt_rule_id} "
            f"target_frame={self.target_frame} zone_split_x={self.zone_split_x} server={self.server_url} "
            f"enable_server={self.enable_server}"
        )


    def _always_accept(self) -> bool:
        return (getattr(self, "plan_accept_policy", "normal") == "always_accept")


    def _swap_str(self, s: str, mapping: dict) -> str:
        if not isinstance(s, str) or not mapping:
            return s
        out = s
        for a in sorted(mapping.keys(), key=len, reverse=True):
            out = re.sub(rf"\b{re.escape(a)}\b", mapping[a], out)
        return out


    def _swap_json(self, obj: Any, mapping: dict) -> Any:
        """
        Recursively replace BOTH dict keys and values.
        This is ONLY used at the LLM boundary.
        """
        if isinstance(obj, dict):
            return {
                self._swap_str(k, mapping): self._swap_json(v, mapping)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [self._swap_json(x, mapping) for x in obj]
        if isinstance(obj, str):
            return self._swap_str(obj, mapping)
        return obj


    def _translate_messages(self, messages: list, mapping: dict) -> list:
        """
        Translate message.content (JSON or raw text).
        """
        out = []
        for m in messages:
            mm = dict(m)
            c = mm.get("content")
            if isinstance(c, str):
                try:
                    parsed = json.loads(c)
                    parsed2 = self._swap_json(parsed, mapping)
                    mm["content"] = json.dumps(parsed2, ensure_ascii=False)
                except Exception:
                    mm["content"] = self._swap_str(c, mapping)
            out.append(mm)
        return out



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

            if p.name == "plan_accept_policy":
                self.plan_accept_policy = str(p.value)



        return SetParametersResult(successful=True, reason="ok")




    def _publish_context_capsule(self, summary_only: bool = False):
        cap = self._context_capsule(summary_only=summary_only)
        self.pub_capsule.publish(StringMsg(data=json.dumps(cap)))



    # ------------------------------ Schema ------------------------------
    def _ensure_schema(self):
        cur = self.conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")

        # Drop-and-create (for clean dev boots)

        for trg in [
            "trg_best_on_current_insert_init",
            "trg_best_on_current_insert_if_better"
        ]:
            cur.execute(f"DROP TRIGGER IF EXISTS {trg};")
        for tbl in [
            "contamination_records", "obj_measurements",
            "nodes_state", "bt_nodes",
            "agent_status", "agent_locations", "agent_node_labels", "robot_exec_evals"
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
        # Plan proposals (LLM speech, optimizer, etc.)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS plan_proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                proposer_id TEXT,
                source TEXT,
                req_id TEXT,
                adopted INTEGER NOT NULL DEFAULT 0,
                better_than_current INTEGER NOT NULL DEFAULT 0,
                score_optimal REAL,
                score_candidate REAL,
                suboptimal_pct REAL
            );
        """)


        # Robot skill execution evals (mainly for sense/dispose actions)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS robot_exec_evals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                agent_id TEXT,
                skill TEXT,
                box_id INTEGER,
                property TEXT,
                kind TEXT,      -- 'sense' or 'dispose'
                status TEXT,
                detected INTEGER,
                probability REAL,
                deadline REAL,
                completed_at REAL,
                fulfilled INTEGER   -- 1 if confirmed in box server state, 0 if not, NULL if unknown
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_robot_exec_ts ON robot_exec_evals(ts);")


        
        cur.execute("CREATE INDEX idx_agent_loc_agent_ts ON agent_locations(agent_id, ts);")

        # Views


       
        
        self.conn.commit()
        cur.close()
        self.get_logger().info(f"Broker schema ready at {self.db_path}")

    def _plan_fingerprint(self, plan: dict) -> str:
        """
        Stable fingerprint so we don't repeatedly announce the same plan.
        """
        try:
            # plan is dict[agent_id] -> list[(box_id, prop, kind)]
            plan_norm = {}
            for aid, actions in (plan or {}).items():
                norm_actions = []
                for a in actions or []:
                    try:
                        box_id, prop, kind = a
                        norm_actions.append([int(box_id), str(prop), str(kind)])
                    except Exception:
                        continue
                norm_actions.sort(key=lambda x: (x[2], x[0], x[1]))  # kind, box_id, prop
                plan_norm[str(aid)] = norm_actions

            s = json.dumps(plan_norm, sort_keys=True)
            return hashlib.sha256(s.encode("utf-8")).hexdigest()
        except Exception:
            return str(time.time())


    def _format_one_action(self, a: dict) -> str:
        """
        a = {"box_id": int, "property": "X"|"Y", "kind": "sense"|"dispose"}
        """
        kind = (a.get("kind") or "").strip()
        box_id = a.get("box_id")
        prop = (a.get("property") or "").strip()

        if kind == "sense":
            return f"sense box {box_id} for {prop}"
        if kind == "dispose":
            return f"dispose box {box_id} ({prop})"
        return f"{kind} box {box_id} ({prop})"


    def _announce_idle_plan(self, plan: dict, current_time: float):
        """
        Called after a trigger_idle optimizer publish.
        Announces what the robot will do + proposes human assignments.
        Debounced to avoid spam.
        """
        # --- debounce identical plans ---
        fp = self._plan_fingerprint(plan)
        now = time.time()

        # small cooldown AND identical-plan suppression
        cooldown_sec = 4.0
        last_t = getattr(self, "_last_idle_announce_ts", None)
        last_fp = getattr(self, "_last_idle_announce_fp", None)

        if last_t is not None and (now - float(last_t)) < cooldown_sec and fp == last_fp:
            return

        self._last_idle_announce_ts = now
        self._last_idle_announce_fp = fp

        # --- turn plan into the published JSON-ish structure (same as _publish_optimizer_plan) ---
        agents_block = {
            aid: [
                {"box_id": int(box_id), "property": prop, "kind": kind}
                for (box_id, prop, kind) in (actions or [])
            ]
            for aid, actions in (plan or {}).items()
        }

        robot_actions = agents_block.get("robot") or []
        human_a_actions = agents_block.get("human_a") or []
        human_b_actions = agents_block.get("human_b") or []

        # pick “next” action for each (keep it short)
        robot_next = robot_actions[0] if robot_actions else None
        ha_next = human_a_actions[0] if human_a_actions else None
        hb_next = human_b_actions[0] if human_b_actions else None

        # resolve display names
        ha_name = self.agent_id_to_human_name.get("human_a", "Human A")
        hb_name = self.agent_id_to_human_name.get("human_b", "Human B")

        parts = []

        # 1) announce robot intent
        if robot_next:
            parts.append(f"I’m going to {self._format_one_action(robot_next)}.")
        else:
            parts.append("I don’t have a robot action queued right now.")

        # 2) propose human parts
        proposals = []
        if ha_next:
            proposals.append(f"{ha_name}, could you {self._format_one_action(ha_next)}?")
        if hb_next:
            proposals.append(f"{hb_name}, could you {self._format_one_action(hb_next)}?")

        if proposals:
            parts.append(" ".join(proposals))

        # Optional: mention that optimizer produced more items (without dumping everything)
        extra = 0
        if robot_actions: extra += max(0, len(robot_actions) - 1)
        if human_a_actions: extra += max(0, len(human_a_actions) - 1)
        if human_b_actions: extra += max(0, len(human_b_actions) - 1)
        if extra > 0:
            parts.append("I also have additional suggested actions if you want them.")

        utterance = " ".join(parts).strip()
        if not utterance:
            return

        try:
            # BrokerMediationMixin should provide this
            self._robot_say(utterance)
        except Exception as e:
            self.get_logger().warn(f"[idle] failed to announce plan via TTS: {e}")


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
        
        # --- NEW: track nested current action using skill events ---
        try:
            self._update_action_stack_from_skill_event(rule, data, ts)
        except Exception as e:
            self.get_logger().debug(f"[action-stack] update failed: {e}")

        
        # --- NEW: log human utterances from speech_final_any events ---
        self._log_human_utterance_from_basic_event(rule, data, ts)
        
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

                elif k == "request" and isinstance(v, dict):
                    # Extract just the scalar bits we care about
                    speaker_id = v.get("speaker_id")
                    if isinstance(speaker_id, str):
                        trace_entry["data"]["speaker_id"] = speaker_id
                    # Optionally keep the request id too:
                    req_id = v.get("id")
                    if isinstance(req_id, str):
                        trace_entry["data"]["request_id"] = req_id
                    req_text = v.get("text")
                    if isinstance(req_text, str):
                        trace_entry["data"]["request_text"] = req_text
                        

                # 2. Keep small scalar fields
                elif isinstance(v, (str, int, float, bool)):
                    trace_entry["data"][k] = v

    
        if trace_entry and "data" in trace_entry and "request_text" in trace_entry["data"]:
            info_utterance = "[on_basic_event] human utterance "
            if "speaker_id" in trace_entry["data"]:
                info_utterance += f'spoken by {trace_entry["data"]["speaker_id"]} '
            info_utterance += f'-> {trace_entry["data"]["request_text"]}'
            self.get_logger().info(info_utterance)

        self._event_trace.append(trace_entry)
        
        # Feed running event summary (async)
        self._record_event_for_summary(trace_entry, ts)


        try:
            if self._maybe_route_speech_to_mediation(rule, trace_entry):
                return
        except Exception as e:
            self.get_logger().warn(f"[mediation] routing speech turn failed: {e}")

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
                if data.get("done", False):
                    # Clear current objective
                    self._robot_objective = None

                    # Log a skill execution record for objective metrics
                    skill_name = (data.get("skill") or "").strip()
                    inner_ctx = data.get("inner_ctx") or {}
                    if skill_name:
                        # For now we assume these are robot skills; you can pass a different
                        # agent_id if you later emit human skill events.
                        self._record_skill_execution(
                            agent_id="robot",
                            skill_name=skill_name,
                            inner_ctx=inner_ctx,
                            ts=ts,
                        )
        except Exception as e:
            self.get_logger().debug(f"robot_objective update failed: {e}")

        # --- LLM multi-agent speech plans (COMMANDS ONLY) ---
        try:
            d = trace_entry.get("data") or {}
            # We stored the request id under "request_id" in the trace
            req_id = d.get("request_id", "")

            # Only consider events that are responses from llm_speech_to_multiagent_plan
            if isinstance(req_id, str) and req_id.startswith("llm_speech_to_multiagent_plan:"):
                # The classifier / planner output is in d["json"] or d["json_text"]
                plan_json = d.get("json")
                if not isinstance(plan_json, dict):
                    json_text = d.get("json_text")
                    if isinstance(json_text, str) and json_text.strip():
                        try:
                            plan_json = json.loads(json_text)
                        except Exception:
                            plan_json = None

                # If we can't parse JSON, bail
                if not isinstance(plan_json, dict):
                    self.get_logger().warn(
                        "[broker] llm_speech plan hook: missing or invalid JSON; skipping"
                    )
                    return

                # We only care about is_command here
                is_command = bool(plan_json.get("is_command"))
                is_addr    = bool(plan_json.get("is_addressed_to_robot"))
                
                if not is_addr:
                    return            
                
                # Non-command utterances should NOT start a new mediation session here.
                # They are handled via _maybe_route_speech_to_mediation on speech_final_any,
                # and during the race window they should just be ignored in this hook.
                if not is_command:
                    # Non-command utterances:
                    #   - do NOT start a new mediation session
                    #   - may trigger a chatty/explanatory reply from Bob if appropriate
                    utterance = (
                        (d.get("request_text") or "")
                        or (plan_json.get("natural_summary") or "")
                    )
                    speaker_id = d.get("speaker_id") or "unknown"

                    self.get_logger().info(
                        "[broker] llm_speech plan is_command=False; "
                        "skipping _handle_llm_speech_plan, maybe chatting back."
                    )

                    try:
                        self._maybe_chat_reply_to_utterance(
                            utterance=utterance,
                            speaker_id=speaker_id,
                            plan_json=plan_json,
                            ts=ts,
                        )
                    except Exception as e:
                        self.get_logger().warn(f"[chat] chat reply handler failed: {e}")

                    return

                # COMMAND path: delegate to the BrokerMediationMixin handler
                self._handle_llm_speech_plan(trace_entry, ts)

        except Exception as e:
            self.get_logger().warn(f"broker: error in LLM plan hook: {e}")


    def _log_human_utterance_from_basic_event(
        self,
        rule: str,
        data: dict,
        default_ts: float,
    ):
        if not rule.startswith("speech_final_any"):
            return

        text = (data.get("text") or "").strip()
        if not text:
            text = (data.get("utterance") or "").strip()

        if not text:
            return

        speaker_id_raw = data.get("speaker_id") or "unknown"
        speaker_id = (speaker_id_raw or "unknown").strip()


        ts = float(data.get("ts") or default_ts or time.time())

        # 1) Log in global chat history (mixin)
        self._append_chat_turn(speaker_id, text, ts)

        # 2) Let the mixin handle per-human reflection counters
        try:
            self._on_human_utterance_for_profile_reflection(speaker_id, ts)
        except Exception as e:
            self.get_logger().warn(f"[profiles] periodic reflection hook failed: {e}")



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

        # --- NEW: idle trigger forces optimization + plan publish ---
        if rid == "trigger_idle":
            self._trigger_optimizer_once(reason="trigger_idle", ts=ts)




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
        frame_id  = (data.get("frame_id") or "").strip()
        ts_epoch  = float(data.get("ts") or envelope.get("ts") or time.time())
        sensed_by_raw = (data.get("sensed_by") or data.get("phone_id") or "robot").strip()
        sensed_by = (sensed_by_raw or "robot").strip()


        agent_id = sensed_by if sensed_by in ("robot", "human_a", "human_b") else "robot"


        self._ensure_node(node_id)
        self._ensure_node_state(node_id)

        x, y = self._tf_to_map(frame_id)
        zone = self._zone_from_xy(x, y)


    # ------------------------------ DB helpers ------------------------------
    def _ensure_node(self, node_id: str):
        self.conn.execute("INSERT OR IGNORE INTO bt_nodes(node_id) VALUES (?)", (node_id,))

    def _ensure_node_state(self, node_id: str):
        self.conn.execute(
            "INSERT OR IGNORE INTO nodes_state(node_id, in_basket, disposed_to) VALUES (?, 0, 'none')",
            (node_id,)
        )


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

    # ------------------------------ Human planning conflict metrics ------------------------------


    def _node_id_from_box(self, box_id: int) -> str:
        """
        Map an integer box_id (from the FastAPI box server) to a canonical node_id
        in the broker DB. Adjust the format if you use a different naming scheme.
        """
        return f"CNode{box_id}"


    def _update_action_stack_from_skill_event(self, rule: str, data: dict, ts: float):
        """
        Maintain a nested stack of active skills using skill_started_any / skill_done_any events.

        Handles nesting:
          start(A) -> start(B) -> done(B) -> done(A)
        Also handles out-of-order / mismatched done by searching from top down.
        """
        if rule not in ("skill_started_any", "skill_done_any"):
            return

        if not isinstance(data, dict):
            return

        kind = (data.get("kind") or "").strip()
        skill = (data.get("skill") or "").strip()
        step_idx = data.get("step_idx")
        inner_ctx = data.get("inner_ctx") if isinstance(data.get("inner_ctx"), dict) else {}

        if not skill:
            return

        entry = {
            "skill": skill,
            "kind": kind,
            "ts": float(data.get("ts") or ts),
            "step_idx": int(step_idx) if isinstance(step_idx, (int, float)) else None,
            "inner_ctx": inner_ctx,
        }

        with self._action_stack_lock:
            if rule == "skill_started_any" or kind == "skill_started":
                # push
                self._action_stack.append(entry)
                return

            # done/finished: pop the most recent matching skill
            if rule == "skill_done_any" or kind in ("skill_finished", "skill_done"):
                # search from top down for matching skill
                for i in range(len(self._action_stack) - 1, -1, -1):
                    if self._action_stack[i].get("skill") == skill:
                        # remove that frame and anything above it (those above are "dangling" nesting)
                        del self._action_stack[i:]
                        return

                # if no match, do nothing (could be restarted node / missed starts)
                return


    def _current_action_brief(self) -> Optional[dict]:
        """
        Return a compact description of the currently active action (top of stack),
        plus a short stack view for debugging/prompts.
        """
        with self._action_stack_lock:
            if not self._action_stack:
                return None

            top = self._action_stack[-1]
            stack_view = []
            for e in self._action_stack[-5:]:  # cap depth in prompt
                stack_view.append({
                    "skill": e.get("skill"),
                    "step_idx": e.get("step_idx"),
                    "ts": e.get("ts"),
                })

        # Extract a compact "what/where" view from inner_ctx
        inner = top.get("inner_ctx") if isinstance(top.get("inner_ctx"), dict) else {}
        prop = inner.get("property")
        node_id = inner.get("target_node_id") or inner.get("node_id")

        nav = inner.get("nav") if isinstance(inner.get("nav"), dict) else {}
        final_goal = nav.get("final_goal") if isinstance(nav.get("final_goal"), dict) else None

        brief = {
            "skill": top.get("skill"),
            "step_idx": top.get("step_idx"),
            "property": prop,
            "target_node_id": node_id,
            "final_goal": final_goal,     # may be None
            "stack": stack_view,
        }
        return brief



    def _record_skill_execution(self, agent_id: str, skill_name: str, inner_ctx: dict, ts: float):
        """
        Store an objective record of a completed skill execution.

        For our current use:
          - We only care about:
              * 'box.sense_nearby' (or skills containing 'sense_nearby')
              * 'box.dispose_nearby' (or skills containing 'dispose_nearby')
              * optimizer skills like 'optimizer.robot.sense.<box_id>' / '.dispose.<box_id>'
          - inner_ctx from EventLayer for sense/dispose_nearby looks like:
              { "property": "X"|"Y", "target_node_id": "CNode1..." }

        We:
          - Infer kind = 'sense' or 'dispose'
          - Infer box_id from:
              1) optimizer-style suffix ".sense.<id>" / ".dispose.<id>", OR
              2) DB mapping: SELECT box_id FROM box_env_state WHERE node_id = target_node_id, OR
              3) Regex on node_id: everything after "CNode1"
          - Call /boxes/state to see what actually happened and fill:
              status, detected, probability, deadline, completed_at, fulfilled
        """
        try:
            s_lower = (skill_name or "").lower()

            # Heuristics for sense vs dispose (we only care about these)
            kind = None
            if "sense_nearby" in s_lower:
                kind = "sense"
            elif "dispose_nearby" in s_lower:
                kind = "dispose"
            else:
                # Not a skill we care about
                return

            if not isinstance(inner_ctx, dict):
                inner_ctx = {}

            # What EventLayer gives us for nearby skills
            prop = inner_ctx.get("property")
            node_id = inner_ctx.get("target_node_id") or inner_ctx.get("node_id")

            box_id = None

            # 1) From optimizer-style skill names: optimizer.robot.sense.7 / .dispose.12
            m = re.search(r"\.(sense|dispose)\.(\d+)$", skill_name)
            if m:
                try:
                    box_id = int(m.group(2))
                except Exception:
                    box_id = None

            # 2) From DB mapping: node_id -> box_id (box_env_state is filled by _sync_box_state_from_server)
            if box_id is None and isinstance(node_id, str):
                try:
                    row = self.conn.execute(
                        "SELECT box_id FROM box_env_state WHERE node_id = ?",
                        (node_id,),
                    ).fetchone()
                    if row is not None:
                        box_id = int(row[0])
                except Exception as e:
                    self.get_logger().warn(
                        f"[exec-log] failed to map node_id={node_id} to box_id via DB: {e}"
                    )

            # 3) Fallback: regex from CNode1... pattern (box id = everything after 'CNode1')
            if box_id is None and isinstance(node_id, str):
                m = re.match(r"^CNode1(.+)$", node_id)
                if m:
                    suffix = m.group(1)
                    try:
                        box_id = int(suffix) if suffix.isdigit() else None
                    except Exception:
                        box_id = None

            status = None
            detected = None
            probability = None
            deadline = None
            completed_at = None
            fulfilled: Optional[int] = None

            self.get_logger().info(
                f"[exec-log] inner_ctx={inner_ctx} skill={skill_name} → "
                f"kind={kind}, node_id={node_id}, box_id={box_id}, prop={prop}"
            )

            # 4) Cross-check with box server to see what actually happened
            if box_id is not None and kind in ("sense", "dispose") and prop is not None:
                try:
                    base = self.optimizer_base_url.rstrip("/")
                    url_state = base + "/boxes/state"
                    r_state = requests.get(url_state, timeout=self.req_timeout)

                    if r_state.status_code != 200:
                        self.get_logger().warn(
                            f"[exec-log] /boxes/state returned {r_state.status_code} "
                            f"when checking fulfillment"
                        )
                    else:
                        boxes_state = r_state.json() or []
                        for b in boxes_state:
                            try:
                                bid = int(b.get("box_id"))
                            except Exception:
                                continue
                            if bid != box_id:
                                continue

                            # Common bits
                            d_val = b.get("deadline")
                            if d_val is not None:
                                deadline = float(d_val)

                            if kind == "sense":
                                # Look for a completed sense result for this (agent, property)
                                sense_results = b.get("sense_results") or []
                                for sr in sense_results:
                                    a_id = str(sr.get("agent_id") or "")
                                    p = sr.get("property")
                                    st = sr.get("status")

                                    if a_id != agent_id or p != prop:
                                        continue

                                    status = st
                                    det_raw = sr.get("detected")
                                    if det_raw is not None:
                                        detected = 1 if det_raw else 0

                                    prob_val = sr.get("probability")
                                    if prob_val is not None:
                                        probability = float(prob_val)

                                    comp_val = sr.get("completed_at")
                                    if comp_val is not None:
                                        completed_at = float(comp_val)

                                    fulfilled = 1 if st == "completed" else 0
                                    break

                                if fulfilled is None:
                                    # We saw the box but no matching completed sense result
                                    fulfilled = 0

                            else:  # kind == "dispose"
                                disposed_X = bool(b.get("disposed_X", False))
                                disposed_Y = bool(b.get("disposed_Y", False))

                                if prop == "X":
                                    status = "disposed" if disposed_X else "not_disposed"
                                    fulfilled = 1 if disposed_X else 0
                                elif prop == "Y":
                                    status = "disposed" if disposed_Y else "not_disposed"
                                    fulfilled = 1 if disposed_Y else 0
                                else:
                                    any_disp = disposed_X or disposed_Y
                                    status = "disposed" if any_disp else "not_disposed"
                                    fulfilled = 1 if any_disp else 0

                            break  # stop after matching box
                except Exception as e:
                    self.get_logger().warn(f"[exec-log] server check failed: {e}")

            # 5) Persist record
            self.get_logger().info(
                f"[exec-log] INSERT robot_exec_evals: "
                f"agent={agent_id}, skill={skill_name}, kind={kind}, "
                f"box_id={box_id}, prop={prop}, status={status}, "
                f"detected={detected}, prob={probability}, "
                f"deadline={deadline}, completed_at={completed_at}, "
                f"fulfilled={fulfilled}"
            )

            self.conn.execute(
                """
                INSERT INTO robot_exec_evals(
                    ts, agent_id, skill, box_id, property,
                    kind, status, detected, probability,
                    deadline, completed_at, fulfilled
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    float(ts),
                    agent_id,
                    skill_name,
                    box_id,
                    prop,
                    kind,
                    status,
                    detected,
                    probability,
                    deadline,
                    completed_at,
                    fulfilled,
                ),
            )
            self.conn.commit()

        except Exception as e:
            self.get_logger().warn(f"[exec-log] failed to record skill execution: {e}")




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


    def _maybe_chat_reply_to_utterance(
        self,
        utterance: str,
        speaker_id: str,
        plan_json: Optional[dict],
        ts: float,
    ) -> None:
        # Never chat during mediation (same semantics as before)
        if self._mediation_in_progress():
            return

        utterance = (utterance or "").strip()
        if not utterance:
            return

        key = (speaker_id or "global").strip() or "global"

        # Buffer the chat event; a worker will coalesce and call the LLM once.
        self._llm_submit(
            channel="CHAT",
            key=key,
            ev_type="utterance",
            payload={
                "speaker_id": speaker_id,
                "utterance": utterance,
                "plan_json": plan_json,
            },
            ts=float(ts),
        )

    def _llm_worker_chat(self, chat_key: str):
        """
        Drain buffered chat utterances for this speaker key and make ONE LLM call.

        Coalescing policy:
          - If multiple utterances arrived, merge them into one "utterance" string.
          - Use the most recent plan_json (last event wins).
          - speaker_id: last event wins (but we keep all text in merged utterance).
        """
        # Gate chat while mediation is pending (double safety)
        if self._mediation_in_progress():
            return

        b = self._llm_bucket("CHAT", chat_key)

        # Drain
        with self._llm_lock:
            events = list(self._llm_buf[b])
            self._llm_buf[b].clear()

        if not events:
            return

        # Build one combined utterance
        merged_lines = []
        last_ts = None
        last_plan_json = None
        last_speaker_id = chat_key

        for ev in events:
            if ev.get("type") != "utterance":
                continue
            p = ev.get("payload") or {}
            spk = (p.get("speaker_id") or chat_key or "unknown").strip()
            txt = (p.get("utterance") or "").strip()
            if txt:
                merged_lines.append(f"{spk}: {txt}")
            if "plan_json" in p:
                last_plan_json = p.get("plan_json")
            last_speaker_id = spk
            last_ts = ev.get("ts") or last_ts

        if not merged_lines:
            return

        merged_utterance = "\n".join(merged_lines)
        ts = float(last_ts or time.time())

        # IMPORTANT: one LLM call (your original chat logic)
        try:
            self._maybe_chat_reply_to_utterance_impl(
                utterance=merged_utterance,
                speaker_id=last_speaker_id,
                plan_json=last_plan_json,
                ts=ts,
            )
        except Exception as e:
            self.get_logger().warn(f"[chat] buffered chat worker failed: {e}")


    def _maybe_chat_reply_to_utterance_impl(
        self,
        utterance: str,
        speaker_id: str,
        plan_json: Optional[dict],
        ts: float,
    ) -> None:
        """
        If LLM chat is enabled, no mediation is in progress, and the utterance
        looks like it's directed at the robot, call a small LLM to craft a
        natural-language reply using:
          - current box state (from box server or last cached)
          - current multi-agent plan (self._last_plan)
          - optional running event summary

        The LLM returns JSON { "robot_utterance": str, "should_reply": bool }.
        If should_reply is true and robot_utterance is non-empty, we TTS it
        via _robot_say.
        """

        if self._mediation_in_progress():
            # Don't do casual chat during a plan mediation.
            return

        utterance = (utterance or "").strip()
        if not utterance:
            return


        # 2) Get box state context (prefer last cached; fall back to fresh /boxes/state).
        boxes_state = getattr(self, "_last_boxes_state", None)
        if boxes_state is None:
            try:
                base = self.optimizer_base_url.rstrip("/")
                url_state = base + "/boxes/state"
                r_state = requests.get(url_state, timeout=self.req_timeout)
                if r_state.status_code == 200:
                    boxes_state = r_state.json()
            except Exception as e:
                self.get_logger().warn(f"[chat] failed to pull /boxes/state for context: {e}")
                boxes_state = None

        # Compact world state for the prompt (avoid dumping everything).
        simple_boxes: List[dict] = []
        if isinstance(boxes_state, list):
            for b in boxes_state[:20]:  # cap at 20 boxes for prompt size
                try:
                    simple_boxes.append(
                        {
                            "box_id": b.get("box_id"),
                            "x": b.get("x"),
                            "y": b.get("y"),
                            "deadline": b.get("deadline"),
                            "disposed_X": bool(b.get("disposed_X", False)),
                            "disposed_Y": bool(b.get("disposed_Y", False)),
                            "sense_summary": self._summarize_sense_results(b),
                        }
                    )
                except Exception:
                    continue

        # 3) Split current plan into:
        #    - agreed_plan: actions originally proposed by humans/robot
        #    - optimizer_suggestions: actions added by the optimizer
        current_plan: Plan = getattr(self, "_last_plan", {}) or {}
        prov = getattr(self, "_last_plan_provenance", {}) or {}

        agreed_plan: dict = {}
        optimizer_suggestions: dict = {}

        for aid, actions in current_plan.items():
            agreed_plan[aid] = []
            optimizer_suggestions[aid] = []

            for (box_id, prop, kind) in actions:
                origin = "optimizer"

                # Look up provenance if available
                for meta in prov.get(aid, []):
                    if (
                        int(meta.get("box_id", -1)) == int(box_id)
                        and meta.get("property") == prop
                        and meta.get("kind") == kind
                    ):
                        origin = meta.get("origin", origin) or origin
                        break

                entry = {
                    "box_id": int(box_id),
                    "property": prop,
                    "kind": kind,
                }

                if origin in ("human", "robot"):
                    agreed_plan[aid].append(entry)
                else:
                    optimizer_suggestions[aid].append(entry)


        # 4) Running event summary if available.
        with self._event_summary_lock:
            summary_text = self._event_summary_text

        recent_chat = self._get_recent_chat_turns(limit=8)

        current_action = self._current_action_brief()

        human_profiles = {}
        try:
            human_profiles = self._load_current_human_profiles() or {}
        except Exception:
            human_profiles = {}


        # 5) Build payload for the LLM.
        context_payload = {
            "ts": ts,
            "speaker_id": speaker_id,
            "utterance": utterance,
            "recent_conversation": recent_chat,
            "human_profiles": human_profiles,
            "current_action": current_action,
            "world_state": {
                "boxes": simple_boxes,
            },
            "plan_view": {
                "agreed_plan": agreed_plan,
                "optimizer_suggestions": optimizer_suggestions,
            },
            "event_summary": summary_text,
        }

        system_msg = {
            "role": "system",
            "content": (
                "You are Bob, a helpful mobile robot collaborating with humans to sense and "
                "dispose dangerous boxes (X/Y).\n"
                "- You see world_state and plan_view = {agreed_plan, optimizer_suggestions}.\n"
                "- agreed_plan: actions that humans (or you earlier) proposed and everyone has agreed on.\n"
                "- optimizer_suggestions: extra actions currently suggested by the optimizer, not yet agreed.\n"
                "- In THIS MODULE you only explain and advise; you cannot change the plan or promise changes.\n"
                "- When humans ask what you/others will do, describe the agreed_plan; you may mention "
                "optimizer_suggestions as additional options.\n"
                "- If humans ask for a recommendation (e.g., 'should we dispose box 1?'), give a tentative, "
                "evidence-based suggestion using world_state, clearly separating advice from agreed_plan.\n"
                "- If they ask you to change the plan, acknowledge the request, restate agreed_plan, and say that "
                "changes must go through the planning/mediation step.\n"
                "- Never claim you will add/drop/change actions that are not in agreed_plan.\n"
                "- Keep answers 1-2 short sentences. No emojis. Use hedged language ('I think', 'probably').\n"
                "- If the utterance is clearly not for you, set should_reply=false and use an empty robot_utterance.\n"
                "- You may also receive current_action describing what you are doing RIGHT NOW.\n"
                "- If asked what you are doing, answer using current_action first (if present).\n"
                "- You also receive human_profiles (human_a/human_b). Use them to tailor tone (direct vs gentle), "
                "explain more for low-expertise, and manage conflict neutrally.\n"

            ),
        }



        user_msg = {
            "role": "user",
            "content": (
                "Here is the current context and the latest human utterance. "
                "Return STRICT JSON of the form:\n"
                '{ "robot_utterance": "...", "should_reply": true/false }.\n\n'
                f"{json.dumps(context_payload, ensure_ascii=False)}"
            ),
        }

        try:
            obj = self._chat_json(
                messages=[system_msg, user_msg],
                temperature=0.4,
                max_tokens=80,
                retries=1,
                schema=CHAT_REPLY_SCHEMA,
                schema_name="BrokerChatReply",
                model=self.model,          # or a dedicated chat model param if you add one
                perf_phase="chat_reply",
            )
        except Exception as e:
            self.get_logger().warn(f"[chat] LLM chat reply failed: {e}")
            return

        robot_utt = (obj.get("robot_utterance") or "").strip()
        should_reply = bool(obj.get("should_reply"))

        if should_reply and robot_utt:
            self.get_logger().info(f"[chat] Bob replies to {speaker_id}: {robot_utt}")
            try:
                self._robot_say(robot_utt)
            except Exception as e:
                self.get_logger().warn(f"[chat] failed to publish TTS reply: {e}")


    def _summarize_sense_results(self, b: dict) -> dict:
        out = {"X": None, "Y": None}
        for prop in ("X", "Y"):
            latest = None
            for sr in (b.get("sense_results") or []):
                if sr.get("property") != prop:
                    continue
                if sr.get("status") != "completed":
                    continue
                # pick newest by completed_at if present, else last seen
                if latest is None:
                    latest = sr
                else:
                    a = latest.get("completed_at")
                    c = sr.get("completed_at")
                    if c is not None and (a is None or float(c) > float(a)):
                        latest = sr
            if latest:
                out[prop] = {
                    "agent_id": latest.get("agent_id"),
                    "detected": latest.get("detected"),
                    "probability": latest.get("probability"),

                }
        return out


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
        used_model = self.model #model or self.model

        # --- LLM NAME TRANSLATION (canonical → display) ---
        fwd = dict(self.agent_id_to_human_name or {})      # human_a → Jacob
        rev = dict(self.human_name_to_agent_id or {})      # Jacob → human_a

        messages_for_llm = self._translate_messages(messages, fwd)

        self.get_logger().info(
            "\n=== LLM PROMPT ===\n" + json.dumps(messages_for_llm, indent=2)
        )



        for attempt in range(retries + 1):
            t0 = time.time()
            ok_api = False


            try:
                if "gpt-oss" in used_model:
                    client = Groq()
                    resp = client.chat.completions.create(
                        model="openai/" + used_model,
                        messages=messages_for_llm,
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
                    client = OpenAI(timeout=60.0, max_retries=0)
                    resp = client.chat.completions.create(
                        model=used_model,
                        messages=messages_for_llm,
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
                obj_display = json.loads(content)

                # --- LLM NAME TRANSLATION BACK (display → canonical) ---
                obj = self._swap_json(obj_display, rev)

                self.get_logger().info(
                    f"\n=== LLM RAW RESPONSE ({schema_name}) ===\n{content}\nLatency: {dt_ms}\n"
                )

                validate(instance=obj, schema=used_schema)
                return obj

            except APITimeoutError as e:
                dt_ms = int((time.time() - t0) * 1000)
                last_exc = e
                self.get_logger().warn(
                    f"[llm] timeout after {dt_ms} ms "
                    f"(attempt {attempt + 1}/{retries + 1})"
                )

                if attempt >= retries:
                    break

                # simple fixed backoff (or remove sleep if you want immediate retry)
                time.sleep(0.5)
                continue


            except (json.JSONDecodeError, ValidationError) as e:
                # schema/json error
                dt_ms = int((time.time() - t0) * 1000)
                last_exc = e
                messages = messages + [{
                    "role": "system",
                    "content": "Return ONLY valid JSON per the given schema. No prose.",
                }]
                self.get_logger().info(
                    "\n=== LLM JSON DECODE ERROR ===\n"
                )
                continue



            except Exception as e:
                dt_ms = int((time.time() - t0) * 1000)
                last_exc = e
                self.get_logger().warn(f"[llm] error?")
                continue

        self.get_logger().info(
            "\n=== lllm ERROR ===\n"
        )

        raise ValueError(f"LLM did not return valid JSON ({schema_name}): {last_exc}")


  





    def _optimizer_tick(self):
        """
        Periodic tick: pull /boxes/state and /time from the box server.

        If the world state fingerprint changed since last tick, recompute plan.
        """
        if self._optimizer_running or not self.optimizer_enabled:
            return


        # NEW: don't replan while a human–robot mediation is underway
        if self._mediation_in_progress():
            self.get_logger().info("[optimizer] mediation in progress; skipping replanning")
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
            self._last_server_time = current_time


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
                        
                        "sense_time_X": b.get("sense_time_X"),
                        "sense_time_Y": b.get("sense_time_Y"),
                        "dispose_time_X": b.get("dispose_time_X"),
                        "dispose_time_Y": b.get("dispose_time_Y"),

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
            kwargs={"publish": False, "publish_reason": "periodic_tick"},
            daemon=True,
        ).start()


    def _run_optimizer_thread(
        self,
        boxes_state: list,
        current_time: float,
        *,
        publish: bool = False,
        publish_reason: str = "",
    ):
        """
        Background worker that:
          1) syncs box-server world into broker DB
          2) builds optimizer inputs
          3) runs Gurobi and (optionally) publishes a plan
        """
        try:
            self._sync_box_state_from_server(boxes_state)

            agents = self._build_agents_for_optimizer()
            boxes, box_positions = self._build_boxes_for_optimizer(boxes_state)

            self._last_boxes_state = boxes_state
            self._last_boxes_for_optimizer = boxes
            self._last_box_positions = box_positions

            if not agents or not boxes:
                self.get_logger().info("[optimizer] no agents or boxes available; skipping")
                return

            horizon = self.optimizer_horizon_sec
            agent_positions = self._snapshot_agent_positions()
            travel_time_fn = self._make_travel_time_fn(agent_positions, box_positions)

            plan = plan_assignments_gurobi(
                agents=agents,
                boxes=boxes,
                current_time=current_time,
                horizon=horizon,
                travel_time_fn=travel_time_fn,
            )

            self._last_plan = plan

            # ✅ Publish ONLY if this run came from a trigger callback
            if publish:
                self._publish_optimizer_plan(plan, current_time, box_positions)
                self.get_logger().info(f"[optimizer] published plan (reason={publish_reason})")

                # NEW: if idle-triggered, announce robot action + propose human parts
                if publish_reason == "trigger_idle":
                    try:
                        self._announce_idle_plan(plan, current_time)
                    except Exception as e:
                        self.get_logger().warn(f"[idle] announce/propose failed: {e}")
                
            else:
                self.get_logger().debug("[optimizer] computed plan (not published; periodic tick)")

        except Exception as e:
            self.get_logger().warn(f"[optimizer] optimization failed: {e}")
        finally:
            self._optimizer_running = False



    def _trigger_optimizer_once(self, reason: str, ts: float):
        """
        Force a single optimizer run ASAP, independent of fingerprint changes.
        Uses the same background thread as _optimizer_tick().
        """
        if self._optimizer_running or not self.optimizer_enabled:
            return

        if self._mediation_in_progress():
            self.get_logger().info(f"[optimizer] {reason}: mediation in progress; skip")
            return

        # Simple cooldown to avoid repeated trigger_idle spam
        cooldown = 2.0  # seconds (tune)
        last = getattr(self, "_last_idle_opt_ts", None)
        if last is not None and (ts - float(last)) < cooldown:
            return
        self._last_idle_opt_ts = float(ts)

        try:
            base = self.optimizer_base_url.rstrip("/")
            url_state = base + "/boxes/state"
            url_time  = base + "/time"

            r_state = requests.get(url_state, timeout=self.req_timeout)
            r_time  = requests.get(url_time, timeout=self.req_timeout)

            if r_state.status_code != 200 or r_time.status_code != 200:
                self.get_logger().warn(
                    f"[optimizer] {reason}: box server unavailable: "
                    f"state={r_state.status_code}, time={r_time.status_code}"
                )
                return

            boxes_state = r_state.json()
            time_resp   = r_time.json()
            current_time = float(time_resp.get("server_time", 0.0))
            self._last_server_time = current_time

        except Exception as e:
            self.get_logger().warn(f"[optimizer] {reason}: failed to contact box server: {e}")
            return

        self.get_logger().info(f"[optimizer] {reason}: forcing optimizer run")
        self._optimizer_running = True
        threading.Thread(
            target=self._run_optimizer_thread,
            args=(boxes_state, current_time),
            kwargs={"publish": True, "publish_reason": reason},
            daemon=True,
        ).start()



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

    def _make_travel_time_fn(
        self,
        agent_positions: Dict[str, Tuple[float, float]],
        box_positions: Dict[int, Tuple[float, float]],
    ):
        """
        Build a travel_time_fn(agent_id, box_id) closure using the latest
        agent_positions and box_positions plus the configured per-agent speeds.
        """
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

        return travel_time_fn


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

            # Per-box durations are now supplied by /boxes/state
            try:
                sense_time_X = float(b["sense_time_X"])
                sense_time_Y = float(b["sense_time_Y"])
                dispose_time_X = float(b["dispose_time_X"])
                dispose_time_Y = float(b["dispose_time_Y"])
            except KeyError as e:
                self.get_logger().warn(
                    f"[optimizer] missing duration field {e} in box {box_id}; skipping"
                )
                continue

            disposed_X = bool(b.get("disposed_X", False))
            disposed_Y = bool(b.get("disposed_Y", False))

            # --- Build already_sensed + crude beliefs from sense_results ---
            # already_sensed_any[prop] = True if ANY agent has a completed sense for (box, prop)
            already_sensed_any = {"X": False, "Y": False}

            already_sensed: Dict[str, Dict[str, bool]] = {}   # keep your per-agent map too
            sense_results = b.get("sense_results") or []

            last_det_X = None
            last_det_Y = None
            count_X = 0
            count_Y = 0

            for sr in sense_results:
                agent_id = str(sr.get("agent_id") or "")
                prop = sr.get("property")
                status = sr.get("status")
                detected = sr.get("detected")

                if prop not in ("X", "Y"):
                    continue

                if status == "completed":
                    # Any-agent completion flag (THIS is what you’ll use for the optimizer constraint)
                    already_sensed_any[prop] = True

                    # Keep your existing per-agent bookkeeping (may still be useful elsewhere)
                    if agent_id:
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
                already_sensed={
                    **already_sensed,
                    "__any__": already_sensed_any,   # << add this
                },
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
                f"{int(box_id)}": {
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

