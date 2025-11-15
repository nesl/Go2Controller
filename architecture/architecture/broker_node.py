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

# LLM reply must be STRICT JSON like: {"sql":"SELECT ...", "params": {...}, "purpose":"..."}
LLM_SQL_SCHEMA = {
  "type": "object",
  "required": ["sql"],
  "properties": {
    "sql":    {"type":"string"},
    "params": {"type":"object"},
    "purpose":{"type":"string"}
  },
  "additionalProperties": False
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
            "speech_final_any": "human_command",
        }))

        # Contamination fetch policy (broker owns it)
        self.declare_parameter('contam_enable_server_calls', True)
        self.declare_parameter('contam_server_url', 'http://127.0.0.1:8000/check')
        self.declare_parameter('contam_request_timeout_sec', 0.6)
        self.declare_parameter('contam_min_refresh_sec', 120.0)    # throttle per (agent_id,node_id)

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
            "vw_backlog_counts","vw_object_sheet"
        ]))

        # Optional: mock LLM for offline dev (pass JSON {"sql": "...", "params": {...}, "purpose": "..."} in param)
        self.declare_parameter('llm_mock_json', '')
        
        self.declare_parameter("llm_model", "gpt-5-nano")
        self.client = OpenAI()
        self.llm_model = self.get_parameter("llm_model").get_parameter_value().string_value

        # Parameters → members
        self.db_path       = self.get_parameter('db_path').get_parameter_value().string_value
        self.target_frame  = self.get_parameter('target_frame').get_parameter_value().string_value
        self.zone_split_x  = float(self.get_parameter('zone_split_x').value)
        self.bus_topic     = self.get_parameter('subscribe_topic').get_parameter_value().string_value
        self.bt_rule_id    = self.get_parameter('bt_rule_id').get_parameter_value().string_value
        self.human3d_rule_id = self.get_parameter('human3d_rule_id').get_parameter_value().string_value

        self.trigger_map   = json.loads(self.get_parameter('trigger_map_json').get_parameter_value().string_value)

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

        # ------------ ROS I/O ------------
        # Events
        self.sub_basic = self.create_subscription(StringMsg, self.bus_topic, self._on_basic_event, 1000)
        self.sub_comp  = self.create_subscription(StringMsg, "/events/composite", self._on_comp_event, 500)


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

        self.get_logger().info(
            f"broker_node up | db={self.db_path} bus={self.bus_topic} rule={self.bt_rule_id} "
            f"target_frame={self.target_frame} zone_split_x={self.zone_split_x} server={self.server_url} "
            f"enable_server={self.enable_server}"
        )


    def _publish_context_capsule(self):
        cap = self._context_capsule()
        cap["schema_card"] = self._schema_card()
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

        # compact trace entry
        trace_entry = {"rule": rule, "ts": ts}
        if isinstance(data, dict):
            trace_entry.update(data)
        self._event_trace.append(trace_entry)

        # trigger state (used by LLM prompt)
        trig_type = self.trigger_map.get(rule)
        if trig_type:
            self._current_trigger = {"type": trig_type, "trigger_event": o, "ts": ts}

        # ingestion
        if rule == self.bt_rule_id:
            self._ingest_bt_reading(data, o)
        elif rule == self.human3d_rule_id:
            self._ingest_human3d(data, o)

        # Proactive: if this event is a planning trigger, immediately run initial LLM-SQL
        if trig_type in ("new_object","finish_or_fail","human_command","idle","presence"):
            try:
            
                self._publish_context_capsule()
                
                pack = self._llm_sql_to_facts(proactive=True)
                self.pub_facts.publish(StringMsg(data=json.dumps(pack)))
                self._emit_sql_debug(pack)
            except Exception as e:
                self.get_logger().warn(f"proactive run failed: {e}")

    def _on_comp_event(self, msg: StringMsg):
        try:
            o = json.loads(msg.data)
        except Exception:
            self.get_logger().warn("broker: invalid JSON on /events/composite")
            return

        rid = str(o.get("rule") or "")
        ts  = float(o.get("ts") or time.time())
        expr = o.get("expr") or ""

        # trace entry
        self._event_trace.append({"rule": rid, "ts": ts, "composite": True, "expr": expr[:160]})

        # map composite rule id → trigger (use the same trigger_map param)
        trig_type = self.trigger_map.get(rid, "composite_hit")
        self._current_trigger = {"type": trig_type, "ts": ts, "composite": True, "rid": rid}

        # (optional) proactive run
        if trig_type:
            try:
                self._current_trigger["trigger_event"] = o
                self._publish_context_capsule()
                
                pack = self._llm_sql_to_facts(proactive=True)
                self.pub_facts.publish(StringMsg(data=json.dumps(pack)))
                self._emit_sql_debug(pack)
            except Exception as e:
                self.get_logger().warn(f"proactive run (composite) failed: {e}")


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
        self._maybe_queue_contamination_refresh(agent_id, node_id, rssi, ts_epoch)

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

    def _refresh_one_label(self, agent_id: str, node_id: str):
        now = time.time()
        key = (agent_id, node_id)
        with self._lock:
            ent = self._contam_cache.get(key)
            if ent and (now - ent["ts"] < self.min_refresh):
                return
        if not self.enable_server or not self.server_url:
            return
        try:
            resp = requests.post(
                self.server_url,
                json={"object_id": node_id, "phone_id": agent_id},
                timeout=self.req_timeout
            )
            if resp.status_code != 200:
                self.get_logger().warn(f"contam server non-200 for {(agent_id,node_id)}: {resp.status_code}")
                return
            data = resp.json()
            contaminated = bool(data.get("contaminated"))
            probability  = float(data.get("probability"))
        except Exception as e:
            self.get_logger().warn(f"contam server failed for {(agent_id,node_id)}: {e}")
            return

        self.conn.execute("""
            INSERT INTO agent_node_labels(agent_id, node_id, contaminated, probability, updated_ts)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, node_id) DO UPDATE SET
                contaminated=excluded.contaminated,
                probability=excluded.probability,
                updated_ts=excluded.updated_ts
        """, (agent_id, node_id, int(1 if contaminated else 0), probability, now))

        with self._lock:
            self._contam_cache[key] = {"ts": now, "contaminated": contaminated, "probability": probability}

    # ------------------------------ LLM SQL layer ------------------------------
    # Strict validators
    _SQL_BAD = re.compile(r'(--|/\*|\*/|;|\b(ATTACH|DETACH|ALTER|DROP|CREATE|INSERT|UPDATE|DELETE|REPLACE|VACUUM|PRAGMA|BEGIN|END|COMMIT|ROLLBACK)\b)', re.I)
    _SQL_TABLES = re.compile(r'\b(from|join)\s+([a-zA-Z0-9_\.]+)', re.I)

    def _validate_sql_readonly(self, sql: str) -> Optional[str]:
        if self._SQL_BAD.search(sql):
            return "sql_contains_prohibited_tokens"
        used = [m.group(2) for m in self._SQL_TABLES.finditer(sql)]
        for name in used:
            base = name.split('.')[-1]
            if base not in self.allowed:
                return f"object_not_allowed:{base}"
        return None

    def _exec_sql_safely(self, sql: str, params: dict, max_rows: int, max_bytes: int, timeout_ms: int):
        con = self.conn


        aborted = {"v": False}
        start = time.time()
        def _progress():
            if (time.time() - start) * 1000.0 > timeout_ms:
                aborted["v"] = True
                return 1
            return 0
        con.set_progress_handler(_progress, 1000)

        cur = con.execute(sql, params or {})
        colnames = [d[0] for d in cur.description] if cur.description else []
        rows, size, i = [], 0, -1
        for i, row in enumerate(cur):
            if i >= max_rows: break
            vals = []
            for v in row:
                if isinstance(v, (bytes, bytearray)): v = "<blob>"
                vals.append(v)
            size += sum(len(str(x)) for x in vals) + 2*len(vals)
            if size > max_bytes: break
            rows.append(vals)

        con.set_progress_handler(None, 0)
        truncated = aborted["v"] or (i+1 >= max_rows) or (size > max_bytes)
        return colnames, rows, truncated, int((time.time()-start)*1000)

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

    def _context_capsule(self) -> dict:
        counts = self.conn.execute("SELECT * FROM vw_backlog_counts").fetchone()
        if counts:
            cc = dict(zip(["to_pick","in_basket","delivered_clean","delivered_contaminated"], counts))
        else:
            cc = {}
        basket = [r[0] for r in self.conn.execute(
            "SELECT node_id FROM nodes_state WHERE in_basket=1 LIMIT 8").fetchall()]
        return {
            "trigger": self._current_trigger,
            "profiles": self._profiles,
            "event_trace": list(self._event_trace)[-20:],
            "world": {"backlog_counts": cc, "basket": basket},
            "budgets": {"max_rows": self.sql_max_rows, "max_bytes": self.sql_max_bytes, "timeout_ms": self.sql_timeout_ms}
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


    def _chat_json(self, messages, temperature=0.2, max_tokens=300, retries=1):
        """Call OpenAI chat.completions and enforce JSON + schema."""
        for _ in range(retries + 1):
            resp = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            try:
                obj = json.loads(content)
                validate(instance=obj, schema=LLM_SQL_SCHEMA)
                return obj
            except (json.JSONDecodeError, ValidationError):
                messages = messages + [{"role":"system","content":"Return ONLY valid JSON per the schema. No prose."}]
        # last-resort fallback handled by caller
        raise ValueError("LLM did not return valid JSON")

    

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
            # safe fallbacks if the call fails or returns garbage
            hints = (context_capsule.get("trigger") or {}).get("hints") or {}
            oid = hints.get("object_id")
            if oid:
                return "SELECT * FROM vw_object_sheet WHERE node_id=:object_id LIMIT 1", {"object_id": oid}, "fallback object sheet"
            return ("SELECT node_id, best_zone, best_rssi, in_basket, disposed_to "
                    "FROM vw_object_sheet WHERE disposed_to='none' ORDER BY best_ts DESC LIMIT 5",
                    {}, "fallback shortlist")



    # ---- Turn LLM SQL into facts ----
    def _llm_sql_to_facts(self, *, proactive: bool, needs: Optional[dict] = None) -> dict:
        ws_id = self._ws_id()
        ent = self._ws.setdefault(ws_id, {"hashes": set(), "iters": 0})
        if not proactive and ent["iters"] >= self.iteration_limit:
            return {"mode": "reactive", "done": True, "reason": "iteration_limit"}

        schema = self._schema_card()
        capsule = self._context_capsule()

        sql, params, purpose = self._llm_plan_sql(proactive=proactive, schema_card=schema, context_capsule=capsule, planner_needs=needs)

        err = self._validate_sql_readonly(sql)
        if err:
            return {"error": err, "sql_meta": {"sql": sql, "params": params}, "mode": ("proactive" if proactive else "reactive")}

        cols, rows, truncated, ms = self._exec_sql_safely(sql, params, self.sql_max_rows, self.sql_max_bytes, self.sql_timeout_ms)

        changed = self._ws_changed(ws_id, sql, rows)
        if changed:
            self._ws_add(ws_id, sql, rows)
        if not proactive:
            ent["iters"] += 1

        pack = {
            "mode": ("proactive" if proactive else "reactive"),
            "ws_id": ws_id,
            "rationale": purpose,
            "table": {"columns": cols, "rows": rows, "truncated": truncated, "changed": changed},
            "sql_meta": {"sql": sql, "params": params, "ms": ms}
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

