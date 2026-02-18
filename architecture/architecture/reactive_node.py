#!/usr/bin/env python3
"""
interaction_loop_node.py

Interaction Loop / Skill Selector node.

- Subscribes:
    /events/basic      (std_msgs/String, JSON: {"rule":..., "data":..., "ts":..., "zone":...})
    /events/composite  (std_msgs/String, JSON: {"rule":..., "expr":..., "ts":..., "zone":...})
    /task_state        (std_msgs/String, JSON: task progress & robot objective)
    /profiles/summary  (std_msgs/String, JSON: HDT human profiles)
    /planner/proposal  (std_msgs/String, JSON: high-level proposals from planner)

- On any "trigger" rule (rule id in trigger_map or starting with trigger_prefix),
  it builds a "loop_capsule" that includes:
    * trigger info (rule, type, text, zone, ts)
    * recent events (small trace)
    * current task_state
    * compact HDT profiles (including active human and robot zone)
    * latest planner proposal (if any)

- When a planner proposal is received on /planner/proposal, it is immediately
  converted into a concrete skill *state_machine* and executed by:
    1) appending it to skills_composite.yaml
    2) calling /skills/reload
    3) publishing {"skill": "<name>", "ctx": {...}} on /skills/execute

- For other triggers (e.g., human speech), it calls a small LLM (fast) to pick
  a SHORT sequence of existing skills:

    {
      "name": "reactive.<something>",
      "skills": [
        {"use": "<existing_skill_name>", "with": { ... }},
        ...
      ]
    }

- That list is similarly turned into a state_machine and executed via
  /skills/execute in the exact same format as router_node.py.
"""

import json
import time
import os
from collections import deque
from typing import Any, Dict, Optional, List

import yaml

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from std_msgs.msg import String as StringMsg
from std_srvs.srv import Trigger  # ### NEW

from jsonschema import validate, ValidationError
from openai import OpenAI
from groq import Groq


# ---------- JSON schema for the skill sequence ----------

FAST_SKILLS_LIST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["skills"],
    "properties": {
        "name": {"type": "string"},
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["use"],
                "properties": {
                    "use": {"type": "string"},
                    "with": {"type": "object"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

FAST_SKILLS_SYSTEM_PROMPT = """
You are the INTERACTION LOOP / SKILL SELECTOR for a mobile robot named Bob.

You receive a single JSON object with:
  - trigger_type: string
  - router_capsule:
      {
        "trigger": {
          "rule": str,
          "type": str,
          "kind": "basic"|"composite",
          "zone": "A"|"B"|null|...,
          "ts": float,
          "text": str          # human utterance snippet if any
        },
        "recent_events": [    # short trace of recent events
          {
            "kind": "basic"|"composite",
            "rule": str,
            "zone": str|null,
            "ts": float,
            "text_snippet"?: str,
            "expr_snippet"?: str
          }, ...
        ],
        "task_state": { ... },   # current task progress and robot objective
        "planner_proposal": { ... }  # latest proposal from planner, if any
        "humans": {
          "active_human": str|null,
          "robot_zone": str|null,
          "humans": {
            "<human_id>": {
              "style": str|null,
              "helpfulness": str|null,
              "risk_aversion": str|null,
              "contamination_bias": str|null,
              "summary": str|null
            },
            ...
          }
        }
        "control": {
          "mode": "autonomous" | "follow_human" | "idle_listen",
          "target": null | "any" | "<human_id>",
          "reason": str,
          "ts": float
        }
      }

  - skills_inventory:
      {
        "primitives": [
          {
            "name": str,
            "kind": "primitive",
            "description": str,
            "params_template": object,
            "param_keys": [str, ...],
            "action": str
          }, ...
        ],
        "composites": [
          {
            "name": str,
            "kind": "composite" or "state_machine",
            "description": str,
            "params_template": object,
            "param_keys": [str, ...]
          }, ...
        ]
      }

Your ONLY job is to propose a SHORT SEQUENCE of existing skills the robot
should execute *immediately* in response to this trigger, taking into account:

  - the human's likely intent (from trigger text and recent events),
  - current task_state (zones, backlogs, robot objective),
  - planner proposals (if present),
  - HDT human profiles (style, helpfulness, risk_aversion, etc.),
  - safety and social appropriateness.

Requirements:
- Use ONLY skill names that exist in skills_inventory (either primitives or composites).
- Do NOT invent new skill names.
- The sequence must be short: 1–3 steps.
- Each step:
    {
      "use": "<existing_skill_name>",
      "with": { ...optional params... }
    }
- When using "with":
    - Use only parameter names that appear in param_keys for that skill.
    - Follow the structure suggested by params_template; keep values small and simple.
- Prefer to call higher-level composite/state_machine skills when appropriate
  (e.g., "interact.greet_human" instead of raw primitives), but you may mix them
  with primitives like "tts.say" or "nav.move_relative".
- If control.mode is "autonomous", the robot is leading and may politely
  acknowledge humans but generally follow its own plan.
- If control.mode is "follow_human", the robot should treat humans as leaders.
  The target may be "any" or a specific human id.
- If control.mode is "idle_listen", the robot should primarily listen / respond.

You MUST return STRICT JSON ONLY, no explanations, of the form:

{
  "name": "reactive.<something>",  // optional, can be any string; you may omit this
  "skills": [
    {
      "use": "<existing_skill_name>",
      "with": { ... }
    },
    ...
  ]
}
""".strip()


class InteractionLoopNode(Node):
    """
    Interaction Loop / Skill Selector.

    - Listens for trigger events.
    - Ingests task_state, HDT profiles, and planner proposals.
    - For planner proposals, converts them into a state_machine and executes it.
    - For other triggers, uses LLM (or fallback) to choose a skills list,
      converts it into a state_machine, and executes it.
    - Execution is done by publishing to /skills/execute in the SAME FORMAT
      as router_node: {"skill": "<name>", "ctx": {...}}.
    """

    def __init__(self):
        super().__init__("interaction_loop_node")

        # ----- Parameters -----
        self.declare_parameter("llm_enabled", True)
        self.declare_parameter("model", "gpt-5-mini")
        self.declare_parameter("groq_model_prefix", "gpt-oss")
        self.declare_parameter("trigger_prefix", "XXXXXXXXXXXX")
        self.declare_parameter(
            "trigger_map_json",
            json.dumps({
                "trigger_speech_final": "human_command",
                #"trigger_idle": "idle",
            }),
        )

        # Where to read skills and rules from (same style as router/orchestrator)
        self.declare_parameter("skills_base_path", "")
        self.declare_parameter("skills_composite_path", "")
        self.declare_parameter("rules_init_path", "")
        self.declare_parameter("rules_path", "")

        self.declare_parameter("profiles_topic", "/profiles/summary")
        self.declare_parameter("task_state_topic", "/task_state")
        self.declare_parameter("planner_proposal_topic", "/planner/proposal")
        self.declare_parameter("context_capsule_topic", "/broker/context_capsule")


        # How many recent events to keep in memory
        self.declare_parameter("event_trace_len", 10)

        # Parameters → attributes
        self.llm_enabled = bool(self.get_parameter("llm_enabled").value)
        self.model = str(self.get_parameter("model").value)
        self.groq_prefix = str(self.get_parameter("groq_model_prefix").value)
        self.trigger_prefix = str(self.get_parameter("trigger_prefix").value)
        self.trigger_map = json.loads(self.get_parameter("trigger_map_json").value)

        self.skills_base_path = self.get_parameter("skills_base_path").value
        self.skills_composite_path = self.get_parameter("skills_composite_path").value
        self.rules_init_path = self.get_parameter("rules_init_path").value
        self.rules_path = self.get_parameter("rules_path").value

        self.profiles_topic = self.get_parameter("profiles_topic").value
        self.task_state_topic = self.get_parameter("task_state_topic").value
        self.planner_proposal_topic = self.get_parameter("planner_proposal_topic").value
        self.context_capsule_topic = self.get_parameter("context_capsule_topic").value


        self.event_trace_len = int(self.get_parameter("event_trace_len").value)

        self.add_on_set_parameters_callback(self._on_param_change)

        # ----- Internal state -----
        self._event_trace = deque(maxlen=self.event_trace_len)

        # NEW: latest global event summary from Broker
        self._last_event_summary: Optional[str] = None

        # Latest task_state snapshot
        self._last_task_state: Dict[str, Any] = {}

        # Latest planner proposal (raw object)
        self._last_planner_proposal: Dict[str, Any] = {}
        self._last_planner_proposal_ts: float = 0.0
        
        # NEW: control-mode snapshot from planner
        # mode ∈ {"autonomous", "follow_human", "idle_listen"}
        # target ∈ {None, "any", "<human_id>"}
        self._control_mode: str = "follow_human"
        self._control_target: Optional[str] = None
        self._control_reason: str = "initial_default"
        self._control_ts: float = self._now()


        # Latest HDT profiles (raw + compact)
        self._profiles_raw: Dict[str, Any] = {}
        self._profiles_compact: Dict[str, Any] = {}
        self._active_human: Optional[str] = None
        self._last_robot_zone: Optional[str] = None

        # ----- ROS I/O -----
        # Events from EventLayer
        self.sub_basic = self.create_subscription(
            StringMsg, "/events/basic", self._on_basic_event, 200
        )
        self.sub_comp = self.create_subscription(
            StringMsg, "/events/composite", self._on_comp_event, 100
        )

        # Task state monitor
        self.sub_task_state = self.create_subscription(
            StringMsg, self.task_state_topic, self._on_task_state, 20
        )

        # NEW: broker context capsule (contains event summary)
        self.sub_context_capsule = self.create_subscription(
            StringMsg,
            self.context_capsule_topic,
            self._on_context_capsule,
            10,
        )


        # HDT profiles
        self.sub_profiles = self.create_subscription(
            StringMsg, self.profiles_topic, self._on_profiles, 10
        )

        # Planner proposals
        self.sub_proposal = self.create_subscription(
            StringMsg, self.planner_proposal_topic, self._on_planner_proposal, 20
        )


        # optimizer plan (Gurobi assignments)
        self.sub_optimizer_plan = self.create_subscription(
            StringMsg,
            "/optimizer/plan",
            self._on_optimizer_plan,
            10,
        )


        # Skills: reload + execute (MATCH router_node)
        self.reload_skills_client = self.create_client(  # ### NEW
            Trigger,
            "/skills/reload",
        )
        self.skills_execute_pub = self.create_publisher(  # ### NEW
            StringMsg,
            "/skills/execute",
            10,
        )

        self.get_logger().info("interaction_loop_node initialized")


    def _box_node_id_from_box_id(self, box_id: int) -> str:
        """
        Map optimizer box_id -> world node_id used by box.* skills.

        Naming convention:
          box_id = 7   -> 'CNode107'
          box_id = 23  -> 'CNode123'

        Pattern: 'CNode1##' where ## is zero-padded box_id.
        """
        try:
            return f"CNode1{int(box_id):02d}"
        except (TypeError, ValueError):
            return ""


    def _on_optimizer_plan(self, msg: StringMsg):
        """
        Ingest Gurobi optimizer plan and execute ONLY the first action
        assigned to the robot, as a short skill sequence.

        Expected payload shape (from Broker):

        {
          "ts": <float>,
          "current_time": <float>,
          "agents": {
            "robot": [
              {"box_id": <int>, "property": "X"|"Y", "kind": "sense"|"dispose"},
              ...
            ],
            "human_a": [...],
            "human_b": [...]
          },
          "nodes": {
            "<node_id>": {
              "x": <float>,
              "y": <float>,
              "yaw": <float optional>
            },
            ...
          }
        }
        """
        try:
            obj = json.loads(msg.data) if msg.data else {}
        except Exception:
            self.get_logger().warn(
                "[interaction_loop] bad JSON on /optimizer/plan: %r" % msg.data
            )
            return

        if not isinstance(obj, dict):
            return

        agents = obj.get("agents") or {}
        if not isinstance(agents, dict):
            return

        robot_actions = agents.get("robot") or []
        if not isinstance(robot_actions, list) or not robot_actions:
            self.get_logger().info(
                "[interaction_loop] optimizer plan has no robot actions; nothing to do"
            )
            return

        first = robot_actions[0]
        if not isinstance(first, dict):
            return

        box_id = first.get("box_id")
        prop = str(first.get("property", "X")).upper()
        kind = str(first.get("kind", "sense")).lower()

        if box_id is None:
            self.get_logger().warn(
                "[interaction_loop] optimizer robot action missing box_id; skipping"
            )
            return

        if prop not in ("X", "Y"):
            self.get_logger().warn(
                f"[interaction_loop] optimizer robot action has invalid property={prop!r}; defaulting to 'X'"
            )
            prop = "X"

        node_id = self._box_node_id_from_box_id(box_id)
        if not node_id:
            self.get_logger().warn(
                f"[interaction_loop] could not map box_id={box_id} to node_id; skipping"
            )
            return

        # Map kind -> skill
        if kind == "sense":
            skill_name = "box.sense_nearby"
        elif kind == "dispose":
            skill_name = "box.dispose_nearby"
        else:
            self.get_logger().warn(
                f"[interaction_loop] unknown optimizer action kind={kind!r}; "
                "defaulting to sense_nearby"
            )
            skill_name = "box.sense_nearby"

        # ---------------------------------------
        # NEW: pull node pose from plan and prepend nav.move_absolute
        # ---------------------------------------
        nodes = obj.get("nodes") or {}
        x = y = yaw = None

        if isinstance(nodes, dict):
            # we assume optimizer keys by node_id (string), which in your setup
            # is just the numeric id (e.g., "7")
            node_entry = nodes.get(str(node_id)) or nodes.get(str(box_id))
            if isinstance(node_entry, dict):
                try:
                    x = float(node_entry.get("x"))
                    y = float(node_entry.get("y"))
                    yaw_val = node_entry.get("yaw", 0.0)
                    yaw = float(yaw_val) if yaw_val is not None else 0.0
                except (TypeError, ValueError):
                    x = y = yaw = None
                    self.get_logger().warn(
                        f"[interaction_loop] error parsing pose {node_entry}"
                    )

        skills_list: List[Dict[str, Any]] = []

        if x is not None and y is not None:
            # First: go to the absolute pose
            skills_list.append(
                {
                    "use": "nav.move_absolute",
                    "with": {
                        "frame": "map",
                        "x": x,
                        "y": y,
                        "yaw": yaw if yaw is not None else 0.0,
                    },
                }
            )
        else:
            self.get_logger().warn(
                f"[interaction_loop] no pose for node_id={node_id}, "
                "skipping nav.move_absolute and going straight to nearby-op"
            )

        # Second: your existing nearby operation
        skills_list.append(
            {
                "use": skill_name,
                "with": {
                    "target_node_id": node_id,
                    "property": prop,
                },
            }
        )

        skills_obj = {
            "name": f"optimizer.robot.{kind}.{box_id}.{prop}",
            "skills": skills_list,
        }

        self.get_logger().info(
            f"[interaction_loop] executing optimizer first robot action: "
            f"kind={kind}, box_id={box_id}, node_id={node_id}, property={prop}, "
            f"pose=({x},{y},{yaw})"
        )

        # Reuse the same publishing path as other plans
        self._publish_skill_plan(
            rule="optimizer_plan",
            trig_type="planner",
            skills_obj=skills_obj,
        )


    # ---------- Parameter updates ----------
    def _on_param_change(self, params):
        for p in params:
            if p.name == "llm_enabled" and p.type_ == Parameter.Type.BOOL:
                self.llm_enabled = p.value
                self.get_logger().info(
                    f"[interaction_loop] llm_enabled -> {self.llm_enabled}"
                )
            elif p.name == "model" and p.type_ == Parameter.Type.STRING:
                self.model = p.value
                self.get_logger().info(
                    f"[interaction_loop] model -> {self.model}"
                )
        return SetParametersResult(successful=True, reason="ok")

    # ---------- Helpers ----------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _read_yaml_if_exists(self, path: str) -> Optional[dict]:
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().warn(
                f"[interaction_loop] failed to read YAML '{path}': {e}"
            )
            return None

    def _build_skills_inventory(self) -> Dict[str, Any]:
        """
        Build a small skills inventory:
          - primitives
          - composites (including state_machines)

        We read both base and composite skills docs if provided.
        """
        primitives: List[Dict[str, Any]] = []
        composites: List[Dict[str, Any]] = []

        for path in [self.skills_base_path]:
            doc = self._read_yaml_if_exists(path) or {}
            for s in doc.get("skills", []) or []:
                if not isinstance(s, dict):
                    continue

                name = str(s.get("name", "")).strip()
                if not name:
                    continue

                kind = str(s.get("kind", "")).strip() or "primitive"
                description = str(s.get("description", "")).strip()
                params_template = s.get("params") or {}
                param_keys = (
                    list(params_template.keys())
                    if isinstance(params_template, dict)
                    else []
                )

                entry: Dict[str, Any] = {
                    "name": name,
                    "kind": kind,
                    "description": description,
                    "params_template": params_template,
                    "param_keys": param_keys,
                }

                if kind == "primitive":
                    entry["action"] = s.get("action", "")
                    primitives.append(entry)
                else:
                    composites.append(entry)

        return {"primitives": primitives, "composites": composites}

    # ---------- Task state ingestion ----------
    def _on_task_state(self, msg: StringMsg):
        try:
            obj = json.loads(msg.data) if msg.data else {}
        except Exception:
            self.get_logger().warn(
                f"[interaction_loop] bad JSON on {self.task_state_topic}: {msg.data}"
            )
            return
        if isinstance(obj, dict):
            self._last_task_state = obj

    def _on_context_capsule(self, msg: StringMsg):
        """
        Ingest Broker's context capsule to access the global event summary.

        Expected shape (from broker_node._context_capsule):
          {
            "trigger": {...},
            "profiles": {...},
            "event_trace": "<short summary>" OR [ ...raw events... ],
            "world": {...}
          }
        We only care about event_trace here.
        """
        try:
            obj = json.loads(msg.data) if msg.data else {}
        except Exception:
            self.get_logger().warn(
                f"[interaction_loop] bad JSON on {self.context_capsule_topic}: {msg.data}"
            )
            return

        event_trace = obj.get("event_trace")

        if isinstance(event_trace, str):
            # LLM-generated short summary
            self._last_event_summary = event_trace.strip()
        elif isinstance(event_trace, list):
            # Fallback: compact stringified version of the last few events
            try:
                s = json.dumps(event_trace[-5:], ensure_ascii=False)
                self._last_event_summary = f"Recent events (raw): {s[:400]}"
            except Exception:
                self._last_event_summary = None


    # ---------- Planner proposals ingestion ----------
    def _on_planner_proposal(self, msg: StringMsg):
        """
        Ingest proposals from the planner/coordination node and immediately
        convert them into a concrete skill plan executed via /skills/execute.
        """
        try:
            obj = json.loads(msg.data) if msg.data else {}
        except Exception:
            self.get_logger().warn(
                f"[interaction_loop] bad JSON on {self.planner_proposal_topic}: {msg.data}"
            )
            return

        if not isinstance(obj, dict):
            return

        # Keep a snapshot for context
        self._last_planner_proposal = obj
        self._last_planner_proposal_ts = self._now()
        self.get_logger().info(
            "[interaction_loop] updated planner proposal snapshot"
        )
        
        # NEW: update local control-mode snapshot
        ctrl = obj.get("control") or {}
        new_mode = ctrl.get("mode", self._control_mode)
        new_target = ctrl.get("target", self._control_target)
        new_reason = ctrl.get("reason", self._control_reason)

        # normalize mode
        if new_mode not in ("autonomous", "follow_human", "idle_listen"):
            new_mode = self._control_mode
            new_target = self._control_target
            new_reason = f"invalid_mode_from_planner_keep_previous ({ctrl.get('mode')!r})"

        # if not follow_human, target must be None
        if new_mode != "follow_human":
            new_target = None

        if new_mode != self._control_mode or new_target != self._control_target:
            self.get_logger().info(
                f"[interaction_loop] control_mode update from planner: "
                f"{self._control_mode}/{self._control_target} → "
                f"{new_mode}/{new_target} (reason={new_reason})"
            )

        self._control_mode = new_mode
        self._control_target = new_target
        self._control_reason = new_reason
        self._control_ts = self._now()


        # Immediately try to turn this planner proposal into a concrete skills list
        skills_inventory = self._build_skills_inventory()
        skills_obj = self._proposal_to_skill_plan(obj, skills_inventory)

        if not skills_obj:
            self.get_logger().info(
                "[interaction_loop] planner proposal received but could not be mapped to concrete skills"
            )
            return

        # Use synthetic rule/type labels so downstream logging still makes sense
        self._publish_skill_plan(
            rule="planner_proposal",
            trig_type="planner",
            skills_obj=skills_obj,
        )
        self.get_logger().info(
            "[interaction_loop] executed planner proposal as a state_machine via /skills/execute"
        )

    # ---------- Control-mode helpers (gating) ----------
    def _extract_human_id_from_payload(self, payload: Dict[str, Any]) -> Optional[str]:
        """
        Best-effort extraction of a human id from an event payload.
        We check several common keys; if none present, fall back to HDT active_human.
        """
        if not isinstance(payload, dict):
            return self._active_human

        for key in ("human_id", "speaker_id", "hdt_id", "who", "source_human"):
            v = payload.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # Fallback to HDT's active human if defined
        return self._active_human

    def _should_handle_trigger(
        self,
        trig_type: str,
        payload: Dict[str, Any],
    ) -> bool:
        """
        Decide if this trigger should produce a reactive skill plan, based on
        the planner's control_mode and target.

        Logic:

        - Non-human triggers (idle, generic, planner, etc.) are ALWAYS handled.
        - trig_type == "human_command" is gated by control_mode:
            * autonomous    → ignore human_command (except logging)
            * idle_listen   → always handle human_command
            * follow_human:
                - target is None or "any" → handle from any human
                - target is "<human_id>" → handle only if event human matches
                  target or (no event id but active_human == target)
        """
        # Only gate human-origin triggers; everything else goes through
        if trig_type != "human_command":
            return True

        mode = self._control_mode
        target = self._control_target

        # 1) Autonomous: robot leads, ignore human_command by default
        if mode == "autonomous":
            self.get_logger().info(
                f"[interaction_loop] ignoring human_command because control_mode=autonomous"
            )
            return False

        # 2) Idle-listen: always respond to human_command
        if mode == "idle_listen":
            return True

        # 3) follow_human
        if mode == "follow_human":
            # No target or "any" → accept from anyone
            if target is None or target == "any":
                return True

            # Otherwise, require specific human id
            event_hid = self._extract_human_id_from_payload(payload)
            if event_hid == target:
                return True

            # Fallback: if we don't have an event id but HDT active matches target
            if event_hid is None and self._active_human == target:
                return True

            self.get_logger().info(
                f"[interaction_loop] ignoring human_command from '{event_hid}' "
                f"because control_mode=follow_human target={target}"
            )
            return False

        # Unknown mode (shouldn't happen): be safe and ignore
        self.get_logger().warn(
            f"[interaction_loop] unknown control_mode={mode!r}, ignoring human_command"
        )
        return False


    # ---------- HDT profiles ingestion ----------
    def _on_profiles(self, msg: StringMsg):
        try:
            obj = json.loads(msg.data) if msg.data else {}
        except Exception as e:
            self.get_logger().warn(
                f"[interaction_loop] bad JSON on {self.profiles_topic}: {e}"
            )
            return

        if not isinstance(obj, dict):
            return

        self._profiles_raw = obj

        meta = obj.get("_meta") or {}
        if isinstance(meta, dict):
            ah = meta.get("active_human")
            rz = meta.get("robot_zone")
            if isinstance(ah, str):
                self._active_human = ah
            if isinstance(rz, str):
                self._last_robot_zone = rz

        compact: Dict[str, Any] = {
            "active_human": self._active_human,
            "robot_zone": self._last_robot_zone,
            "humans": {},
        }
        for hid, prof in obj.items():
            if hid == "_meta":
                continue
            if not isinstance(prof, dict):
                continue
            compact["humans"][hid] = {
                "style": prof.get("style"),
                "helpfulness": prof.get("helpfulness"),
                "risk_aversion": prof.get("risk_aversion"),
                "contamination_bias": prof.get("contamination_bias"),
                "summary": prof.get("summary"),
            }

        self._profiles_compact = compact

    # ---------- Event handlers ----------
    def _on_basic_event(self, msg: StringMsg):
        try:
            evt = json.loads(msg.data)
        except Exception:
            self.get_logger().warn(
                "[interaction_loop] invalid JSON on /events/basic"
            )
            return

        if not isinstance(evt, dict):
            return

        rule = str(evt.get("rule") or "")
        data = evt.get("data") or {}
        ts = float(evt.get("ts") or self._now())
        zone = evt.get("zone")

        # Add to event trace (compact)
        entry = {
            "kind": "basic",
            "rule": rule,
            "ts": ts,
        }
        if zone is not None:
            entry["zone"] = zone

        txt = ""
        parsed_json = None

        if isinstance(data, dict):
            # --- NEW: try to parse structured json_text from VLM/LLM ---
            jt = data.get("json_text")
            if isinstance(jt, str) and jt.strip():
                try:
                    parsed_json = json.loads(jt)
                except Exception:
                    parsed_json = None

            # keep a short snippet of text for context
            txt = str(
                data.get("text")
                or data.get("utterance")
                or data.get("speech")
                or ""
            )

        if txt:
            entry["text_snippet"] = txt[:80]

        # --- NEW: attach parsed JSON (or a short snippet) if available ---
        if parsed_json is not None:
            entry["json"] = parsed_json
        elif isinstance(data, dict):
            jt = data.get("json_text")
            if isinstance(jt, str) and jt.strip():
                entry["json_text_snippet"] = jt[:160]

        self._event_trace.append(entry)

        # Decide if this is a trigger
        trig_type = self.trigger_map.get(rule)
        if not trig_type and rule.startswith(self.trigger_prefix):
            trig_type = "generic_trigger"

        if trig_type:
            self.get_logger().info(
                f"[interaction_loop] trigger basic rule={rule}, type={trig_type}"
            )
            self._run_for_trigger(
                rule=rule,
                trig_type=trig_type,
                kind="basic",
                ts=ts,
                zone=zone,
                payload=data,
            )


    def _on_comp_event(self, msg: StringMsg):
        try:
            evt = json.loads(msg.data)
        except Exception:
            self.get_logger().warn(
                "[interaction_loop] invalid JSON on /events/composite"
            )
            return

        if not isinstance(evt, dict):
            return

        rule = str(evt.get("rule") or "")
        expr = evt.get("expr") or ""
        ts = float(evt.get("ts") or self._now())
        zone = evt.get("zone")
        data = evt.get("data") or {}   # NEW: in case we start attaching payloads

        entry = {
            "kind": "composite",
            "rule": rule,
            "ts": ts,
        }
        if zone is not None:
            entry["zone"] = zone
        if expr:
            entry["expr_snippet"] = str(expr)[:120]

        # --- NEW: if composite carries json_text, parse and store it ---
        parsed_json = None
        if isinstance(data, dict):
            jt = data.get("json_text")
            if isinstance(jt, str) and jt.strip():
                try:
                    parsed_json = json.loads(jt)
                except Exception:
                    parsed_json = None

        if parsed_json is not None:
            entry["json"] = parsed_json
        elif isinstance(data, dict):
            jt = data.get("json_text")
            if isinstance(jt, str) and jt.strip():
                entry["json_text_snippet"] = jt[:160]

        self._event_trace.append(entry)

        trig_type = self.trigger_map.get(rule)
        if not trig_type and rule.startswith(self.trigger_prefix):
            trig_type = "generic_trigger"

        if trig_type:
            self.get_logger().info(
                f"[interaction_loop] trigger composite rule={rule}, type={trig_type}"
            )
            # If you later attach more to evt["data"], you can pass it here instead of just {"expr": expr}
            payload = {"expr": expr}
            self._run_for_trigger(
                rule=rule,
                trig_type=trig_type,
                kind="composite",
                ts=ts,
                zone=zone,
                payload=payload,
            )


    # ---------- Planner proposal → skill plan mapping ----------
    def _proposal_to_skill_plan(
        self,
        proposal_root: Dict[str, Any],
        skills_inventory: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Map a planner proposal into the standard skills list format.

        We assume either:
          proposal_root = { "summary": ..., "steps":[...] }
        or:
          proposal_root = { "objective": {...}, "proposal": { "summary":..., "steps":[...] }, ... }
        """
        # Extract the inner proposal dict
        if "proposal" in proposal_root and isinstance(proposal_root["proposal"], dict):
            proposal = proposal_root["proposal"]
        else:
            proposal = proposal_root

        steps = proposal.get("steps") or []
        if not isinstance(steps, list) or not steps:
            self.get_logger().info(
                "[interaction_loop] planner proposal has no steps; cannot map to skill plan"
            )
            return None

        # Build set of available skill names
        all_skill_names = {
            s["name"]
            for group in skills_inventory.values()
            for s in group
        }

        skill_steps: List[Dict[str, Any]] = []
        for step in steps:
            if not isinstance(step, dict):
                continue

            skill_name = None
            ctx: Dict[str, Any] = {}

            # Priority 1: explicit skill_hint (expected from planner)
            hint = step.get("skill_hint")
            if isinstance(hint, str) and hint in all_skill_names:
                skill_name = hint

            # Priority 2: explicit skill field that matches a skill
            if not skill_name:
                sname = step.get("skill")
                if isinstance(sname, str) and sname in all_skill_names:
                    skill_name = sname

            # Priority 3: crude mapping from type/zone to known skills
            if not skill_name:
                stype = step.get("type")
                zone = step.get("zone")
                if isinstance(stype, str):
                    if stype == "move" and isinstance(zone, str):
                        cand1 = f"goto_zone_{zone}"
                        cand2 = f"nav.goto_zone_{zone}"
                        if cand1 in all_skill_names:
                            skill_name = cand1
                        elif cand2 in all_skill_names:
                            skill_name = cand2
                    elif stype == "survey":
                        for cand in ["survey_scene", "survey_scene_or_patrol"]:
                            if cand in all_skill_names:
                                skill_name = cand
                                break
                    elif stype == "summarize":
                        for cand in ["summarize_progress", "summarize_state"]:
                            if cand in all_skill_names:
                                skill_name = cand
                                break

            # Priority 4: fall back to tts.say to at least narrate the step
            if not skill_name and "tts.say" in all_skill_names:
                desc = step.get("description") or proposal.get("summary") or "I will continue working on the task."
                skill_name = "tts.say"
                ctx = {"text": str(desc)}

            if not skill_name:
                # If we can't map this particular step, skip it
                continue

            skill_steps.append({
                "use": skill_name,
                "with": ctx,
            })

        if not skill_steps:
            self.get_logger().info(
                "[interaction_loop] unable to map proposal steps to known skills"
            )
            return None

        summary = proposal.get("summary") or "planner_proposal"
        plan_name = f"planner_proposal.{int(self._now() * 1000)}"

        return {
            "name": plan_name,
            "skills": skill_steps,
        }

    # ---------- Main trigger path ----------
    def _run_for_trigger(
        self,
        rule: str,
        trig_type: str,
        kind: str,
        ts: float,
        zone: Any,
        payload: Dict[str, Any],
    ):
    
        # 🔹 NEW: gating based on planner control_mode
        if not self._should_handle_trigger(trig_type, payload):
            self.get_logger().info(
                f"[interaction_loop] trigger {rule} (type={trig_type}) ignored due to control_mode="
                f"{self._control_mode}/{self._control_target}"
            )
            return
    
        # Extract text snippet from payload if any
        text = ""
        if isinstance(payload, dict):
            text = str(
                payload.get("text")
                or payload.get("utterance")
                or payload.get("speech")
                or ""
            )

        trigger_dict = {
            "rule": rule,
            "type": trig_type,
            "kind": kind,
            "zone": zone,
            "ts": ts,
            "text": text[:200] if text else "",
        }

        capsule: Dict[str, Any] = {
            "trigger": trigger_dict,
            "recent_events": list(self._event_trace),
            "task_state": self._last_task_state,
        }

        # NEW: include control-mode snapshot for the LLM (for context only)
        capsule["control"] = {
            "mode": self._control_mode,
            "target": self._control_target,
            "reason": self._control_reason,
            "ts": self._control_ts,
        }


        # NEW: attach global event summary from Broker if available
        if self._last_event_summary:
            capsule["event_summary"] = self._last_event_summary


        # Include latest planner proposal (if any) just for context to the LLM
        if self._last_planner_proposal:
            capsule["planner_proposal"] = self._last_planner_proposal

        if self._profiles_compact:
            capsule["humans"] = self._profiles_compact

        # Build skills inventory
        skills_inventory = self._build_skills_inventory()

        # Normal path: LLM-based or heuristic selector
        payload_for_llm = {
            "trigger_type": trig_type,
            "router_capsule": capsule,
            "skills_inventory": skills_inventory,
        }

        # Decide whether to call LLM or use fallback
        if self.llm_enabled:
            try:
                skills_obj = self._call_skill_selector_llm(payload_for_llm)
            except Exception as e:
                self.get_logger().error(
                    f"[interaction_loop] LLM skill selector failed: {e}"
                )
                skills_obj = self._fallback_skill_plan(trig_type, capsule, skills_inventory)
        else:
            skills_obj = self._fallback_skill_plan(trig_type, capsule, skills_inventory)

        if not skills_obj:
            self.get_logger().info(
                "[interaction_loop] no skill plan produced (LLM + fallback failed)"
            )
            return

        # Execute via /skills/execute in SAME FORMAT as router
        self._publish_skill_plan(rule, trig_type, skills_obj)

    # ---------- LLM call ----------
    def _call_skill_selector_llm(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": FAST_SKILLS_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        self.get_logger().info(
            f"\n[interaction_loop] === SKILL SELECTOR PROMPT ===\n{messages}\n"
        )

        t0 = time.time()
        model = self.model

        # Groq vs OpenAI routing (same pattern as other nodes)
        if self.groq_prefix and self.groq_prefix in model:
            client = Groq()
            resp = client.chat.completions.create(
                model="openai/" + model,
                reasoning_effort="medium",
                messages=messages,
                response_format={"type": "json_object"},
            )
        else:
            client = OpenAI()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )

        t1 = time.time()
        lat_ms = (t1 - t0) * 1000.0

        content = resp.choices[0].message.content
        self.get_logger().info(
            "\n[interaction_loop] === SKILL SELECTOR RAW RESPONSE ===\n"
            + content
            + f"\nLatency: {lat_ms:.1f} ms\n"
        )

        obj = json.loads(content)
        validate(instance=obj, schema=FAST_SKILLS_LIST_SCHEMA)
        return obj

    # ---------- Fallback heuristic ----------
    def _fallback_skill_plan(
        self,
        trig_type: str,
        capsule: Dict[str, Any],
        skills_inventory: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Very simple heuristic if LLM is disabled or fails:
          - If human_command and tts.say exists, say something back.
          - Otherwise: no plan.
        """
        all_skill_names = {
            s["name"]
            for group in skills_inventory.values()
            for s in group
        }

        trigger = capsule.get("trigger") or {}
        text = str(trigger.get("text") or "").strip()

        if trig_type == "human_command":
            if "tts.say" in all_skill_names:
                say_text = text or "Okay, I'm on it."
                return {
                    "name": "reactive.fallback_tts",
                    "skills": [
                        {
                            "use": "tts.say",
                            "with": {"text": say_text},
                        }
                    ],
                }

        # No suitable fallback
        return None

    # ---------- Skills list → state_machine (MATCH router logic) ----------
    def _skills_list_to_state_machine(
        self,
        skills_obj: Dict[str, Any],
        rule: str,
        trig_type: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Turn {"name":?, "skills":[{use,with},...]} into a state_machine skill
        that skills_node understands, same pattern as router_node._skills_list_to_state_machine.
        """
        skills = skills_obj.get("skills") or []
        if not isinstance(skills, list) or not skills:
            self.get_logger().warn(
                "[interaction_loop] skills list is empty; nothing to build"
            )
            return None

        base_name = str(skills_obj.get("name") or "").strip()
        suffix = int(self._now() * 1000)

        if not base_name:
            base_name = f"{trig_type}.{rule}.{suffix}"

        # We don't enforce a prefix here (router uses router_fast.), but you can if you want.
        states: List[Dict[str, Any]] = []
        for idx, step in enumerate(skills):
            use = str(step.get("use") or "").strip()
            if not use:
                continue
            with_params = step.get("with") or {}
            state_id = f"s{idx+1}"
            next_state = f"s{idx+2}" if idx < len(skills) - 1 else None
            states.append(
                {
                    "id": state_id,
                    "type": "action",
                    "action": {
                        "use": use,
                        "with": with_params,
                    },
                    "on_complete": next_state,
                    "on_failure": None,
                }
            )

        if not states:
            self.get_logger().warn(
                "[interaction_loop] after filtering, no valid states in skills list"
            )
            return None

        sm = {
            "name": base_name,
            "kind": "state_machine",
            "description": (
                f"interaction_loop auto-generated sequence for {trig_type}/{rule}"
            ),
            "params_template": {},
            "param_keys": [],
            "when": {},          # arm immediately
            "until": {},
            "initial_state": states[0]["id"],
            "states": states,
        }
        return sm

    def _execute_skill(self, entry: Dict[str, Any]):  # ### NEW
        """
        Publish a /skills/execute command for a state_machine or composite skill.
        This matches RouterNode._execute_fast_skill format exactly.
        """
        try:
            skill_name = str(entry.get("skill") or "")
            if not skill_name:
                return
            ctx = entry.get("ctx") or {}
            payload = {"skill": skill_name, "ctx": ctx}
            self.skills_execute_pub.publish(
                StringMsg(data=json.dumps(payload))
            )
            self.get_logger().info(
                f"[interaction_loop] execute: skill='{skill_name}' ctx={ctx}"
            )
        except Exception as e:
            self.get_logger().error(
                f"[interaction_loop] failed to publish /skills/execute: {e}"
            )

    def _append_composite_and_reload(  # ### NEW
        self,
        composite_skill: Dict[str, Any],
        to_execute: Dict[str, Any],
    ):
        """
        Append composite_skill to skills_composite.yaml, reload skills,
        then execute to_execute via /skills/execute.

        Mirrors router_node._append_composite_and_reload, but scoped to this node.
        """
        if not self.skills_composite_path:
            self.get_logger().warn(
                "[interaction_loop] no skills_composite_path set; cannot persist sequence"
            )
            return

        # Read or init doc
        doc = self._read_yaml_if_exists(self.skills_composite_path) or {
            "version": 2,
            "defaults": {"window_ms": 3000},
            "skills": [],
        }
        skills_list = doc.get("skills") or []
        skills_list.append(composite_skill)
        doc["skills"] = skills_list

        try:
            with open(self.skills_composite_path, "w") as f:
                yaml.safe_dump(doc, f, sort_keys=False)
            self.get_logger().info(
                f"[interaction_loop] wrote composite '{composite_skill['name']}' to skills_composite.yaml"
            )
        except Exception as e:
            self.get_logger().error(
                f"[interaction_loop] failed to write skills_composite.yaml: {e}"
            )
            return

        # Reload skills, then execute
        def _after_reload(_future):
            try:
                res = _future.result()
                if res and res.success:
                    self.get_logger().info(
                        f"[interaction_loop] /skills/reload after sequence ok: {res.message}"
                    )
                else:
                    self.get_logger().warn(
                        "[interaction_loop] /skills/reload after sequence failed or returned None"
                    )
            except Exception as e:
                self.get_logger().warn(
                    f"[interaction_loop] /skills/reload call error after sequence: {e}"
                )

            self._execute_skill(to_execute)

        if self.reload_skills_client.wait_for_service(timeout_sec=1.0):
            req = Trigger.Request()
            fut = self.reload_skills_client.call_async(req)
            fut.add_done_callback(_after_reload)
        else:
            self.get_logger().warn(
                "[interaction_loop] /skills/reload not available; executing sequence without reload"
            )
            self._execute_skill(to_execute)

    # ---------- Publishing (now via /skills/execute) ----------
    def _publish_skill_plan(
        self,
        rule: str,
        trig_type: str,
        skills_obj: Dict[str, Any],
    ):
        """
        Take a skills_obj (FAST_SKILLS_LIST_SCHEMA) and:
          1) Convert it to a state_machine skill.
          2) Append it to skills_composite.yaml.
          3) Reload skills.
          4) Execute via /skills/execute with {"skill":..., "ctx":{}}.

        This matches router_node's pattern so the over-the-wire format is identical.
        """
        sm = self._skills_list_to_state_machine(skills_obj, rule, trig_type)
        if not sm:
            self.get_logger().warn(
                "[interaction_loop] _publish_skill_plan: could not build state_machine from skills list"
            )
            return

        to_execute = {
            "skill": sm["name"],
            "ctx": {},  # you could inject trigger text here if you want
        }
        self._append_composite_and_reload(sm, to_execute)


def main(args=None):
    rclpy.init(args=args)
    node = InteractionLoopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

