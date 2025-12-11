#!/usr/bin/env python3
"""
planner_node.py

High-level task planner / critic.

- Subscribes:
    /events/basic      (std_msgs/String, JSON: {"rule":..., "data":..., "ts":..., "zone":...})
    /events/composite  (std_msgs/String, JSON: {"rule":..., "expr":..., "ts":..., "zone":...})
    /task_state        (std_msgs/String, JSON: task progress & robot objective)
    /profiles/summary  (std_msgs/String, JSON: HDT human profiles)

- Parameters:
    llm_enabled (bool): use LLM or fallback heuristics
    model (str):       OpenAI model id (e.g. "gpt-5.1-mini")
    groq_model_prefix (str): prefix to detect Groq models (e.g. "gpt-oss")
    trigger_prefix (str): rules starting with this are treated as triggers
    trigger_map_json (str): JSON mapping rule_id -> trigger_type (e.g. "trigger_idle" -> "idle")
    run_period_sec (float): periodic planning interval
    skills_base_path (str): YAML with base skills
    skills_composite_path (str): YAML with composite/state_machine skills
    rules_init_path (str): YAML for initial rules (optional, for context only)
    rules_path (str): YAML for dynamic rules (optional, for context only)
    profiles_topic (str): topic for HDT profiles
    task_state_topic (str): topic for task state
    event_trace_len (int): number of recent events to keep

- Publishes:
    /planner/proposal  (std_msgs/String, JSON proposal)

The proposal shape is intentionally compatible with interaction_loop_node:

Either:
  {
    "summary": str,
    "steps": [ {...}, ... ]
  }

or:
  {
    "objective": {...},
    "proposal": {
      "summary": str,
      "steps": [
        {
          "description": str,
          "type": str,             # e.g., "move", "survey", "summarize", ...
          "zone": "A"|"B"|null,
          "skill_hint": str|null,  # MUST be an existing skill name, if set
          "params": {...}          # optional, small param dict
        },
        ...
      ]
    },
    "reason": str,
    "ts": float
  }

interaction_loop_node will look for proposal["steps"][i]["skill_hint"] first.
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

from jsonschema import validate, ValidationError
from openai import OpenAI
from groq import Groq


# ---------- JSON Schema for planner proposal ----------

PLANNER_PROPOSAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["proposal"],
    "properties": {
        "objective": {"type": ["object", "null"]},
        "proposal": {
            "type": "object",
            "required": ["summary", "steps"],
            "properties": {
                "summary": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "type": {"type": ["string", "null"]},
                            "zone": {"type": ["string", "null"]},
                            "skill_hint": {"type": ["string", "null"]},
                            "params": {"type": "object"},
                        },
                        "required": ["description"],
                        "additionalProperties": True,
                    },
                },
            },
            "additionalProperties": True,
        },
        "reason": {"type": ["string", "null"]},
        "ts": {"type": ["number", "null"]},
    },
    "additionalProperties": True,
}

PLANNER_SYSTEM_PROMPT = """
You are the HIGH-LEVEL TASK PLANNER and CRITIC for a mobile robot named Bob.

You receive a single JSON object with:
  - planner_capsule: {
      "task_state": {
         ...  # current task progress, backlogs, high_level_objective, robot_idle, etc.
      },
      "recent_events": [
        {
          "kind": "basic"|"composite",
          "rule": str,
          "zone": str|null,
          "ts": float,
          "text_snippet"?: str,
          "expr_snippet"?: str
        }, ...
      ],
      "last_trigger": {
          "rule": str,
          "type": str,     # e.g. "idle", "human_command", ...
          "kind": "basic"|"composite",
          "zone": str|null,
          "ts": float,
          "text": str
      } | null,
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
      } | null,
      "skills_inventory": {
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
            "kind": "composite" | "state_machine",
            "description": str,
            "params_template": object,
            "param_keys": [str, ...]
          }, ...
        ]
      }
    }

Your job:

1. CRITICALLY ASSESS the current task progress and objective:
   - Is the current high_level_objective still appropriate?
   - Is the robot stuck, idle, or under-utilized?
   - Are there obvious opportunities to reduce backlog or help humans?

2. OPTIONALLY UPDATE the high-level objective (conceptually) and then:

3. PRODUCE a SHORT PROPOSAL of concrete next steps for the robot to take.
   These steps will be handed to another module that maps them to executable skills.

Each step should:
  - Describe what to do in 1–2 sentences ("description").
  - Optionally specify:
      * "type":   a coarse action type (e.g., "move", "survey", "summarize",
                  "interact", "check_objects", "assist_human", etc.)
      * "zone":   which zone to focus on ("A" | "B" | null)
      * "skill_hint": a specific skill name from skills_inventory (primitive or composite),
                      but ONLY if it truly exists there.
      * "params": a small JSON object with key parameters (zone, counts, thresholds, etc.).

Guidelines:
- Keep the number of steps small (1–4).
- Be honest and critical: if the robot is wasting time, say so in the "reason".
- When choosing "skill_hint", use only names that appear in skills_inventory.
  If no appropriate skill exists, leave "skill_hint" as null.
- Prefer to reference composite/state_machine skills where appropriate
  (for example, "interact.greet_human" or "survey_scene_or_patrol"),
  but ONLY if they exist in skills_inventory.

You MUST respond with STRICT JSON only, of the form:

{
  "objective": {
    "summary": str,               // updated high-level objective in one sentence
    "focus_zone": "A"|"B"|null,   // where the robot should primarily work next
    "rationale": str              // short justification
  },
  "proposal": {
    "summary": str,               // summary of the proposed course of action
    "steps": [
      {
        "description": str,       // natural language description
        "type": "move" | "survey" | "summarize" | "interact" |
                "check_objects" | "assist_human" | "other" | null,
        "zone": "A"|"B"|null,
        "skill_hint": str|null,   // MUST be a real skill name if non-null
        "params": { ... small JSON ... }
      },
      ...
    ]
  },
  "reason": str,                  // short critique and justification of the proposal
  "ts": float                     // planning timestamp
}
""".strip()


class PlannerNode(Node):
    """
    High-level planner that periodically (and on idle triggers) assesses task
    progress, critiques the current objective, and publishes proposals to
    /planner/proposal for the interaction_loop_node to consume.
    """

    def __init__(self):
        super().__init__("planner_node")

        # ----- Parameters -----
        self.declare_parameter("llm_enabled", True)
        self.declare_parameter("model", "gpt-5.1-mini")
        self.declare_parameter("groq_model_prefix", "gpt-oss")

        self.declare_parameter("trigger_prefix", "trigger_")
        self.declare_parameter(
            "trigger_map_json",
            json.dumps({
                "trigger_idle": "idle",
            }),
        )

        # Run periodically even if not idle (seconds)
        self.declare_parameter("run_period_sec", 30.0)

        # Skills & rules paths (same shape as interaction loop)
        self.declare_parameter("skills_base_path", "")
        self.declare_parameter("skills_composite_path", "")
        self.declare_parameter("rules_init_path", "")
        self.declare_parameter("rules_path", "")

        self.declare_parameter("profiles_topic", "/profiles/summary")
        self.declare_parameter("task_state_topic", "/task_state")

        # Event trace length
        self.declare_parameter("event_trace_len", 50)

        # ----- Read parameters -----
        self.llm_enabled = bool(self.get_parameter("llm_enabled").value)
        self.model = str(self.get_parameter("model").value)
        self.groq_prefix = str(self.get_parameter("groq_model_prefix").value)

        self.trigger_prefix = str(self.get_parameter("trigger_prefix").value)
        self.trigger_map = json.loads(self.get_parameter("trigger_map_json").value)

        self.run_period_sec = float(self.get_parameter("run_period_sec").value)

        self.skills_base_path = self.get_parameter("skills_base_path").value
        self.skills_composite_path = self.get_parameter("skills_composite_path").value
        self.rules_init_path = self.get_parameter("rules_init_path").value
        self.rules_path = self.get_parameter("rules_path").value

        self.profiles_topic = self.get_parameter("profiles_topic").value
        self.task_state_topic = self.get_parameter("task_state_topic").value

        self.event_trace_len = int(self.get_parameter("event_trace_len").value)

        self.add_on_set_parameters_callback(self._on_param_change)

        # ----- Internal state -----
        self._event_trace = deque(maxlen=self.event_trace_len)

        self._last_task_state: Dict[str, Any] = {}
        self._profiles_raw: Dict[str, Any] = {}
        self._profiles_compact: Dict[str, Any] = {}
        self._active_human: Optional[str] = None
        self._last_robot_zone: Optional[str] = None

        # We remember the last trigger that invoked _run_planner (optional)
        self._last_trigger_ctx: Optional[Dict[str, Any]] = None

        # ----- ROS I/O -----
        # Events
        self.sub_basic = self.create_subscription(
            StringMsg, "/events/basic", self._on_basic_event, 200
        )
        self.sub_comp = self.create_subscription(
            StringMsg, "/events/composite", self._on_comp_event, 100
        )

        # Task state
        self.sub_task_state = self.create_subscription(
            StringMsg, self.task_state_topic, self._on_task_state, 20
        )

        # HDT profiles
        self.sub_profiles = self.create_subscription(
            StringMsg, self.profiles_topic, self._on_profiles, 10
        )

        # Proposals publisher
        self.pub_proposal = self.create_publisher(
            StringMsg, "/planner/proposal", 10
        )

        # Periodic timer
        self.timer = self.create_timer(
            self.run_period_sec, self._on_timer_tick
        )

        self.get_logger().info(
            f"planner_node initialized (run_period_sec={self.run_period_sec})"
        )

    # ---------- Param change ----------
    def _on_param_change(self, params):
        for p in params:
            if p.name == "llm_enabled" and p.type_ == Parameter.Type.BOOL:
                self.llm_enabled = p.value
                self.get_logger().info(
                    f"[planner] llm_enabled -> {self.llm_enabled}"
                )
            elif p.name == "model" and p.type_ == Parameter.Type.STRING:
                self.model = p.value
                self.get_logger().info(
                    f"[planner] model -> {self.model}"
                )
            elif p.name == "run_period_sec" and p.type_ in (
                Parameter.Type.DOUBLE,
                Parameter.Type.INTEGER,
            ):
                self.run_period_sec = float(p.value)
                # Update timer
                try:
                    self.timer.cancel()
                except Exception:
                    pass
                self.timer = self.create_timer(
                    self.run_period_sec, self._on_timer_tick
                )
                self.get_logger().info(
                    f"[planner] run_period_sec -> {self.run_period_sec}"
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
                f"[planner] failed to read YAML '{path}': {e}"
            )
            return None

    def _build_skills_inventory(self) -> Dict[str, Any]:
        """
        Build a small skills inventory for the planner:
          - primitives
          - composites (including state_machines)
        """
        primitives: List[Dict[str, Any]] = []
        composites: List[Dict[str, Any]] = []

        for path in [self.skills_base_path, self.skills_composite_path]:
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

    # ---------- Ingestion: task_state ----------
    def _on_task_state(self, msg: StringMsg):
        try:
            obj = json.loads(msg.data) if msg.data else {}
        except Exception:
            self.get_logger().warn(
                f"[planner] bad JSON on {self.task_state_topic}: {msg.data}"
            )
            return
        if isinstance(obj, dict):
            self._last_task_state = obj

    # ---------- Ingestion: HDT profiles ----------
    def _on_profiles(self, msg: StringMsg):
        try:
            obj = json.loads(msg.data) if msg.data else {}
        except Exception as e:
            self.get_logger().warn(
                f"[planner] bad JSON on {self.profiles_topic}: {e}"
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

    # ---------- Ingestion: events ----------
    def _on_basic_event(self, msg: StringMsg):
        try:
            evt = json.loads(msg.data)
        except Exception:
            self.get_logger().warn(
                "[planner] invalid JSON on /events/basic"
            )
            return

        if not isinstance(evt, dict):
            return

        rule = str(evt.get("rule") or "")
        data = evt.get("data") or {}
        ts = float(evt.get("ts") or self._now())
        zone = evt.get("zone")

        entry = {
            "kind": "basic",
            "rule": rule,
            "ts": ts,
        }
        if zone is not None:
            entry["zone"] = zone

        txt = ""
        if isinstance(data, dict):
            txt = str(
                data.get("text")
                or data.get("utterance")
                or data.get("speech")
                or ""
            )
        if txt:
            entry["text_snippet"] = txt[:80]

        self._event_trace.append(entry)

        # Detect triggers (including idle)
        trig_type = self.trigger_map.get(rule)
        if not trig_type and rule.startswith(self.trigger_prefix):
            trig_type = "generic_trigger"

        if trig_type == "idle":
            self.get_logger().info(
                "[planner] idle trigger via /events/basic; running planner now"
            )
            self._last_trigger_ctx = {
                "rule": rule,
                "type": trig_type,
                "kind": "basic",
                "zone": zone,
                "ts": ts,
                "text": txt[:200] if txt else "",
            }
            self._run_planner(trigger_ctx=self._last_trigger_ctx, idle_invocation=True)


    def _on_comp_event(self, msg: StringMsg):
        try:
            evt = json.loads(msg.data)
        except Exception:
            self.get_logger().warn(
                "[planner] invalid JSON on /events/composite"
            )
            return

        if not isinstance(evt, dict):
            return

        rule = str(evt.get("rule") or "")
        expr = evt.get("expr") or ""
        ts = float(evt.get("ts") or self._now())
        zone = evt.get("zone")

        entry = {
            "kind": "composite",
            "rule": rule,
            "ts": ts,
        }
        if zone is not None:
            entry["zone"] = zone
        if expr:
            entry["expr_snippet"] = str(expr)[:120]

        self._event_trace.append(entry)

        trig_type = self.trigger_map.get(rule)
        if not trig_type and rule.startswith(self.trigger_prefix):
            trig_type = "generic_trigger"

        # Idle trigger: run planner right away
        if trig_type == "idle":
            self.get_logger().info(
                "[planner] idle trigger via /events/composite; running planner now"
            )
            self._last_trigger_ctx = {
                "rule": rule,
                "type": trig_type,
                "kind": "composite",
                "zone": zone,
                "ts": ts,
                "text": "",
            }
            self._run_planner(trigger_ctx=self._last_trigger_ctx, idle_invocation=True)


    # ---------- Timer ----------
    def _on_timer_tick(self):
        """
        Periodic planning tick, even if not idle. We can pass the last trigger
        context (if any) but mark this as non-idle.
        """
        self.get_logger().info("[planner] periodic tick; running planner")
        self._run_planner(trigger_ctx=self._last_trigger_ctx, idle_invocation=False)


    # ---------- Planner core ----------
    def _run_planner(self, trigger_ctx: Optional[Dict[str, Any]], idle_invocation: bool = False):
        # Build capsule
        capsule: Dict[str, Any] = {
            "task_state": self._last_task_state,
            "recent_events": list(self._event_trace),
            "last_trigger": trigger_ctx,
            "skills_inventory": self._build_skills_inventory(),
        }
        if self._profiles_compact:
            capsule["humans"] = self._profiles_compact

        payload_for_llm = {"planner_capsule": capsule}

        if self.llm_enabled:
            try:
                proposal_obj = self._call_planner_llm(payload_for_llm)
            except Exception as e:
                self.get_logger().error(
                    f"[planner] LLM planner failed: {e}"
                )
                proposal_obj = self._fallback_proposal(capsule)
        else:
            proposal_obj = self._fallback_proposal(capsule)

        if not proposal_obj:
            self.get_logger().info(
                "[planner] no proposal produced (LLM + fallback failed)"
            )
            return

        # Indicate why this run happened
        if idle_invocation:
            # Explicitly idle-triggered
            trigger_type = (trigger_ctx or {}).get("type", "idle")
        else:
            # Non-idle: periodic or other
            trigger_type = (trigger_ctx or {}).get("type", "periodic")

        proposal_obj["idle_trigger"] = bool(idle_invocation)
        proposal_obj["trigger_type"] = trigger_type

        # Fill ts if missing
        if proposal_obj.get("ts") is None:
            proposal_obj["ts"] = self._now()

        # Publish
        try:
            s = json.dumps(proposal_obj, ensure_ascii=False)
        except Exception as e:
            self.get_logger().warn(
                f"[planner] failed to serialize proposal: {e}"
            )
            return

        self.pub_proposal.publish(StringMsg(data=s))
        self.get_logger().info(
            "[planner] published proposal to /planner/proposal"
        )


    # ---------- LLM call ----------
    def _call_planner_llm(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        self.get_logger().info(
            f"\n[planner] === PLANNER PROMPT ===\n{messages}\n"
        )

        t0 = time.time()
        model = self.model

        if self.groq_prefix and self.groq_prefix in model:
            client = Groq()
            resp = client.chat.completions.create(
                model="openai/" + model,
                messages=messages,
                reasoning_effort="medium",
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
            "\n[planner] === PLANNER RAW RESPONSE ===\n"
            + content
            + f"\nLatency: {lat_ms:.1f} ms\n"
        )

        obj = json.loads(content)
        validate(instance=obj, schema=PLANNER_PROPOSAL_SCHEMA)
        return obj

    # ---------- Fallback ----------
    def _fallback_proposal(self, capsule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Very coarse heuristic fallback:
          - If backlog exists in any zone, pick that as focus zone.
          - Suggest a generic "survey" step there.
        """
        ts_now = self._now()
        task_state = capsule.get("task_state") or {}

        # Try to infer a focus zone from backlog, default to "A"
        focus_zone = "A"
        backlog = task_state.get("backlog") or {}
        # Example expected shape:
        # backlog = {"zone_A": {...}, "zone_B": {...}, ...}
        try:
            # crude heuristic: more to_pick or in_basket => focus
            score_a = (
                (backlog.get("zone_A") or {}).get("to_pick", 0)
                + (backlog.get("zone_A") or {}).get("in_basket", 0)
            )
            score_b = (
                (backlog.get("zone_B") or {}).get("to_pick", 0)
                + (backlog.get("zone_B") or {}).get("in_basket", 0)
            )
            focus_zone = "A" if score_a >= score_b else "B"
        except Exception:
            pass

        objective = {
            "summary": f"Continue making progress in zone {focus_zone}",
            "focus_zone": focus_zone,
            "rationale": "LLM disabled; heuristic: focus where backlog appears highest.",
        }

        proposal = {
            "summary": f"Heuristic: survey zone {focus_zone} and look for objects to act on.",
            "steps": [
                {
                    "description": f"Move to zone {focus_zone} and survey the area.",
                    "type": "survey",
                    "zone": focus_zone,
                    "skill_hint": None,  # interaction_loop will map this if possible
                    "params": {"zone": focus_zone},
                }
            ],
        }

        reason = "LLM planner disabled or failed; using simple backlog-based heuristic."

        return {
            "objective": objective,
            "proposal": proposal,
            "reason": reason,
            "ts": ts_now,
        }


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

