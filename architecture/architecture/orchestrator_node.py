#!/usr/bin/env python3
# orchestrator_node.py
#
# Orchestrator linking planner_node and skills_node, using an OpenAI LLM
# to parse planner outputs into:
#   - new composite skills (in skills_composite.yaml)
#   - new rules (in rules.yaml)
#   - a list of skills to execute now (via /skills/execute)
#
# Base skills and base rules remain read-only:
#   skills_base_path  -> immutable skills.yaml
#   skills_composite_path -> writable skills_composite.yaml
#   rules_init_path   -> immutable rules_init.yaml
#   rules_path        -> writable rules.yaml
#
import json
import os
import time  # NEW
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter              # NEW
from std_msgs.msg import String as StringMsg
from std_srvs.srv import Trigger
from rcl_interfaces.srv import SetParameters       # NEW
from rcl_interfaces.msg import (                  # NEW
    Parameter as ParamMsg,
    ParameterValue,
    SetParametersResult,
    ParameterType
)

import yaml
from openai import OpenAI
from jsonschema import validate, ValidationError

# ---------- JSON schema for LLM output ----------

PLAN_PARSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["to_execute"],
    "properties": {
        "new_composites": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "kind", "steps"],
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": ["composite"]},
                    "when": {"type": "object"},
                    "until": {"type": "object"},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["use"],
                            "properties": {
                                "use": {"type": "string"},
                                "with": {"type": "object"},
                                "when": {"type": "object"},
                                "until": {"type": "object"},
                            },
                            "additionalProperties": True,
                        },
                    },
                },
                "additionalProperties": True,
            },
        },
        "new_rules": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "expr"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": ["basic", "composite"]},
                    "enabled": {"type": "boolean"},
                    "task": {"type": "string"},
                    "output": {"type": "string"},
                    "model_id": {"type": "string"},
                    "expr": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
        "to_execute": {
            "type": "object",
            "required": ["skill"],
            "properties": {
                "skill": {"type": "string"},
                "ctx": {"type": "object"},
            },
            "additionalProperties": True,
        },
        # NEW: optional LLM model choices per role ("broker", "planner", "orchestrator", etc.)
        "llm_models": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
    "additionalProperties": False,
}

SYSTEM_PARSE = """
You are the ORCHESTRATOR for a mobile robot collaborating with humans.

The planner has already produced a high-level plan (plan_header, plan_doc, _hints).
You also have:
  - A skills library (primitives and composites)
  - A rules library
  - A tasks_inventory describing all perception, LLM, and VLM tools
  - The task registry includes performance metrics for all models (latency, fps, etc.)

Your job is to:
  1) Decide which skill (state-machine) to activate next
  2) Optionally define NEW composite skills as finite state machines
  3) Optionally define NEW rules (prefer composite rules using exists('rule_id', ms))
  4) Optionally recommend updated LLM/VLM models in llm_models, based on performance
     metrics available in tasks_inventory and perf.json outputs
  5) Return STRICT JSON ONLY following the required schema


======================================================================
 GENERAL RULES
======================================================================
- Do NOT invent new task names or outputs. Only use tasks/outputs from tasks_inventory.
- You SHOULD NOT reference existing rules whose id starts with "trigger_".
- You MAY reference existing composite or state_machine skills (from skills_inventory) as actions inside state machines (treat them as black-box actions that run until they finish).
- Use ctx to pass small argument sets (object_ids, zones, text, etc.).
- When proposing model upgrades or downgrades in llm_models:
    * Use only performance metrics documented in tasks_inventory (and perf.json outputs if present).
    * Prefer models with lower latency and acceptable quality.
    * If no clear improvement is indicated, omit the role.


======================================================================
 STATE-MACHINE SKILLS
======================================================================
All NEW composite skills must be expressed as a finite state machine:

{
  "name": "string",
  "kind": "state_machine",
  "params_template": { ... },
  "param_keys": [ ... ],
  "when": {},                    // must be empty for skills used in to_execute
  "until": { ...optional... },
  "initial_state": "state_id",
  "states": [ ... ]
}

Two allowed state types:

1) ACTION STATE:
   {
     "id": "string",
     "type": "action",
     "action": { "use": "<primitive_or_composite>", "with": { ... } },
     "on_complete": "next_state_or_done",
     "on_failure":  "optional_state_or_done"
   }

2) WAIT STATE:
   {
     "id": "string",
     "type": "wait",
     "wait_for": {
       "any_of": [
         { "rule_id": "rule_name", "within_ms": <timeout> }
       ]
     },
     "on_event":   "next_state_or_done",
     "on_timeout": "timeout_state_or_done"
   }

Behavior guidelines:
- Each state machine represents ONE SHORT INTERACTION SEGMENT.
- action.use may be a primitive or another composite/state_machine skill.
  - Nested skills run to completion before on_complete / on_failure is evaluated.
- Examples:
  * ask clarifying question then wait for reply
  * call LLM/VLM primitive then wait for rule based on its output
  * move short distance then wait for a perceptual event
- If future actions depend on new speech, gesture, BT events, LLM output, or VLM output:
  * end the current skill, then let rules trigger a new one.
- Avoid recursion: do NOT have a state machine that eventually calls itself via action.use.



======================================================================
 LLM AND VLM MICRO-SERVICES
======================================================================
Some primitives wrap lower-level LLM or VLM modules and allow custom prompts and
output schemas.

Example primitives from skills_inventory:

1) llm.check_speech
   Params may include:
     - check_kind: short mode label (example: "carry_intent")
     - prompt: optional free-text instructions (overrides default template if non-empty)
     - output_schema: optional JSON schema string describing expected fields
     - text: utterance to analyze
   The underlying node:
     - Builds final prompt
     - Calls an LLM and returns structured JSON
     - Publishes output on its task output (example: check.json)

2) vlm.run_query
   Params may include:
     - mode: short mode label (example: "pointing_zone")
     - prompt: optional free-text question/instructions
   The node:
     - Builds prompt
     - Calls VLM
     - Publishes structured JSON on output topic

When using these:
- Keep prompts short and precise
- Define small JSON output schemas
- Use ctx variables such as {{ctx.last_utterance}}
- After calling an LLM or VLM primitive in an action state, follow it with a WAIT state
  listening for a rule that checks the parsed JSON output.


======================================================================
 RULE CREATION (new_rules)
======================================================================
Allowed rule types:

1) Composite rules:
   - type: "composite"
   - expr uses exists('rule_id', ms)
   - Preferred whenever possible.

2) Basic rules:
   - type: "basic"
   - Must specify: id, task, output, mode, expr, enabled, and model_id if required
   - May only reference tasks/outputs from tasks_inventory
   - Useful for interpreting JSON from LLM/VLM (example: is_carry, zone, confidence, etc.)

Do NOT:
- invent task names
- invent outputs
- create new rules whose id starts with "trigger_"


======================================================================
 SKILL TO EXECUTE NOW (to_execute)
======================================================================
Your JSON must contain exactly one skill to run:

"to_execute": {
  "skill": "<state_machine_name>",
  "ctx": { ... small context ... }
}

Requirements:
- The selected skill must exist in skills_inventory or new_composites.
- It must have top-level "when": {} to arm immediately.
- Execution flow must be fully determined by its state-machine states.


======================================================================
 LLM MODEL SELECTION (llm_models)
======================================================================
The llm_models section may override which model each role uses:

{
  "broker": "model_id",
  "planner": "model_id",
  "orchestrator": "model_id",
  "vlm": "model_id"
}

Rules for selecting models:
- Base decisions only on performance metrics from tasks_inventory and perf.json outputs.
- Look at fields like latency_ms.det_mean, utter_infer_mean, pose_mean, run_trigger_typical, etc.
- Prefer lower latency and stable performance.
- If no strong evidence for improvement, omit role.


======================================================================
 STRICT JSON OUTPUT SCHEMA
======================================================================
You MUST output STRICT JSON ONLY:

{
  "new_composites": [
    {
      "name": "...",
      "kind": "state_machine",
      "params_template": { ... },
      "param_keys": [ ... ],
      "when": {},
      "until": { ...optional... },
      "initial_state": "state_id",
      "states": [
        {
          "id": "state_id",
          "type": "action",
          "action": { "use": "primitive_or_composite", "with": { ... } },
          "on_complete": "next_state_or_done",
          "on_failure": "optional_state_or_done"
        },
        {
          "id": "state_id_2",
          "type": "wait",
          "wait_for": {
            "any_of": [
              { "rule_id": "rule_id", "within_ms": <int> }
            ]
          },
          "on_event": "next_state_or_done",
          "on_timeout": "timeout_state_or_done"
        }
      ]
    }
  ],
  "new_rules": [
    {
      "id": "...",
      "type": "basic_or_composite",
      "task": "...",        // only for basic
      "output": "...",      // only for basic
      "model_id": "...",    // optional
      "mode": "edge_or_level",
      "expr": "expression OR exists('other_rule', ms)",
      "enabled": true
    }
  ],
  "to_execute": {
    "skill": "<state_machine_name>",
    "ctx": { ... }
  },
  "llm_models": {
    "optional_role": "optional_model_id"
  }
}
"""

class OrchestratorNode(Node):
    """
    Bridges PlannerNode and SkillsAgent with an LLM-driven planner-to-skills+rules parser.
    Base skills/rules are read-only; dynamic composites and rules are appended to
    skills_composite.yaml and rules.yaml respectively.

    LLM responsibilities now also include suggesting which LLM models should be used
    by broker / planner / orchestrator based on the performance info in the
    broker ContextCapsule (passed into the prompt).
    """

    def __init__(self):
        super().__init__("orchestrator_node")

        # ---------------- Parameters ----------------
        self.declare_parameter("skills_base_path", "")
        self.declare_parameter("skills_composite_path", "")
        self.declare_parameter("rules_init_path", "")
        self.declare_parameter("rules_path", "")
        self.declare_parameter("registry_path", "")
        self.declare_parameter("dynamic_prefix", "")
        self.declare_parameter("max_dynamic_skills", 10)
        self.declare_parameter("max_dynamic_rules", 20)
        self.declare_parameter("model", "gpt-5-mini")
        self.declare_parameter("temperature", 0.1)

        # NEW: where to listen for broker context (which includes perf DB)
        self.declare_parameter("capsule_topic", "/broker/context_capsule")

        # NEW: where to publish THIS node's LLM perf so broker can ingest it
        self.declare_parameter("perf_topic", "/llm/orchestrator_perf")

        # NEW: broker service that writes perf EMA into task_registry.yaml
        self.declare_parameter(
            "registry_refresh_service",
            "/broker/save_task_registry_with_perf",
        )

        # NEW: mapping from logical roles to (node, param) for remote model changes
        self.declare_parameter(
            "llm_targets_json",
            json.dumps({
                "broker": {"node": "broker_node", "param": "llm_model"},
                "planner": {"node": "planner_node", "param": "model"},
                "orchestrator": {"node": "orchestrator_node", "param": "model"},
            }),
        )

        self.skills_base_path: str = (
            self.get_parameter("skills_base_path")
            .get_parameter_value()
            .string_value
        )
        self.skills_composite_path: str = (
            self.get_parameter("skills_composite_path")
            .get_parameter_value()
            .string_value
        )
        self.rules_init_path: str = (
            self.get_parameter("rules_init_path")
            .get_parameter_value()
            .string_value
        )
        self.rules_path: str = (
            self.get_parameter("rules_path")
            .get_parameter_value()
            .string_value
        )
        self.registry_path: str = (
            self.get_parameter("registry_path")
            .get_parameter_value()
            .string_value
        )


        # NEW: registry refresh service name and client
        self.registry_refresh_service: str = (
            self.get_parameter("registry_refresh_service")
            .get_parameter_value()
            .string_value
        )

        self.registry_refresh_client = None
        if self.registry_refresh_service:
            self.registry_refresh_client = self.create_client(
                Trigger, self.registry_refresh_service
            )

        self.dynamic_prefix: str = (
            self.get_parameter("dynamic_prefix")
            .get_parameter_value()
            .string_value
        )
        self.max_dynamic_skills: int = int(
            self.get_parameter("max_dynamic_skills").value
        )
        self.max_dynamic_rules: int = int(
            self.get_parameter("max_dynamic_rules").value
        )
        self.model: str = (
            self.get_parameter("model").get_parameter_value().string_value
        )
        self.temperature: float = float(
            self.get_parameter("temperature").value
        )

        self.capsule_topic: str = (
            self.get_parameter("capsule_topic")
            .get_parameter_value()
            .string_value
        )
        self.perf_topic: str = (
            self.get_parameter("perf_topic")
            .get_parameter_value()
            .string_value
        )
        self.llm_targets: Dict[str, Any] = json.loads(
            self.get_parameter("llm_targets_json")
            .get_parameter_value()
            .string_value
        )

        if not self.skills_composite_path:
            self.get_logger().warn(
                "No skills_composite_path set; orchestrator cannot persist new composites."
            )
        if not self.rules_path:
            self.get_logger().warn(
                "No rules_path set; orchestrator cannot persist new rules."
            )

        # ---------------- OpenAI client ----------------
        self.client = OpenAI()

        # ---------------- State ----------------
        self._capsule: Dict[str, Any] = {}  # NEW: last broker ContextCapsule

        # ---------------- ROS I/O ----------------
        self.sub_plan = self.create_subscription(
            StringMsg, "/planner/plan_out", self._on_plan, 10
        )

        # NEW: subscribe to broker context (which contains perf summaries, etc.)
        self.sub_capsule = self.create_subscription(
            StringMsg, self.capsule_topic, self._on_capsule, 10
        )

        self.pub_execute = self.create_publisher(
            StringMsg, "/skills/execute", 10
        )

        # NEW: publish orchestrator LLM perf so broker can ingest it
        self.pub_perf = self.create_publisher(
            StringMsg, self.perf_topic, 10
        )

        self.reload_skills_client = self.create_client(
            Trigger, "/skills/reload"
        )
        self.reload_rules_client = self.create_client(
            Trigger, "/event_layer/reload_rules"
        )

        self.cancel_skills_client = self.create_client(
            Trigger, "/skills/cancel_all"
        )

        # NEW: SetParameters clients for remote LLM model changes
        self._param_clients: Dict[str, Any] = {}
        for role, info in self.llm_targets.items():
            node_name = info.get("node")
            if not node_name:
                continue
            srv_name = f"/{node_name}/set_parameters"
            self._param_clients[role] = self.create_client(
                SetParameters, srv_name
            )

        # NEW: allow dynamic change of THIS node's own model / temperature
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            f"orchestrator_node up | base='{self.skills_base_path}' | "
            f"composite='{self.skills_composite_path}' | "
            f"rules_init='{self.rules_init_path}' | rules='{self.rules_path}' | "
            f"model={self.model} | capsule_topic={self.capsule_topic} | perf_topic={self.perf_topic}"
        )


    # =====================================================================
    # Dynamic params for orchestrator itself (model / temperature)
    # =====================================================================

    def _on_set_parameters(self, params):
        result = SetParametersResult()
        result.successful = True
        result.reason = "ok"
        for p in params:
            if p.name == "model" and p.type_ == Parameter.Type.STRING:
                self.model = p.value
                self.get_logger().info(f"[orchestrator] model changed to: {self.model}")
            elif p.name == "temperature":
                # Accept double or int
                try:
                    self.temperature = float(p.value)
                    self.get_logger().info(f"[orchestrator] temperature -> {self.temperature}")
                except Exception:
                    pass
        return result

    # =====================================================================
    # YAML helpers
    # =====================================================================

    def _read_yaml_if_exists(self, path: str) -> Optional[dict]:
        if not path:
            return None
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().error(f"Failed to read YAML '{path}': {e}")
            return None

    # --- skills (composite) ---

    def _ensure_composite_yaml(self) -> dict:
        """
        Load or create the composite skills YAML doc:
        {version, defaults, skills:[]}
        """
        doc = self._read_yaml_if_exists(self.skills_composite_path)
        if doc is None:
            doc = {"version": 2, "defaults": {"window_ms": 3000}, "skills": []}
        if "skills" not in doc or not isinstance(doc["skills"], list):
            doc["skills"] = []
        return doc

    def _write_composite_yaml(self, doc: Dict[str, Any]) -> bool:
        if not self.skills_composite_path:
            return False
        try:
            with open(self.skills_composite_path, "w") as f:
                yaml.safe_dump(doc, f, sort_keys=False)
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to write composite YAML: {e}")
            return False

    def _build_skills_inventory(self) -> Dict[str, Any]:
        """
        Build a combined inventory of base + state_machine skills for the LLM.

        We expose:
          - primitives: name, kind, action, params_template, param_keys, when, until
          - composites: state machines only (kind == "state_machine"), including:
              * name
              * kind ("state_machine")
              * params_template / param_keys
              * when / until
              * initial_state
              * states (FULL array, no truncation)

        NOTE:
        - Legacy 'composite' + 'steps' skills are ignored here. If you still have them
          in your YAML, they just won’t show up in skills_inventory.
        """
        primitives: List[Dict[str, Any]] = []
        state_machines: List[Dict[str, Any]] = []

        for path in [self.skills_base_path, self.skills_composite_path]:
            doc = self._read_yaml_if_exists(path) or {}
            for s in doc.get("skills", []) or []:
                if not isinstance(s, dict):
                    continue

                name = str(s.get("name", "")).strip()
                if not name:
                    continue

                kind = str(s.get("kind", "")).strip() or "primitive"
                params_template = s.get("params") or {}
                when_block = s.get("when") or {}
                until_block = s.get("until") or {}

                base_entry: Dict[str, Any] = {
                    "name": name,
                    "kind": kind,
                    "params_template": params_template,
                    "param_keys": (
                        list(params_template.keys())
                        if isinstance(params_template, dict)
                        else []
                    ),
                    "when": when_block,
                    "until": until_block,
                }

                if kind == "primitive":
                    # Primitive: expose action + params
                    base_entry["action"] = s.get("action", "")
                    primitives.append(base_entry)

                elif kind == "state_machine":
                    # State machine: expose full structure
                    base_entry["initial_state"] = s.get("initial_state")
                    base_entry["states"] = s.get("states") or []
                    # optional extras if you later add them in YAML (timeouts, docs, etc.)
                    if "description" in s:
                        base_entry["description"] = s.get("description")
                    state_machines.append(base_entry)

                else:
                    # Unknown / legacy kinds (e.g., old "composite" with steps) are ignored.
                    # If you still need them, either convert to state_machine in YAML
                    # or add a compatibility branch here.
                    continue

        # For compatibility with the prompt + planner code, we still call this key "composites"
        # even though it now contains only kind=="state_machine" entries.
        return {
            "primitives": primitives,
            "composites": state_machines,
        }



    def _prune_old_dynamic_skills(
        self, skills: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Keep at most max_dynamic_skills entries whose names start with dynamic_prefix.
        Only applied to the *composite* file.
        """
        if not self.dynamic_prefix:
            return skills

        dynamic_indices = [
            i
            for i, s in enumerate(skills)
            if isinstance(s, dict)
            and str(s.get("name", "")).startswith(self.dynamic_prefix)
        ]
        if len(dynamic_indices) <= self.max_dynamic_skills:
            return skills

        to_drop = dynamic_indices[
            0 : len(dynamic_indices) - self.max_dynamic_skills
        ]
        keep = [
            s for idx, s in enumerate(skills) if idx not in to_drop
        ]
        return keep

    # --- rules (dynamic) ---

    def _ensure_rules_yaml(self) -> dict:
        """
        Load or create the dynamic rules YAML doc:
        {version, defaults?, rules:[]}
        Note: defaults usually live in rules_init.yaml; this file can omit them.
        """
        doc = self._read_yaml_if_exists(self.rules_path)
        if doc is None:
            doc = {"version": 1, "rules": []}
        if "rules" not in doc or not isinstance(doc["rules"], list):
            doc["rules"] = []
        return doc

    def _write_rules_yaml(self, doc: Dict[str, Any]) -> bool:
        if not self.rules_path:
            return False
        try:
            with open(self.rules_path, "w") as f:
                yaml.safe_dump(doc, f, sort_keys=False)
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to write rules YAML: {e}")
            return False

    def _build_rules_inventory(self) -> Dict[str, Any]:
        """
        Build a combined inventory of base + dynamic rules for the LLM.
        """
        rules = []

        for path in [self.rules_init_path, self.rules_path]:
            doc = self._read_yaml_if_exists(path) or {}
            for r in doc.get("rules", []) or []:
                if not isinstance(r, dict):
                    continue
                rules.append(
                    {
                        "id": str(r.get("id", "")),
                        "type": r.get("type", "basic"),
                        "task": r.get("task"),
                        "output": r.get("output"),
                        "model_id": r.get("model_id"),
                        "mode": r.get("mode"),   # <── added
                        "expr": r.get("expr", ""),
                        "enabled": bool(r.get("enabled", True)),
                    }
                )

        return {"rules": rules}

    def _prune_old_dynamic_rules(
        self, rules: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Keep at most max_dynamic_rules entries whose ids start with dynamic_prefix.
        Only applied to the *dynamic* rules file.
        """
        if not self.dynamic_prefix:
            return rules

        dynamic_indices = [
            i
            for i, r in enumerate(rules)
            if isinstance(r, dict)
            and str(r.get("id", "")).startswith(self.dynamic_prefix)
        ]
        if len(dynamic_indices) <= self.max_dynamic_rules:
            return rules

        to_drop = dynamic_indices[
            0 : len(dynamic_indices) - self.max_dynamic_rules
        ]
        keep = [
            r for idx, r in enumerate(rules) if idx not in to_drop
        ]
        return keep

    # --- tasks (registry; optional) ---


    def _refresh_registry_from_broker(self):
        if not self.registry_refresh_client:
            return

        if not self.registry_refresh_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn("registry refresh service unavailable")
            return

        req = Trigger.Request()
        future = self.registry_refresh_client.call_async(req)

        # Optional: log result asynchronously
        def _cb(fut):
            try:
                res = fut.result()
                if res and res.success:
                    self.get_logger().info(f"registry refresh ok: {res.message}")
                else:
                    self.get_logger().warn("registry refresh failed or returned None")
            except Exception as e:
                self.get_logger().warn(f"registry refresh error: {e}")

        future.add_done_callback(_cb)


    def _build_tasks_inventory(self) -> Dict[str, Any]:
        """
        For rule & model selection, expose:
          - task name
          - outputs (id/topic/msg/fields)
          - models: id, version, role, and latency_ms metrics (if present)

        The registry file is assumed to follow task_registry.yaml structure.
        """
        if not self.registry_path:
            return {"tasks": []}

        # Ask broker to flush latest perf EMAs into the registry before reading
        self._refresh_registry_from_broker()

        doc = self._read_yaml_if_exists(self.registry_path) or {}
        out = []

        for tname, tdoc in (doc.get("tasks") or {}).items():
            # ---- Outputs as before ----
            outputs = []
            for o in tdoc.get("outputs") or []:
                context = o.get("context") or {}
                fields = (
                    context.get("fields")
                    or context.get("per_detection_fields")
                    or context.get("per_person_fields")
                    or []
                )
                outputs.append(
                    {
                        "id": o.get("id"),
                        "topic": (o.get("ros") or {}).get("topic"),
                        "msg": (o.get("ros") or {}).get("msg"),
                        "fields": fields,
                    }
                )

            # ---- Models: id + latency_ms (and a bit of structure) ----
            models_info = []
            for m in tdoc.get("models") or []:
                metrics = m.get("metrics") or {}
                latency_ms = None
                if isinstance(metrics, dict):
                    # this will pass through whatever shape you have under latency_ms
                    # e.g. {det_mean: 8.0} or {typical: 40.0, utter_infer_mean: 160.0}
                    latency_ms = metrics.get("latency_ms")

                models_info.append(
                    {
                        "id": m.get("id"),
                        "metrics": {
                            "latency_ms": latency_ms
                        } if latency_ms is not None else {},
                    }
                )

            out.append(
                {
                    "task": tname,
                    "description": tdoc.get("description"),
                    "outputs": outputs,
                    "models": models_info,
                }
            )

        return {"tasks": out}


    # =====================================================================
    # Context & perf I/O
    # =====================================================================

    def _on_capsule(self, msg: StringMsg):
        """Store latest broker ContextCapsule (which may include llm_perf, etc.)."""
        try:
            self._capsule = json.loads(msg.data)
        except Exception:
            # keep last capsule if this one is bad
            return

    def _publish_perf(self, lat_ms: float, ok: bool, phase: str = "parse_plan"):
        """Publish orchestrator LLM latency so broker can record it in its perf DB."""
        payload = {
            "node": "orchestrator",
            "model": self.model,
            "lat_ms": float(lat_ms),
            "ok": bool(ok),
            "phase": phase,
        }
        try:
            self.pub_perf.publish(StringMsg(data=json.dumps(payload)))
        except Exception:
            pass

    # =====================================================================
    # OpenAI helper
    # =====================================================================

    def _chat_json(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        temperature: float,
        retries: int = 1,
        phase: str = "parse_plan",  # NEW: phase label for perf
    ) -> Dict[str, Any]:

        self.get_logger().info("starting llm in orchestrator")

        last_error: Optional[Exception] = None

        for _ in range(retries + 1):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages)
                '''
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "OrchestratorOutput",
                        "schema": schema,
                    },
                },
                '''
                
                t1 = time.time()
                lat_ms = (t1 - t0) * 1000.0
                self._publish_perf(lat_ms=lat_ms, ok=True, phase=phase)

                content = resp.choices[0].message.content
                
                self.get_logger().info("\n=== LLM PROMPT ===\n" + json.dumps(messages, indent=2))
                
                self.get_logger().info(
                    f"=== ORCHESTRATOR LLM RAW RESPONSE ===\n{content}\n"
                )
                try:
                    obj = json.loads(content)
                    #validate(instance=obj, schema=schema)
                    return obj
                except (json.JSONDecodeError, ValidationError) as e:
                    self.get_logger().warn(
                        f"orchestrator: LLM JSON/schema error: {e}"
                    )
                    last_error = e
                    messages = messages + [
                        {
                            "role": "system",
                            "content": (
                                "Return ONLY valid JSON that passes the schema. "
                                "No extra text."
                            ),
                        }
                    ]
            except Exception as e:
                t1 = time.time()
                lat_ms = (t1 - t0) * 1000.0
                self._publish_perf(lat_ms=lat_ms, ok=False, phase=phase)
                self.get_logger().warn(f"orchestrator: LLM call failed: {e}")
                last_error = e

        raise ValueError(f"LLM did not return valid JSON for schema: {last_error}")

    # =====================================================================
    # Example inventory helpers (for prompt demo only)
    # =====================================================================

    def _build_example_skills_inventory(
        self, skills_inventory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Tiny skills_inventory only for the demonstration turn.

        We keep just:
          - nav.approach_speaker
          - beacons.report_top3
          - tts.say
          - db.query_top_beacons

        The full skills_inventory is still used for the real plan payload.
        """
        wanted = {
            "nav.approach_speaker",
            "beacons.report_top3",
            "tts.say",
            "db.query_top_beacons",
        }

        prims = [
            p
            for p in skills_inventory.get("primitives", [])
            if p.get("name") in wanted
        ]
        comps = [
            c
            for c in skills_inventory.get("composites", [])
            if c.get("name") in wanted
        ]

        return {"primitives": prims, "composites": comps}

    def _build_example_rules_inventory(
        self, rules_inventory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Tiny rules_inventory only for the example.

        Keep just the rules referenced in the example rule:
          - bt_rssi_seen
          - speech_final_any
        """
        wanted_ids = {"bt_rssi_seen", "speech_final_any"}

        out_rules = [
            r
            for r in rules_inventory.get("rules", [])
            if r.get("id") in wanted_ids
        ]
        return {"rules": out_rules}

    def _build_example_tasks_inventory(
        self, tasks_inventory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Tiny tasks_inventory only for the example.

        Keep just the tasks backing the example rules:
          - bt_proximity  (bt_rssi_seen)
          - audio_asr     (speech_final_any)
        """
        wanted_tasks = {"bt_proximity", "audio_asr"}

        out_tasks = [
            t
            for t in tasks_inventory.get("tasks", [])
            if t.get("task") in wanted_tasks
        ]
        return {"tasks": out_tasks}


    # =====================================================================
    # Plan handling
    # =====================================================================



    def _build_llm_messages_for_plan(
        self,
        plan: Dict[str, Any],
        skills_inventory: Dict[str, Any],
        rules_inventory: Dict[str, Any],
        tasks_inventory: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        header = plan.get("plan_header") or {}
        doc = plan.get("plan_doc", "")
        hints = plan.get("_hints") or {}
        meta = plan.get("_meta") or {}

        # --- Build minimal example inventories (demo only) ---
        example_skills_inventory = self._build_example_skills_inventory(
            skills_inventory
        )
        example_rules_inventory = self._build_example_rules_inventory(
            rules_inventory
        )
        example_tasks_inventory = self._build_example_tasks_inventory(
            tasks_inventory
        )

        example_user = {
            "plan_header": {
                "objective": "Approach the caller and summarize nearby beacons",
                "time_horizon": "short",
                "priority": [
                    "approach human caller",
                    "report top 3 beacons at current location",
                ],
            },
            "plan_doc": (
                "Situation: A human just called the robot (speech_keyword detected)\n"
                "and the robot is near a rack with Bluetooth-tagged objects.\n"
                "Intent: Move close enough to the caller for a brief interaction,\n"
                "then summarize the strongest 3 beacon signals 'here'.\n"
                "Action Sketch: if speech_keyword and pose_present_precise, approach\n"
                "speaker by ~0.8m; query_top_beacons; speak concise summary.\n"
                "Evidence & Uncertainty: beacon DB contains last RSSI per object;\n"
                "uncertainty in contamination labels remains, but this step only\n"
                "reports signals, not bin decisions.\n"
                "Coordination & Tone: concise, status-style speech; no long dialogue.\n"
                "Hard Constraints: do not move if no human is present or user interrupts."
            ),
            "_hints": {
                "object_ids": ["CNode12"],
                "areas": ["A"],
                "bins": [],
                "open_items": ["user might ask follow-up about CNode12 later"],
            },
            "_meta": {"ws_id": 0},
            "ContextCapsule": {
                "trigger": {"type": "speech_keyword"},
                "llm_perf": {
                    "broker": [{"model": "gpt-5-nano", "ema_ms": 120.0}],
                    "planner": [{"model": "gpt-5-mini", "ema_ms": 180.0}],
                },
            },
        }


        example_assistant = {
            "new_composites": [
                {
                    "name": "interact.beacon_brief_here",
                    "kind": "composite",
                    "when": {},
                    "until": {"not_exists": "human_detected_3d"},
                    "steps": [
                        {
                            "use": "nav.approach_speaker",
                            "when": {},
                            "with": {"dist_m": 0.8},
                        },
                        {
                            "use": "beacons.report_top3",
                            "when": {
                                "exists": "bt_rssi_seen",
                            },
                            "with": {},
                        },
                    ],
                }
            ],
            "new_rules": [
                {
                    "id": "interact.beacon_brief_here_ready",
                    "type": "composite",
                    "enabled": True,
                    "expr": (
                        "exists('speech_keyword', 3000) "
                        "and exists('pose_present_precise', 2500) "
                        "and exists('bt_rssi_seen', 2000)"
                    ),
                }
            ],
            "to_execute": {
                "skill": "interact.beacon_brief_here",
                "ctx": {
                    "objective": "approach caller and summarize nearby beacons",
                    "object_ids": ["CNode12"],
                    "areas": ["A"],
                },
            },
            "llm_models": {
                "broker": "gpt-5-nano",
                "planner": "gpt-5-mini",
                "orchestrator": "gpt-4o-mini",
            },
        }


        payload = {
            "plan_header": header,
            "plan_doc": doc,
            "_hints": hints,
            "_meta": meta,
            "skills_inventory": skills_inventory,
            "rules_inventory": rules_inventory,
            "tasks_inventory": tasks_inventory,
            # NEW: give orchestrator LLM access to broker context capsule, including perf DB
            #"ContextCapsule": self._capsule or {},
        }

        '''
        return [
            {"role": "system", "content": SYSTEM_PARSE},
            
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "example": True,
                        "plan": example_user,
                        "skills_inventory": example_skills_inventory,
                        "rules_inventory": example_rules_inventory,
                        "tasks_inventory": example_tasks_inventory,
                    },
                    ensure_ascii=False,
                ),
            },
            
            {
                "role": "assistant",
                "content": json.dumps(
                    example_assistant, ensure_ascii=False
                ),
            },
            
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]
        '''

        return [
            {"role": "system", "content": SYSTEM_PARSE},
            
            
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]

    def _on_plan(self, msg: StringMsg):
        try:
            plan = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(
                f"orchestrator: bad JSON on /planner/plan_out: {e}"
            )
            return

        self.get_logger().info("orchestrator: received /planner/plan_out")

        skills_inventory = self._build_skills_inventory()
        rules_inventory = self._build_rules_inventory()
        tasks_inventory = self._build_tasks_inventory()

        # Call LLM to translate plan -> {new_composites, new_rules, to_execute, llm_models?}
        try:
            messages = self._build_llm_messages_for_plan(
                plan, skills_inventory, rules_inventory, tasks_inventory
            )
            orchestrator_obj = self._chat_json(
                messages,
                PLAN_PARSE_SCHEMA,
                temperature=self.temperature,
                retries=1,
                phase="parse_plan",
            )
        except Exception as e:
            self.get_logger().error(
                f"orchestrator: LLM parse failed, aborting: {e}"
            )
            return

        new_composites = orchestrator_obj.get("new_composites") or []
        new_rules = orchestrator_obj.get("new_rules") or []
        to_execute_obj = orchestrator_obj.get("to_execute") or {}
        llm_models = orchestrator_obj.get("llm_models") or {}

        # --- Apply any LLM-chosen model changes first ---
        if isinstance(llm_models, dict):
            for role, model in llm_models.items():
                if isinstance(model, str) and model:
                    self._set_remote_model(role, model)

        # --- Update composite YAML only ---
        if new_composites and self.skills_composite_path:
            comp_doc = self._ensure_composite_yaml()
            skills = comp_doc.get("skills", [])

            for c in new_composites:
                name = str(c.get("name", ""))
                c["name"] = name
                skills.append(c)
                self.get_logger().info(
                    f"orchestrator: planning to append composite '{c['name']}'"
                )

            skills = self._prune_old_dynamic_skills(skills)
            comp_doc["skills"] = skills

            if not self._write_composite_yaml(comp_doc):
                self.get_logger().warn(
                    "orchestrator: failed to update skills_composite.yaml; aborting."
                )
                return

            self.get_logger().info(
                f"orchestrator: wrote {len(new_composites)} new composites to skills_composite.yaml"
            )

        # --- Update rules YAML only ---
        if new_rules and self.rules_path:
            rules_doc = self._ensure_rules_yaml()
            rules_list = rules_doc.get("rules", [])

            for r in new_rules:
                rid = str(r.get("id", ""))
                if self.dynamic_prefix and not rid.startswith(self.dynamic_prefix):
                    r["id"] = f"{self.dynamic_prefix}{rid}"
                # default type if missing
                if "type" not in r:
                    r["type"] = "composite" if not r.get("task") else "basic"
                if "enabled" not in r:
                    r["enabled"] = True
                rules_list.append(r)
                self.get_logger().info(
                    f"orchestrator: planning to append rule '{r['id']}'"
                )

            rules_list = self._prune_old_dynamic_rules(rules_list)
            rules_doc["rules"] = rules_list

            if not self._write_rules_yaml(rules_doc):
                self.get_logger().warn(
                    "orchestrator: failed to update rules.yaml; aborting."
                )
                return

            self.get_logger().info(
                f"orchestrator: wrote {len(new_rules)} new rules to rules.yaml"
            )

        # Reload rules + skills, then execute
        if isinstance(to_execute_obj, dict) and "skill" in to_execute_obj:
            to_execute_list = [to_execute_obj]    # wrap single skill as a list
        else:
            to_execute_list = []

        self._reload_all_and_execute(to_execute_list)

    # =====================================================================
    # Remote LLM model changes
    # =====================================================================

    def _set_remote_model(self, role: str, model: str):
        """Ask the target node to switch its LLM model param via SetParameters."""
        info = self.llm_targets.get(role) or {}
        node_name = info.get("node")
        param_name = info.get("param", "model")
        if not node_name:
            self.get_logger().warn(f"no node mapping for role='{role}'")
            return
        client = self._param_clients.get(role)
        if client is None:
            self.get_logger().warn(f"no SetParameters client for role='{role}'")
            return

        if not client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn(
                f"SetParameters service not available for role='{role}' (node={node_name})"
            )
            return

        req = SetParameters.Request()
        p = ParamMsg()
        p.name = param_name
        pv = ParameterValue()
        pv.type = ParameterType.PARAMETER_STRING   
        pv.string_value = str(model)
        p.value = pv
        req.parameters.append(p)

        self.get_logger().info(
            f"orchestrator: requesting LLM model change role='{role}' node='{node_name}' param='{param_name}' -> '{model}'"
        )
        _ = client.call_async(req)

    # =====================================================================
    # Reload + execute
    # =====================================================================

    def _reload_all_and_execute(self, to_execute: List[Dict[str, Any]]):
        """
        Reload rules + skills, then cancel any currently active skills
        and finally execute the new ones from this plan.
        """

        # fire-and-forget rules reload
        if self.reload_rules_client.wait_for_service(timeout_sec=1.0):
            req_r = Trigger.Request()
            _ = self.reload_rules_client.call_async(req_r)

        def _after_cancel(_future_cancel):
            # ignore cancel result errors for now, just log
            try:
                res = _future_cancel.result()
                if not res.success:
                    self.get_logger().warn(f"/skills/cancel_all failed: {res.message}")
                else:
                    self.get_logger().info(f"/skills/cancel_all ok: {res.message}")
            except Exception as e:
                self.get_logger().warn(f"/skills/cancel_all call error: {e}")

            # now arm the new skills
            self._execute_skills(to_execute)

        def _after_skills_reload(_future_reload):
            try:
                res = _future_reload.result()
                if not res.success:
                    self.get_logger().warn(f"/skills/reload failed: {res.message}")
                else:
                    self.get_logger().info(f"/skills/reload ok: {res.message}")
            except Exception as e:
                self.get_logger().warn(f"/skills/reload call error: {e}")

            # after reload, cancel all active skills, then execute new plan
            if self.cancel_skills_client.wait_for_service(timeout_sec=1.0):
                req_c = Trigger.Request()
                future_c = self.cancel_skills_client.call_async(req_c)
                future_c.add_done_callback(_after_cancel)
            else:
                self.get_logger().warn(
                    "/skills/cancel_all not available; executing skills without cancel."
                )
                self._execute_skills(to_execute)

        # trigger skills reload, then chain cancel → execute
        if self.reload_skills_client.wait_for_service(timeout_sec=2.0):
            req_s = Trigger.Request()
            future_s = self.reload_skills_client.call_async(req_s)
            future_s.add_done_callback(_after_skills_reload)
        else:
            self.get_logger().warn(
                "/skills/reload not available; executing skills without reload."
            )
            # even if reload is missing, still try to cancel
            if self.cancel_skills_client.wait_for_service(timeout_sec=1.0):
                req_c = Trigger.Request()
                future_c = self.cancel_skills_client.call_async(req_c)
                future_c.add_done_callback(_after_cancel)
            else:
                self.get_logger().warn(
                    "/skills/cancel_all not available; executing skills without cancel."
                )
                self._execute_skills(to_execute)

    def _execute_skills(self, to_execute: List[Dict[str, Any]]):
        for entry in to_execute:
            try:
                name = str(entry["skill"])
            except Exception:
                continue
            ctx = entry.get("ctx") or {}
            payload = {"skill": name, "ctx": ctx}
            self.pub_execute.publish(
                StringMsg(data=json.dumps(payload))
            )
            self.get_logger().info(
                f"orchestrator: armed skill '{name}' with ctx={ctx}"
            )


def main():
    rclpy.init()
    node = OrchestratorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

