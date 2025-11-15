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
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String as StringMsg
from std_srvs.srv import Trigger

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
            "type": "array",
            "items": {
                "type": "object",
                "required": ["skill"],
                "properties": {
                    "skill": {"type": "string"},
                    "ctx": {"type": "object"},
                },
                "additionalProperties": True,
            },
        },
    },
    "additionalProperties": False,
}

SYSTEM_PARSE = (
    "You are the ORCHESTRATOR for a mobile robot collaborating with humans.\n"
    "The planner has already produced a high-level plan (plan_header, plan_doc, _hints).\n"
    "You also have a skills library with primitive and composite skills, and a rules library.\n\n"
    "Your job is to:\n"
    " 1) Decide which skills to activate next (existing primitives or composites),\n"
    " 2) Optionally define NEW composite skills that encode short sequences of existing primitives,\n"
    " 3) Optionally define NEW rules (prefer composite rules using exists('rule_id', ms)),\n"
    " 4) Return STRICT JSON ONLY following the provided schema.\n\n"
    "Rules:\n"
    " - You CANNOT invent new primitive actions. You may only reference existing skill names.\n"
    " - New composite skills must use 'use: <skill_name>' where <skill_name> is in the skills list.\n"
    " - Prefer short, focused composites (2–6 steps).\n"
    " - 'to_execute' may reference either existing composites or new composites you define.\n"
    " - For new rules:\n"
    "     * Prefer composite rules over basic rules: type='composite', expr using exists('rule_id', ms).\n"
    "     * Only create basic rules (with task/output) if necessary and ONLY using tasks/outputs from tasks_inventory.\n"
    "     * Do NOT invent new task names or outputs.\n"
    " - Use 'ctx' to pass simple arguments (e.g., object_ids, areas) when helpful, but keep it compact.\n"
    " - If the plan is mostly about sensing or scanning, prefer existing skills like sense.here, beacons.report_top3,\n"
    "   nav.go_absolute, tts.say, etc.\n"
    " - If you are unsure, default to a simple tts.say composite that describes what you intend to do.\n"
)

class OrchestratorNode(Node):
    """
    Bridges PlannerNode and SkillsAgent with an LLM-driven planner-to-skills+rules parser.
    Base skills/rules are read-only; dynamic composites and rules are appended to
    skills_composite.yaml and rules.yaml respectively.
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

        self.cancel_skills_client = self.create_client(
            Trigger, "/skills/cancel_all"
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

        # ---------------- ROS I/O ----------------
        self.sub_plan = self.create_subscription(
            StringMsg, "/planner/plan_out", self._on_plan, 10
        )

        self.pub_execute = self.create_publisher(
            StringMsg, "/skills/execute", 10
        )

        self.reload_skills_client = self.create_client(
            Trigger, "/skills/reload"
        )
        self.reload_rules_client = self.create_client(
            Trigger, "/event_layer/reload_rules"
        )

        self.get_logger().info(
            f"orchestrator_node up | base='{self.skills_base_path}' | "
            f"composite='{self.skills_composite_path}' | "
            f"rules_init='{self.rules_init_path}' | rules='{self.rules_path}' | "
            f"model={self.model}"
        )

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
        Build a combined inventory of base + composite skills for the LLM.
        """
        primitives = []
        composites = []

        for path in [self.skills_base_path, self.skills_composite_path]:
            doc = self._read_yaml_if_exists(path) or {}
            for s in doc.get("skills", []) or []:
                if not isinstance(s, dict):
                    continue
                name = str(s.get("name", ""))
                kind = str(s.get("kind", ""))
                if kind == "primitive":
                    primitives.append(
                        {
                            "name": name,
                            "action": s.get("action", ""),
                            "params": list((s.get("params") or {}).keys()),
                        }
                    )
                elif kind == "composite":
                    steps = s.get("steps") or []
                    composites.append(
                        {"name": name, "num_steps": len(steps)}
                    )

        return {"primitives": primitives, "composites": composites}

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

    def _build_tasks_inventory(self) -> Dict[str, Any]:
        """
        For rule creation, expose task->output IDs so the LLM doesn’t invent new names.
        """
        if not self.registry_path:
            return {"tasks": []}
        doc = self._read_yaml_if_exists(self.registry_path) or {}
        out = []
        for tname, tdoc in (doc.get("tasks") or {}).items():
            outputs = []
            for o in tdoc.get("outputs") or []:
                outputs.append(
                    {
                        "id": o.get("id"),
                        "topic": (o.get("ros") or {}).get("topic"),
                        "msg": (o.get("ros") or {}).get("msg"),
                    }
                )
            out.append({"task": tname, "outputs": outputs})
        return {"tasks": out}

    # =====================================================================
    # OpenAI helper
    # =====================================================================

    def _chat_json(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        temperature: float,
        retries: int = 1,
    ) -> Dict[str, Any]:
    
        self.get_logger().info(
            f"starting llm in orchestrator\n"
        )
        for _ in range(retries + 1):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "OrchestratorOutput",
                        "schema": schema,
                    },
                },
            )
            content = resp.choices[0].message.content
            self.get_logger().info(
                f"=== ORCHESTRATOR LLM RAW RESPONSE ===\n{content}\n"
            )
            try:
                obj = json.loads(content)
                validate(instance=obj, schema=schema)
                return obj
            except (json.JSONDecodeError, ValidationError) as e:
                self.get_logger().warn(
                    f"orchestrator: LLM JSON/schema error: {e}"
                )
                messages = messages + [
                    {
                        "role": "system",
                        "content": (
                            "Return ONLY valid JSON that passes the schema. "
                            "No extra text."
                        ),
                    }
                ]
        raise ValueError("LLM did not return valid JSON for schema")

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

        example_user = {
            "plan_header": {
                "objective": "Bin all nodes in this short horizon",
                "time_horizon": "short",
                "priority": ["scan new node CNode12"],
            },
            "plan_doc": (
                "Situation: New object_id=CNode12 detected in area=A.\n"
                "Intent: Confirm label and scan nearby items.\n"
                "Action Sketch: approach area=A; ask_scan CNode12; "
                "confirm contamination; deliver to bin=contaminated in area=B.\n"
                "Evidence & Uncertainty: OPEN: need stronger signal.\n"
                "Coordination & Tone: concise prompts.\n"
                "Hard Constraints: limited time."
            ),
            "_hints": {
                "object_ids": ["CNode12"],
                "areas": ["A", "B"],
                "bins": ["contaminated"],
                "open_items": ["need stronger signal"],
            },
            "_meta": {"ws_id": 0},
        }

        example_assistant = {
            "new_composites": [
                {
                    "name": "plan.ws_0",
                    "kind": "composite",
                    "when": {},
                    "until": {"exists": "speech_final_any", "within_ms": 1},
                    "steps": [
                        {"use": "sense.here", "when": {}, "with": {}},
                        {
                            "use": "beacons.report_top3",
                            "when": {},
                            "with": {},
                        },
                    ],
                }
            ],
            "new_rules": [
                {
                    "id": "plan.ws_0_ready",
                    "type": "composite",
                    "enabled": True,
                    "expr": "exists('bt_rssi_seen', 2000) and not exists('speech_final_any', 500)",
                }
            ],
            "to_execute": [
                {
                    "skill": "plan.ws_0",
                    "ctx": {
                        "objective": "scan and confirm CNode12",
                        "object_ids": ["CNode12"],
                        "areas": ["A", "B"],
                        "bins": ["contaminated"],
                    },
                }
            ],
        }

        payload = {
            "plan_header": header,
            "plan_doc": doc,
            "_hints": hints,
            "_meta": meta,
            "skills_inventory": skills_inventory,
            "rules_inventory": rules_inventory,
            "tasks_inventory": tasks_inventory,
        }

        return [
            {"role": "system", "content": SYSTEM_PARSE},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "example": True,
                        "plan": example_user,
                        "skills_inventory": skills_inventory,
                        "rules_inventory": rules_inventory,
                        "tasks_inventory": tasks_inventory,
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

        # Call LLM to translate plan -> {new_composites, new_rules, to_execute}
        try:
            messages = self._build_llm_messages_for_plan(
                plan, skills_inventory, rules_inventory, tasks_inventory
            )
            orchestrator_obj = self._chat_json(
                messages,
                PLAN_PARSE_SCHEMA,
                temperature=self.temperature,
                retries=1,
            )
        except Exception as e:
            self.get_logger().error(
                f"orchestrator: LLM parse failed, aborting: {e}"
            )
            return

        new_composites = orchestrator_obj.get("new_composites") or []
        new_rules = orchestrator_obj.get("new_rules") or []
        to_execute = orchestrator_obj.get("to_execute") or []

        # --- Update composite YAML only ---
        if new_composites and self.skills_composite_path:
            comp_doc = self._ensure_composite_yaml()
            skills = comp_doc.get("skills", [])

            for c in new_composites:
                name = str(c.get("name", ""))
                #if not name.startswith(self.dynamic_prefix):
                #    c["name"] = f"{self.dynamic_prefix}{name}"
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
                if not rid.startswith(self.dynamic_prefix):
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
        self._reload_all_and_execute(to_execute)

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

