#!/usr/bin/env python3
"""
interaction_loop_node.py

Interaction Loop / Skill Selector node.

- Subscribes:
    /events/basic      (std_msgs/String, JSON: {"rule":..., "data":..., "ts":..., "zone":...})
    /events/composite  (std_msgs/String, JSON: {"rule":..., "expr":..., "ts":..., "zone":...})
    /task_state        (std_msgs/String, JSON: task progress & robot objective)
    /profiles/summary  (std_msgs/String, JSON: HDT human profiles)
    /planner/proposal  (std_msgs/String, JSON: high-level proposals from planner)  <-- NEW

- On any "trigger" rule (rule id in trigger_map or starting with trigger_prefix),
  it builds a "loop_capsule" that includes:
    * trigger info (rule, type, text, zone, ts)
    * recent events (small trace)
    * current task_state
    * compact HDT profiles (including active human and robot zone)
    * latest planner proposal (if any)

- When a planner proposal is received on /planner/proposal, it is immediately
  converted into a concrete skill sequence and executed (published to
  /skills/execute_plan) WITHOUT calling the LLM.

- For other triggers (e.g., human speech), it calls a small LLM (fast) to pick
  a SHORT sequence of existing skills:

    {
      "name": "reactive.<something>",
      "skills": [
        {"use": "<existing_skill_name>", "with": { ... }},
        ...
      ]
    }

- Publishes this plan to /skills/execute_plan (std_msgs/String, JSON).
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
    - For idle triggers, executes the planner proposal directly (if available).
    - Otherwise, uses LLM (or heuristic fallback) to choose a skill sequence.
    - Publishes the plan to /skills/execute_plan.
    """

    def __init__(self):
        super().__init__("interaction_loop_node")

        # ----- Parameters -----
        self.declare_parameter("llm_enabled", True)
        self.declare_parameter("model", "gpt-5.1-mini")
        self.declare_parameter("groq_model_prefix", "gpt-oss")
        self.declare_parameter("trigger_prefix", "trigger_")
        self.declare_parameter(
            "trigger_map_json",
            json.dumps({
                "trigger_speech_final": "human_command",
                "trigger_idle": "idle",
            }),
        )

        # Where to read skills and rules from (same style as router/orchestrator)
        self.declare_parameter("skills_base_path", "")
        self.declare_parameter("skills_composite_path", "")
        self.declare_parameter("rules_init_path", "")
        self.declare_parameter("rules_path", "")

        self.declare_parameter("profiles_topic", "/profiles/summary")
        self.declare_parameter("task_state_topic", "/task_state")
        # NEW: planner proposals
        self.declare_parameter("planner_proposal_topic", "/planner/proposal")

        # How many recent events to keep in memory
        self.declare_parameter("event_trace_len", 20)

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

        self.event_trace_len = int(self.get_parameter("event_trace_len").value)

        self.add_on_set_parameters_callback(self._on_param_change)

        # ----- Internal state -----
        self._event_trace = deque(maxlen=self.event_trace_len)

        # Latest task_state snapshot
        self._last_task_state: Dict[str, Any] = {}

        # Latest planner proposal (raw object)
        self._last_planner_proposal: Dict[str, Any] = {}    # NEW
        self._last_planner_proposal_ts: float = 0.0         # NEW

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

        # HDT profiles
        self.sub_profiles = self.create_subscription(
            StringMsg, self.profiles_topic, self._on_profiles, 10
        )

        # Planner proposals  <-- NEW
        self.sub_proposal = self.create_subscription(
            StringMsg, self.planner_proposal_topic, self._on_planner_proposal, 20
        )

        # Skill plan publisher (for skills_node)
        self.pub_execute_plan = self.create_publisher(
            StringMsg, "/skills/execute_plan", 10
        )

        self.get_logger().info("interaction_loop_node initialized")

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

    # ---------- Planner proposals ingestion ----------
    def _on_planner_proposal(self, msg: StringMsg):
        """
        Ingest proposals from the planner/coordination node and immediately
        convert them into a concrete skill plan if possible.

        Expected payload (flexible):
          EITHER:
            {
              "objective": {...},
              "proposal": {
                "summary": str,
                "steps": [ {...}, ...],
                ...
              },
              "reason": str,
              "ts": float,
              ...
            }
          OR directly:
            {
              "summary": str,
              "steps": [ {...}, ...],
              ...
            }
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

        # Keep a snapshot for context (e.g., for the LLM path if needed)
        self._last_planner_proposal = obj
        self._last_planner_proposal_ts = self._now()
        self.get_logger().info(
            "[interaction_loop] updated planner proposal snapshot"
        )

        # Immediately try to turn this planner proposal into a concrete skill plan
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
            "[interaction_loop] executed planner proposal as a skill plan"
        )

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

        if trig_type:
            self.get_logger().info(
                f"[interaction_loop] trigger composite rule={rule}, type={trig_type}"
            )
            payload = {"expr": expr}
            self._run_for_trigger(
                rule=rule,
                trig_type=trig_type,
                kind="composite",
                ts=ts,
                zone=zone,
                payload=payload,
            )

    # ---------- Planner proposal → skill plan mapping (NEW) ----------
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
                        # e.g. goto_zone_A / goto_zone_B
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

        # Publish to /skills/execute_plan
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
                messages=messages,
                response_format={"type": "json_object"},
            )
        else:
            client = OpenAI()
            resp = client.chat.completions.create(
                model=model,
                reasoning_effort="medium",
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

    # ---------- Publishing ----------
    def _publish_skill_plan(
        self,
        rule: str,
        trig_type: str,
        skills_obj: Dict[str, Any],
    ):
        name = str(skills_obj.get("name") or "").strip()
        if not name:
            # Auto-generate a name if missing
            suffix = int(self._now() * 1000)
            name = f"reactive.{trig_type}.{rule}.{suffix}"

        plan = {
            "name": name,
            "skills": skills_obj.get("skills", []),
        }

        try:
            s = json.dumps(plan, ensure_ascii=False)
        except Exception as e:
            self.get_logger().warn(
                f"[interaction_loop] failed to serialize skill plan {name}: {e}"
            )
            return

        self.pub_execute_plan.publish(StringMsg(data=s))
        self.get_logger().info(
            f"[interaction_loop] published skill plan '{name}' with "
            f"{len(plan['skills'])} step(s) to /skills/execute_plan"
        )


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

