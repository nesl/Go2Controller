#!/usr/bin/env python3
"""
router_node.py

Mini router LLM node that runs *before* the broker/planner/HDT/orchestrator.

- Subscribes:
    /events/basic      (std_msgs/String, JSON: {"rule":..., "data":..., "ts":..., "zone":...})
    /events/composite  (std_msgs/String, JSON: {"rule":..., "expr":..., "ts":..., "zone":...})
    /llm/*_perf        (std_msgs/String, JSON: {"node":"broker", "model":..., "lat_ms":...,...})

- On any "trigger" rule (rule id starting with a given prefix, or present in trigger_map),
  it builds a lightweight "router_capsule", calls a mini router LLM, and uses the result to:

    * Enable/disable each reasoning node's LLM (llm_enabled: bool)
    * Select llm_model for each node (tier -> concrete model_id)

Assumes each target node exposes ROS parameters:
  - llm_enabled (bool)
  - llm_model   (string)   # or change name below if you use "llm_model" vs "model_id"
"""

import json
import time
import os
from collections import deque
from typing import Any, Dict, Optional

import yaml

import rclpy
from rclpy.node import Node

from std_msgs.msg import String as StringMsg

from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import (
    Parameter as ParameterMsg,
    ParameterValue,
    ParameterType,
    SetParametersResult,
)

from openai import OpenAI
from jsonschema import validate, ValidationError
from groq import Groq
from std_srvs.srv import Trigger


# ─────────────────────────────
# JSON Schema for router output
# ─────────────────────────────

ROUTER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["latency_profile", "nodes"],
    "properties": {
        "latency_profile": {
            "type": "string",
            "enum": ["fast_reactive", "normal", "deliberative"],
        },
        "nodes": {
            "type": "object",
            "required": ["broker", "planner", "hdt", "orchestrator"],
            "properties": {
                "broker": {
                    "type": "object",
                    "required": ["tier"],
                    "properties": {
                        "tier": {
                            "type": "string",
                            "enum": ["off", "fast", "balanced", "thorough"],
                        },
                    },
                },
                "planner": {
                    "type": "object",
                    "required": ["tier"],
                    "properties": {
                        "tier": {
                            "type": "string",
                            "enum": ["off", "fast", "balanced", "thorough"],
                        },
                    },
                },
                "hdt": {
                    "type": "object",
                    "required": ["tier"],
                    "properties": {
                        "tier": {
                            "type": "string",
                            "enum": ["off", "fast", "balanced", "thorough"],
                        },
                    },
                },
                "orchestrator": {
                    "type": "object",
                    "required": ["tier"],
                    "properties": {
                        "tier": {
                            "type": "string",
                            "enum": ["off", "fast", "balanced", "thorough"],
                        },
                    },
                },
            },
        },
    },
    "additionalProperties": False,
}


# ─────────────────────────────
# Static router system prompt
# ─────────────────────────────

ROUTER_SYSTEM_PROMPT = """
You are the ROUTER for a mobile robot's cognitive pipeline.

You receive a single JSON "router_capsule" describing:
- the triggering event (rule id, text, zone, timestamp)
- a short trace of recent events
- optional performance summary per node/model
- optional human profiles.

Your job is NOT to plan actions or SQL.
Your ONLY job is to choose:

  - a latency_profile: "fast_reactive" | "normal" | "deliberative"
  - for each node (broker, planner, hdt, orchestrator) a tier:
      "off" | "fast" | "balanced" | "thorough"

Return a single JSON object with EXACTLY this shape:

{
  "latency_profile": "<fast_reactive|normal|deliberative>",
  "nodes": {
    "broker":       { "tier": "<off|fast|balanced|thorough>" },
    "planner":      { "tier": "<off|fast|balanced|thorough>" },
    "hdt":          { "tier": "<off|fast|balanced|thorough>" },
    "orchestrator": { "tier": "<off|fast|balanced|thorough>" }
  }
}

Semantics:
- If tier == "off" for a node, the router will disable its LLM.
- If tier != "off", the router will enable its LLM with a model matching that tier.

Do NOT:
- put "broker", "planner", "hdt", "orchestrator" at the top level.
- return an "enabled" field anywhere.
- add any extra top-level keys.
- return explanations or comments. STRICT JSON only.

Guidelines:
- The orchestrator must never be fully disabled:
  do NOT set orchestrator.tier = "off". Use "fast" instead for
  very reactive episodes.
""".strip()



# ─────────────────────────────
# Model catalog (tiers -> model_id)
# ─────────────────────────────

MODEL_CATALOG: Dict[str, Dict[str, str]] = {
    "broker": {
        "fast":     "gpt-oss-120b",
        "balanced": "gpt-5-nano",
        "thorough": "gpt-5-mini",   # or a bigger one if you want
    },
    "planner": {
        "fast":     "gpt-oss-120b",
        "balanced": "gpt-5-nano",
        "thorough": "gpt-5-mini",   # or e.g. "gpt-5"
    },
    "orchestrator": {
        "fast":     "gpt-oss-120b",
        "balanced": "gpt-5-nano",
        "thorough": "gpt-5-mini",
    },
    "hdt": {
        "fast":     "gpt-oss-120b",
        "balanced": "gpt-5-nano",
        "thorough": "gpt-5-mini",
    },
}

class RouterNode(Node):

    _ROLE_TO_TASK = {
            "broker":       "llm_broker",
            "planner":      "llm_planner",
            "hdt":          "llm_hdt",
            "orchestrator": "llm_orchestrator",
        }

    def __init__(self):
        super().__init__("router_node")

        # Router LLM enable/disable flag (for the router itself)
        self.declare_parameter("llm_enabled", True)
        self.llm_enabled = (
            self.get_parameter("llm_enabled")
            .get_parameter_value()
            .bool_value
        )

        # Which model to use for the *router itself*
        self.declare_parameter("model", "gpt-oss-120b")
        self.model = (
            self.get_parameter("model")
            .get_parameter_value()
            .string_value
        )

        # Which rules should be considered "triggers"
        # You can extend this mapping to give semantics if you want.
        self.declare_parameter(
            "trigger_map_json",
            json.dumps({
                "trigger_speech_final": "human_command",
            }),
        )
        self.trigger_map = json.loads(
            self.get_parameter("trigger_map_json")
            .get_parameter_value()
            .string_value
        )

        # Any rule whose id starts with this prefix will also be treated as a trigger
        self.declare_parameter("trigger_prefix", "trigger_")
        self.trigger_prefix = (
            self.get_parameter("trigger_prefix")
            .get_parameter_value()
            .string_value
        )

        # Perf topics per role (these should match what your nodes publish)
        self.declare_parameter(
            "perf_topics_json",
            json.dumps(
                {
                    "broker": "/llm/broker_perf",
                    "planner": "/llm/planner_perf",
                    "hdt": "/llm/hdt_perf",
                    "orchestrator": "/llm/orchestrator_perf",
                }
            ),
        )
        
        # Topic to read HDT profiles from
        self.declare_parameter("profiles_topic", "/profiles/summary")
        self.profiles_topic = (
            self.get_parameter("profiles_topic")
            .get_parameter_value()
            .string_value
        )

        # Last known profiles from HDT + compact view for the router LLM
        self._profiles_raw: Dict[str, Any] = {}
        self._profiles_compact: Dict[str, Any] = {}
        self._active_human: Optional[str] = None
        self._last_robot_zone: Optional[str] = None

        # Subscription to HDT profiles
        self.sub_profiles = self.create_subscription(
            StringMsg,
            self.profiles_topic,
            self._on_profiles,
            10,
        )
        
        
        self.perf_topics = json.loads(
            self.get_parameter("perf_topics_json")
            .get_parameter_value()
            .string_value
        )

        # Optional: path to task_registry.yaml for initial latency estimates
        self.declare_parameter("registry_path", "")
        self.registry_path = (
            self.get_parameter("registry_path")
            .get_parameter_value()
            .string_value
        )

        # Baseline latencies loaded from registry: {role: {model_id: latency_ms}}
        self._llm_baseline_ms: Dict[str, Dict[str, float]] = {}

        # Seed baselines once at startup (only for the LLM roles we already track)
        self._load_initial_llm_baselines_from_registry()



        # Target node names per role (make sure these match your actual nodes)
        self.role_nodes = {
            "broker": "/broker_node",
            "planner": "/planner_node",
            "hdt": "/hdt_node",
            "orchestrator": "/orchestrator_node",
        }


        # Track recent event trace for context (only very compact info)
        self._event_trace = deque(maxlen=20)

        # Perf EMA: perf[role][model] = {"lat_ms_ema": float, "n": int, "last_ts": float}
        self._perf_ema: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._perf_alpha = 0.3

        # Subscriptions: events
        self.sub_basic = self.create_subscription(
            StringMsg, "/events/basic", self._on_basic_event, 1000
        )
        self.sub_comp = self.create_subscription(
            StringMsg, "/events/composite", self._on_comp_event, 500
        )

        # Subscriptions: perf topics
        self._perf_subs = []
        for role, topic in self.perf_topics.items():
            if not topic:
                continue
            self.get_logger().info(
                f"[router] subscribing to perf topic '{topic}' for role '{role}'"
            )
            sub = self.create_subscription(
                StringMsg,
                topic,
                lambda msg, r=role: self._on_perf_msg(r, msg),
                50,
            )
            self._perf_subs.append(sub)

        # SetParameters clients for each role node
        self.param_clients: Dict[str, rclpy.client.Client] = {}
        for role, node_name in self.role_nodes.items():
            client = self.create_client(
                SetParameters, f"{node_name}/set_parameters"
            )
            self.param_clients[role] = client
            self.get_logger().info(
                f"[router] created SetParameters client for role={role} node={node_name}"
            )

        # Allow dynamic change of router_model
        self.add_on_set_parameters_callback(self._on_param_change)

        # Service client to tell broker when to run an initial turn
        self.broker_run_initial_client = self.create_client(
            Trigger,
            "/broker/run_initial",
        )

        self.get_logger().info("router_node initialized")


    def _maybe_call_broker_run_initial(self, trig_type: str, policy: Dict[str, Any]):
        """
        Decide whether to ask broker to do an initial SQL reasoning pass
        for this episode, and if so, call /broker/run_initial.
        """


        # Simple heuristic based on trigger type:
        # You can tweak this set depending on your trigger_map semantics.
        broker_relevant_triggers = {
            "new_object",
            "human_command",
            "planner_trigger",
            "finish_or_fail",
            "idle",
            "presence",
            "generic_trigger",
        }
        if trig_type not in broker_relevant_triggers:
            self.get_logger().info(
                f"[router] trigger type '{trig_type}' not in broker_relevant_triggers; skipping broker run"
            )
            return

        if not self.broker_run_initial_client.service_is_ready():
            self.get_logger().warn(
                "[router] /broker/run_initial service not ready; cannot trigger broker yet"
            )
            return

        self.get_logger().info(
            f"[router] calling /broker/run_initial (trig_type={trig_type})"
        )

        req = Trigger.Request()
        future = self.broker_run_initial_client.call_async(req)

        def _done(fut):
            try:
                resp = fut.result()
                if resp.success:
                    self.get_logger().info(
                        f"[router] broker run_initial OK: {resp.message}"
                    )
                else:
                    self.get_logger().warn(
                        f"[router] broker run_initial FAILED: {resp.message}"
                    )
            except Exception as e:
                self.get_logger().error(
                    f"[router] exception calling broker run_initial: {e}"
                )

        future.add_done_callback(_done)


    # ─────────────────────────
    # HDT profiles ingestion
    # ─────────────────────────
    def _on_profiles(self, msg: StringMsg):
        """
        Ingest profiles from HDT on /profiles/summary.

        Expected payload is the HDT's JSON, e.g.:
          {
            "H1": {...},
            "H2": {...},
            "_meta": {
              "active_human": "H1",
              "robot_zone": "A",
              "ts": ...
            }
          }
        """
        try:
            obj = json.loads(msg.data) if msg.data else {}
        except Exception as e:
            self.get_logger().warn(f"[router] bad JSON on profiles_topic: {e}")
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

        # Build a compact view for the router LLM:
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
            # keep only a few routing-relevant fields
            compact["humans"][hid] = {
                "style": prof.get("style"),
                "helpfulness": prof.get("helpfulness"),
                "risk_aversion": prof.get("risk_aversion"),
                "contamination_bias": prof.get("contamination_bias"),
                "summary": prof.get("summary"),
            }

        self._profiles_compact = compact
        self.get_logger().debug(f"[router] updated compact profiles: {self._profiles_compact}")



    # ─────────────────────────
    # Param change
    # ─────────────────────────

    def _on_param_change(self, params):
        for p in params:
            if p.name == "model" and p.type_ == Parameter.Type.STRING:
                self.model = p.value
                self.get_logger().info(
                    f"[router] router_model updated to {self.model}"
                )
            elif p.name == "llm_enabled" and p.type_ == Parameter.Type.BOOL:
                self.llm_enabled = p.value
                self.get_logger().info(
                    f"[router] llm_enabled updated to {self.llm_enabled}"
                )
        return SetParametersResult(successful=True, reason="ok")
        
        
    def fast_only_policy(self, capsule: Dict[str, Any]) -> Dict[str, Any]:
        """
        Policy used when router LLM is disabled via llm_enabled=False.
        All roles set to 'fast'.
        """
        return {
            "latency_profile": "fast_reactive",
            "nodes": {
                "broker":       {"tier": "fast"},
                "planner":      {"tier": "fast"},
                "hdt":          {"tier": "fast"},
                "orchestrator": {"tier": "fast"},
            },
        }
    # ─────────────────────────
    # Event handlers
    # ─────────────────────────

    def _on_basic_event(self, msg: StringMsg):
        try:
            o = json.loads(msg.data)
        except Exception:
            self.get_logger().warn("[router] invalid JSON on /events/basic")
            return

        rule = str(o.get("rule") or "")
        data = o.get("data") or {}
        ts = float(o.get("ts") or time.time())
        zone = o.get("zone")

        # Append compact entry to trace
        trace_entry = {
            "kind": "basic",
            "rule": rule,
            "ts": ts,
        }
        if zone is not None:
            trace_entry["zone"] = zone
        # include a tiny snippet if there's text
        txt = ""
        if isinstance(data, dict):
            # guess a text key; you can adjust based on your event schema
            txt = str(
                data.get("text")
                or data.get("utterance")
                or data.get("speech")
                or ""
            )
        if txt:
            trace_entry["text_snippet"] = txt[:60]
        self._event_trace.append(trace_entry)

        # Decide if this is a trigger
        trig_type = self.trigger_map.get(rule)
        if not trig_type and rule.startswith(self.trigger_prefix):
            trig_type = "generic_trigger"

        if trig_type:
            self.get_logger().info(f"[router] trigger basic rule={rule}")
            self._run_router_for_trigger(
                rule=rule,
                trig_type=trig_type,
                kind="basic",
                ts=ts,
                zone=zone,
                payload=data,
            )

    def _on_comp_event(self, msg: StringMsg):
        try:
            o = json.loads(msg.data)
        except Exception:
            self.get_logger().warn("[router] invalid JSON on /events/composite")
            return

        rule = str(o.get("rule") or "")
        expr = o.get("expr") or ""
        ts = float(o.get("ts") or time.time())
        zone = o.get("zone")

        trace_entry = {
            "kind": "composite",
            "rule": rule,
            "ts": ts,
        }
        if zone is not None:
            trace_entry["zone"] = zone
        if expr:
            trace_entry["expr_snippet"] = str(expr)[:80]
        self._event_trace.append(trace_entry)

        trig_type = self.trigger_map.get(rule)
        if not trig_type and rule.startswith(self.trigger_prefix):
            trig_type = "generic_trigger"

        if trig_type:
            self.get_logger().info(f"[router] trigger composite rule={rule}")
            payload = {"expr": expr}
            self._run_router_for_trigger(
                rule=rule,
                trig_type=trig_type,
                kind="composite",
                ts=ts,
                zone=zone,
                payload=payload,
            )

    # ─────────────────────────
    # Registry → initial LLM baselines
    # ─────────────────────────

    def _read_yaml_if_exists(self, path: str) -> Optional[dict]:
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().error(f"[router] Failed to read YAML '{path}': {e}")
            return None

    def _load_initial_llm_baselines_from_registry(self):
        """
        Load initial latency baselines from task_registry.yaml for the
        LLM roles this router already knows about (broker, planner, hdt, orchestrator).

        Populates:
            self._llm_baseline_ms[role][model_id] = latency_ms
        Only used as a fallback until live /llm/*_perf messages arrive.
        """
        if not self.registry_path:
            self.get_logger().info("[router] registry_path not set; skipping initial LLM baselines")
            return

        doc = self._read_yaml_if_exists(self.registry_path) or {}
        tasks_doc = doc.get("tasks") or {}
        baselines: Dict[str, Dict[str, float]] = {}

        for role, task_id in self._ROLE_TO_TASK.items():
            tdoc = tasks_doc.get(task_id)
            if not isinstance(tdoc, dict):
                continue

            for m in tdoc.get("models") or []:
                mid = m.get("id")
                if not mid:
                    continue

                # metrics.latency_ms can be e.g. {typical: 150.0} or {utter_infer_mean: 160.0}
                metrics = (m.get("metrics") or {}).get("latency_ms") or {}
                if not isinstance(metrics, dict) or not metrics:
                    continue

                try:
                    if "typical" in metrics:
                        lat = float(metrics["typical"])
                    else:
                        # fall back to the smallest numeric metric
                        lat = min(float(v) for v in metrics.values())
                except Exception:
                    continue

                baselines.setdefault(role, {})[mid] = lat

        self._llm_baseline_ms = baselines

        if baselines:
            self.get_logger().info(f"[router] initial LLM baselines: {self._llm_baseline_ms}")
        else:
            self.get_logger().info("[router] no LLM baselines found in registry")


    # ─────────────────────────
    # Perf ingestion
    # ─────────────────────────

    def _on_perf_msg(self, role: str, msg: StringMsg):
        """
        Expect payload like broker's _publish_llm_perf:
          {"node":"broker","model":str,"lat_ms":num,"ok":bool,"phase":str,...}
        """
        try:
            payload = json.loads(msg.data) if msg.data else {}
        except Exception as e:
            self.get_logger().warn(
                f"[router] perf JSON parse error for role={role}: {e}"
            )
            return

        if not isinstance(payload, dict):
            return

        model = payload.get("model")
        lat_ms = payload.get("lat_ms")

        if not isinstance(model, str) or not isinstance(lat_ms, (int, float)):
            return

        role_dict = self._perf_ema.setdefault(role, {})
        ent = role_dict.get(model)
        ts = time.time()
        lat_ms = float(lat_ms)

        if ent is None:
            ent = {
                "lat_ms_ema": lat_ms,
                "n": 1,
                "last_ts": ts,
            }
        else:
            a = self._perf_alpha
            ent["lat_ms_ema"] = (1.0 - a) * ent["lat_ms_ema"] + a * lat_ms
            ent["n"] = ent.get("n", 0) + 1
            ent["last_ts"] = ts

        role_dict[model] = ent
        self._perf_ema[role] = role_dict

    def _perf_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a compact summary for the router LLM, combining:
          - initial baselines from task_registry.yaml (if any)
          - live EMAs from /llm/*_perf topics (which override baselines)

        Shape:
          {
            "broker": {
              "gpt-5-nano": {"lat_ms_ema": 40.0, "n": 0},    # baseline only
              "gpt-5-mini": {"lat_ms_ema": 120.0, "n": 12},  # live EMA
              ...
            },
            "planner": {...}
          }
        """
        out: Dict[str, Dict[str, Any]] = {}

        # union of roles for which we have baselines or live perf
        roles = set(self._llm_baseline_ms.keys()) | set(self._perf_ema.keys())

        for role in roles:
            role_entry: Dict[str, Any] = {}

            # 1) start from registry baselines
            base_models = self._llm_baseline_ms.get(role, {})
            for model_id, lat in base_models.items():
                role_entry[model_id] = {"lat_ms_ema": float(lat), "n": 0}

            # 2) overlay live perf (EMAs override baselines)
            live_models = self._perf_ema.get(role, {})
            for model_id, ent in live_models.items():
                role_entry[model_id] = {
                    "lat_ms_ema": ent.get("lat_ms_ema"),
                    "n": ent.get("n", 0),
                }

            if role_entry:
                out[role] = role_entry

        return out


    # ─────────────────────────
    # Router episode logic
    # ─────────────────────────

    def _run_router_for_trigger(
        self,
        rule: str,
        trig_type: str,
        kind: str,
        ts: float,
        zone: Any,
        payload: Dict[str, Any],
    ):
        """
        Build router_capsule and call mini router LLM once per trigger.
        """
        # Try to extract some text from payload if present
        text = ""
        if isinstance(payload, dict):
            text = str(
                payload.get("text")
                or payload.get("utterance")
                or payload.get("speech")
                or ""
            )

        router_capsule = {
            "trigger": {
                "rule": rule,
                "type": trig_type,
                "kind": kind,          # "basic" or "composite"
                "zone": zone,
                "ts": ts,
                "text": text[:200] if text else "",
            },
            "recent_events": list(self._event_trace),
            "perf": self._perf_summary(),
        }

        # NEW: include compact HDT profiles if we have them
        if self._profiles_compact:
            router_capsule["humans"] = self._profiles_compact

        self.get_logger().info(
            f"[router] router_capsule trigger={rule} type={trig_type}, calling LLM"
        )


        # Decide whether to use router LLM or a fixed fast-only policy
        if not self.llm_enabled:
            self.get_logger().info(
                "[router] llm_enabled is False; skipping router LLM call and using fast-only policy"
            )
            policy = self.fast_only_policy(router_capsule)
        else:
            self.get_logger().info(
                f"[router] router_capsule trigger={rule} type={trig_type}, calling LLM"
            )
            try:
                policy = self.call_router_llm(router_capsule)
            except Exception as e:
                self.get_logger().error(
                    f"[router] LLM router call failed, falling back to default policy: {e}"
                )
                policy = self.default_policy(router_capsule)

        # Validate
        try:
            validate(instance=policy, schema=ROUTER_SCHEMA)
        except ValidationError as e:
            self.get_logger().error(
                f"[router] policy failed schema validation, using default: {e}"
            )
            policy = self.default_policy(router_capsule)

        self.apply_policy(policy)


        # NEW: after we know the policy, decide whether to tell the broker to run
        self._maybe_call_broker_run_initial(trig_type=trig_type, policy=policy)

    # ─────────────────────────
    # LLM call
    # ─────────────────────────

    def call_router_llm(self, capsule: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call mini router LLM and return a policy dict.
        """
        capsule_str = json.dumps(capsule, ensure_ascii=False)

        messages = [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": capsule_str},
            ]

        self.get_logger().info("\n=== ROUTER PROMPT ===\n" + json.dumps(messages, indent=2))

        t0 = time.time()

        if "gpt-oss" in self.model:
            client = Groq()
            resp = client.chat.completions.create(
                model='openai/' + self.model,
                messages=messages,
                reasoning_effort="medium",
                response_format={"type": "json_object"})  
        else:
            client = OpenAI()
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
            )
        
      

        t1 = time.time()
        lat_ms = (t1 - t0) * 1000.0
        
        content = resp.choices[0].message.content
        
        self.get_logger().info("\n=== ROUTER LLM RAW RESPONSE ===\n" + content + "\n" + "Latency: " + str(lat_ms) + '\n')
        
        policy = json.loads(content)
        return policy

    # ─────────────────────────
    # Fallback policy
    # ─────────────────────────

    def default_policy(self, capsule: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "latency_profile": "normal",
            "nodes": {
                "broker": {"tier": "fast"},
                "planner": {"tier": "balanced"},
                "hdt": {"tier": "fast"},
                "orchestrator": {"tier": "balanced"},
            },
        }
    


    # ─────────────────────────
    # Apply policy via ROS params
    # ─────────────────────────

    def apply_policy(self, policy: Dict[str, Any]):
        nodes_cfg = policy["nodes"]
        for role, cfg in nodes_cfg.items():
            if role not in self.role_nodes:
                self.get_logger().warn(f"[router] unknown role in policy: {role}")
                continue

            tier = cfg.get("tier", "fast")

            # Orchestrator is NEVER allowed to be fully off.
            if role == "orchestrator" and tier == "off":
                self.get_logger().warn(
                    "[router] policy requested orchestrator.tier='off'; "
                    "forcing tier='fast' and keeping orchestrator enabled."
                )
                tier = "fast"

            # For non-orchestrator roles, tier == "off" means LLM disabled.
            enabled = tier != "off" or role == "orchestrator"

            model_id = None
            if enabled:
                model_id = MODEL_CATALOG.get(role, {}).get(tier)
                if not model_id:
                    self.get_logger().warn(
                        f"[router] no model in MODEL_CATALOG for role={role} tier={tier}, using mini fallback"
                    )
                    model_id = "gpt-5.1-mini"

            self.set_llm_params_for_role(role, enabled, model_id)


    def set_llm_params_for_role(self, role: str, enabled: bool, model_id: str):
        node_name = self.role_nodes[role]
        client = self.param_clients[role]

        if not client.service_is_ready():
            self.get_logger().warn(
                f"[router] SetParameters service not ready for {node_name}"
            )
            return

        req = SetParameters.Request()

        # llm_enabled param
        p_enabled = ParameterMsg()
        p_enabled.name = "llm_enabled"
        p_enabled.value = ParameterValue(
            type=ParameterType.PARAMETER_BOOL,
            bool_value=bool(enabled),
        )
        req.parameters.append(p_enabled)

        # llm_model (if enabled)
        if model_id is not None:
            p_model = ParameterMsg()
            p_model.name = "model"  # change to "model_id" if that's your param name
            p_model.value = ParameterValue(
                type=ParameterType.PARAMETER_STRING,
                string_value=model_id,
            )
            req.parameters.append(p_model)

        future = client.call_async(req)
        future.add_done_callback(
            lambda fut, role=role, node_name=node_name: self._on_param_set_done(
                fut, role, node_name
            )
        )

    def _on_param_set_done(self, future, role: str, node_name: str):
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().error(
                f"[router] failed to set params for role={role} node={node_name}: {e}"
            )
            return

        if not all(r.successful for r in resp.results):
            self.get_logger().warn(
                f"[router] some params not accepted for {node_name}: {resp.results}"
            )
        else:
            self.get_logger().info(
                f"[router] updated llm params for role={role} node={node_name}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = RouterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

