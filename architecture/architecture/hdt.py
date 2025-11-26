#!/usr/bin/env python3
import json, time
from collections import deque
from typing import Dict, Any, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from std_msgs.msg import String as StringMsg
from std_srvs.srv import Trigger, SetBool

from openai import OpenAI
from jsonschema import validate, ValidationError

# ---------- Profile JSON Schema (per human) ----------

PROFILE_SCHEMA = {
    "type": "object",
    "required": ["id", "summary"],
    "properties": {
        "id":          {"type": "string"},
        "short_name":  {"type": "string"},
        "style":       {"type": "string"},  # e.g. concise|chatty|cautious
        "helpfulness": {"type": "string"},  # low|medium|high
        "risk_aversion": {"type": "string"},
        "contamination_bias": {"type": "string"},
        "instruction_pref": {
            "type": "object",
            "properties": {
                "detail_level":  {"type": "string"},
                "pace":          {"type": "string"},
                "confirmation":  {"type": "string"}
            },
            "additionalProperties": True
        },
        "interaction_history": {
            "type": "object",
            "properties": {
                "utterances_seen": {"type": "integer"},
                "last_command_ts": {"type": "number"}
            },
            "additionalProperties": True
        },
        "summary":  {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "additionalProperties": True
}

# Wrapper schema for multiple humans, keyed by id
PROFILES_SCHEMA = {
    "type": "object",
    "description": "Mapping from human id (e.g. 'H1','H2') to a profile object.",
    "properties": {},
    # Any key is allowed; each value must be a PROFILE_SCHEMA
    "additionalProperties": PROFILE_SCHEMA,
}


SYSTEM_HDT = (
    "You are a HUMAN DIGITAL TWIN module for a mobile-robot team.\n"
    "Your job is to maintain a compact, grounded psychological/interaction profile "
    "for each human teammate based ONLY on the provided event_trace, trigger, world, "
    "and prior_profile.\n\n"
    "HUMAN MAPPING:\n"
    "- Assume human 'H1' is primarily associated with zone 'A'.\n"
    "- Assume human 'H2' is primarily associated with zone 'B'.\n"
    "- The field 'robot_zone' indicates where the robot currently is (A/B/unknown).\n"
    "- The field 'active_human' is the human the robot is most likely interacting with now.\n\n"
    "Each profile should describe:\n"
    "- How this human tends to talk to the robot (style, tone, verbosity).\n"
    "- How helpful they seem (helpfulness).\n"
    "- How cautious they are about contamination / safety (risk_aversion, contamination_bias).\n"
    "- How they prefer instructions (instruction_pref: detail_level, pace, confirmation).\n"
    "- A short natural-language summary and a few bullet evidence items.\n\n"
    "STRICT RULES:\n"
    "- Use ONLY the provided events and prior_profile; do NOT invent specific past events.\n"
    "- If there is not enough data, mark attributes as 'unknown' and explain this in summary.\n"
    "- You may update multiple human profiles at once if the input clearly references them.\n"
    "- Use robot_zone/active_human to guess which human is currently speaking or being addressed.\n"
    "- Output STRICT JSON ONLY: an object whose keys are human ids (e.g., 'H1','H2'),\n"
    "  and whose values follow the Profile schema.\n"
)


class HumanDigitalTwinNode(Node):
    """
    Human Digital Twin (HDT) node.

    - Subscribes to /broker/context_capsule.
    - Maintains short history of context capsules (event_trace, trigger).
    - Uses an LLM to maintain/update profiles per human (H1, H2, ...).
    - Publishes profiles for planner on /profiles/summary (std_msgs/String, JSON).
    - Publishes perf on /llm/hdt_perf.
    """

    def __init__(self):
        super().__init__("human_digital_twin")

        # ----- Parameters -----
        self.declare_parameter("model", "gpt-5-mini")
        self.declare_parameter("temperature", 0.2)
        self.declare_parameter("model_H1", "gpt-5-mini")
        self.declare_parameter("model_H2", "gpt-5-mini")
        self.declare_parameter("capsule_topic", "/broker/context_capsule")
        self.declare_parameter("profiles_topic", "/profiles/summary")
        self.declare_parameter("perf_topic", "/llm/hdt_perf")
        self.declare_parameter("max_capsules", 20)
        self.declare_parameter("human_ids", ["H1", "H2"])  # can be customized
        self.declare_parameter("update_period_s", 2.0)     # throttle LLM calls

        self.declare_parameter("enabled", True)
        self.enabled = bool(self.get_parameter("enabled").value)

        # NEW: LLM-specific enable flag (toggled by orchestrator via SetParameters)
        self.declare_parameter("llm_enabled", True)
        self.llm_enabled = bool(self.get_parameter("llm_enabled").value)

        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.temperature = float(self.get_parameter("temperature").value)
        self.capsule_topic = self.get_parameter("capsule_topic").get_parameter_value().string_value
        self.profiles_topic = self.get_parameter("profiles_topic").get_parameter_value().string_value
        self.model_H1 = self.get_parameter("model_H1").get_parameter_value().string_value
        self.model_H2 = self.get_parameter("model_H2").get_parameter_value().string_value
        self.perf_topic = self.get_parameter("perf_topic").get_parameter_value().string_value
        self.max_capsules = int(self.get_parameter("max_capsules").value)
        self.human_ids = [str(x) for x in self.get_parameter("human_ids").value]
        self.update_period_s = float(self.get_parameter("update_period_s").value)

        # ----- State -----
        self._capsules: deque[Dict[str, Any]] = deque(maxlen=self.max_capsules)
        self._profiles: Dict[str, Dict[str, Any]] = {}   # id -> profile
        self._last_update_ts: float = 0.0

        # Last known capsule + active human
        self._last_capsule: Dict[str, Any] = {}
        self._last_robot_zone: Optional[str] = None
        self._active_human_id: Optional[str] = None   # "H1" or "H2"

        # OpenAI client
        self.client = OpenAI()

        # ROS I/O
        self.sub_capsule = self.create_subscription(
            StringMsg, self.capsule_topic, self._on_capsule, 10
        )
        self.pub_profiles = self.create_publisher(
            StringMsg, self.profiles_topic, 10
        )
        self.pub_perf = self.create_publisher(
            StringMsg, self.perf_topic, 10
        )

        # Optional service to force immediate update
        self.srv_update = self.create_service(
            Trigger, "/digital_twin/update_profiles", self._srv_update_profiles
        )

        # Optional service to force immediate update
        self.srv_update = self.create_service(
            Trigger, "/digital_twin/update_profiles", self._srv_update_profiles
        )

        # NEW: enable/disable HDT node at runtime
        self.srv_enable = self.create_service(
            SetBool, "/digital_twin/enable", self._srv_enable
        )


        # Allow runtime model/temp changes
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            f"human_digital_twin up | model={self.model} "
            f"capsule_topic={self.capsule_topic} profiles_topic={self.profiles_topic}"
        )

    def _srv_enable(self, req, res):
        self.enabled = bool(req.data)
        res.success = True
        res.message = f"HDT {'ENABLED' if self.enabled else 'DISABLED'}"
        self.get_logger().info(res.message)
        return res


    def _zone_to_human(self, zone: Optional[str]) -> Optional[str]:
        """
        Map robot zone to human ID.
        Adjust if your mapping changes.
        """
        if zone == "A":
            return "H1"
        if zone == "B":
            return "H2"
        return None


    # ---------- Dynamic parameters ----------
    def _on_set_parameters(self, params):
        for p in params:
            if p.name == "model" and p.type_ == Parameter.Type.STRING:
                self.model = p.value
                self.get_logger().info(f"[hdt] model changed to: {self.model}")

            elif p.name == "temperature" and p.type_ in (
                Parameter.Type.DOUBLE,
                Parameter.Type.INTEGER,
            ):
                self.temperature = float(p.value)
                self.get_logger().info(f"[hdt] temperature -> {self.temperature}")

            # NEW: toggle LLM participation via bool param (used by orchestrator)
            elif p.name == "llm_enabled" and p.type_ == Parameter.Type.BOOL:
                self.llm_enabled = bool(p.value)
                state = "ENABLED" if self.llm_enabled else "DISABLED"
                self.get_logger().info(f"[hdt] llm_enabled -> {state}")

        return SetParametersResult(successful=True, reason="ok")

    # ---------- Subscribers ----------
    def _on_capsule(self, msg: StringMsg):
        try:
            cap = json.loads(msg.data) if msg.data else {}
        except Exception:
            self.get_logger().warn("HDT: bad JSON on context_capsule")
            return

        # Trim trace length here if needed
        trace = cap.get("event_trace") or cap.get("trace") or []
        max_trace = 40
        cap["event_trace"] = trace[-max_trace:]

        # --- NEW: capture robot_zone from broker.world ---
        world = cap.get("world") or {}
        robot_zone = world.get("robot_zone")
        if isinstance(robot_zone, str):
            self._last_robot_zone = robot_zone

        # Decide which human is currently "active"
        self._active_human_id = self._zone_to_human(self._last_robot_zone)

        # Keep full last capsule around if you ever need it later
        self._last_capsule = cap

        # Store capsule in short history
        self._capsules.append(cap)

        # Do not call the LLM when HDT or its LLM is disabled, but keep history.
        if not self.enabled or not self.llm_enabled:
            # Do not call the LLM when HDT is disabled, but keep history.
            return

        # Throttled automatic update
        now = time.time()
        if now - self._last_update_ts >= self.update_period_s:
            try:
                self._update_profiles()
                self._last_update_ts = now
            except Exception as e:
                self.get_logger().warn(f"HDT: auto update failed: {e}")


    # ---------- Services ----------
    def _srv_update_profiles(self, req, res):
        try:
            self._update_profiles()
            res.success = True
            res.message = "profiles updated"
        except Exception as e:
            res.success = False
            res.message = f"HDT update error: {e}"
        return res

    # ---------- LLM glue ----------
    def _build_llm_messages(self) -> List[Dict[str, Any]]:
        """
        Build an LLM prompt that gives:
          - recent capsules
          - current profiles (if any)
          - list of human ids we care about
          - current robot_zone and active_human hint
        """
        payload = {
            "human_ids": self.human_ids,
            "recent_capsules": list(self._capsules),
            "prior_profiles": self._profiles,
            "robot_zone": self._last_robot_zone,          # NEW
            "active_human": self._active_human_id,        # NEW
        }
        return [
            {"role": "system", "content": SYSTEM_HDT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def _publish_perf(self, lat_ms: float, ok: bool, phase: str = "profile_update"):
        payload = {
            "node": "hdt",
            "model": self.model,
            "lat_ms": float(lat_ms),
            "ok": bool(ok),
            "phase": phase,
            "ts": time.time(),
        }
        try:
            self.pub_perf.publish(StringMsg(data=json.dumps(payload)))
        except Exception as e:
            self.get_logger().warn(f"[hdt] failed to publish perf: {e}")

    def _chat_profiles_json(self, messages: List[Dict[str, Any]], model_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Call OpenAI, expect STRICT JSON keyed by human id, validate with PROFILES_SCHEMA.
        """
        model_id = model_override or self.model
        last_exc: Optional[Exception] = None
        for attempt in range(2):  # 1 retry
            t0 = time.time()
            ok = False
            try:
            
                self.get_logger().info("\n=== HDT PROMPT ===\n" + json.dumps(messages, indent=2))
            
                resp = self.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "HumanProfiles",
                            "schema": PROFILES_SCHEMA,
                        },
                    },
                )
                t1 = time.time()
                lat_ms = (t1 - t0) * 1000.0
                ok = True

                
                

                content = resp.choices[0].message.content
                self.get_logger().info("\n=== HDT LLM RAW RESPONSE ===\n" + content + "\n")

                obj = json.loads(content)
                
                # --- NEW: ensure each profile has an 'id' field matching its key ---
                if isinstance(obj, dict):
                    for human_id, prof in obj.items():
                        if isinstance(prof, dict) and "id" not in prof:
                            prof["id"] = human_id
                
                validate(instance=obj, schema=PROFILES_SCHEMA)

                self._publish_perf(lat_ms=lat_ms, ok=True, phase="profile_update")
                return obj

            except Exception as e:
                last_exc = e
                t1 = time.time()
                lat_ms = (t1 - t0) * 1000.0
                self._publish_perf(lat_ms=lat_ms, ok=False, phase="profile_update")

                # tighten instructions
                messages = messages + [
                    {
                        "role": "system",
                        "content": "Return ONLY valid JSON per the schema. No prose.",
                    }
                ]

        raise ValueError(f"HDT LLM did not return valid profiles JSON: {last_exc}")

    # ---------- Core ----------
    def _update_profiles(self):
    
        # Node or LLM disabled → do nothing
        if not self.enabled or not self.llm_enabled:
            return
    
        if not self._capsules:
            # nothing to do yet
            return

        # --- NEW: pick an LLM model based on active human, if any ---
        active = self._active_human_id
        if active == "H1":
            effective_model = self.model_H1 or self.model
        elif active == "H2":
            effective_model = self.model_H2 or self.model
        else:
            effective_model = self.model

        msgs = self._build_llm_messages()
        self.get_logger().info("HDT: calling LLM to update profiles")

        profiles_obj = self._chat_profiles_json(msgs, model_override=effective_model)

        # Merge: LLM output replaces/extends our current profiles
        # (You could also do more careful merging; for now we trust the LLM to carry over.)
        self._profiles = profiles_obj

        self._profiles["_meta"] = {
            "active_human": self._active_human_id,
            "robot_zone": self._last_robot_zone,
            "ts": time.time(),
        }

        # Publish summary for planner
        out_msg = StringMsg()
        out_msg.data = json.dumps(self._profiles, ensure_ascii=False)
        self.pub_profiles.publish(out_msg)

        self.get_logger().info(f"HDT: published profiles: {self._profiles}")

    # ---------- Shutdown ----------
    def destroy_node(self):
        super().destroy_node()


def main():
    rclpy.init()
    node = HumanDigitalTwinNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

