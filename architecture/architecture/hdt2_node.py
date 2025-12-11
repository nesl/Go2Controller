#!/usr/bin/env python3
"""
hdt_node.py

Human Digital Twin (HDT) node.

- Maintains lightweight, interpretable per-human profiles.
- Ingests:
    * /events/basic      (EventLayer basic rule hits)
    * /events/composite  (EventLayer composite rule hits)
    * /task_state        (Task State Monitor summary)

- Publishes:
    * /profiles/summary (std_msgs/String, JSON):
        {
          "human_a": {...profile...},
          "human_b": {...profile...},
          "_meta": {
            "active_human": "human_a" | "human_b" | "human_unknown" | null,
            "robot_zone": "A" | "B" | "unknown" | null,
            "ts": <float>
          }
        }

Profiles are free-form but structured, and must at least include:
  - id, summary
  - style, helpfulness, risk_aversion, contamination_bias
  - communication_self  (how the human prefers to communicate, possibly nonverbal)
  - communication_robot (how they prefer the robot to communicate, possibly nonverbal)
"""

import json
import time
from collections import defaultdict
from typing import Any, Dict, Optional, List  # UPDATED: added List

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from std_msgs.msg import String as StringMsg

from jsonschema import validate, ValidationError
from openai import OpenAI
from groq import Groq


# ---------- Profile JSON Schema (per human) ----------
PROFILE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["id", "summary"],
    "properties": {
        "id":           {"type": "string"},
        "short_name":   {"type": "string"},

        # Router-expected-ish fields
        "style":            {"type": "string"},
        "helpfulness":      {"type": "string"},
        "risk_aversion":    {"type": "string"},
        "contamination_bias": {"type": "string"},

        "language":         {"type": "string"},
        "notes":            {"type": "string"},

        # How the HUMAN prefers to communicate (may be nonverbal)
        "communication_self": {
            "type": "object",
            "properties": {
                "preferred_modalities": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "nonverbal_cues": {
                    "type": "string"
                },
                "typical_utterance_style": {
                    "type": "string"
                }
            },
            "additionalProperties": True
        },

        # How they prefer the ROBOT to communicate (may be nonverbal)
        "communication_robot": {
            "type": "object",
            "properties": {
                "preferred_modalities": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "nonverbal_signals": {
                    "type": "string"
                },
                "verbosity_preference": {
                    "type": "string"
                },
                "confirmation_preference": {
                    "type": "string"
                }
            },
            "additionalProperties": True
        },

        "recent_frustration":      {"type": "number"},
        "proactivity_preference":  {"type": "number"},

        "summary": {"type": "string"},
    },
    "additionalProperties": True,
}


PROFILE_SYSTEM_PROMPT = """
You maintain compact HUMAN PROFILES for a mobile robot teammate "Bob".

You receive:
  - human_id
  - recent_features: statistics and observations about that human
  - task_state: a snapshot of the team/task context (zones, backlogs, etc.)
  - old_profile: previous profile for that human (may be null)
  - event_trace: a recent sequence of events that have been linked to this human.
    Each event includes: timestamp, source ("basic" or "composite"), rule id,
    zone, and a small excerpt of the event data (e.g., text, kind, intent, etc.).

Your job:
  - Interpret the features, the event_trace, and context into a small, stable
    profile that helps the robot adapt its behavior and communication.

You MUST output a JSON object matching PROFILE_SCHEMA with at least:
  - id, short_name
  - style: free-text description of their interaction style
  - helpfulness: how willing they seem to help or collaborate
  - risk_aversion: how cautious or bold they are (especially about contamination)
  - contamination_bias: how they seem to classify objects (e.g., "overcautious",
    "underestimates risk", "roughly calibrated")
  - recent_frustration: a number (0.0–1.0) representing how frustrated they seem lately
  - proactivity_preference: a number (0.0–1.0) for how much proactive help they like
  - communication_self: object describing how THEY usually communicate with the robot,
    including:
      * preferred_modalities: a list of free-form strings, e.g.,
        ["speech", "pointing", "head nods"]
      * nonverbal_cues: short description (gestures, gaze, posture, etc.)
      * typical_utterance_style: e.g., "short commands", "polite questions"
  - communication_robot: object describing how they prefer the ROBOT to communicate, including:
      * preferred_modalities: free-form, e.g., ["speech", "lights", "navigation motions"]
      * nonverbal_signals: description of nonverbal cues the robot should use or avoid
      * verbosity_preference: e.g., "very brief", "step-by-step explanations"
      * confirmation_preference: e.g., "confirm important actions only", 
        "confirm every step when stakes are high"
  - summary: 1–3 sentences giving concrete advice for how Bob should interact with them.

Rules:
  - Use free-form natural language; do NOT restrict yourself to fixed vocabularies.
  - Be conservative; do not make extreme claims from weak evidence.
  - If something is uncertain, you can say so in the text, but still make a best guess.
  - Keep the profile small and practical; focus on aspects that matter for coordination
    and communication.
  - If old_profile exists, you may refine or slightly adjust it, not completely overwrite
    without good reason.

Output STRICT JSON only, no extra explanations.
""".strip()


class HumanDigitalTwinNode(Node):
    """
    Human Digital Twin node:
      - Aggregates features per human from EventLayer (/events/basic + /events/composite)
        and task-level state (/task_state).
      - Optionally refines profiles with an LLM.
      - Publishes /profiles/summary JSON snapshots.
    """

    def __init__(self):
        super().__init__("hdt_node")

        # --------- Parameters ---------
        self.declare_parameter("llm_enabled", True)
        self.declare_parameter("model", "gpt-5.1-mini")
        self.declare_parameter("groq_model_prefix", "gpt-oss")
        self.declare_parameter("publish_period_s", 1.0)
        self.declare_parameter("profile_update_period_s", 20.0)
        self.declare_parameter("frustration_alpha", 0.3)
        self.declare_parameter("help_alpha", 0.3)
        self.declare_parameter("utterance_alpha", 0.3)
        self.declare_parameter("default_language", "en")

        # NEW: how many events to keep per human for event traces
        self.declare_parameter("event_trace_max_len", 50)

        self.llm_enabled = self.get_parameter("llm_enabled").value
        self.model = self.get_parameter("model").value
        self.groq_prefix = self.get_parameter("groq_model_prefix").value
        self.publish_period_s = float(self.get_parameter("publish_period_s").value)
        self.profile_update_period_s = float(self.get_parameter("profile_update_period_s").value)
        self.frustration_alpha = float(self.get_parameter("frustration_alpha").value)
        self.help_alpha = float(self.get_parameter("help_alpha").value)
        self.utterance_alpha = float(self.get_parameter("utterance_alpha").value)
        self.default_language = self.get_parameter("default_language").value
        self.event_trace_max_len = int(self.get_parameter("event_trace_max_len").value)  # NEW

        self.add_on_set_parameters_callback(self._on_param_change)

        # --------- Internal state ---------
        # features[human_id] = {...aggregated features...}
        self.features: Dict[str, Dict[str, Any]] = defaultdict(dict)
        # profiles[human_id] = {...PROFILE_SCHEMA...}
        self.profiles: Dict[str, Dict[str, Any]] = {}
        # last LLM update per human
        self._last_profile_update_ts: Dict[str, float] = {}

        self._active_human: Optional[str] = None
        self._robot_zone: Optional[str] = None
        self._last_task_state: Dict[str, Any] = {}

        # NEW: per-human event traces (recent event history)
        # _event_traces[human_id] = [ {ts, src, rule, zone, data_excerpt, ...}, ... ]
        self._event_traces: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # --------- ROS I/O ---------
        self.sub_events_basic = self.create_subscription(
            StringMsg, "/events/basic", self._on_event_basic, 50
        )
        self.sub_events_comp = self.create_subscription(
            StringMsg, "/events/composite", self._on_event_composite, 50
        )
        self.sub_task_state = self.create_subscription(
            StringMsg, "/task_state", self._on_task_state, 10
        )

        self.pub_profiles = self.create_publisher(
            StringMsg, "/profiles/summary", 10
        )

        self.create_timer(self.publish_period_s, self._publish_profiles_summary)

        self.get_logger().info("hdt_node initialized (basic + composite events)")

    # ---------- Param updates ----------
    def _on_param_change(self, params):
        for p in params:
            if p.name == "llm_enabled" and p.type_ == Parameter.Type.BOOL:
                self.llm_enabled = p.value
                self.get_logger().info(f"[hdt] llm_enabled -> {self.llm_enabled}")
            elif p.name == "model" and p.type_ == Parameter.Type.STRING:
                self.model = p.value
                self.get_logger().info(f"[hdt] model -> {self.model}")  # FIXED typo
            elif p.name == "event_trace_max_len" and p.type_ == Parameter.Type.INTEGER:
                self.event_trace_max_len = int(p.value)
                self.get_logger().info(f"[hdt] event_trace_max_len -> {self.event_trace_max_len}")
        return SetParametersResult(successful=True, reason="ok")

    # ---------- Helpers ----------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _ensure_human(self, human_id: str) -> Dict[str, Any]:
        f = self.features.get(human_id)
        if f is None:
            f = {
                "utterances": 0,
                "avg_len": 0.0,
                "frustration_ema": 0.0,
                "help_accept_ema": 0.5,
                "language": self.default_language,
                "spoken_events": 0,
                "nonverbal_events": 0,
                "last_ts": self._now(),
            }
            self.features[human_id] = f
        return f

    def _ewma_update(self, old: float, new: float, alpha: float) -> float:
        return (1.0 - alpha) * old + alpha * new

    def _guess_human_id_from_zone(self, zone: Optional[str]) -> str:
        if zone == "A":
            return "human_a"
        if zone == "B":
            return "human_b"
        return "human_unknown"

    def _guess_human_id_from_event(self, evt: Dict[str, Any]) -> str:
        data = evt.get("data") or {}
        hid = data.get("human_id") or data.get("speaker_id") or data.get("agent_id")
        if isinstance(hid, str) and hid:
            return hid
        zone = evt.get("zone") or data.get("zone")
        return self._guess_human_id_from_zone(zone)

    # NEW: record an event into this human's event trace
    def _record_event_for_human(self, human_id: str, evt: Dict[str, Any], src: str):
        """
        Store a slimmed-down version of an event for later LLM contextualization.
        Assumes human_id has already been inferred (e.g., from zone).
        """
        if not human_id or human_id == "human_unknown":
            return

        ts = float(evt.get("ts") or self._now())
        zone = evt.get("zone")
        rule_id = evt.get("rule")
        data = evt.get("data") or {}
        etype = evt.get("type")  # for composite events

        # Build a small data excerpt to avoid massive payloads
        data_excerpt: Dict[str, Any] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                # keep small scalar fields; truncate strings
                if isinstance(v, str):
                    data_excerpt[k] = v[:160]
                elif isinstance(v, (int, float, bool)):
                    data_excerpt[k] = v
                # keep small nested dicts with simple scalars if you like
                # else ignore to keep things compact

        trace_entry = {
            "ts": ts,
            "src": src,
            "rule": rule_id,
            "zone": zone,
            "type": etype,
            "data": data_excerpt,
        }

        buf = self._event_traces[human_id]
        buf.append(trace_entry)
        # enforce max length
        if len(buf) > self.event_trace_max_len:
            # drop oldest entries
            del buf[0 : len(buf) - self.event_trace_max_len]

    def _get_event_trace(self, human_id: str) -> List[Dict[str, Any]]:
        """
        Return a shallow copy of the recent event trace for this human.
        """
        return list(self._event_traces.get(human_id, []))

    # ---------- Task state ingestion ----------
    def _on_task_state(self, msg: StringMsg):
        try:
            obj = json.loads(msg.data)
        except Exception:
            self.get_logger().warn(f"[hdt] bad JSON on /task_state: {msg.data}")
            return
        if isinstance(obj, dict):
            self._last_task_state = obj

    # ---------- BASIC events ----------
    def _on_event_basic(self, msg: StringMsg):
        """
        Handle /events/basic from EventLayer.

        EventLayer payload:
          {"ts":..., "rule":<id>, "data":{...}, "zone":<A|B|...>}
        """
        try:
            evt = json.loads(msg.data)
        except Exception:
            self.get_logger().warn(f"[hdt] bad JSON on /events/basic: {msg.data}")
            return

        if not isinstance(evt, dict):
            return

        # mark source for tracing
        evt["_src"] = "basic"
        rule_id = str(evt.get("rule") or "")
        data = evt.get("data") or {}
        zone = evt.get("zone")
        ts = float(evt.get("ts") or self._now())

        if isinstance(zone, str) and zone:
            self._robot_zone = zone

        # Link this basic event to a human based on explicit ids or zone
        human_id = self._guess_human_id_from_event(evt)
        if human_id:
            # ensure we have a feature slot for them
            f = self._ensure_human(human_id)
            f["last_ts"] = ts
            # record the event in their trace
            self._record_event_for_human(human_id, evt, src="basic")

        # Mark "active human" on final speech events that are commands to Bob
        if rule_id in ("speech_final_any", "trigger_speech_final"):
            if human_id:
                self._active_human = human_id
            else:
                self._active_human = self._guess_human_id_from_event(evt)

        # Meta info from llm_speech_check: interpret style, frustration, etc.
        if "kind" in data and "intent" in data:
            # use zone and ts we already parsed
            self._ingest_speech_meta_event(rule_id, data, ts, zone)

    # ---------- COMPOSITE events ----------
    def _on_event_composite(self, msg: StringMsg):
        """
        Handle /events/composite from EventLayer.

        Payload is typically:
          {"type":"composite","rule":<id>,"expr":<expr>,"ts":<float>,"zone":<A|B|...>}
        """
        try:
            evt = json.loads(msg.data)
        except Exception:
            self.get_logger().warn(f"[hdt] bad JSON on /events/composite: {msg.data}")
            return

        if not isinstance(evt, dict):
            return

        # mark source for tracing
        evt["_src"] = "composite"
        rule_id = str(evt.get("rule") or "")
        zone = evt.get("zone")
        ts = float(evt.get("ts") or self._now())

        if isinstance(zone, str) and zone:
            self._robot_zone = zone

        # Link composite event to a human via zone (one human per zone assumption)
        human_id = self._guess_human_id_from_event(evt)
        if human_id:
            f = self._ensure_human(human_id)
            f["last_ts"] = ts
            self._record_event_for_human(human_id, evt, src="composite")

        # Example: human_ping_here represents a nonverbal “ping” from a human
        if rule_id == "human_ping_here":
            hid = human_id or self._guess_human_id_from_zone(zone)
            f = self._ensure_human(hid)
            f["nonverbal_events"] = int(f.get("nonverbal_events", 0)) + 1
            f["last_ts"] = ts

        # Example: trigger_idle → no recent activity; we can clear active_human
        if rule_id == "trigger_idle":
            self._active_human = None

        # Other composite triggers (llm_infer_intent, vlm_describe_scene_with_human)
        # are implicitly handled via their downstream basic events (e.g. /llm/speech_check)

    # ---------- Feature updates ----------
    def _ingest_speech_meta_event(
        self,
        rule_id: str,
        data: Dict[str, Any],
        ts: float,
        zone: Optional[str],
    ):
        """
        Ingest meta about a spoken utterance, e.g. from llm_speech_check.

        Expected (best-effort):
          text: str
          language: optional str
          frustration: optional [0,1] float
          verbosity: optional numeric proxy
          human_id/speaker_id: optional
        """
        txt = (data.get("text") or "").strip()
        lang = (data.get("language") or "").strip()
        frustration = data.get("frustration")
        verbosity = data.get("verbosity")

        human_id = data.get("human_id") or data.get("speaker_id")
        if not human_id:
            human_id = self._guess_human_id_from_zone(zone)

        f = self._ensure_human(human_id)
        f["last_ts"] = ts

        if lang:
            f["language"] = lang

        length = len(txt.split()) if txt else 0
        old_avg = float(f.get("avg_len", 0.0))
        f["avg_len"] = self._ewma_update(old_avg, float(length), self.utterance_alpha)
        f["utterances"] = int(f.get("utterances", 0)) + 1

        if isinstance(frustration, (int, float)):
            val = max(0.0, min(1.0, float(frustration)))
            old_f = float(f.get("frustration_ema", 0.0))
            f["frustration_ema"] = self._ewma_update(old_f, val, self.frustration_alpha)

        # speech meta → spoken communication evidence
        f["spoken_events"] = int(f.get("spoken_events", 0)) + 1

        if isinstance(verbosity, (int, float)):
            f["verbosity"] = float(verbosity)

    def _update_help_preference(self, human_id: str, event: str):
        # Hook if later you map some composite/basic rule to help_accepted / help_rejected
        f = self._ensure_human(human_id)
        old = float(f.get("help_accept_ema", 0.5))
        val = 1.0 if event == "help_accepted" else 0.0
        f["help_accept_ema"] = self._ewma_update(old, val, self.help_alpha)

    # ---------- LLM profile refinement ----------
    def _maybe_update_profile_with_llm(self, human_id: str):
        now = self._now()
        last = self._last_profile_update_ts.get(human_id, 0.0)
        if now - last < self.profile_update_period_s:
            return

        feats = self.features.get(human_id)
        if not feats:
            return

        old_profile = self.profiles.get(human_id)
        event_trace = self._get_event_trace(human_id)  # NEW

        if not self.llm_enabled:
            profile = self._heuristic_profile(human_id, feats, old_profile)
            self.profiles[human_id] = profile
            self._last_profile_update_ts[human_id] = now
            return

        payload = {
            "human_id": human_id,
            "recent_features": feats,
            "task_state": self._last_task_state,
            "old_profile": old_profile,
            "event_trace": event_trace,  # NEW: pass per-human event trace to LLM
        }

        messages = [
            {"role": "system", "content": PROFILE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        try:
            raw = self._call_llm(messages)
            validate(instance=raw, schema=PROFILE_SCHEMA)
            self.profiles[human_id] = raw
            self._last_profile_update_ts[human_id] = now
        except ValidationError as e:
            self.get_logger().warn(f"[hdt] profile JSON failed schema validation for {human_id}: {e}")
            profile = self._heuristic_profile(human_id, feats, old_profile)
            self.profiles[human_id] = profile
            self._last_profile_update_ts[human_id] = now
        except Exception as e:
            self.get_logger().warn(f"[hdt] LLM profile update error for {human_id}: {e}")

    def _heuristic_profile(
        self,
        human_id: str,
        feats: Dict[str, Any],
        old_profile: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        lang = feats.get("language") or self.default_language
        avg_len = float(feats.get("avg_len", 0.0))
        frustration = float(feats.get("frustration_ema", 0.0))
        help_accept = float(feats.get("help_accept_ema", 0.5))
        spoken = int(feats.get("spoken_events", 0))
        nonverbal = int(feats.get("nonverbal_events", 0))

        # Basic style heuristic
        if avg_len < 5:
            style = "brief and to the point"
        elif avg_len < 15:
            style = "moderately detailed"
        else:
            style = "very detailed and talkative"

        helpfulness = "generally helpful" if help_accept >= 0.5 else "sometimes reluctant to accept help"
        risk_aversion = "cautious" if frustration > 0.6 else "moderate"
        contamination_bias = "unknown"

        preferred_self_modalities = []
        if spoken >= nonverbal:
            preferred_self_modalities.append("speech")
        if nonverbal >= spoken:
            preferred_self_modalities.append("nonverbal gestures")
        if not preferred_self_modalities:
            preferred_self_modalities = ["speech"]

        preferred_robot_modalities = ["speech"]
        verbosity_pref = "brief explanations" if avg_len < 8 else "step-by-step explanations"
        confirmation_pref = "confirm important actions only"

        proactivity = max(0.0, min(1.0, help_accept))

        summary = (
            f"{human_id} usually communicates in {lang}, "
            f"tends to be {style}, and appears "
            f"{'quite' if frustration > 0.6 else 'not very'} frustrated recently. "
            f"They seem to {'appreciate' if proactivity >= 0.5 else 'prefer limited'} proactive help. "
            f"Use {', '.join(preferred_robot_modalities)} with {verbosity_pref}."
        )

        profile = {
            "id": human_id,
            "short_name": human_id,
            "style": style,
            "helpfulness": helpfulness,
            "risk_aversion": risk_aversion,
            "contamination_bias": contamination_bias,
            "language": lang,
            "recent_frustration": float(frustration),
            "proactivity_preference": float(proactivity),
            "communication_self": {
                "preferred_modalities": preferred_self_modalities,
                "nonverbal_cues": "may use gestures or stance; details unknown",
                "typical_utterance_style": style,
            },
            "communication_robot": {
                "preferred_modalities": preferred_robot_modalities,
                "nonverbal_signals": "neutral; use gentle motion and simple cues",
                "verbosity_preference": verbosity_pref,
                "confirmation_preference": confirmation_pref,
            },
            "notes": "",
            "summary": summary,
        }

        return profile

    def _call_llm(self, messages: list) -> Dict[str, Any]:
        model = self.model
        self.get_logger().info(
            f"\n[hdt] === PROFILE LLM PROMPT ===\n{messages}\n"
        )

        t0 = time.time()
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
                messages=messages,
                reasoning_effort="medium",
                response_format={"type": "json_object"},
            )

        t1 = time.time()
        lat_ms = (t1 - t0) * 1000.0
        content = resp.choices[0].message.content
        self.get_logger().info(
            "\n[hdt] === PROFILE LLM RAW RESPONSE ===\n"
            + content
            + f"\nLatency: {lat_ms:.1f} ms\n"
        )
        return json.loads(content)

    # ---------- Publishing ----------
    def _publish_profiles_summary(self):
        now = self._now()

        # Ensure each known human has some profile
        for human_id, feats in list(self.features.items()):
            self._maybe_update_profile_with_llm(human_id)
            if human_id not in self.profiles:
                self.profiles[human_id] = self._heuristic_profile(
                    human_id, feats, self.profiles.get(human_id)
                )

        payload = {hid: prof for hid, prof in self.profiles.items()}

        meta = {
            "active_human": self._active_human,
            "robot_zone": self._robot_zone,
            "ts": now,
        }
        payload["_meta"] = meta

        try:
            s = json.dumps(payload, ensure_ascii=False)
            self.pub_profiles.publish(StringMsg(data=s))
        except Exception as e:
            self.get_logger().warn(f"[hdt] failed to publish profiles/summary: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = HumanDigitalTwinNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

