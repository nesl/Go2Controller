#!/usr/bin/env python3
# planner_node.py
#
# LLM-driven high-level planner.
# - Ingests broker facts + context capsule + human profiles.
# - Produces an unstructured plan + a small machine-readable header.
# - Optionally produces open information needs to iterate broker queries.
#
import os, json, time, re
from typing import List, Dict, Any, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String as StringMsg
from std_srvs.srv import Trigger

from openai import OpenAI
from jsonschema import validate, ValidationError

# ---------- JSON Schemas ----------
PLAN_SCHEMA = {
    "type": "object",
    "required": ["plan_header", "plan_doc"],
    "properties": {
        "plan_header": {
            "type": "object",
            "required": ["objective", "time_horizon", "priority", "communication_policy"],
            "properties": {
                "objective": {"type": "string"},
                "time_horizon": {"type": "string", "enum": ["short", "medium", "long"]},
                "priority": {"type": "array", "items": {"type": "string"}},
                "communication_policy": {"type": "object"},
                "risk_flags": {"type": "array", "items": {"type": "string"}},
                "assumptions": {"type": "array", "items": {"type": "string"}},
                "exploration_hint": {"type": "string"}
            },
            "additionalProperties": True
        },
        "plan_doc": {"type": "string"}
    },
    "additionalProperties": False
}

NEEDS_SCHEMA = {
    "type": "object",
    "required": ["needs"],
    "properties": {
        "needs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["why"],
                "properties": {
                    "why": {"type": "string"},
                    "focus": {"type": "string", "enum": ["object","zone","person","policy","uncertainty","route","other"]},
                    "object_id": {"type": "string"},
                    "zone": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low","medium","high","critical"]}
                },
                "additionalProperties": True
            }
        }
    },
    "additionalProperties": False
}

# ---------- Prompt snippets ----------
SYSTEM_PLAN = (
  "You are the HIGH-LEVEL PLANNER for a mobile robot collaborating with two humans across areas A/B.\n"
  "Objective: find all Bluetooth-tagged objects and place each into the correct bin (clean vs contaminated).\n"
  "Inputs:\n"
  " • ContextCapsule (trigger, budgets, recent event trace, phase/goals)\n"
  " • Profiles (digital-twin summaries for Human A / Human B)\n"
  " • Facts (compact SQL-derived tables from the broker)\n\n"
  "OUTPUT: STRICT JSON ONLY matching the schema.\n"
  "Inside plan_doc, write a short narrative (≤ 220 words) using THIS fixed section template and rules:\n"
  "  Situation: <what just happened, who’s present, basket state, backlog highlights>.\n"
  "  Intent: <goal and subgoals in plain words>.\n"
  "  Action Sketch: <2–5 step sketch using ONLY these verbs: approach, ask_scan, confirm, pick, carry, deliver, verify, handoff>.\n"
  "  Evidence & Uncertainty: <cite key evidence; mark gaps as OPEN: ...>.\n"
  "  Coordination & Tone: <how to speak/gesture to each human>.\n"
  "  Hard Constraints: <capacity/bin access/time/safety>.\n"
  "Refer to entities with canonical tags inline, e.g., object_id=CNode12, area=A|B|H1_zone|H2_zone, bin=clean|contaminated.\n"
  "Keep it groundable, do not invent IDs; if unknown, mark OPEN.\n"
  "Be proactive when idle; confirm when uncertainty is high; minimize travel; resolve label disagreements.\n"
)

EX_USER_PLAN = {
  "ContextCapsule": {"trigger":{"type":"new_object","hints":{"object_id":"CNode12"}}},
  "Profiles": {"human_a":{"style":"concise"}, "human_b":{"style":"confirm"}},
  "Facts": {"vw_object_sheet":[{"node_id":"CNode12","in_basket":0,"best_zone":"B",
                               "robot_probability":0.63,"human_a_probability":0.76}]}
}
EX_ASSISTANT_PLAN = {
  "plan_header":{
    "objective":"Bin all nodes correctly",
    "time_horizon":"short",
    "priority":["resolve CNode12 disagreement","reduce travel B→A"],
    "communication_policy":{"human_a":"proactive+brief","human_b":"reactive+confirm"},
    "risk_flags":["label_disagreement:CNode12"],
    "assumptions":["contaminated bin in H2_zone"]
  },
  "plan_doc":
    "Situation: New item object_id=CNode12 scanned by Human A; robot in area=H1_zone; bin=contaminated is in area=H2_zone; basket 1/4 full.\n"
    "Intent: Confirm label for CNode12 and place it in the correct bin.\n"
    "Action Sketch: approach H1; ask_scan CNode12 via NFC; confirm contamination; carry to bin=contaminated in area=H2_zone if p≥0.7; deliver; verify with Human B.\n"
    "Evidence & Uncertainty: strongest RSSI in H1_zone; human_a p=0.76 vs robot p=0.63. OPEN: reconcile.\n"
    "Coordination & Tone: brief single-step prompts for Human A; confirmation question for Human B.\n"
    "Hard Constraints: capacity OK; cross-area travel allowed; keep path short via corridor C2."
}


SYSTEM_NEEDS = (
    "Given the same inputs, list the most critical OPEN information needs that, if answered, "
    "would materially improve the plan quality or safety. Keep the list small and specific. "
    "Output STRICT JSON ONLY matching the provided schema."
)

_TAG_RE = re.compile(r"(object_id=\w+|area=\w+|bin=(?:clean|contaminated)|OPEN:[^.\n]+)")
def _extract_hints(plan_doc: str) -> dict:
    tags = _TAG_RE.findall(plan_doc or "")
    return {
        "object_ids": sorted(set([t.split("=",1)[1] for t in tags if t.startswith("object_id=")])),
        "areas":      sorted(set([t.split("=",1)[1] for t in tags if t.startswith("area=")])),
        "bins":       sorted(set([t.split("=",1)[1] for t in tags if t.startswith("bin=")])),
        "open_items": [t[5:].strip() for t in tags if t.startswith("OPEN:")]
    }

# ---------- Helper: compact representations ----------
def _bound_list(items: List[Any], n: int) -> List[Any]:
    return list(items[:max(0, int(n))])

def _truncate_str(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else (s[:n-1] + "…")

class PlannerNode(Node):
    def __init__(self):
        super().__init__("planner_node")

        # --- Parameters ---
        self.declare_parameter("model", "gpt-5-nano")
        self.declare_parameter("temperature_plan", 0.2)
        self.declare_parameter("temperature_needs", 0.2)
        # Budgets for what we pass to the LLM (keep tokens in check)
        self.declare_parameter("max_facts_packs", 6)       # how many recent fact packs
        self.declare_parameter("max_rows_per_table", 20)   # rows per table inside each pack
        self.declare_parameter("max_trace_events", 40)     # from ContextCapsule.trace
        # Topics
        self.declare_parameter("facts_topic", "/broker/facts")
        self.declare_parameter("capsule_topic", "/broker/context_capsule")
        self.declare_parameter("profiles_topic", "/profiles/summary")
        
        self.declare_parameter("max_open_iterations", 0)
        self.max_open_iters = int(self.get_parameter("max_open_iterations").value)

        self._waiting_for_info = False
        self._open_items = []        # last OPEN tags extracted from plan_doc
        self._open_iter = 0          # how many refine rounds we’ve done in this cycle


        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.temp_plan = float(self.get_parameter("temperature_plan").value)
        self.temp_needs = float(self.get_parameter("temperature_needs").value)
        self.max_packs = int(self.get_parameter("max_facts_packs").value)
        self.max_rows = int(self.get_parameter("max_rows_per_table").value)
        self.max_trace = int(self.get_parameter("max_trace_events").value)

        self.facts_topic = self.get_parameter("facts_topic").get_parameter_value().string_value
        self.capsule_topic = self.get_parameter("capsule_topic").get_parameter_value().string_value
        self.profiles_topic = self.get_parameter("profiles_topic").get_parameter_value().string_value

        # --- State (working set) ---
        self._facts_buffer: List[Dict[str, Any]] = []   # list of fact packs
        self._capsule: Dict[str, Any] = {}              # last context capsule
        self._profiles: Dict[str, Any] = {}             # last profiles summary
        self._ws_id = 0                                 # simple counter for iterations

        # --- OpenAI client ---
        self.client = OpenAI()

        # --- ROS I/O ---
        self.sub_facts = self.create_subscription(StringMsg, self.facts_topic, self._on_facts, 40)
        
        self.sub_facts_delta = self.create_subscription(
            StringMsg,
            "/broker/facts_delta",
            self._on_facts_delta,
            40
        )
        
        self.sub_capsule = self.create_subscription(StringMsg, self.capsule_topic, self._on_capsule, 10)
        self.sub_profiles = self.create_subscription(StringMsg, self.profiles_topic, self._on_profiles, 10)

        self.sub_plan_req = self.create_subscription(StringMsg, "/planner/plan_req", self._on_plan_req, 10)

        self.pub_plan = self.create_publisher(StringMsg, "/planner/plan_out", 10)


        self.srv_plan = self.create_service(Trigger, "/planner/plan", self._srv_plan)
        self.pub_needs = self.create_publisher(StringMsg, "/planner/needs", 10)

        self.get_logger().info(f"planner_node up | model={self.model}")

    # ---------- Subscribers ----------
    def _on_facts(self, msg: StringMsg):
        
        self.get_logger().info(f'received facts: {msg}')
    
        try:
            pack = json.loads(msg.data)
        except Exception:
            self.get_logger().warn("planner: bad JSON on facts")
            return

        compact = self._compact_fact_pack(pack)
        self._facts_buffer.insert(0, compact)
        self._facts_buffer = _bound_list(self._facts_buffer, self.max_packs)

        # Start a new planning cycle only on proactive broker runs
        if pack.get("mode") == "proactive":
            self._open_iter = 0
            self._waiting_for_info = False
            self._open_items = []
            self._step_planning_cycle()

    def _on_facts_delta(self, msg: StringMsg):
        self.get_logger().info("received facts delta")
        try:
            pack = json.loads(msg.data)
        except Exception:
            self.get_logger().warn("planner: bad JSON on facts_delta")
            return

        compact = self._compact_fact_pack(pack)
        self._facts_buffer.insert(0, compact)
        self._facts_buffer = _bound_list(self._facts_buffer, self.max_packs)

        # Only react if we were actually waiting for extra info
        if self._waiting_for_info:
            self._step_planning_cycle()

    def _step_planning_cycle(self):
        # 1) Plan once given current facts/capsule/profiles
        plan = self._plan_once()
        hints = plan.get("_hints") or {}
        open_items = hints.get("open_items") or []

        # Decide if we still need more info
        need_more = bool(open_items) and (self._open_iter < self.max_open_iters)

        if need_more:
            # 2) Emit needs and mark waiting
            self._open_items = open_items
            self._open_iter += 1
            self._waiting_for_info = True

            needs = self._needs_once()
            # (optionally, you can inject open_items into needs payload as a hint)
            self.pub_needs.publish(StringMsg(data=json.dumps(needs, ensure_ascii=False)))
            self.get_logger().info(f"planner: requesting more info, iter={self._open_iter}, open={open_items}")
            # DO NOT publish the plan as 'final' yet; it's provisional.
            # If you want, you can add plan["_meta"]["status"] = "draft".
            return

        # 3) We are done (no OPEN gaps or hit iteration cap)
        plan.setdefault("_meta", {})
        plan["_meta"].update({
            "status": "final" if not open_items else "final_with_gaps",
            "open_items": open_items,
            "iterations": self._open_iter
        })

        self._waiting_for_info = False
        self.pub_plan.publish(StringMsg(data=json.dumps(plan, ensure_ascii=False)))
        self.get_logger().info(f"planner: published final plan (iterations={self._open_iter}, open_items={len(open_items)}): {plan}")


    def _on_capsule(self, msg: StringMsg):
        try:
            capsule = json.loads(msg.data)
        except Exception:
            return
        # compact trace
        trace = _bound_list((capsule.get("trace") or []), self.max_trace)
        cap = dict(capsule)
        cap["trace"] = trace
        self._capsule = cap

    def _on_profiles(self, msg: StringMsg):
        try:
            profiles = json.loads(msg.data)
        except Exception:
            return
        # keep as-is; expected to be a small summary
        self._profiles = profiles

    def _on_plan_req(self, msg: StringMsg):
        _ = msg.data  # not used; presence means "plan now"
        try:
            plan = self._plan_once()
            self.pub_plan.publish(StringMsg(data=json.dumps(plan, ensure_ascii=False)))
        except Exception as e:
            self.get_logger().warn(f"plan_req failed: {e}")

    # ---------- Services ----------
    def _srv_plan(self, req, res):
        try:
            plan = self._plan_once()
            res.success = True
            res.message = json.dumps(plan, ensure_ascii=False)
            # also publish
            self.pub_plan.publish(StringMsg(data=res.message))
        except Exception as e:
            res.success = False
            res.message = f"plan error: {e}"
        return res

    def _srv_needs(self, req, res):
        try:
            needs = self._needs_once()
            res.success = True
            res.message = json.dumps(needs, ensure_ascii=False)
            self.pub_needs.publish(StringMsg(data=res.message))
        except Exception as e:
            res.success = False
            res.message = f"needs error: {e}"
        return res

    # ---------- Core ----------
    def _compact_fact_pack(self, pack: Dict[str, Any]) -> Dict[str, Any]:
        """Trim rows inside each table to max_rows to keep token usage bounded."""
        compact = {}
        for k, v in (pack or {}).items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                compact[k] = v[: self.max_rows]
            else:
                compact[k] = v
        return compact

    def _build_llm_messages_plan(self) -> list[dict]:
        payload = {
            "ContextCapsule": self._capsule or {},
            "Profiles": self._profiles or {},
            "Facts": {"packs": list(reversed(self._facts_buffer))}
        }
        return [
            {"role":"system","content": SYSTEM_PLAN},
            {"role":"user","content": json.dumps(EX_USER_PLAN, ensure_ascii=False)},
            {"role":"assistant","content": json.dumps(EX_ASSISTANT_PLAN, ensure_ascii=False)},
            {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
        ]


    def _build_llm_messages_needs(self) -> List[Dict[str, str]]:
        ex_user = {
            "ContextCapsule": {"trigger":{"type":"finish_or_fail"}},
            "Facts": {"vw_backlog_counts":[{"to_pick":3,"in_basket":1}]}
        }
        ex_assistant = {
            "needs": [
                {"why":"resolve conflicting label for CNode37","focus":"object","object_id":"CNode37","severity":"high"},
                {"why":"missing bin access info for Human A","focus":"policy","severity":"medium"}
            ]
        }
        payload = {
            "ContextCapsule": self._capsule or {},
            "Profiles": self._profiles or {},
            "Facts": {"packs": list(reversed(self._facts_buffer))}
        }
        return [
            {"role":"system","content": SYSTEM_NEEDS},
            {"role":"user","content": json.dumps(ex_user, ensure_ascii=False)},
            {"role":"assistant","content": json.dumps(ex_assistant, ensure_ascii=False)},
            {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
        ]

    def _chat_json(self, messages: List[Dict[str,str]],
                   schema: Dict[str,Any],
                   temperature: float,
                   max_tokens: int = 700,
                   retries: int = 1) -> Dict[str, Any]:
        
        self.get_logger().info("starting llm")            
       
        for _ in range(retries + 1):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "PlannerOutput",
                        "schema": schema     # ← your PLAN_SCHEMA or NEEDS_SCHEMA
                    }
                },
            )

            # DEBUG PRINT — raw prompt + raw assistant content
            try:
                self.get_logger().info("\n=== LLM PROMPT ===\n" + json.dumps(messages, indent=2))
            except Exception:
                # occasionally messages may contain non-serializable fields
                self.get_logger().info("=== LLM PROMPT (non-JSON-printable) ===")

            content = resp.choices[0].message.content

            self.get_logger().info(f"\n=== LLM RAW RESPONSE ===\n{content}\n")

            try:
                obj = json.loads(content)
                validate(instance=obj, schema=schema)
                return obj
            except (json.JSONDecodeError, ValidationError):
                messages = messages + [{"role": "system", "content": "Return ONLY valid JSON per the schema. No prose."}]
        raise ValueError("LLM did not return valid JSON for schema")

    def _plan_once(self) -> Dict[str, Any]:
        msgs = self._build_llm_messages_plan()
        obj = self._chat_json(msgs, PLAN_SCHEMA, temperature=self.temp_plan, max_tokens=900, retries=1)

        # cap narrative length a bit (LLM should comply, but belt-and-suspenders)
        doc = obj.get("plan_doc","").strip()
        words = doc.split()
        if len(words) > 230:
            doc = " ".join(words[:230]) + " …"
        obj["plan_doc"] = doc

        # quick hints for orchestrator (optional)
        obj["_hints"] = _extract_hints(doc)
        obj["_meta"]  = {"ws_id": self._ws_id, "ts": time.time(), "packs_used": len(self._facts_buffer)}
        self._ws_id += 1
        return obj


    def _needs_once(self) -> Dict[str, Any]:
        msgs = self._build_llm_messages_needs()
        obj = self._chat_json(msgs, NEEDS_SCHEMA, temperature=self.temp_needs, max_tokens=500, retries=1)
        obj["_meta"] = {"ws_id": self._ws_id, "ts": time.time()}
        return obj

    # ---------- Shutdown ----------
    def destroy_node(self):
        super().destroy_node()

# ---------- main ----------
def main():
    rclpy.init()
    node = PlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

