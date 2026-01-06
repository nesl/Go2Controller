# plan_mediator.py
#
# Flexible conversational mediator for planning conflicts.
#
# - Represents a "mediation session" around a baseline (optimizer) plan and a
#   candidate (human/LLM) plan.
# - At each step(), it calls an LLM with:
#       * objective metrics (suboptimality, risk, etc.)
#       * social / interaction context (proposer, conflict, recent dialogue)
#   and gets back:
#       * decision: pending | accept | reject
#       * robot_utterance: what the robot should say
#       * planner_action: how to change or keep the plan
#       * optional candidate_plan_delta for structured edits
#
# You can:
#   - create a session when a new candidate appears,
#   - feed human utterances + updated metrics as they arrive,
#   - keep calling step() until decision != "pending" or max_turns reached.
#
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    List,
    Dict,
    Optional,
    Literal,
    Callable,
    Any,
    Tuple,
)

from .optimizer_client import Plan  # Plan = Dict[str, List[Tuple[int, Property, str]]]


Decision = Literal["pending", "accept", "reject"]
PlannerActionKind = Literal[
    "keep_baseline",
    "adopt_candidate",
    "merge_plans",
    "request_new_plan",
]


@dataclass
class MediationTurn:
    """
    One dialogue turn in the mediation session.

    role: "human_a" | "human_b" | "robot" | "system"
    text: natural language utterance (or short description for system).
    meta: optional metadata (timestamps, tags, etc.).
    """
    role: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MediationObjectiveMetrics:
    """
    Objective metrics summarizing baseline vs candidate.

    You can extend this freely — the LLM will see it as JSON.
    """
    suboptimality_pct: Optional[float] = None
    baseline_score: Optional[float] = None
    candidate_score: Optional[float] = None
    deadline_risk: Optional[str] = None      # "low" | "medium" | "high"
    imbalance_XY: Optional[float] = None     # |totalX - totalY|
    fulfillment_history_ok: Optional[bool] = None
    notes: Optional[str] = None


@dataclass
class MediationSocialContext:
    """
    Social / team context relevant for negotiation.
    """
    proposer_id: str                          # "human_a" | "human_b" | "robot" | "llm"
    proposer_success_rate: Optional[float] = None
    conflict_index: Optional[float] = None    # e.g. 0..1 estimated conflict / disagreement
    override_frequency: Optional[str] = None  # "low" | "medium" | "high"
    leadership_contestation: Optional[str] = None  # "none" | "emerging" | "strong"
    comments: Optional[str] = None


@dataclass
class MediationInteractionContext:
    """
    Interaction-level context: summaries, recent utterances, etc.
    """
    event_summary: Optional[str] = None       # your running summary from Broker
    recent_utterances: List[MediationTurn] = field(default_factory=list)
    robot_role_description: Optional[str] = None  # short sentence: "Bob is a cooperative teammate..."
    session_notes: Optional[str] = None


@dataclass
class MediationLLMConfig:
    """
    Configuration for the LLM call used by the mediator.

    llm_call: a function that will be used to call the LLM and must return a
              dict already validated against the schema you use.
              Signature:
                llm_call(messages: List[dict]) -> dict
    """
    model_name: str
    temperature: float = 0.3
    max_turns: int = 4
    llm_call: Optional[Callable[[List[dict]], Dict[str, Any]]] = None


@dataclass
class MediationState:
    """
    Complete state of a mediation session.
    """
    session_id: str
    baseline_plan: Plan
    candidate_plan: Plan
    objective: MediationObjectiveMetrics
    social: MediationSocialContext
    interaction: MediationInteractionContext

    status: Decision = "pending"
    turns: List[MediationTurn] = field(default_factory=list)
    turns_used: int = 0


class PlanMediator:
    """
    Core mediator object. Stateless across sessions; state is in MediationState.
    """

    def __init__(self, config: MediationLLMConfig):
        if config.llm_call is None:
            raise ValueError("MediationLLMConfig.llm_call must be provided")
        self.config = config

    # ---- Public API -----------------------------------------------------

    def step(
        self,
        state: MediationState,
        new_human_turn: Optional[MediationTurn] = None,
    ) -> Tuple[MediationState, Dict[str, Any]]:
        """
        Perform ONE mediation step.

        - Optionally add a new human utterance (e.g., from speech input).
        - Build an LLM prompt using:
            * baseline & candidate plans (summarized),
            * objective metrics,
            * social context,
            * recent dialogue turns.
        - Call LLM via self.config.llm_call.
        - Update state (status, turns, turns_used, candidate_plan if changed).
        - Return (new_state, llm_raw_output).

        The LLM is expected to output JSON like:

            {
              "decision": "pending | accept | reject",
              "planner_action": {
                "kind": "keep_baseline | adopt_candidate | merge_plans | request_new_plan",
                "candidate_plan_delta": {
                  "robot":   [{"box_id": 3, "property": "X", "kind": "sense"}, ...],
                  "human_a": [...],
                  "human_b": [...]
                }
              },
              "robot_utterance": "natural language to say to the team",
              "log_tags": {
                "strategy": "persuade | concede | negotiate | assert",
                "rationale": "one short sentence explanation"
              }
            }
        """
        if state.status != "pending":
            # Session already resolved; nothing to do.
            return state, {"skipped": True, "reason": "status_not_pending"}

        if state.turns_used >= self.config.max_turns:
            # Graceful fail-safe: we stop the conversation and keep baseline.
            state.status = "reject"
            return state, {
                "decision": "reject",
                "reason": "max_turns_reached",
                "planner_action": {"kind": "keep_baseline"},
            }

        # Incorporate new human turn if provided
        if new_human_turn is not None:
            state.turns.append(new_human_turn)
            state.interaction.recent_utterances.append(new_human_turn)

        # Build LLM messages
        messages = self._build_messages_for_llm(state)

        # Call LLM
        raw = self.config.llm_call(messages)

        # Interpret LLM response
        decision: Decision = raw.get("decision", "pending")
        planner_action = raw.get("planner_action") or {}
        robot_utt = (raw.get("robot_utterance") or "").strip()
        log_tags = raw.get("log_tags") or {}

        # Append robot turn to dialogue history (if any)
        if robot_utt:
            robot_turn = MediationTurn(role="robot", text=robot_utt, meta={"log_tags": log_tags})
            state.turns.append(robot_turn)
            state.interaction.recent_utterances.append(robot_turn)

        # Apply plan deltas if provided
        candidate_delta = planner_action.get("candidate_plan_delta")
        if candidate_delta:
            state.candidate_plan = self._apply_candidate_delta(state.candidate_plan, candidate_delta)

        # Update status + turn count
        state.turns_used += 1
        if decision in ("accept", "reject"):
            state.status = decision

        return state, raw

    # ---- Internal helpers -----------------------------------------------

    def _build_messages_for_llm(self, state: MediationState) -> List[Dict[str, Any]]:
        """
        Build a compact prompt exposing:
          - baseline vs candidate plans (summarized),
          - objective metrics,
          - social / interaction context,
          - recent dialogue history.
        """
        # Summarize plans as short, inspectable structures suited for a prompt.
        baseline_summary = self._summarize_plan(state.baseline_plan)
        candidate_summary = self._summarize_plan(state.candidate_plan)

        # Prepare a compact dialogue transcript (last N turns)
        transcript = [
            {"role": t.role, "text": t.text, "meta": t.meta}
            for t in state.turns[-8:]
        ]

        system_msg = {
            "role": "system",
            "content": (
                "You are Bob's planning mediator in a mixed human-robot team.\n"
                "- You see an OPTIMIZER baseline plan, a CANDIDATE plan proposed by a human or LLM,\n"
                "  and objective metrics about both. You also see recent dialogue and social context.\n"
                "- Your job is to decide whether Bob should ACCEPT, REJECT, or CONTINUE NEGOTIATING\n"
                "  about the candidate plan, and to propose what Bob should SAY in one short utterance.\n"
                "- Bob's goals are:\n"
                "    1) Keep the team safe and task-effective (use objective metrics!),\n"
                "    2) Maintain good collaboration and trust with humans,\n"
                "    3) Avoid endless arguing; be decisive when needed.\n"
                "- If the context (objective_metrics.notes or interaction.session_notes) indicates that some\n"
                "  requested actions are impossible (deadline already passed, or action already fulfilled),\n"
                "  you MUST explicitly mention this to the humans and briefly say which actions cannot\n"
                "  be executed and why.\n"
                "- Be concise, cooperative, and honest about trade-offs.\n"
            ),
        }

        user_payload = {
            "session_id": state.session_id,
            "baseline_plan": baseline_summary,
            "candidate_plan": candidate_summary,
            "objective_metrics": state.objective.__dict__,
            "social_context": state.social.__dict__,
            "interaction_context": {
                "event_summary": state.interaction.event_summary,
                "robot_role_description": state.interaction.robot_role_description,
                "session_notes": state.interaction.session_notes,
                "recent_dialogue": transcript,
            },
        }

        user_msg = {
            "role": "user",
            "content": (
                "Here is the current mediation context.\n"
                "Return STRICT JSON with keys:\n"
                '{\n'
                '  "decision": "pending | accept | reject",\n'
                '  "planner_action": {\n'
                '    "kind": "keep_baseline | adopt_candidate | merge_plans | request_new_plan",\n'
                '    "candidate_plan_delta": { ... optional ... }\n'
                '  },\n'
                '  "robot_utterance": "one short sentence Bob will say to the team",\n'
                '  "log_tags": {\n'
                '    "strategy": "persuade | concede | negotiate | assert | other",\n'
                '    "rationale": "very short explanation (<= 1 sentence)"\n'
                '  }\n'
                '}\n'
            ),
        }

        # Add payload as a second user message to keep structure clean
        user_payload_msg = {
            "role": "user",
            "content": self._to_json_str(user_payload),
        }

        return [system_msg, user_msg, user_payload_msg]

    def _summarize_plan(self, plan: Plan) -> Dict[str, Any]:
        """
        Convert internal Plan into a summary better suited for LLM prompts.
        """
        summary: Dict[str, Any] = {}
        for aid, actions in (plan or {}).items():
            out_actions = []
            for (box_id, prop, kind) in actions:
                out_actions.append(
                    {
                        "box_id": int(box_id),
                        "property": prop,
                        "kind": kind,
                    }
                )
            summary[aid] = out_actions
        return summary

    def _apply_candidate_delta(self, current: Plan, delta: Dict[str, Any]) -> Plan:
        """
        Apply a delta from the LLM to the candidate plan.

        For now: if delta[aid] is present, it REPLACES the entire list
        for that agent. This is simple and predictable. You can extend
        it later with operations like 'append', 'remove', etc.
        """
        new_plan: Plan = dict(current) if current is not None else {}

        for aid, actions in (delta or {}).items():
            if not isinstance(actions, list):
                continue
            new_actions = []
            for step in actions:
                if not isinstance(step, dict):
                    continue
                box_id = step.get("box_id")
                prop = step.get("property")
                kind = step.get("kind")
                if prop not in ("X", "Y"):
                    continue
                if kind not in ("sense", "dispose"):
                    continue
                try:
                    box_id_int = int(box_id)
                except Exception:
                    continue
                new_actions.append((box_id_int, prop, kind))
            new_plan[aid] = new_actions

        return new_plan

    @staticmethod
    def _to_json_str(obj: Any) -> str:
        import json
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

