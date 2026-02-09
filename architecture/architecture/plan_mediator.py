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



def _level_3(x: Optional[float]) -> Optional[str]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if v <= 0.33:
        return "low"
    if v <= 0.66:
        return "medium"
    return "high"

def _pct_to_level(pct: Optional[float]) -> Optional[str]:
    if pct is None:
        return None
    try:
        v = float(pct)
    except Exception:
        return None
    # tune freely:
    if v <= 5.0:
        return "very_low"
    if v <= 15.0:
        return "low"
    if v <= 35.0:
        return "medium"
    if v <= 60.0:
        return "high"
    return "very_high"

def _rate_to_level(r: Optional[float]) -> Optional[str]:
    if r is None:
        return None
    try:
        v = float(r)
    except Exception:
        return None
    if v >= 0.85:
        return "very_high"
    if v >= 0.70:
        return "high"
    if v >= 0.50:
        return "medium"
    if v >= 0.30:
        return "low"
    return "very_low"



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
    suboptimality_pct: float
    baseline_score: Optional[float] = None
    candidate_score: Optional[float] = None
    deadline_risk: Any = None   # you already sometimes store strings like "low"
    imbalance_XY: Optional[float] = None
    fulfillment_history_ok: bool = True
    notes: Optional[str] = None

    def for_llm(self) -> Dict[str, Any]:
        """
        Qualitative-only view for the LLM prompt (no raw numbers).
        """
        # deadline_risk may already be "low"/"medium"/"high"
        dr = self.deadline_risk
        if isinstance(dr, (int, float)):
            # if it’s numeric in some runs, bucket it
            dr = _level_3(float(dr))

        return {
            "suboptimality_level": _pct_to_level(self.suboptimality_pct),
            "deadline_risk": dr,
            "imbalance_level": _level_3(self.imbalance_XY),
            "fulfillment_history_ok": bool(self.fulfillment_history_ok),
            "warnings": self.notes,
        }


@dataclass
class MediationSocialContext:
    proposer_id: str
    proposer_success_rate: Optional[float] = None
    conflict_index: Optional[float] = None
    override_frequency: Optional[str] = None  # already low/medium/high in your code
    leadership_contestation: Optional[str] = None  # none/emerging/strong
    comments: Optional[str] = None

    def for_llm(self) -> Dict[str, Any]:
        """
        Qualitative-only view for the LLM prompt (no raw numbers).
        """
        return {
            "proposer_id": self.proposer_id,
            "proposer_success_level": _rate_to_level(self.proposer_success_rate),
            "conflict_level": _level_3(self.conflict_index),
            "override_frequency": self.override_frequency,
            "leadership_contestation": self.leadership_contestation,
            "comments": self.comments,
        }


@dataclass
class MediationInteractionContext:
    """
    Interaction-level context: summaries, recent utterances, etc.
    """
    event_summary: Optional[str] = None       # your running summary from Broker
    recent_utterances: List[MediationTurn] = field(default_factory=list)
    robot_role_description: Optional[str] = None  # short sentence: "Bob is a cooperative teammate..."
    session_notes: Optional[str] = None
    human_profiles: Optional[Dict[str, Any]] = None


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
    include_baseline_proposer: bool = True

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
    prefix_plan: Optional[Plan] = None
    status: Decision = "pending"
    turns: List[MediationTurn] = field(default_factory=list)
    turns_used: int = 0
    baseline_provenance: Optional[Dict[str, List[Dict[str, Any]]]] = None
    human_ids: List[str] = field(default_factory=list)


    
class PlanMediator:
    """
    Core mediator object. Stateless across sessions; state is in MediationState.
    """

    def __init__(self, config: MediationLLMConfig):
        self.capabilities =  {
          "Sam": {"sense": ["X"], "dispose": ["X"]},
          "Jacob": {"sense": ["Y"], "dispose": ["Y"]},
          "robot": {"sense": ["X","Y"], "dispose": ["X","Y"]}
        }

    
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

        '''
        if state.turns_used >= self.config.max_turns:
            # Graceful fail-safe: we stop the conversation and keep baseline.
            state.status = "reject"
            return state, {
                "decision": "reject",
                "reason": "max_turns_reached",
                "planner_action": {"kind": "keep_baseline"},
            }
        '''
        # Incorporate new human turn if provided
        if new_human_turn is not None:
            state.turns.append(new_human_turn)
            state.interaction.recent_utterances.append(new_human_turn)

        # Build LLM messages
        messages = self._build_messages_for_llm(state)

        # Call LLM
        raw = self.config.llm_call(messages)

        # Interpret LLM response
        planner_action = raw.get("planner_action") or {}
        robot_utt = (raw.get("robot_utterance") or "").strip()
        log_tags = raw.get("log_tags") or {}
        kind_raw = (planner_action.get("kind") or "").strip()
        decision_raw = (raw.get("decision") or "").strip()

        # --- Normalize decision ---
        # Allowed canonical decisions
        canonical_decisions = {"pending", "accept", "reject"}
        # Planner action kinds that imply "accept" if used as decision
        kind_values = {
            "keep_baseline",
            "adopt_candidate",
            "merge_plans",
            "request_new_plan",
        }

        if decision_raw in canonical_decisions:
            decision: Decision = decision_raw  # type: ignore[assignment]
        elif decision_raw in kind_values:
            # e.g. decision="merge_plans" → treat as accept
            decision = "accept"
        elif not decision_raw and kind_raw in kind_values:
            # If model forgets 'decision' but sets a kind, also treat as accept
            decision = "accept"
        else:
            # Fallback: stay in the conversation
            decision = "pending"

        # Append robot turn to dialogue history (if any)
        if robot_utt:
            robot_turn = MediationTurn(role="robot", text=robot_utt, meta={"log_tags": log_tags})
            state.turns.append(robot_turn)
            state.interaction.recent_utterances.append(robot_turn)

        # Apply plan deltas if provided
        candidate_delta = planner_action.get("candidate_plan_delta")
        if candidate_delta:
            state.candidate_plan = self._apply_candidate_delta(
                state=state,
                current=state.candidate_plan,
                delta=candidate_delta,
            )


        # Update status + turn count
        state.turns_used += 1
        if decision in ("accept", "reject"):
            state.status = decision

        return state, raw

    # ---- Internal helpers -----------------------------------------------
    '''
    def _build_planning_view(self, state: MediationState) -> Dict[str, Any]:
        """
        Build a structured view for the LLM:
          - baseline_human_agreed_actions
          - baseline_optimizer_suggestions
          - human_proposed_changes
          - optimizer_suggestions_for_changes
        """
        baseline = self._summarize_plan(state.baseline_plan)
        candidate = self._summarize_candidate_with_sources(
            state.candidate_plan,
            state.prefix_plan,
        )
        provenance = state.baseline_provenance or {}

        # ---- OPTIONAL OVERRIDE: human-agreed baseline carryover ----
        # Broker can attach a pruned agreed baseline that should be treated as "human agreed"
        # even if the optimizer's current baseline_plan no longer contains those actions.
        agreed_override_plan = getattr(state, "baseline_human_agreed_override", None)
        agreed_override_prov = getattr(state, "baseline_human_agreed_override_provenance", None)

        if isinstance(agreed_override_plan, dict) and agreed_override_plan:
            # Replace what we consider "baseline human agreed" source with the override
            baseline_human_source = self._summarize_plan(agreed_override_plan)
            provenance_human_source = agreed_override_prov if isinstance(agreed_override_prov, dict) else {}
        else:
            baseline_human_source = baseline  # default: current baseline plan
            provenance_human_source = provenance  # default: current provenance


        # --- Split baseline into human vs optimizer origins ---
        baseline_human_agreed = {}
        baseline_opt_suggestions = {}
        include_proposer = getattr(self.config, "include_baseline_proposer", True)

        for aid, actions in (baseline_human_source or {}).items():
            prov_actions = (provenance_human_source or {}).get(aid) or []

            # (box, prop, kind) -> full provenance dict
            origin_map = {
                (int(p["box_id"]), p["property"], p["kind"]): p
                for p in prov_actions
                if "box_id" in p and "property" in p and "kind" in p
            }

            human_list = []
            opt_list = []
            for a in actions:
                key = (int(a["box_id"]), a["property"], a["kind"])
                pinfo = origin_map.get(key) or {}
                origin = (pinfo.get("origin") or "optimizer").strip()
                proposed_by = pinfo.get("proposed_by")

                is_human_agreed = (origin == "human") or bool(proposed_by)

                if is_human_agreed:
                    if include_proposer and proposed_by:
                        a_with_meta = dict(a)
                        a_with_meta["original_proposer"] = proposed_by
                        human_list.append(a_with_meta)
                    else:
                        human_list.append(a)
                else:
                    opt_list.append(a)


            if human_list:
                baseline_human_agreed[aid] = human_list
            if opt_list:
                baseline_opt_suggestions[aid] = opt_list

        # --- Human-proposed changes in this session (from prefix_plan) ---
        human_proposed_changes = self._summarize_plan(state.prefix_plan or {})

        # --- Optimizer suggestions tied to those changes ---
        optimizer_suggestions_for_changes = {}
        for aid, actions in (candidate or {}).items():
            sugg = [
                a for a in actions
                if a.get("source") == "optimizer_completion"
            ]
            if sugg:
                optimizer_suggestions_for_changes[aid] = sugg


        planning_view = {
            "baseline_human_agreed_actions": baseline_human_agreed,
            "baseline_optimizer_suggestions": baseline_opt_suggestions,
            "human_proposed_changes": human_proposed_changes,
            "optimizer_suggestions_for_changes": optimizer_suggestions_for_changes,
        }
        proposer = state.social.proposer_id


        # --- NEW: detect conflicts ONLY w.r.t. human-agreed actions ---
        conflicts = self._detect_human_conflicts(
            planning_view=planning_view,
            proposer_id=proposer,
        )
        if conflicts:
            planning_view["direct_conflicts_with_other_human"] = conflicts

        return planning_view

    '''
    
    def _summarize_committed_plan_with_proposer(
        self,
        plan: Plan,
        provenance: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        default_proposer_id: str = "optimizer",
    ) -> Dict[str, Any]:
        """
        Summarize the committed plan for the LLM, adding proposer_id per action
        using baseline_provenance when available.

        baseline_provenance format you already use:
          provenance[agent_id] = [
            {"box_id": 2, "property": "Y", "kind": "sense", "origin": "...", "proposed_by": "..."},
            ...
          ]
        """
        prov = provenance or {}
        summary: Dict[str, Any] = {}

        for aid, actions in (plan or {}).items():
            prov_actions = prov.get(aid) or []

            # (box_id, property, kind) -> proposer_id
            proposer_map: Dict[Tuple[int, str, str], str] = {}
            for p in prov_actions:
                try:
                    k = (int(p.get("box_id")), p.get("property"), p.get("kind"))
                except Exception:
                    continue
                if k[1] not in ("X", "Y") or k[2] not in ("sense", "dispose"):
                    continue

                proposed_by = p.get("proposed_by")
                origin = (p.get("origin") or "").strip()

                # Prefer explicit proposer; otherwise bucket to optimizer/system
                proposer_id = proposed_by or (default_proposer_id if origin != "human" else "human")
                proposer_map[k] = str(proposer_id)

            out_actions = []
            for (box_id, prop, kind) in actions:
                k = (int(box_id), prop, kind)
                out_actions.append(
                    {
                        "box_id": int(box_id),
                        "property": prop,
                        "kind": kind,
                        "proposer_id": proposer_map.get(k, default_proposer_id),
                    }
                )

            summary[aid] = out_actions

        return summary

    
    def _build_planning_view(self, state: MediationState) -> Dict[str, Any]:
        """
        New planning view (committed-plan based):

          - committed_plan = what the robot/team is currently committed to do
          - human_proposed_changes = explicit tasks requested by the CURRENT proposer (prefix_plan)
          - optimizer_suggestions_for_changes = extra tasks optimizer added when completing the prefix into candidate

        We intentionally remove baseline_human_agreed_actions and baseline_optimizer_suggestions.
        """

        # ✅ committed plan is the single baseline
        committed_plan = self._summarize_committed_plan_with_proposer(
            state.baseline_plan,
            provenance=state.baseline_provenance,
            default_proposer_id="robot",
        )

        # ✅ explicit human request in this session
        human_proposed_changes = self._summarize_plan(state.prefix_plan or {})

        # ✅ optimizer completion relative to prefix
        candidate = self._summarize_candidate_with_sources(
            state.candidate_plan,
            state.prefix_plan,
        )

        MAX_OPT_SUGG_PER_AGENT = 2

        optimizer_suggestions_for_changes: Dict[str, Any] = {}
        for aid, actions in (candidate or {}).items():
            sugg = [
                {k: v for k, v in a.items() if k != "source"}
                for a in actions
                if a.get("source") == "optimizer_completion"
            ]
            if sugg:
                #optimizer_suggestions_for_changes[aid] = sugg
                optimizer_suggestions_for_changes[aid] = sugg[:MAX_OPT_SUGG_PER_AGENT]

        planning_view = {
            "committed_plan": committed_plan,
            "human_proposed_changes": human_proposed_changes,
            "optimizer_suggestions_for_changes": optimizer_suggestions_for_changes,
        }

        # Conflicts are now defined w.r.t. committed_plan (not "human agreed")
        proposer = state.social.proposer_id
        conflicts = self._detect_committed_conflicts(planning_view=planning_view, proposer_id=proposer)
        if conflicts:
            planning_view["direct_conflicts_with_committed_plan"] = conflicts

        if not planning_view["human_proposed_changes"]:
            planning_view["human_proposed_changes"] = "keep committed plan as it is, no changes"

        return planning_view

    def _detect_committed_conflicts(
        self,
        planning_view: Dict[str, Any],
        proposer_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Conflict if CURRENT proposer's requested robot action contradicts the committed plan's robot action.

        Conflict condition we care about:
          - same box_id on robot
          - but property differs OR kind differs
        """
        conflicts: List[Dict[str, Any]] = []

        committed = planning_view.get("committed_plan") or {}
        changes = planning_view.get("human_proposed_changes") or {}

        committed_robot = committed.get("robot") or []
        change_robot = changes.get("robot") or []

        by_box: Dict[int, List[Dict[str, Any]]] = {}
        for base in committed_robot:
            try:
                b_box = int(base.get("box_id"))
            except Exception:
                continue
            by_box.setdefault(b_box, []).append(base)

        for change in change_robot:
            try:
                c_box = int(change.get("box_id"))
            except Exception:
                continue

            c_prop = change.get("property")
            c_kind = change.get("kind")
            if c_prop not in ("X", "Y") or c_kind not in ("sense", "dispose"):
                continue

            for base in by_box.get(c_box, []):
                b_prop = base.get("property")
                b_kind = base.get("kind")

                if b_prop != c_prop or b_kind != c_kind:
                    conflicts.append(
                        {
                            "type": "override_committed_action",
                            "previous_action": {
                                "agent": "robot",
                                "box_id": c_box,
                                "property": b_prop,
                                "kind": b_kind,
                            },
                            "new_action": {
                                "agent": "robot",
                                "box_id": c_box,
                                "property": c_prop,
                                "kind": c_kind,
                                "proposer_id": proposer_id,
                            },
                            "reason": (
                                f"Proposer requests robot to {c_kind} box {c_box} ({c_prop}), "
                                f"but committed plan has robot {b_kind} box {c_box} ({b_prop})."
                            ),
                        }
                    )

        return conflicts

    
    def _detect_human_conflicts(
        self,
        planning_view: Dict[str, Any],
        proposer_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts where the current proposer's changes contradict past
        HUMAN-AGREED actions originally proposed by a different human.

        Primary conflict we care about (matches your log):
          - same agent (robot)
          - same box_id
          - (property differs OR kind differs)

        Example:
          baseline: robot sense box 2 (Y), original_proposer=Jacob
          change:   robot sense box 2 (X)  proposer=Sam
        """
        conflicts: List[Dict[str, Any]] = []

        baseline_human = planning_view.get("baseline_human_agreed_actions") or {}
        human_changes = planning_view.get("human_proposed_changes") or {}

        baseline_robot = baseline_human.get("robot") or []
        change_robot = human_changes.get("robot") or []

        # Index baseline human-agreed robot actions by box_id for quick match
        by_box: Dict[int, List[Dict[str, Any]]] = {}
        for base in baseline_robot:
            try:
                b_box = int(base.get("box_id"))
            except Exception:
                continue
            by_box.setdefault(b_box, []).append(base)

        for change in change_robot:
            try:
                c_box = int(change.get("box_id"))
            except Exception:
                continue

            c_prop = change.get("property")
            c_kind = change.get("kind")

            if c_prop not in ("X", "Y") or c_kind not in ("sense", "dispose"):
                continue

            for base in by_box.get(c_box, []):
                b_prop = base.get("property")
                b_kind = base.get("kind")
                orig = base.get("original_proposer")

                # Only treat as social conflict if baseline action was human-origin
                # and proposed by someone other than the current proposer.
                if not orig or orig == proposer_id:
                    continue

                # Conflict condition: same box, but different task (prop/kind differs)
                if b_prop != c_prop or b_kind != c_kind:
                    conflicts.append(
                        {
                            "type": "override_baseline_human_action",
                            "previous_action": {
                                "agent": "robot",
                                "box_id": c_box,
                                "property": b_prop,
                                "kind": b_kind,
                                "original_proposer": orig,
                            },
                            "new_action": {
                                "agent": "robot",
                                "box_id": c_box,
                                "property": c_prop,
                                "kind": c_kind,
                                "proposer_id": proposer_id,
                            },
                            "reason": (
                                f"Current proposer requests robot to {c_kind} box {c_box} ({c_prop}), "
                                f"but baseline human-agreed robot action was {b_kind} box {c_box} ({b_prop}) "
                                f"from {orig}."
                            ),
                            "previous_proposer": orig,
                            "current_proposer": proposer_id,
                        }
                    )

        return conflicts



    def build_messages_for_autoresolve(
        self,
        state: MediationState,
        reason: str,
    ) -> List[Dict[str, Any]]:
        """
        Build messages for a timeout-based auto-resolve.
        Intentionally almost identical to _build_messages_for_llm, with ONLY:
          - phase forced to "autoresolve"
          - negotiation_suffix replaced with a finalize-now suffix
          - payload includes autoresolve_reason
        """
        planning_view = self._build_planning_view(state)

        # Recent dialogue (same)
        source_turns = state.interaction.recent_utterances or []
        N = 12
        tail = source_turns[-N:] if N and N > 0 else source_turns
        transcript = []
        for idx, t in enumerate(tail):
            ts_val = None
            try:
                if isinstance(t.meta, dict):
                    ts_val = t.meta.get("ts")
            except Exception:
                ts_val = None
            transcript.append(
                {
                    "role": t.role,
                    "text": t.text,
                    "ts": ts_val,
                    "is_last": (idx == len(tail) - 1),
                }
            )

        # --- DIFF #1: phase is forced ---
        phase = "autoresolve"

        # --- DIFF #2: negotiation suffix becomes finalize suffix ---
        finalize_suffix = (
            "\n- AUTO-RESOLVE mode: humans did not respond in time.\n"
            "- You MUST end this step with decision=\"accept\" or decision=\"reject\" (NOT pending).\n"
            "- Prefer a compromise: if there is a direct conflict, choose a middle-ground plan "
            "or a minimal safe merge that reduces conflict.\n"
            "- If unsure, keep_baseline.\n"
        )

        '''
        # System prompt: identical body + only this suffix change
        system_msg = {
            "role": "system",
            "content": (
                "You are Bob's planning mediator in a mixed human-robot team. Bob is the robot.\n"
                
                "- You receive a planning view with four groups:\n"
                "  - baseline_human_agreed_actions = tasks previously proposed by humans and already adopted\n"
                "  - baseline_optimizer_suggestions = tasks the optimizer added earlier (not explicitly agreed)\n"
                "  - human_proposed_changes = new tasks requested by the CURRENT proposer\n"
                "  - optimizer_suggestions_for_changes = extra optimizer tasks supporting those changes (suggestions only)\n"
                "- Treat human_proposed_changes as explicit wishes of the proposer.\n"
                "- Treat both optimizer_* groups as robot suggestions that still require human agreement.\n"
                "- Social conflict ONLY applies to changes that contradict baseline_human_agreed_actions "
                "(especially where another human originally proposed them). Edits that only affect "
                "optimizer_suggestions are NOT social conflict.\n"
                "- When building planner_action.candidate_plan_delta you are NOT limited to only the latest human change.\n"
                "  You may:\n"
                "    - adopt the human_proposed_changes exactly, OR\n"
                "    - keep the baseline plan unchanged, OR\n"
                "    - combine human_proposed_changes with some optimizer_* suggestions so that humans and the robot\n"
                "      cover more useful boxes (e.g., the human senses their requested box while the robot takes an\n"
                "      optimizer-suggested box).\n"
                "- Prefer such combined plans when they clearly improve safety/coverage/balance according to "
                "objective_metrics (e.g., better deadlines, less imbalance) and do NOT create new social conflict.\n"
                "- candidate_plan_delta should reflect the full recommended plan after this step (for all relevant agents), "
                "not just the single last human request.\n"
                "\n"
                "- Use objective_metrics to modulate how assertive you are:\n"
                "    - If deadline_risk is medium/high or suboptimality_pct is high/very_high, be more willing to suggest extra\n"
                "      optimizer tasks that improve safety, while still framing them as options for humans.\n"
                "    - If metrics look already good and there is no urgency, prefer minimal changes and lighter suggestions.\n"
                "\n"
                "- Never speak as if suggested tasks are already agreed; use wording like "
                "\"If you agree, one option is...\".\n"
                "- If notes indicate impossible actions (expired, already fulfilled), explicitly mention them.\n"
                f"{finalize_suffix}"
            ),
        }
        '''
        system_msg = {
            "role": "system",
            "content": (
                "You are Bob's planning mediator in a mixed human-robot team. Bob is the robot.\n"
                "- You receive a planning view with three groups:\n"
                "  - committed_plan = tasks the team/robot is currently committed to execute or executing at the moment\n"
                "  - human_proposed_changes = explicit tasks requested by the CURRENT proposer\n"
                "  - optimizer_suggestions_for_changes = extra optimizer tasks supporting those changes (suggestions only)\n"
                "- Treat human_proposed_changes as explicit wishes of the proposer.\n"
                "- Treat optimizer_suggestions_for_changes as robot suggestions that still require human agreement.\n"
                "- Conflict ONLY applies when human_proposed_changes contradict committed_plan.\n"
                "\n"
                "- When building planner_action.candidate_plan_delta you are NOT limited to only the latest human change.\n"
                "  You may:\n"
                "    - adopt the human_proposed_changes exactly, OR\n"
                "    - keep the committed_plan unchanged, OR\n"
                "    - combine human_proposed_changes with some optimizer_suggestions_for_changes so that humans and the robot\n"
                "      cover more useful boxes.\n"
                "- Prefer such combined plans when they clearly improve safety/coverage/balance according to objective_metrics "
                "and do NOT create new conflict with committed_plan.\n"
                "- candidate_plan_delta should reflect the full recommended plan after this step (for all relevant agents).\n"
                "\n"
                "- Never speak as if suggested tasks are already agreed; use wording like "
                "\"If you agree, one option is…\".\n"
                "- If notes indicate impossible actions (expired, already fulfilled), explicitly mention them.\n"
                "- Always be explicit on who has to perform what.\n"
                f"{finalize_suffix}"
            ),
        }


        # User payload: identical + only reason field
        user_payload = {
            "capabilities": self.capabilities,
            "planning_view": planning_view,
            "phase": phase,
            "autoresolve_reason": reason,  # --- DIFF #3 ---
            "objective_metrics": state.objective.for_llm(),
            "social_context": state.social.for_llm(),
            "interaction_context": {
                "event_summary": state.interaction.event_summary,
                "robot_role_description": state.interaction.robot_role_description,
                "session_notes": state.interaction.session_notes,
                "recent_dialogue": transcript,
                "human_profiles": state.interaction.human_profiles or {},
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

        user_payload_msg = {
            "role": "user",
            "content": self._to_json_str(user_payload),
        }

        return [system_msg, user_msg, user_payload_msg]


    def _build_messages_for_llm(self, state: MediationState) -> List[Dict[str, Any]]:
        # Build planning view
        planning_view = self._build_planning_view(state)

        # Recent dialogue as before (with ts + is_last)
        source_turns = state.interaction.recent_utterances or []
        N = 12
        tail = source_turns[-N:] if N and N > 0 else source_turns
        transcript = []
        for idx, t in enumerate(tail):
            ts_val = None
            try:
                if isinstance(t.meta, dict):
                    ts_val = t.meta.get("ts")
            except Exception:
                ts_val = None
            transcript.append(
                {
                    "role": t.role,
                    "text": t.text,
                    "ts": ts_val,
                    "is_last": (idx == len(tail) - 1),
                }
            )

        phase = "initial" if state.turns_used == 0 else "negotiation"

        negotiation_suffix = ""
        if phase == "negotiation":
            negotiation_suffix = (
                "\n- Ongoing negotiation mode: prefer decision=\"pending\" when humans disagree "
                "or when changes affect another human’s agreed actions. In pending mode, acknowledge "
                "both preferences, mention the key trade-off, and invite them to confirm a shared choice. "
                "Use ACCEPT only when there is clear consensus; use REJECT only for unsafe or impossible actions.\n"

            )


        '''
        system_msg = {
            "role": "system",
            "content": (
                "You are Bob's planning mediator in a mixed human-robot team.\n"
                "- You receive a planning view with four groups:\n"
                "  - baseline_human_agreed_actions = tasks previously proposed by humans and already adopted\n"
                "  - baseline_optimizer_suggestions = tasks the optimizer added earlier (not explicitly agreed)\n"
                "  - human_proposed_changes = new tasks requested by the CURRENT proposer\n"
                "  - optimizer_suggestions_for_changes = extra optimizer tasks supporting those changes (suggestions only)\n"
                "- Treat human_proposed_changes as explicit wishes of the proposer.\n"
                "- Treat both optimizer_* groups as robot suggestions that still require human agreement.\n"
                "- Social conflict ONLY applies to changes that contradict baseline_human_agreed_actions "
                "(especially where another human originally proposed them). Edits that only affect "
                "optimizer_suggestions are NOT social conflict.\n"
                "- If direct_conflicts_with_other_human is present, prefer CONTINUE NEGOTIATING and prompt for consensus, "
                "rather than immediately accepting one side.\n"
                "- Your output decides: ACCEPT | REJECT | PENDING, and provides one short utterance Bob will say.\n"
                "- Never speak as if suggested tasks are already agreed; use wording like "
                "\"If you agree, one option is…\".\n"
                "- If notes indicate impossible actions (expired, already fulfilled), explicitly mention them.\n"
                f"{negotiation_suffix}"
            ),
        }
        '''
        '''
        system_msg = {
            "role": "system",
            "content": (
                "You are Bob's planning mediator in a mixed human-robot team.\n"
                "- You receive a planning view with four groups:\n"
                "  - baseline_human_agreed_actions = tasks previously proposed by humans and already adopted\n"
                "  - baseline_optimizer_suggestions = tasks the optimizer added earlier (not explicitly agreed)\n"
                "  - human_proposed_changes = new tasks requested by the CURRENT proposer\n"
                "  - optimizer_suggestions_for_changes = extra optimizer tasks supporting those changes (suggestions only)\n"
                "- Treat human_proposed_changes as explicit wishes of the proposer.\n"
                "- Treat both optimizer_* groups as robot suggestions that still require human agreement.\n"
                "- Social conflict ONLY applies to changes that contradict baseline_human_agreed_actions "
                "(especially where another human originally proposed them). Edits that only affect "
                "optimizer_suggestions are NOT social conflict.\n"
                "- If direct_conflicts_with_other_human is present, prefer CONTINUE NEGOTIATING and prompt for consensus, "
                "rather than immediately accepting one side.\n"
                "\n"
                "- When building planner_action.candidate_plan_delta you are NOT limited to only the latest human change.\n"
                "  You may:\n"
                "    - adopt the human_proposed_changes exactly, OR\n"
                "    - keep the baseline plan unchanged, OR\n"
                "    - combine human_proposed_changes with some optimizer_* suggestions so that humans and the robot\n"
                "      cover more useful boxes (e.g., the human senses their requested box while the robot takes an\n"
                "      optimizer-suggested box).\n"
                "- Prefer such combined plans when they clearly improve safety/coverage/balance according to "
                "objective_metrics (e.g., better deadlines, less imbalance) and do NOT create new social conflict.\n"
                "- candidate_plan_delta should reflect the full recommended plan after this step (for all relevant agents), "
                "not just the single last human request.\n"
                "\n"
                "- Use objective_metrics to modulate how assertive you are:\n"
                "    - If deadline_risk is medium/high or suboptimality_pct is high/very_high, be more willing to suggest extra\n"
                "      optimizer tasks that improve safety, while still framing them as options for humans.\n"
                "    - If metrics look already good and there is no urgency, prefer minimal changes and lighter suggestions.\n"
                "\n"
                "- Your output decides: ACCEPT | REJECT | PENDING, and provides one short utterance Bob will say.\n"
                "- Never speak as if suggested tasks are already agreed; use wording like "
                "\"If you agree, one option is…\".\n"
                "- If notes indicate impossible actions (expired, already fulfilled), explicitly mention them.\n"
                "- Mediation is always grounded in concrete actions.\n"
                "- In robot_utterance, you MUST explicitly mention at least one concrete task from the proposed plan (agent, box_id, property, kind).\n"
                "- Avoid abstract phrases like 'some optimizer suggestions' or 'improving coverage' unless tied to specific actions.\n"
                "- Always be explicit on who has to perform what.\n"
                f"{negotiation_suffix}"
            ),
        }
        '''
        system_msg = {
            "role": "system",
            "content": (
                "You are Bob's planning mediator in a mixed human-robot team.\n"
                "- You receive a planning view with three groups:\n"
                "  - committed_plan = tasks the team/robot is currently committed to execute or executing at the moment\n"
                "  - human_proposed_changes = explicit tasks requested by the CURRENT proposer\n"
                "  - optimizer_suggestions_for_changes = extra optimizer tasks supporting those changes (suggestions only)\n"
                "- Treat human_proposed_changes as explicit wishes of the proposer.\n"
                "- Treat optimizer_suggestions_for_changes as robot suggestions that still require human agreement.\n"
                "- Conflict ONLY applies when human_proposed_changes contradict committed_plan.\n"
                "- If direct_conflicts_with_committed_plan is present, prefer CONTINUE NEGOTIATING and prompt for consensus, "
                "rather than immediately accepting one side.\n"
                "\n"
                "- When building planner_action.candidate_plan_delta you are NOT limited to only the latest human change.\n"
                "  You may:\n"
                "    - adopt the human_proposed_changes exactly, OR\n"
                "    - keep the committed_plan unchanged, OR\n"
                "    - combine human_proposed_changes with some optimizer_suggestions_for_changes so that humans and the robot\n"
                "      cover more useful boxes.\n"
                "- Prefer such combined plans when they clearly improve safety/coverage/balance according to objective_metrics "
                "and do NOT create new conflict with committed_plan.\n"
                "- candidate_plan_delta should reflect the full recommended plan after this step (for all relevant agents), "
                "not just the single last human request.\n"
                "\n"
                "- Use objective_metrics to modulate how assertive you are:\n"
                "    - If deadline_risk is medium/high or suboptimality_level is high/very_high, be more willing to suggest extra\n"
                "      optimizer tasks that improve safety, while still framing them as options for humans.\n"
                "    - If metrics look already good and there is no urgency, prefer minimal changes and lighter suggestions.\n"
                "\n"
                "- Your output must be valid JSON matching the requested schema.\n"
                "- Never speak as if suggested tasks are already agreed; use wording like "
                "\"If you agree, one option is…\".\n"
                "- If notes indicate impossible actions (expired, already fulfilled), explicitly mention them.\n"
                "- Mediation is always grounded in concrete actions.\n"
                "- In robot_utterance, you MUST explicitly mention at least one concrete task (agent, box_id, property, kind).\n"
                "- Avoid abstract phrases like 'some optimizer suggestions' unless tied to specific actions.\n"
                "- Always be explicit on who has to perform what.\n"
                "- If optimizer_suggestions_for_changes contains ANY action with kind=\"dispose\": you MUST spend this step trying to get agreement from both humans.\n"
                f"{negotiation_suffix}"
            ),
        }



        user_payload = {
            "capabilities": self.capabilities,
            "planning_view": planning_view,
            "phase": phase,
            "objective_metrics": state.objective.for_llm(),
            "social_context": state.social.for_llm(),
            "interaction_context": {
                "event_summary": state.interaction.event_summary,
                "robot_role_description": state.interaction.robot_role_description,
                "session_notes": state.interaction.session_notes,
                "recent_dialogue": transcript,
                "human_profiles": state.interaction.human_profiles or {},
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

    def _summarize_candidate_with_sources(
        self,
        candidate: Plan,
        prefix: Optional[Plan],
    ) -> Dict[str, Any]:
        """
        Summarize the candidate plan, tagging each action with a 'source':
          - 'human_prefix' if it was in the original prefix_plan
          - 'optimizer_completion' otherwise
        """
        summary: Dict[str, Any] = {}

        # Build a set of (agent_id, (box_id, prop, kind)) for prefix actions
        prefix_set = set()
        if prefix:
            for aid, actions in (prefix or {}).items():
                for tup in actions:
                    # tup is (box_id, prop, kind)
                    prefix_set.add((aid, tup))

        for aid, actions in (candidate or {}).items():
            out_actions = []
            for (box_id, prop, kind) in actions:
                source = "optimizer_completion"
                if (aid, (box_id, prop, kind)) in prefix_set:
                    source = "human_prefix"

                out_actions.append(
                    {
                        "box_id": int(box_id),
                        "property": prop,
                        "kind": kind,
                        "source": source,  # NEW
                    }
                )
            summary[aid] = out_actions

        return summary


    def _apply_candidate_delta(self, state: MediationState, current: Plan, delta: Dict[str, Any]) -> Plan:
        new_plan: Plan = dict(current) if current is not None else {}

        for agent_key, actions in (delta or {}).items():
            aid = (agent_key or "").strip()
            if aid not in ("robot", "human_a", "human_b"):
                continue

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

