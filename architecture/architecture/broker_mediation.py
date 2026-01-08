# broker_mediation.py
#
# Mediation / reflection / conflict-metrics logic for BrokerNode,
# split out as a mixin so broker_node.py is less of a monster.
#
# Assumes `self` is a rclpy Node with:
#   - self.get_logger()
#   - self.conn (sqlite3 connection)
#   - self.pub_mediation_ctrl (Publisher<std_msgs/String>)
#   - self._plan_mediator : PlanMediator
#   - self._mediation_sessions : dict[str, MediationState]
#   - self._active_mediation_id : Optional[str]
#   - self._event_summary_text : Optional[str]
#   - self.optimizer_base_url, self.req_timeout
#   - self.optimizer_horizon_sec, self._last_plan, self._last_server_time
#   - self._build_agents_for_optimizer(), self._build_boxes_for_optimizer()
#   - self._snapshot_agent_positions(), self._make_travel_time_fn()
#   - self._publish_optimizer_plan()
#   - self._chat_json()
#   - self.event_summary_model
#
from __future__ import annotations

import json
import time
import sqlite3
from typing import Dict, Any, Optional, List, Tuple

from std_msgs.msg import String as StringMsg
from std_srvs.srv import Trigger

from jsonschema import validate, ValidationError  # if you need it for your own schemas

from std_msgs.msg import String as StringMsg

import threading
import copy

from .optimizer_client import (
    Plan,
    build_plan_from_llm_agents_plan,
    evaluate_candidate_plan,
    extend_plan_with_prefix,
    PlannerWeights,
    estimate_deadline_risk_for_plan,
    compute_xy_imbalance_for_plan,
    summarize_plan_parse_issues
)
from .plan_mediator import (
    PlanMediator,
    MediationLLMConfig,
    MediationState,
    MediationObjectiveMetrics,
    MediationSocialContext,
    MediationInteractionContext,
    MediationTurn
)

from collections import deque


# ------------------------ Profile reflection schema ------------------------

PROFILE_REFLECTION_SCHEMA = {
    "type": "object",
    "required": ["humans"],
    "properties": {
        "humans": {
            "type": "object",
            # Each key under "humans" is a human_id ("human_a", "human_b", ...)
            "additionalProperties": {
                "type": "object",
                # summary_delta + trait_updates are *optional*
                "properties": {
                    "summary_delta": {"type": "string"},
                    "trait_updates": {
                        "type": "object",
                        # Each key under trait_updates is a trait name
                        "additionalProperties": {
                            "type": "object",
                            "required": ["new_value"],
                            "properties": {
                                "new_value": {},
                                "confidence_delta": {"type": "number"},
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                # Allow extra keys like:
                # "leadership_preference": "emerging"
                # "trust_in_robot_plans": "low"
                # We'll normalize them in _apply_profile_updates.
                "additionalProperties": True,
            },
        }
    },
    # IMPORTANT: allow top-level extra keys the model sometimes adds,
    # like "type" and "properties".
    "additionalProperties": True,
}


class BrokerMediationMixin:
    """
    Mixin that adds:
      - LLM mediation call wrapper
      - mediation session lifecycle
      - speech routing into mediation
      - human-profile reflection
      - conflict metrics & service
    to a Broker-like node.
    """

    def _init_mediation_tts(self):
        self.pub_tts_immediate = self.create_publisher(
            StringMsg, "/skills/tts_immediate", 10
        )
        
        # Global chat history: humans + robot utterances
        # Each entry: {"role": "human_a"/"human_b"/"unknown"/"robot", "text": str, "ts": float}
        self._chat_history = deque(maxlen=50)  # tune window as you like
        # In __init__ or _init_mediation_tts or BrokerNode init
        self._last_plan_provenance = {}  # type: Dict[str, List[Dict[str, str]]]

        # ---- Mediation liveness controls ----
        self.mediation_turn_timeout_sec = float(getattr(self, "mediation_turn_timeout_sec", 60.0))
        self.mediation_max_turns = int(getattr(self, "mediation_max_turns", 6))   # human+robot steps combined


        self._mediation_pending_deadline_ts: Optional[float] = None
        self._mediation_pending_started_ts: Optional[float] = None

        self._mediation_watchdog_timer = None  # rclpy Timer

    def _start_pending_deadlines(self, now_ts: float):
        self._mediation_pending_started_ts = float(now_ts)
        self._mediation_pending_deadline_ts = float(now_ts) + float(self.mediation_turn_timeout_sec)

    def _bump_pending_deadline(self, now_ts: float):
        # “waiting for next utterance” timeout
        self._mediation_pending_deadline_ts = float(now_ts) + float(self.mediation_turn_timeout_sec)




    def _append_chat_turn(self, role: str, text: str, ts: float):
        text = (text or "").strip()
        if not text:
            return
        try:
            self._chat_history.append({
                "role": role or "unknown",
                "text": text,
                "ts": float(ts),
            })
        except Exception:
            # Never let history logging crash the node
            pass

    def _get_recent_chat_turns(self, limit: int = 8) -> List[dict]:
        hist = list(getattr(self, "_chat_history", []))
        if limit is not None and limit > 0:
            hist = hist[-limit:]
        return hist

    # ---------- Per-human periodic reflection ----------

    def _run_periodic_profile_reflection_for(self, human_id: str, ts: float):
        """
        Lightweight reflection that runs every N utterances for a given human,
        based on GLOBAL chat history (not only mediation sessions).
        """
        # Recent conversation focused on this human
        recent_turns = [
            t for t in self._get_recent_chat_turns(limit=50)
            if t.get("role") == human_id
        ]
        if len(recent_turns) < 2:
            # Not enough signal to reflect
            return

        current_profiles = self._load_current_human_profiles() or {}
        cur_prof = current_profiles.get(human_id) or {}

        payload = {
            "mode": "periodic",
            "human_id": human_id,
            "timestamp": ts,
            "recent_utterances": recent_turns,
            "current_profile": cur_prof,
        }

        system_msg = {
            "role": "system",
            "content": (
                "You are Bob's reflection module.\n"
                "- You see recent utterances from ONE human interacting with Bob.\n"
                "- Infer small, conservative updates to this human's profile.\n"
                "- Focus on traits like: leadership_preference, trust_in_robot_plans,\n"
                "  need_for_explanations, risk_attitude, and tolerance_for_disagreement.\n"
                "- Only change traits you have clear evidence for.\n"
            ),
        }

        user_msg = {
            "role": "user",
            "content": (
                "Given these recent utterances and the current profile, propose updates "
                "for this ONE human.\n"
                "Return STRICT JSON of the same form as the ProfileReflection.humans entry, e.g.:\n"
                "{\n"
                f'  "humans": {{ "{human_id}": {{\n'
                '      "summary_delta": "...",\n'
                '      "trait_updates": {\n'
                '        "leadership_preference": {"new_value": "emerging", "confidence_delta": 0.1}\n'
                "      }\n"
                "  }}}\n"
                "}\n"
            ),
        }

        payload_msg = {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        }

        obj = self._chat_json(
            messages=[system_msg, user_msg, payload_msg],
            temperature=0.3,
            max_tokens=250,
            retries=0,
            schema=PROFILE_REFLECTION_SCHEMA,
            schema_name="ProfileReflection",
            model=self.event_summary_model,  # or another small model
            perf_phase="profile_reflection_periodic",
        )

        # Apply updates (will safely ignore if nothing for this human)
        updates = obj.get("humans", {}) or {}
        self._apply_profile_updates(updates, current_profiles)


    def _on_human_utterance_for_profile_reflection(self, speaker_id: str, ts: float):
        """
        Called for every final human utterance (speech_final_any).
        We:
          - keep a per-human message count,
          - every `profile_reflection_every` messages, kick off a periodic reflection
            based on recent global chat history.
        """
        # Only track real humans
        if not isinstance(speaker_id, str) or not speaker_id.startswith("human_"):
            return

        # Config: how often to reflect
        every = getattr(self, "profile_reflection_every", 0)
        if every <= 0:
            return

        # Make sure the counter dict exists
        if not hasattr(self, "_profile_msg_counts"):
            self._profile_msg_counts = {}

        old = int(self._profile_msg_counts.get(speaker_id, 0))
        new = old + 1
        self._profile_msg_counts[speaker_id] = new

        # Not yet at the threshold → nothing to do
        if new % every != 0:
            return

        # Threshold hit: run periodic reflection asynchronously
        self.get_logger().info(
            f"[profiles] periodic reflection trigger for {speaker_id} after {new} utterances"
        )

        def _worker():
            try:
                self._run_periodic_profile_reflection_for(speaker_id, ts)
            except Exception as e:
                self.get_logger().warn(
                    f"[profiles] periodic reflection for {speaker_id} failed: {e}"
                )

        threading.Thread(target=_worker, daemon=True).start()

        # After we kick off a reflection, reset this human's counter so the next
        # reflection happens after another full `every` utterances.
        self._profile_msg_counts[speaker_id] = 0



    def _robot_say(self, text: str):
        """
        Send one-shot TTS utterance to SkillsAgent.
        """
        if not text:
            return
        try:
            self.pub_tts_immediate.publish(StringMsg(data=text))
        except Exception as e:
            self.get_logger().warn(f"[mediation][tts] failed to publish utterance: {e}")
            
        # Log robot utterance in global history
        try:
            self._append_chat_turn("robot", text, time.time())
        except Exception:
            pass

    # ---------- LLM wrapper for mediator ----------

    def _mediate_llm_call(self, messages: list) -> dict:
        """
        Adapter used by PlanMediator; calls self._chat_json with
        a small, mediation-specific schema.
        """
        schema = {
            "type": "object",
            "required": ["decision", "planner_action", "robot_utterance", "log_tags"],
            "properties": {
                "decision": {
                    "type": "string",
                },
                "planner_action": {
                    "type": "object",
                    "required": ["kind"],
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": [
                                "keep_baseline",
                                "adopt_candidate",
                                "merge_plans",
                                "request_new_plan",
                            ],
                        },
                        "candidate_plan_delta": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                        "notes": {
                            "type": "string"
                        },
                    },
                    "additionalProperties": False,
                },
                "robot_utterance": {"type": "string"},
                "log_tags": {
                    "type": "object",
                    "properties": {
                        "strategy": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            },
            "additionalProperties": False,
        }

        obj = self._chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=200,
            retries=1,
            schema=schema,
            schema_name="PlanMediation",
            model=self.model,
            perf_phase="plan_mediation",
        )
        return obj

    # ---------- Candidate sanity: impossible / already-fulfilled actions ----------

    def _summarize_impossible_actions(
        self,
        plan: Plan,
        boxes,
        current_time: float,
    ) -> Optional[str]:
        """
        Look for actions in the (prefix) plan that are already fulfilled or impossible
        given current box state:

        - disposal where that (box, property) is already disposed
        - disposal where the deadline has already passed
        - sensing where that agent already sensed (box, property)
        - sensing or disposal on a box whose deadline is already past
        """
        if not plan:
            return None

        box_by_id = {b.box_id: b for b in boxes or []}
        lines: List[str] = []

        for aid, actions in (plan or {}).items():
            for (box_id, prop, kind) in actions:
                reasons = []
                b = box_by_id.get(box_id)

                if b is None:
                    reasons.append("unknown box")
                else:
                    # Deadline fully expired for this box?
                    if b.deadline is not None and current_time >= float(b.deadline):
                        reasons.append("deadline already passed")

                    if kind == "dispose":
                        # already disposed?
                        if prop == "X" and b.disposed_X:
                            reasons.append("already disposed for X")
                        if prop == "Y" and b.disposed_Y:
                            reasons.append("already disposed for Y")

                    elif kind == "sense":
                        # already sensed by this agent?
                        already = (b.already_sensed or {}).get(aid, {}).get(prop, False)
                        if already:
                            reasons.append("already sensed by this agent")

                if reasons:
                    reason_str = " and ".join(reasons)
                    lines.append(
                        f"- {aid} {kind} box {box_id} ({prop}) → {reason_str}"
                    )

        if not lines:
            return None

        header = (
            "Some of the actions in the requested plan are no longer possible "
            "because they are already fulfilled or past their deadline:\n"
        )
        return header + "\n".join(lines)



    # ---------- Mediation status plumbing ----------

    def _publish_mediation_status(self, session_id: str, status: str):
        """
        Tell EventLayer whether mediation is pending or finished.

        status: "pending", "accept", "reject", "cancelled", "idle", ...
        """
        payload = {
            "session_id": session_id,
            "status": status,
            "ts": time.time(),
        }
        try:
            self.pub_mediation_ctrl.publish(
                StringMsg(data=json.dumps(payload))
            )
        except Exception as e:
            self.get_logger().warn(f"[mediation] failed to publish status: {e}")

    def _mediation_in_progress(self) -> bool:
        """
        Returns True if there is an active mediation session whose status is 'pending'.
        """
        sid = getattr(self, "_active_mediation_id", None)
        if not sid:
            return False
        sess = self._mediation_sessions.get(sid)
        return bool(sess and sess.status == "pending")

    def _summarize_plan_diff(self, baseline: Plan, candidate: Plan) -> str:
        """
        Build a tiny human-readable diff between baseline and candidate,
        mainly to help the mediator decide what to verbalize.
        """
        lines = []
        agents = sorted(set(list(baseline.keys()) + list(candidate.keys())))
        for aid in agents:
            b = baseline.get(aid) or []
            c = candidate.get(aid) or []
            if b == c:
                continue
            lines.append(f"- {aid}: baseline={b}, candidate={c}")
        if not lines:
            return "No changes between baseline and candidate plans."
        return "Plan changes:\n" + "\n".join(lines)


    # ---------- Main entry: LLM speech plan arrives ----------

    def _handle_llm_speech_plan(self, trace_entry: dict, ts: float):
        """
        Handle a speech → multi-agent plan result coming back from the LLM.

        Expects trace_entry['data'] to contain either:
          - 'json'      -> parsed dict
          - or 'json_text' -> JSON string we can parse

        Also expects (if present):
          - 'speaker_id'  -> which human proposed this plan ('human_a', 'human_b', ...)
          - 'request_id'  -> llm_speech_to_multiagent_plan:<...>
        """
        # If we already have a mediation session in progress, ignore new plans.
        if self._mediation_in_progress():
            self.get_logger().info("[llm-plan] mediation in progress; ignoring new speech-plan result")
            return

        data = trace_entry.get("data") or {}

        # Who proposed this?
        proposer_id = data.get("speaker_id") or "unknown"
        req_id = data.get("request_id") or data.get("id")

        # 1) Get the parsed plan JSON, or parse json_text as a fallback
        plan_json = data.get("json")
        if not isinstance(plan_json, dict):
            json_text = data.get("json_text")
            if isinstance(json_text, str) and json_text.strip():
                try:
                    plan_json = json.loads(json_text)
                except Exception as e:
                    self.get_logger().warn(f"[llm-plan] failed to parse json_text: {e}")
                    plan_json = None

        if not isinstance(plan_json, dict):
            self.get_logger().warn("[llm-plan] missing or invalid plan JSON; aborting")
            return

        # 2) Extract agents_plan
        agents_plan = plan_json.get("agents_plan")
        if not isinstance(agents_plan, dict):
            self.get_logger().warn("[llm-plan] agents_plan missing or not an object")
            return

        # 3) Build a PREFIX plan from the LLM (robot-only for now),
        #    and collect any parse issues so Bob can tell humans what's missing.
        try:
            prefix_plan, parse_issues = build_plan_from_llm_agents_plan(
                agents_plan,
                allowed_agents=["robot", "human_a", "human_b"],
                collect_issues=True,
            )
        except Exception as e:
            self.get_logger().warn(f"[llm-plan] build_plan_from_llm_agents_plan failed: {e}")
            return

        parse_issues_note = None
        if parse_issues:
            parse_issues_note = summarize_plan_parse_issues(parse_issues)
            self.get_logger().info(f"[llm-plan] plan parse issues: {parse_issues_note}")

        # If there are NO valid robot actions but there WERE issues, Bob should
        # just ask for clarification and stop – there's nothing to optimize.
        if not prefix_plan:
            if parse_issues_note:
                self._robot_say(parse_issues_note)
            else:
                self.get_logger().info("[llm-plan] no valid robot actions in prefix; ignoring")
                self._robot_say("I could not understand you.")
            return



        # 4) Build agents + boxes as in optimizer
        try:
            agents = self._build_agents_for_optimizer()
            boxes = getattr(self, "_last_boxes_for_optimizer", None)
            if boxes is None:
                boxes, _ = self._build_boxes_for_optimizer(
                    getattr(self, "_last_boxes_state", []) or []
                )
        except Exception as e:
            self.get_logger().warn(f"[llm-plan] failed to build agents/boxes: {e}")
            return

        current_plan = getattr(self, "_last_plan", {}) or {}

        # 4b) Build travel_time_fn + current_time + horizon as in optimizer thread
        box_positions = getattr(self, "_last_box_positions", {}) or {}
        agent_positions = self._snapshot_agent_positions()
        horizon = self.optimizer_horizon_sec
        current_time = getattr(self, "_last_server_time", None)
        if current_time is None:
            current_time = time.time()

        travel_time_fn = self._make_travel_time_fn(agent_positions, box_positions)

        # Use a single PlannerWeights instance for both completion and evaluation
        weights = PlannerWeights()

        # 5a) Check the human/LLM prefix plan for impossible or already-fulfilled actions
        impossible_note = self._summarize_impossible_actions(
            plan=prefix_plan,
            boxes=boxes,
            current_time=current_time,
        )

        # If the plan was partially usable but had missing fields, let Bob
        # proactively tell humans what he had to ignore.
        if prefix_plan and parse_issues_note:
            self._robot_say(parse_issues_note)


        # 5) First, extend the prefix plan into a full candidate via optimizer
        try:
            candidate_plan = extend_plan_with_prefix(
                prefix_plan=prefix_plan,
                agents=agents,
                boxes=boxes,
                current_time=current_time,
                horizon=horizon,
                travel_time_fn=travel_time_fn,
                weights=weights,
            )
        except Exception as e:
            self.get_logger().warn(f"[llm-plan] failed to extend prefix plan via optimizer: {e}")
            return

        if not candidate_plan:
            self.get_logger().info("[llm-plan] optimizer returned empty candidate plan; ignoring")
            return

        # 6) Unified evaluation: score + constraints + adopt + suboptimality
        try:
            eval_res = evaluate_candidate_plan(
                current_plan=current_plan,
                candidate_plan=candidate_plan,
                agents=agents,
                boxes=boxes,
                current_time=current_time,
                travel_time_fn=travel_time_fn,
                horizon=horizon,
                weights=weights,
                margin=getattr(self, "optimizer_llm_margin", 0.01),
            )

        except Exception as e:
            self.get_logger().warn(f"[llm-plan] plan evaluation failed: {e}")
            return

        better = bool(eval_res["adopt"])
        score_optimal = eval_res["score_current"]
        score_candidate = eval_res["score_candidate"]
        suboptimal_pct = eval_res["suboptimal_pct"]

        xy_metrics = compute_xy_imbalance_for_plan(
            plan=candidate_plan,
            boxes=boxes,
        )
        imbalance_XY = xy_metrics["imbalance"]

        deadline_risk = estimate_deadline_risk_for_plan(
            plan=candidate_plan,
            boxes=boxes,
            current_time=current_time,
            travel_time_fn=travel_time_fn,
            horizon=horizon,
        )

        # 6) Build mediation state (baseline vs candidate + objective/social context)
        session_id = f"mediation:{req_id}"

        # Combine different diagnostics into a single notes field
        notes_parts: List[str] = []
        if parse_issues_note:
            notes_parts.append(parse_issues_note)
        if impossible_note:
            notes_parts.append(impossible_note)
        combined_notes = "\n\n".join(notes_parts) if notes_parts else None


        objective = MediationObjectiveMetrics(
            suboptimality_pct=suboptimal_pct,
            baseline_score=score_optimal,
            candidate_score=score_candidate,
            deadline_risk=deadline_risk,   
            imbalance_XY=imbalance_XY,           # TODO: compute from score components if needed
            fulfillment_history_ok=True, # or from robot_exec_evals
            notes=combined_notes,       # <--- NEW: surface impossible actions here
        )

        # --- NEW: compute social metrics for this proposer from recent history ---
        (
            proposer_success_rate,
            conflict_index,
            override_frequency,
            leadership_contestation,
        ) = self._compute_social_context_for_proposer(
            proposer_id=proposer_id,
            window_sec=600.0,   # e.g., last 5 minutes; tune as needed
        )

        social = MediationSocialContext(
            proposer_id=proposer_id,
            proposer_success_rate=proposer_success_rate,
            conflict_index=conflict_index,
            override_frequency=override_frequency,
            leadership_contestation=leadership_contestation,
        )


        plan_diff_note = self._summarize_plan_diff(
            baseline=current_plan,
            candidate=candidate_plan,
        )

        # Seed session_notes with impossible-action note if present
        base_notes = "Initial mediation triggered by LLM speech plan."
        if parse_issues_note:
            base_notes = base_notes + "\n\n" + parse_issues_note
        if impossible_note:
            base_notes = base_notes + "\n\n" + impossible_note

        base_notes = base_notes + "\n\n" + plan_diff_note

        current_profiles = self._load_current_human_profiles()


        # Build recent dialogue from GLOBAL chat history (last N turns)
        recent_chat = self._get_recent_chat_turns(limit=12)
        recent_utterances: List[MediationTurn] = []
        for i, t in enumerate(recent_chat):
            try:
                role = t.get("role") or "unknown"
                text = (t.get("text") or "").strip()
                if not text:
                    continue
                ts_chat = float(t.get("ts") or ts)
                recent_utterances.append(
                    MediationTurn(
                        role=role,
                        text=text,
                        meta={"ts": ts_chat},
                    )
                )
            except Exception:
                continue

        interaction = MediationInteractionContext(
            event_summary=self._event_summary_text,
            robot_role_description=(
                "Bob is a cooperative teammate that balances safety, efficiency, and human preferences."
            ),
            session_notes=base_notes,
            human_profiles=current_profiles or {},
            recent_utterances=recent_utterances,
        )


        human_text = data.get("request_text")
        if not human_text:
            # Fallback: use natural_summary from the parsed JSON if present
            j = data.get("json") or {}
            human_text = j.get("natural_summary")

        initial_turns = []
        if isinstance(human_text, str) and human_text.strip():
            initial_turns.append(
                MediationTurn(
                    role=proposer_id,
                    text=human_text.strip(),
                    meta={
                        "ts": ts,
                        "req_id": req_id,
                        "source_rule": trace_entry.get("rule"),
                    },
                )
            )



        state = MediationState(
            session_id=session_id,
            baseline_plan=current_plan,
            candidate_plan=candidate_plan,
            objective=objective,
            social=social,
            interaction=interaction,
            turns=initial_turns,
            prefix_plan=prefix_plan,
            baseline_provenance=getattr(self, "_last_plan_provenance", None),
        )

        self._mediation_sessions[session_id] = state
        self._active_mediation_id = session_id

        # 7) Optionally do a first mediation step (robot explains / asks questions)
        self._robot_say("Let me think.")
        state, raw = self._plan_mediator.step(state)
        self._mediation_sessions[session_id] = state

        # --- NEW: if mediation is now pending, block llm_speech_plan in EventLayer ---
        if state.status == "pending":
            self._publish_mediation_status(session_id, "pending")
            self._start_pending_deadlines(ts)

        robot_utt = (raw.get("robot_utterance") or "").strip()
        if robot_utt:
            self.get_logger().info(f"[mediation] Bob says: {robot_utt}")
            self._robot_say(robot_utt)

        # If the mediator already reached a decision in this first step, finalize now.
        if state.status in ("accept", "reject"):
            self._finalize_mediation_session(
                session_id=session_id,
                session=state,
                raw_decision=raw,
                ts=ts,
            )

        # 8) Log plan proposal for objective planning stats (no adoption yet: humans decide)
        try:
            self.conn.execute(
                """
                INSERT INTO plan_proposals(
                    ts, proposer_id, source, req_id,
                    adopted, better_than_current,
                    score_optimal, score_candidate, suboptimal_pct
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    float(ts),
                    proposer_id,
                    "llm_speech",
                    req_id,
                    0,  # adopted = 0 here; human mediation will decide later
                    int(1 if better else 0),
                    score_optimal,
                    score_candidate,
                    suboptimal_pct,
                ),
            )
            self.conn.commit()
        except Exception as e:
            self.get_logger().warn(f"[llm-plan] failed to log plan_proposal: {e}")

    # ---------- Finalize mediation session ----------

    def _finalize_mediation_session(
        self,
        session_id: str,
        session: MediationState,
        raw_decision: dict,
        ts: float,
    ):
        """
        Called when a mediation session reaches a terminal status ('accept' or 'reject').

        - Applies the planner_action (adopt / keep baseline), but NEVER adopts
          a candidate plan that has impossible actions (expired deadlines, already
          fulfilled actions, etc.).
        - Publishes the resulting plan exactly once.
        - Triggers a reflection pass to update human profiles.
        - Clears the active mediation flag so future plans can start.
        """
        try:
            pa = (raw_decision or {}).get("planner_action") or {}
            kind = pa.get("kind")

            # --- SAFETY GUARD: block adoption of impossible candidate plans ---
            # Rebuild latest boxes snapshot and check candidate for impossible actions.
            candidate_plan = session.candidate_plan or {}
            keep_baseline_reason = None

            if kind in ("adopt_candidate", "merge_plans") and candidate_plan:
                try:
                    boxes = getattr(self, "_last_boxes_for_optimizer", None)
                    if boxes is None:
                        boxes, _ = self._build_boxes_for_optimizer(
                            getattr(self, "_last_boxes_state", []) or []
                        )

                    current_time = getattr(self, "_last_server_time", None) or time.time()
                    impossible_note = self._summarize_impossible_actions(
                        plan=candidate_plan,
                        boxes=boxes,
                        current_time=current_time,
                    )
                    if impossible_note:
                        keep_baseline_reason = impossible_note
                except Exception as e:
                    # If something goes wrong computing impossibility, be conservative:
                    self.get_logger().warn(
                        f"[mediation] failed to re-check candidate feasibility: {e}; "
                        "keeping baseline plan."
                    )
                    keep_baseline_reason = "candidate feasibility check failed"

            # If we detected impossible actions, force keep_baseline
            if keep_baseline_reason is not None:
                self.get_logger().warn(
                    f"[mediation] LLM requested {kind} but candidate plan has "
                    f"impossible actions; keeping baseline. Details:\n{keep_baseline_reason}"
                )
                kind = "keep_baseline"

            # Decide which plan to keep (after safety override)
            new_plan = session.baseline_plan
            if kind in ("adopt_candidate", "merge_plans"):
                new_plan = candidate_plan

            # Publish plan (if any) once mediation is over
            if new_plan:
                box_positions = getattr(self, "_last_box_positions", {}) or {}
                current_time = getattr(self, "_last_server_time", None) or time.time()
                self._last_plan = new_plan

                try:
                    # Start with everything as optimizer-origin
                    prov = self._build_optimizer_provenance(new_plan)

                    # If we have a prefix_plan, upgrade origins to "human"
                    # for actions explicitly requested in this session.
                    prefix = getattr(session, "prefix_plan", None) or {}
                    prefix_set = set()
                    for aid, actions in (prefix or {}).items():
                        for (box_id, prop, kind) in (actions or []):
                            prefix_set.add((aid, int(box_id), prop, kind))
                            
                    proposer = session.social.proposer_id or "unknown"

                    for aid, actions in (new_plan or {}).items():
                        updated = []
                        for (box_id, prop, kind) in (actions or []):
                            origin = "optimizer"
                            proposed_by = None
                            
                            if (aid, int(box_id), prop, kind) in prefix_set:
                                origin = "human"
                                proposed_by = proposer
                                
                            updated.append(
                                {
                                    "box_id": int(box_id),
                                    "property": prop,
                                    "kind": kind,
                                    "origin": origin,
                                    "proposed_by": proposed_by,
                                }
                            )
                        prov[aid] = updated

                    self._last_plan_provenance = prov
                except Exception as e:
                    self.get_logger().warn(f"[mediation] failed to build provenance: {e}")
                    self._last_plan_provenance = self._build_optimizer_provenance(new_plan)

                self._publish_optimizer_plan(new_plan, current_time, box_positions)

                self.get_logger().info(
                    f"[mediation] finalized session {session_id} with planner_action={kind}"
                )
            else:
                self.get_logger().info(
                    f"[mediation] session {session_id} finalized with no plan change (kind={kind})"
                )

            # Reflection: update human profiles based on the whole dialogue.
            # Run this asynchronously so plan publishing is not blocked.
            try:
                self._kickoff_async_reflection(session)
            except Exception as e:
                self.get_logger().warn(f"[mediation] failed to start async reflection: {e}")


        except Exception as e:
            self.get_logger().warn(f"[mediation] finalize session failed: {e}")
        finally:
            # Conversation is over → allow new speech-plan proposals again
            self._active_mediation_id = None

            # Tell EventLayer that mediation is no longer pending
            final_status = session.status or "idle"
            if final_status not in ("accept", "reject", "cancelled"):
                final_status = "idle"

            self._mediation_pending_deadline_ts = None

            try:
                self._publish_mediation_status(session_id, final_status)
            except Exception as e:
                self.get_logger().warn(f"[mediation] failed to publish final status: {e}")
            # keep self._mediation_sessions[session_id] for logging/analysis


    def _build_optimizer_provenance(self, plan: Plan) -> dict:
        prov = {}
        for aid, actions in (plan or {}).items():
            prov[aid] = [
                {
                    "box_id": int(box_id),
                    "property": prop,
                    "kind": kind,
                    "origin": "optimizer",
                    "proposed_by": None,
                }
                for (box_id, prop, kind) in (actions or [])
            ]
        return prov


    # ---------- Async reflection helper ----------

    def _kickoff_async_reflection(self, session: MediationState):
        """
        Run human-profile reflection in a background thread so it doesn't
        block the main ROS executor.
        """
        session_snapshot = copy.deepcopy(session)

        def _worker():
            try:
                if self._should_run_reflection_for_session(session_snapshot):
                    self._reflect_on_mediation_session(session_snapshot)
                    # After a full mediation-based reflection, reset per-human
                    # periodic counters so they start fresh.
                    self._reset_profile_msg_counts_for_session(session_snapshot)
                else:
                    self.get_logger().info(
                        f"[mediation] skipping profile reflection for {session_snapshot.session_id} "
                        f"(turns_used={session_snapshot.turns_used}, "
                        f"humans={self._count_human_turns(session_snapshot)})"
                    )
            except Exception as e:
                self.get_logger().warn(f"[mediation] reflection failed: {e}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()


    def _reset_profile_msg_counts_for_session(self, session: MediationState):
        """
        After we run a mediation-based reflection, reset the per-human
        periodic counters for any humans that spoke in this session.
        """
        if not hasattr(self, "_profile_msg_counts"):
            return

        humans_in_session = {
            t.role
            for t in (session.turns or [])
            if isinstance(t.role, str) and t.role.startswith("human_")
        }
        if not humans_in_session:
            return

        for hid in humans_in_session:
            self._profile_msg_counts[hid] = 0

        self.get_logger().info(
            f"[profiles] reset periodic counters for humans in session "
            f"{session.session_id}: {sorted(humans_in_session)}"
        )


    # ---------- Post-hoc reflection / profile updates ----------

    def _count_human_turns(self, session: MediationState) -> int:
        """
        Count how many utterances in this session came from humans.
        """
        return sum(
            1
            for t in (session.turns or [])
            if isinstance(t.role, str) and t.role.startswith("human_")
        )

    def _should_run_reflection_for_session(self, session: MediationState) -> bool:
        """
        Decide whether to run profile reflection for this mediation.

        Heuristics:
          - Run if there was at least one *real* pending phase: i.e., multiple
            mediation steps (turns_used >= 2).
          - OR run if the candidate was adopted (status == 'accept').
          - OR run if there was enough human dialogue (>= min_reflection_human_turns).

        You can tune min_reflection_human_turns as a ROS param or attribute.
        """
        # Did we have more than one LLM mediation step? (pending → new human turns → step again)
        nontrivial_mediation = (session.turns_used or 0) >= 2


        # How much human talk happened in this session?
        human_turns = self._count_human_turns(session)

        # Minimal human turns threshold (configurable)
        min_turns = getattr(self, "min_reflection_human_turns", 3)

        if nontrivial_mediation:
            return True
        if human_turns >= min_turns:
            return True

        return False



    def _reflect_on_mediation_session(self, session: MediationState):
        """
        Post-hoc reflection: infer human preference updates from the mediation dialogue.
        """
        # 1) Build a compact transcript
        transcript = [
            {
                "role": t.role,
                "text": t.text,
                "meta": t.meta,
            }
            for t in session.turns
        ]

        # 2) Optional: get current profiles from HDT / DB
        current_profiles = self._load_current_human_profiles()  # {"human_a": {...}, ...}

        # NEW: load realized disposal outcomes and compute planning accuracy per human
        disposal_outcomes = self._load_disposal_outcomes_for_session(session)
        from .optimizer_client import compute_planning_accuracy_by_human
        planning_stats = compute_planning_accuracy_by_human(disposal_outcomes)

        # 3) Prepare payload for LLM
        payload = {
            "session_id": session.session_id,
            "status": session.status,
            "baseline_plan": self._plan_mediator._summarize_plan(session.baseline_plan),
            "final_candidate_plan": self._plan_mediator._summarize_plan(session.candidate_plan),
            "objective_metrics": session.objective.__dict__,
            "social_context": session.social.__dict__,
            "interaction_context": {
                "event_summary": session.interaction.event_summary,
                "session_notes": session.interaction.session_notes,
            },
            "dialogue": transcript,
            "current_profiles": current_profiles or {},
            "planning_outcomes": planning_stats,
        }

        system_msg = {
            "role": "system",
            "content": (
                "You are Bob's reflection module.\n"
                "- You see a complete planning conversation between Bob and the humans.\n"
                "- You also see statistics about how often each human's proposed disposals "
                "ended up being correct or incorrect.\n"
                "- Infer how each human prefers to plan and interact with Bob.\n"
                "- Penalize humans whose disposal plans frequently send Bob to boxes that "
                "do NOT have the requested property (low correct_rate in planning_outcomes).\n"
                "- Update only traits you have evidence for. Be conservative.\n"
            ),
        }


        user_msg = {
            "role": "user",
            "content": (
                "Given this mediation session, propose updates to each human's profile.\n"
                "Focus on leadership preference, trust in robot plans, need for explanations,\n"
                "risk aversion or deadline focus, and tolerance for disagreement.\n\n"
                "Return STRICT JSON of the form:\n"
                "{\n"
                '  "humans": {\n'
                '    "human_a": {\n'
                '      "summary_delta": "...",  # optional\n'
                '      "trait_updates": {\n'
                '        "leadership_preference": {"new_value": "emerging", "confidence_delta": 0.2},\n'
                '        "trust_in_robot_plans": {"new_value": "low", "confidence_delta": 0.1}\n'
                "      }\n"
                "    },\n"
                "    \"human_b\": { ... }\n"
                "  }\n"
                "}\n"
            ),
        }

        payload_msg = {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        }

        obj = self._chat_json(
            messages=[system_msg, user_msg, payload_msg],
            temperature=0.3,
            max_tokens=300,
            retries=0,
            schema=PROFILE_REFLECTION_SCHEMA,
            schema_name="ProfileReflection",
            model=self.event_summary_model,  # or another cheap model
            perf_phase="profile_reflection",
        )

        # 4) Apply updates
        self._apply_profile_updates(obj.get("humans", {}), current_profiles)

    def _apply_profile_updates(self, updates: dict, current_profiles: dict):
        """
        Merge trait updates into DB / HDT.

        Supports both:

          A) Structured:
             {
               "summary_delta": "...",
               "trait_updates": {
                 "leadership_preference": {"new_value": "emerging", "confidence_delta": 0.3},
                 ...
               }
             }

          B) Flat:
             {
               "leadership_preference": "emerging",
               "trust_in_robot_plans": "low",
               ...
             }
        """
        now = time.time()

        for human_id, upd in (updates or {}).items():
            if not isinstance(upd, dict):
                continue

            # 1) Normalize into trait_updates
            trait_updates = upd.get("trait_updates")
            if not isinstance(trait_updates, dict):
                trait_updates = {}
                for key, val in upd.items():
                    if key in ("summary_delta", "trait_updates"):
                        continue
                    trait_updates[key] = {
                        "new_value": val,
                        "confidence_delta": 0.0,
                    }

            # 2) Get or init current profile
            cur = current_profiles.get(human_id) or {
                "id": human_id,
                "summary": "",
                "traits": {},
                "last_updated_ts": now,
            }
            traits = cur.setdefault("traits", {})

            # 3) Apply trait updates
            for trait_name, tinfo in (trait_updates or {}).items():
                if not isinstance(tinfo, dict):
                    tinfo = {"new_value": tinfo}
                new_val = tinfo.get("new_value")
                if new_val is None:
                    continue
                traits[trait_name] = new_val

            cur["last_updated_ts"] = now
            self._save_human_profile(human_id, cur)

    def _load_disposal_outcomes_for_session(self, session: MediationState) -> list[DisposalOutcome]:
        """
        Hook: load realized disposal outcomes that are linked to this mediation
        session (or recent time window).

        Implementation depends on how you log disposals in your DB.
        For now, stub to an empty list or replace with a real query.
        """
        outcomes: list[DisposalOutcome] = []

        try:
            # Example if you had a disposal_outcomes table with planner_id, session_id:
            rows = self.conn.execute(
                """
                SELECT agent_id, box_id, prop, completed_at, success, correct, planner_id
                FROM disposal_outcomes
                WHERE session_id = ?
                """,
                (session.session_id,),
            ).fetchall()

            from .optimizer_client import DisposalOutcome  # avoid circulars at top

            for (agent_id, box_id, prop, completed_at, success, correct, planner_id) in rows:
                outcomes.append(
                    DisposalOutcome(
                        agent_id=str(agent_id),
                        box_id=int(box_id),
                        prop=str(prop),
                        completed_at=float(completed_at) if completed_at is not None else None,
                        success=bool(success),
                        correct=None if correct is None else bool(correct),
                        planner_id=str(planner_id) if planner_id is not None else None,
                    )
                )
        except Exception:
            # If table doesn't exist yet, just return empty list
            return []

        return outcomes


    def _load_current_human_profiles(self) -> dict:
        """
        TODO: replace with real DB/HDT integration.
        For now, return an empty dict or read from a profiles table if you add one.
        """
        try:
            rows = self.conn.execute(
                "SELECT human_id, json_blob FROM human_profiles"
            ).fetchall()
        except sqlite3.Error:
            # table not present → simple fallback
            return {}

        profiles = {}
        for human_id, blob in rows:
            try:
                profiles[human_id] = json.loads(blob)
            except Exception:
                continue
        return profiles

    def _save_human_profile(self, human_id: str, profile: dict):
        """
        TODO: replace with real DB/HDT integration.
        """
        try:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS human_profiles (
                    human_id TEXT PRIMARY KEY,
                    json_blob TEXT NOT NULL
                );
                """
            )
            self.conn.execute(
                """
                INSERT INTO human_profiles(human_id, json_blob)
                VALUES (?, ?)
                ON CONFLICT(human_id) DO UPDATE SET
                    json_blob = excluded.json_blob
                """,
                (human_id, json.dumps(profile)),
            )
            self.conn.commit()
        except Exception as e:
            self.get_logger().warn(f"[profiles] failed to save profile for {human_id}: {e}")

    # ---------- Speech routing into an active mediation ----------

    def _maybe_route_speech_to_mediation(self, rule: str, trace_entry: dict) -> bool:
        """
        If a mediation session is pending and this basic event looks like human speech,
        feed it as a new human turn into the PlanMediator.

        This now supports:
          - speech_final_any*  (ASR final hypotheses)
          - speech_intent_inferred (LLM intent / plan results), which we treat
            as a speech turn when mediation is already in progress, so that
            close-in-time utterances from other humans are not lost.
        """
        if not self._mediation_in_progress():
            return False

        # 1) Decide whether this rule should count as speech
        is_speech_rule = False

        # Normal speech pipeline: speech_final_any*
        if rule.startswith("speech_final_any"):
            is_speech_rule = True

        # NEW: while mediation is pending, also treat speech_intent_inferred
        # as a speech turn (using request_text / natural_summary).
        elif rule == "speech_intent_inferred":
            is_speech_rule = True

        if not is_speech_rule:
            return False

        data = trace_entry.get("data") or {}

        # 2) Extract text depending on the rule type
        if rule == "speech_intent_inferred":
            # For intent events, the JSON text itself is in `json_text`/`text`,
            # but the *original* human utterance is in `request_text`.
            text = (
                (data.get("request_text") or "")
                or (data.get("natural_summary") or "")
            )
        else:
            # For speech_final_any*, EventLayer should be putting the human
            # transcript in `text` or `utterance`.
            text = (data.get("text") or data.get("utterance") or "")

        text = text.strip()
        if not text:
            return False

        speaker_id = data.get("speaker_id") or "unknown"
        ts = trace_entry.get("ts") or time.time()

        sid = self._active_mediation_id
        sess = self._mediation_sessions.get(sid)
        if not sid or not sess:
            return False

        human_turn = MediationTurn(
            role=speaker_id,
            text=text,
            meta={"ts": ts, "rule": rule},
        )

        # reset the “wait for next utterance” timer
        self._bump_pending_deadline(ts)


        # Step the mediator with this new human turn
        sess, raw = self._plan_mediator.step(sess, new_human_turn=human_turn)
        
        # Hard cap: if too many steps/turns, resolve
        turns_used = int(getattr(sess, "turns_used", 0) or 0)
        if turns_used >= int(self.mediation_max_turns):
            self.get_logger().warn(
                f"[mediation] max turns reached ({turns_used}); auto-resolving session {sid}"
            )
            self._auto_resolve_pending_session(sid, sess, reason="max_turns", now_ts=ts)
            return True

        
        self._mediation_sessions[sid] = sess

        robot_utt = (raw.get("robot_utterance") or "").strip()
        if robot_utt:
            self.get_logger().info(f"[mediation] Bob says: {robot_utt}")
            self._robot_say(robot_utt)

        # If the mediator has reached a decision, finalize the session
        if sess.status in ("accept", "reject"):
            self._finalize_mediation_session(
                session_id=sid,
                session=sess,
                raw_decision=raw,
                ts=ts,
            )

        return True


    def _mediation_watchdog_tick(self):
        if not self._mediation_in_progress():
            return

        sid = getattr(self, "_active_mediation_id", None)
        if not sid:
            return

        sess = self._mediation_sessions.get(sid)
        if not sess or sess.status != "pending":
            return

        now_ts = time.time()

        # Turn-wait timeout (no new utterance)
        dl = getattr(self, "_mediation_pending_deadline_ts", None)
        if dl is not None and now_ts >= float(dl):
            self.get_logger().warn(
                f"[mediation] turn timeout hit (no utterance); auto-resolving {sid}"
            )
            self._auto_resolve_pending_session(sid, sess, reason="turn_timeout", now_ts=now_ts)
            return

    def _normalize_mediation_decision(self, raw: dict) -> dict:
        if not isinstance(raw, dict):
            return raw
        decision = (raw.get("decision") or "").strip()
        kind = ((raw.get("planner_action") or {}).get("kind") or "").strip()

        allowed_decisions = {"pending", "accept", "reject"}
        if decision in allowed_decisions:
            return raw

        # If model put kind into decision, interpret as accept
        allowed_kinds = {"keep_baseline", "adopt_candidate", "merge_plans", "request_new_plan"}
        if decision in allowed_kinds and not kind:
            raw.setdefault("planner_action", {})["kind"] = decision

        # If decision matches a known kind (even if kind is set), coerce to accept/reject
        if decision in allowed_kinds:
            # keep_baseline is effectively rejecting candidate; others accept some change
            raw["decision"] = "reject" if decision == "keep_baseline" else "accept"
            return raw

        # Fallback: map by kind if decision is garbage
        if kind == "keep_baseline":
            raw["decision"] = "reject"
        elif kind in ("adopt_candidate", "merge_plans", "request_new_plan"):
            raw["decision"] = "accept"
        else:
            raw["decision"] = "reject"
            raw["planner_action"] = {"kind": "keep_baseline", "notes": "normalized from invalid decision"}
        return raw


    def _auto_resolve_pending_session(self, session_id: str, sess: MediationState, reason: str, now_ts: float):
        """
        Called by watchdog when per-turn timeout fires or max_turns reached.
        Uses LLM to produce a compromise/final action and then finalizes.

        IMPORTANT: prompt must be the SAME as normal mediation, with only small diffs,
        so we delegate message construction to PlanMediator.
        """
        # Defensive: only act if still pending and still active
        if not sess or sess.status != "pending":
            return
        if getattr(self, "_active_mediation_id", None) != session_id:
            return

        try:
            # 1) Build messages using SAME mediator prompt template (only a few diffs inside)
            #    You add build_messages_for_autoresolve(...) to PlanMediator (as discussed).
            messages = self._plan_mediator.build_messages_for_autoresolve(
                sess,
                reason=reason,
            )

            # 2) Call the SAME LLM wrapper/schema as normal mediation
            raw = self._mediate_llm_call(messages)

        except Exception as e:
            self.get_logger().warn(f"[mediation] autoresolve LLM failed: {e}; falling back to keep_baseline")
            raw = {
                "decision": "reject",
                "planner_action": {"kind": "keep_baseline", "notes": f"fallback due to autoresolve failure: {e}"},
                "robot_utterance": "No response in time—I'll keep the current plan for now.",
                "log_tags": {"strategy": "timeout_fallback", "rationale": "LLM autoresolve failed"},
            }

        # 3) Normalize decision if the model returns a kind
        raw = self._normalize_mediation_decision(raw)

        # 3b) HARDEN: autoresolve must finalize (never remain pending)
        if (raw.get("decision") or "").strip() == "pending":
            raw["decision"] = "reject"
            raw["planner_action"] = {"kind": "keep_baseline", "notes": "autoresolve forced non-pending"}
            raw["robot_utterance"] = "No response in time—I'll keep the current plan for now."
            raw["log_tags"] = {"strategy": "timeout_forced", "rationale": "autoresolve must finalize"}

        # 4) Mark session terminal so _finalize_mediation_session publishes/unblocks
        sess.status = "accept" if raw.get("decision") == "accept" else "reject"

        # 5) Speak, then finalize
        robot_utt = (raw.get("robot_utterance") or "").strip()
        if robot_utt:
            self._robot_say(robot_utt)

        self._finalize_mediation_session(session_id=session_id, session=sess, raw_decision=raw, ts=now_ts)

    # ---------- Social context metrics for a proposer ----------

    def _compute_social_context_for_proposer(
        self,
        proposer_id: str,
        window_sec: float = 300.0,
    ) -> tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
        """
        Compute:
          - proposer_success_rate: fraction of this proposer's proposals adopted
          - conflict_index: 0..1 summary of how conflictual recent planning is
          - override_frequency: 'low' | 'medium' | 'high' based on cross-overrides
          - leadership_contestation: 'none' | 'emerging' | 'strong'

        Uses the same plan_proposals table as _compute_plan_conflict_metrics.
        """
        # Start from conflict metrics over a time window
        metrics = self._compute_plan_conflict_metrics(window_sec=window_sec)

        humans = ("human_a", "human_b")
        total_proposals = metrics.get("total_proposals", {})
        adopted_proposals = metrics.get("adopted_proposals", {})
        cross_overrides = metrics.get("cross_overrides", {})
        conflict_episodes = metrics.get("conflict_episodes", 0) or 0
        tug_of_war_episodes = metrics.get("tug_of_war_episodes", 0) or 0

        # --- 1) proposer_success_rate ---
        tot = float(total_proposals.get(proposer_id, 0) or 0)
        adopted = float(adopted_proposals.get(proposer_id, 0) or 0)
        proposer_success_rate: Optional[float]
        if tot > 0.0:
            proposer_success_rate = adopted / tot
        else:
            proposer_success_rate = None

        # --- 2) conflict_index (0..1) ---
        # Normalize conflict intensity by how many human proposals we’ve seen.
        total_human_props = sum(total_proposals.get(h, 0) or 0 for h in humans)
        denom = max(1.0, float(total_human_props))
        # Tug-of-war counts as “stronger” conflict than a plain alternation.
        conflict_raw = float(conflict_episodes) + 0.5 * float(tug_of_war_episodes)
        conflict_index = min(1.0, conflict_raw / denom)

        # If proposer is not a human, we treat conflict_index as global but still valid.
        # You can set it to None for non-humans if you prefer; leaving it as-is is fine.

        # --- 3) override_frequency label for this proposer ---
        overrides = int(cross_overrides.get(proposer_id, 0) or 0)
        if overrides == 0:
            override_frequency = "low"
        elif overrides <= 2:
            override_frequency = "medium"
        else:
            override_frequency = "high"

        # --- 4) leadership_contestation (global human-vs-human signal) ---
        # Look at adoption rates and conflict episodes between human_a/human_b.
        adoption_rate = metrics.get("adoption_rate", {})
        ar_a = adoption_rate.get("human_a")
        ar_b = adoption_rate.get("human_b")

        # Default: no visible contestation
        leadership_contestation = "none"

        # If we have both adoption rates:
        if ar_a is not None and ar_b is not None:
            diff = abs(ar_a - ar_b)
            # Lots of conflict events + similar adoption rates → strong contestation
            if conflict_index > 0.3 and diff < 0.15:
                leadership_contestation = "strong"
            # Some conflict or overrides, but not as intense → emerging
            elif conflict_index > 0.1 or (cross_overrides.get("human_a", 0) or 0) > 0 or (cross_overrides.get("human_b", 0) or 0) > 0:
                leadership_contestation = "emerging"
            else:
                leadership_contestation = "none"
        else:
            # If we lack good stats, downweight to 'emerging' only if conflict is high.
            if conflict_index > 0.3:
                leadership_contestation = "emerging"

        return proposer_success_rate, conflict_index, override_frequency, leadership_contestation


    # ---------- Conflict metrics over plan_proposals ----------

    def _compute_plan_conflict_metrics(self, window_sec: float = 60.0) -> dict:
        """
        Compute simple, objective metrics about planning conflicts between humans,
        based on the plan_proposals table.

        Interpretation:
          - Each row in plan_proposals is a *proposal* (usually from LLM speech).
          - proposer_id: 'human_a', 'human_b', 'unknown', ...
          - adopted: 1 if this proposal was actually adopted into the current plan.

        Metrics:
          - total_proposals[agent]
          - adopted_proposals[agent]
          - adoption_rate[agent]
          - cross_overrides[agent]:
                how many times this agent's adopted proposal replaced the other
                human's last adopted proposal.
          - conflict_episodes:
                consecutive proposals within `window_sec` where proposer_id flips
                between human_a and human_b.
          - tug_of_war_episodes:
                same as conflict_episodes, but where both proposals are marked
                better_than_current=1 (both trying to "improve" the plan).
        """
        cur = self.conn.cursor()
        try:
            rows = cur.execute(
                """
                SELECT ts, proposer_id, adopted, better_than_current, source
                FROM plan_proposals
                WHERE source = 'llm_speech'
                ORDER BY ts ASC
                """
            ).fetchall()
        except Exception as e:
            self.get_logger().warn(f"[plan-metrics] failed to read plan_proposals: {e}")
            cur.close()
            return {}

        cur.close()

        humans = ("human_a", "human_b")

        total_proposals = {h: 0 for h in humans}
        adopted_proposals = {h: 0 for h in humans}
        cross_overrides = {h: 0 for h in humans}

        last_adopted_proposer: Optional[str] = None

        # First pass: per-human proposal & override stats
        for ts, proposer_id, adopted, better_than_current, source in rows:
            pid = (proposer_id or "unknown").strip()

            # Count only human_a/human_b for these stats
            if pid in humans:
                total_proposals[pid] += 1

                if int(adopted or 0) == 1:
                    adopted_proposals[pid] += 1

                    # If previous adopted plan was from the *other* human,
                    # this is a cross-human override.
                    if last_adopted_proposer in humans and last_adopted_proposer != pid:
                        cross_overrides[pid] += 1

                    last_adopted_proposer = pid

        adoption_rate = {}
        for h in humans:
            if total_proposals[h] > 0:
                adoption_rate[h] = adopted_proposals[h] / float(total_proposals[h])
            else:
                adoption_rate[h] = None

        # Second pass: temporal conflict episodes (A vs B within a short time window)
        conflict_episodes = 0
        tug_of_war_episodes = 0

        # We treat only proposals from humans; ignore 'unknown' or robot/optimizer.
        filtered = [
            (ts, (proposer_id or "unknown").strip(), int(adopted or 0), int(better_than_current or 0))
            for (ts, proposer_id, adopted, better_than_current, source) in rows
            if (proposer_id or "unknown").strip() in humans
        ]

        for i in range(1, len(filtered)):
            ts_i, pid_i, adopted_i, better_i = filtered[i]
            ts_prev, pid_prev, adopted_prev, better_prev = filtered[i - 1]

            # A conflict episode = immediate alternation human_a ↔ human_b within window
            if pid_i != pid_prev and abs(ts_i - ts_prev) <= window_sec:
                conflict_episodes += 1

                # tug-of-war: both proposals claim to be better_than_current
                if better_i == 1 and better_prev == 1:
                    tug_of_war_episodes += 1

        metrics = {
            "total_proposals": total_proposals,
            "adopted_proposals": adopted_proposals,
            "adoption_rate": adoption_rate,
            "cross_overrides": cross_overrides,
            "conflict_episodes": conflict_episodes,
            "tug_of_war_episodes": tug_of_war_episodes,
            "window_sec": window_sec,
        }
        return metrics

    def _srv_plan_conflict_metrics(self, request: Trigger.Request, context) -> Trigger.Response:
        """
        Trigger service that returns current human planning conflict metrics
        as a JSON string in 'message'. 'success' is always True unless we hit an error.
        """
        resp = Trigger.Response()
        try:
            metrics = self._compute_plan_conflict_metrics(window_sec=60.0)
            resp.success = True
            resp.message = json.dumps(metrics, ensure_ascii=False)
        except Exception as e:
            self.get_logger().warn(f"[plan-metrics] service failed: {e}")
            resp.success = False
            resp.message = f"error: {e}"
        return resp

