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

from .optimizer_client import (
    Plan,
    build_plan_from_llm_agents_plan,
    evaluate_candidate_plan,
    extend_plan_with_prefix,
    PlannerWeights,
    estimate_deadline_risk_for_plan,
    compute_xy_imbalance_for_plan,
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
                    "enum": ["pending", "accept", "reject"],
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

        Returns a short human-readable note, or None if everything is OK.
        """
        if not plan:
            return None

        # boxes is a List[BoxInfo]
        box_by_id = {b.box_id: b for b in boxes or []}
        lines: List[str] = []

        for aid, actions in (plan or {}).items():
            for (box_id, prop, kind) in actions:
                reasons = []
                b = box_by_id.get(box_id)

                if b is None:
                    reasons.append("unknown box")
                else:
                    if kind == "dispose":
                        # already disposed?
                        if prop == "X" and b.disposed_X:
                            reasons.append("already disposed for X")
                        if prop == "Y" and b.disposed_Y:
                            reasons.append("already disposed for Y")

                        # deadline passed?
                        if b.deadline is not None and current_time >= float(b.deadline):
                            reasons.append("deadline already passed")

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

        # 3) Build a PREFIX plan from the LLM (robot-only for now)
        try:
            prefix_plan = build_plan_from_llm_agents_plan(
                agents_plan,
                allowed_agents=["robot"],
            )
        except Exception as e:
            self.get_logger().warn(f"[llm-plan] build_plan_from_llm_agents_plan failed: {e}")
            return

        if not prefix_plan:
            self.get_logger().info("[llm-plan] no valid robot actions in prefix; ignoring")
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

        objective = MediationObjectiveMetrics(
            suboptimality_pct=suboptimal_pct,
            baseline_score=score_optimal,
            candidate_score=score_candidate,
            deadline_risk=deadline_risk,   
            imbalance_XY=imbalance_XY,           # TODO: compute from score components if needed
            fulfillment_history_ok=True, # or from robot_exec_evals
            notes=impossible_note,       # <--- NEW: surface impossible actions here
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


        # Seed session_notes with impossible-action note if present
        base_notes = "Initial mediation triggered by LLM speech plan."
        if impossible_note:
            base_notes = base_notes + "\n\n" + impossible_note

        interaction = MediationInteractionContext(
            event_summary=self._event_summary_text,
            robot_role_description=(
                "Bob is a cooperative teammate that balances safety, efficiency, and human preferences."
            ),
            session_notes=base_notes,
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

            # Also seed interaction.recent_utterances
            interaction.recent_utterances = list(initial_turns)


        state = MediationState(
            session_id=session_id,
            baseline_plan=current_plan,
            candidate_plan=candidate_plan,
            objective=objective,
            social=social,
            interaction=interaction,
            turns=initial_turns,
        )

        self._mediation_sessions[session_id] = state
        self._active_mediation_id = session_id

        # 7) Optionally do a first mediation step (robot explains / asks questions)
        state, raw = self._plan_mediator.step(state)
        self._mediation_sessions[session_id] = state

        # --- NEW: if mediation is now pending, block llm_speech_plan in EventLayer ---
        if state.status == "pending":
            self._publish_mediation_status(session_id, "pending")

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

        - Applies the planner_action (adopt / keep baseline).
        - Publishes the resulting plan exactly once.
        - Triggers a reflection pass to update human profiles.
        - Clears the active mediation flag so future plans can start.
        """
        try:
            pa = (raw_decision or {}).get("planner_action") or {}
            kind = pa.get("kind")

            # Decide which plan to keep
            new_plan = session.baseline_plan
            if kind in ("adopt_candidate", "merge_plans"):
                new_plan = session.candidate_plan

            # Publish plan (if any) once mediation is over
            if new_plan:
                box_positions = getattr(self, "_last_box_positions", {}) or {}
                current_time = getattr(self, "_last_server_time", None) or time.time()
                self._last_plan = new_plan
                self._publish_optimizer_plan(new_plan, current_time, box_positions)
                self.get_logger().info(
                    f"[mediation] finalized session {session_id} with planner_action={kind}"
                )
            else:
                self.get_logger().info(
                    f"[mediation] session {session_id} finalized with no plan change (kind={kind})"
                )

            # Reflection: update human profiles based on the whole dialogue
            try:
                self._reflect_on_mediation_session(session)
            except Exception as e:
                self.get_logger().warn(f"[mediation] reflection failed: {e}")

        except Exception as e:
            self.get_logger().warn(f"[mediation] finalize session failed: {e}")
        finally:
            # Conversation is over → allow new speech-plan proposals again
            self._active_mediation_id = None

            # Tell EventLayer that mediation is no longer pending
            final_status = session.status or "idle"
            if final_status not in ("accept", "reject", "cancelled"):
                final_status = "idle"

            try:
                self._publish_mediation_status(session_id, final_status)
            except Exception as e:
                self.get_logger().warn(f"[mediation] failed to publish final status: {e}")
            # keep self._mediation_sessions[session_id] for logging/analysis

    # ---------- Post-hoc reflection / profile updates ----------

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

        Returns True if the event was consumed by mediation (and callers may early-return).
        """
        if not self._mediation_in_progress():
            return False

        # Heuristic: only treat certain rules as speech.
        if not rule.startswith("speech_final_any"):
            return False

        data = trace_entry.get("data") or {}
        text = (data.get("text") or data.get("utterance") or "").strip()
        if not text:
            return False

        speaker_id = data.get("speaker_id") or "unknown"
        ts = trace_entry.get("ts") or time.time()

        sid = self._active_mediation_id
        sess = self._mediation_sessions.get(sid)
        if not sess:
            return False

        human_turn = MediationTurn(
            role=speaker_id,
            text=text,
            meta={"ts": ts, "rule": rule},
        )

        # Step the mediator with this new human turn
        sess, raw = self._plan_mediator.step(sess, new_human_turn=human_turn)
        self._mediation_sessions[sid] = sess

        robot_utt = (raw.get("robot_utterance") or "").strip()
        if robot_utt:
            # TODO: publish this to your InteractionLoop / TTS instead of only logging
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

