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
#TODO consider what happens if in the middle of an action??

from __future__ import annotations

import json
import time
import sqlite3
from typing import Dict, Any, Optional, List, Tuple

from std_msgs.msg import String as StringMsg
from std_srvs.srv import Trigger

from jsonschema import validate, ValidationError  # if you need it for your own schemas

from std_msgs.msg import String as StringMsg

from collections import defaultdict, deque

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
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "summary_delta": {"type": "string"},
                    "trait_updates": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "required": ["new_value"],
                            "properties": {
                                "new_value": {},
                                # Qualitative strength instead of numeric deltas
                                "evidence_level": {
                                    "type": "string",
                                    "enum": ["very_low", "low", "medium", "high", "very_high"],
                                },
                                "evidence_notes": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": True,
            },
        }
    },
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

        # Single source of truth for "what we're actually doing"
        self._committed_plan = {}               # type: Plan
        self._committed_plan_provenance = {}    # type: Optional[dict]


        # ---- Mediation liveness controls ----
        self.mediation_turn_timeout_sec = float(getattr(self, "mediation_turn_timeout_sec", 60.0))
        self.mediation_max_turns = int(getattr(self, "mediation_max_turns", 3))   # human+robot steps combined


        # ---- Mediation single-flight (per session across INIT / MEDIATION / AUTORESOLVE) ----
        self._mediation_llm_busy = {}   # session_id -> bool
        self._mediation_llm_phase = {}  # session_id -> "init"|"turn"|"autoresolve"|None

        self._baseline_human_agreed_plan = {}  # type: Plan
        self._baseline_human_agreed_provenance = {}  # type: Optional[dict]

        self._mediation_pending_deadline_ts: Optional[float] = None
        self._mediation_pending_started_ts: Optional[float] = None

        self._mediation_watchdog_timer = None  # rclpy Timer
        
        # ---- LLM buffering / single-flight ----
        # Buffer buckets: (channel, key) -> deque[event_dict]
        # event_dict is: {"ts": float, "type": "...", "payload": {...}}
        self._llm_buf = defaultdict(lambda: deque(maxlen=1000))
        self._llm_inflight = defaultdict(bool)
        self._llm_lock = threading.Lock()

        # Optional: debounce to coalesce bursts (seconds)
        self._llm_debounce_sec = {
            "MEDIATION": 0.15,   # short: natural speech bursts
            "MEDIATION_INIT": 0.05,
            "CHAT": 0.20,
            "REFLECTION": 1.00,  # slow: let a few messages accumulate
            "MEDIATION_AUTORESOLVE": 0.00,
        }

        # ---- Debounce without thread timers ----
        # Per-bucket "do not run before this timestamp"
        self._llm_next_run_ts = {}  # (channel,key) -> float epoch seconds

        # Dispatcher timer: checks buckets and starts workers when due.
        self._llm_dispatch_period_sec = float(getattr(self, "llm_dispatch_period_sec", 0.05))
        self._llm_dispatch_timer = self.create_timer(
            self._llm_dispatch_period_sec, self._llm_dispatch_tick
        )

        # NOTE: We no longer use thread-based debounce timers
        self._llm_debounce_timer = {}  # can be removed; kept only if referenced elsewhere

        
    
        
        # ---- Always-accept override tracking ----
        # Remember the last committed robot first-action and who requested it,
        # so we can announce overrides when another human issues a new request.
        self._last_always_accept_commit = None  # type: Optional[dict]
        # How soon two requests count as "consecutive" (seconds)
        self.always_accept_override_window_sec = float(
            getattr(self, "always_accept_override_window_sec", 25.0)
        )


    def _action_sig(self, action: Tuple[int, str, str]) -> Tuple[str, int, str]:
        # action is (box_id, prop, kind)
        (box_id, prop, kind) = action
        return (str(kind).lower(), int(box_id), str(prop).upper())

    def _plan_robot_sig_set(self, plan: Plan, limit: int = 8) -> set:
        s = set()
        ra = (plan or {}).get("robot") or []
        for a in ra[: max(0, int(limit))]:
            try:
                s.add(self._action_sig(a))
            except Exception:
                pass
        return s

    def _canon_action(self, a) -> Optional[Tuple[int, str, str]]:
        """
        Normalize an action tuple to (box_id:int, prop:upper, kind:lower).
        """
        try:
            (box_id, prop, kind) = a
            return (int(box_id), str(prop).upper(), str(kind).lower())
        except Exception:
            return None


    def _agent_actions(self, plan: Plan, aid: str, limit: int = 50) -> List[Tuple[int, str, str]]:
        acts = (plan or {}).get(aid) or []
        out = []
        for a in acts[: max(0, int(limit))]:
            ca = self._canon_action(a)
            if ca is not None:
                out.append(ca)
        return out


    def _subtract_committed_prefix_in_order_all_agents(
        self,
        requested: Plan,
        committed: Plan,
        agents: Optional[List[str]] = None,
        lookahead: int = 12,
    ) -> Plan:
        """
        For each agent in `agents`, remove the longest in-order prefix of requested[agent]
        that already matches committed[agent] prefix.

        Returns a new Plan containing ONLY the remaining (true) changes.
        """
        if not isinstance(requested, dict) or not requested:
            return {}

        if agents is None:
            # default to all keys seen in either requested or committed
            agents = sorted(set(list(requested.keys()) + list((committed or {}).keys())))

        out: Plan = {}

        for aid in agents:
            req = self._agent_actions(requested, aid, limit=lookahead)
            if not req:
                continue

            com = self._agent_actions(committed, aid, limit=lookahead)

            k = 0
            while k < len(req) and k < len(com) and req[k] == com[k]:
                k += 1

            remaining = req[k:]
            if remaining:
                out[str(aid)] = [(box_id, prop, kind) for (box_id, prop, kind) in remaining]

        return out


    def _plan_has_any_actions(self, plan: Plan) -> bool:
        if not isinstance(plan, dict) or not plan:
            return False
        for aid, acts in plan.items():
            if acts:
                return True
        return False


    def _is_prefix_plan_already_committed_in_order_all_agents(
        self,
        requested: Plan,
        committed: Plan,
        agents: Optional[List[str]] = None,
        lookahead: int = 12,
    ) -> bool:
        """
        True if requested[aid] is an in-order prefix of committed[aid] for all agents with requests.
        """
        if not isinstance(requested, dict) or not requested:
            return False

        if agents is None:
            agents = list(requested.keys())

        for aid in agents:
            req = self._agent_actions(requested, aid, limit=lookahead)
            if not req:
                continue
            com = self._agent_actions(committed, aid, limit=lookahead)
            if len(com) < len(req):
                return False
            if com[:len(req)] != req:
                return False

        return True


    def _confirm_already_committed(self, proposer_id: str, proposed: Plan) -> None:
        """
        Quick acknowledgement (no mediation).
        """
        ra = (proposed or {}).get("robot") or []
        target = self._to_display_text(proposer_id)
        if ra:
            try:
                (box_id, prop, kind) = ra[0]
                self._robot_say(f"Hey {target}, yes — I'm already doing that: I'll {kind} box {box_id} ({prop}).")
                return
            except Exception:
                pass
        self._robot_say(f"Hey {target}, yes — that plan is already in progress.")


    def _llm_dispatch_tick(self):
        """
        Single periodic dispatcher (ROS timer) that starts bucket workers when:
          - bucket has buffered events
          - bucket is not already inflight
          - debounce window has elapsed (now >= next_run_ts)
        This replaces threading.Timer-based debounce and prevents thread explosion.
        """
        now = time.time()
        to_start: List[Tuple[str, str]] = []

        with self._llm_lock:
            # Snapshot candidate buckets quickly under lock
            for (channel, key), q in list(self._llm_buf.items()):
                if not q:
                    continue

                b = (channel, key)

                if self._llm_inflight.get(b, False):
                    continue

                next_ts = float(self._llm_next_run_ts.get(b, 0.0) or 0.0)
                if now < next_ts:
                    continue

                # Eligible to start
                to_start.append((channel, key))

        # Start outside lock
        for (channel, key) in to_start:
            try:
                self._start_llm_worker(channel, key)
            except Exception:
                # Don't let dispatcher crash
                pass


    def _try_acquire_mediation_session_llm(self, session_id: str, phase: str) -> bool:
        """
        Session-global single-flight across mediation channels (INIT / TURN / AUTORESOLVE).
        Returns True if acquired, False if another mediation phase is running.
        """
        sid = str(session_id)
        with self._llm_lock:
            if self._mediation_llm_busy.get(sid, False):
                return False
            self._mediation_llm_busy[sid] = True
            self._mediation_llm_phase[sid] = str(phase)
            return True

    def _release_mediation_session_llm(self, session_id: str):
        sid = str(session_id)
        with self._llm_lock:
            self._mediation_llm_busy[sid] = False
            self._mediation_llm_phase[sid] = None

    def _mediation_session_llm_phase(self, session_id: str) -> Optional[str]:
        sid = str(session_id)
        with self._llm_lock:
            return self._mediation_llm_phase.get(sid)


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



    def _llm_bucket(self, channel: str, key: str) -> tuple:
        return (str(channel), str(key))

    def _llm_submit(self, channel: str, key: str, ev_type: str, payload: dict, ts: float):
        """
        Buffer an event into (channel,key) bucket and schedule execution via dispatcher timer.

        Debounce behavior:
          - next_run_ts is pushed to now + debounce_sec on every submit
          - worker will only start when now >= next_run_ts
        """
        self.get_logger().info(f'submit {channel}')
        
        b = self._llm_bucket(channel, key)
        now = time.time()

        start_now = False

        with self._llm_lock:
            self._llm_buf[b].append({"ts": float(ts), "type": ev_type, "payload": payload})

            delay = float(self._llm_debounce_sec.get(channel, 0.0) or 0.0)

            # Push next run time forward (debounce extends on every new event)
            self._llm_next_run_ts[b] = now + delay

            # If not inflight and delay==0, we can start immediately (outside lock)
            if (not self._llm_inflight.get(b, False)) and delay <= 0.0:
                start_now = True

        if start_now:
            self._start_llm_worker(channel, key)

    def _should_promote_mediation_to_init(self, session_id: str) -> bool:
        sess = self._mediation_sessions.get(session_id)
        active_sid = getattr(self, "_active_mediation_id", None)

        # no session or not the active one
        if not sess or active_sid != session_id:
            return True

        st = getattr(sess, "status", None)
        if st in ("accept", "reject"):
            return True


        # otherwise keep as normal mediation turn
        return False


    def _start_llm_worker(self, channel: str, key: str):
        """
        Start ONE worker thread for this bucket (if not already inflight).
        Worker drains in a LOOP until empty or until debounce says "wait".
        """
        b = self._llm_bucket(channel, key)
        now = time.time()

        with self._llm_lock:
            # Already running or nothing to do
            if self._llm_inflight.get(b, False):
                return
            if not self._llm_buf.get(b):
                return

            # Respect debounce gate
            next_ts = float(self._llm_next_run_ts.get(b, 0.0) or 0.0)
            if now < next_ts:
                return

            # Mark inflight: one worker per bucket
            self._llm_inflight[b] = True

        def _worker_loop():
            try:
                while True:
                    # 1) Check debounce and drain a batch
                    with self._llm_lock:
                        # If bucket emptied, we're done
                        if not self._llm_buf.get(b):
                            break

                        # If new events arrived and debounce window hasn't elapsed, stop.
                        next_ts2 = float(self._llm_next_run_ts.get(b, 0.0) or 0.0)
                        if time.time() < next_ts2:
                            break

                        events = list(self._llm_buf[b])
                        self._llm_buf[b].clear()

                    # 2) Process drained batch (outside lock)
                    try:
                        if channel == "MEDIATION":
                            if self._should_promote_mediation_to_init(key):
                                # re-enqueue as mediation_init events
                                for ev in events:
                                    self._llm_submit(
                                        channel="MEDIATION_INIT",
                                        key=key,
                                        ev_type=ev.get("type") or "turn",
                                        payload=ev.get("payload") or {},
                                        ts=ev.get("ts") or time.time(),
                                    )
                                continue  # do not process as MEDIATION
                            self._process_mediation_events(key, events)

                        elif channel == "CHAT":
                            self._process_chat_events(key, events)
                        elif channel == "REFLECTION":
                            self._process_reflection_events(key, events)
                        elif channel == "MEDIATION_INIT":
                            self._process_mediation_init_events(key, events)
                        elif channel == "MEDIATION_AUTORESOLVE":
                            self._process_mediation_autoresolve_events(key, events)
                        else:
                            pass
                    except Exception as e:
                        try:
                            self.get_logger().warn(f"[llm] worker for {channel}:{key} failed: {e}")
                        except Exception:
                            pass


                    # Loop continues: if more events arrived and debounce is satisfied,
                    # we will drain again in the same thread.

            finally:
                # Clear inflight flag; dispatcher will restart if there is leftover work
                with self._llm_lock:
                    self._llm_inflight[b] = False

        threading.Thread(target=_worker_loop, daemon=True).start()



    def _get_human_turns_with_prev_context(
        self,
        human_id: str,
        limit: int = 50,
        ctx_prev: int = 1,
    ) -> List[dict]:
        """
        Returns a chronologically-ordered list of chat turns drawn from the global history,
        where every turn spoken by `human_id` is included, PLUS up to `ctx_prev` turns
        immediately before it (any role) as context.

        Uniqueness is enforced by index in the filtered window (not by text),
        so duplicates aren't accidentally dropped if the same text appears twice.
        """
        hist = list(getattr(self, "_chat_history", []))
        if limit is not None and limit > 0:
            hist = hist[-limit:]

        if not hist or not isinstance(human_id, str):
            return []

        include_idx = set()

        for i, t in enumerate(hist):
            if t.get("role") != human_id:
                continue

            # include this human turn
            include_idx.add(i)

            # include up to ctx_prev previous turns for context
            if ctx_prev and ctx_prev > 0:
                start = max(0, i - int(ctx_prev))
                for j in range(start, i):
                    include_idx.add(j)

        # return in original chronological order
        out = []
        for i in sorted(include_idx):
            try:
                out.append(hist[i])
            except Exception:
                pass
        return out


    def _get_recent_chat_turns(self, limit: int = 8) -> List[dict]:
        hist = list(getattr(self, "_chat_history", []))
        if limit is not None and limit > 0:
            hist = hist[-limit:]
        return hist

    def _level_3(self, x: Optional[float]) -> Optional[str]:
        """
        Map a numeric value to low/medium/high.
        If x is None -> None.
        """
        try:
            if x is None:
                return None
            v = float(x)
        except Exception:
            return None

        # Generic buckets; tune if you want
        if v <= 0.33:
            return "low"
        if v <= 0.66:
            return "medium"
        return "high"


    def _level_5(self, x: Optional[float]) -> Optional[str]:
        """
        Map a numeric value to very_low/low/medium/high/very_high.
        """
        try:
            if x is None:
                return None
            v = float(x)
        except Exception:
            return None

        if v <= 0.20:
            return "very_low"
        if v <= 0.40:
            return "low"
        if v <= 0.60:
            return "medium"
        if v <= 0.80:
            return "high"
        return "very_high"


    def _pct_to_level(self, pct: Optional[float]) -> Optional[str]:
        """
        For percentages like suboptimality_pct (0..100).
        """
        try:
            if pct is None:
                return None
            v = float(pct)
        except Exception:
            return None

        # Example buckets (tune):
        # 0-5 excellent, 5-15 low, 15-35 medium, 35-60 high, >60 very_high
        if v <= 5.0:
            return "very_low"
        if v <= 15.0:
            return "low"
        if v <= 35.0:
            return "medium"
        if v <= 60.0:
            return "high"
        return "very_high"


    def _rate_to_trust_level(self, rate: Optional[float]) -> Optional[str]:
        """
        For correct_rate / success_rate style values (0..1).
        Higher rate -> higher trust.
        """
        try:
            if rate is None:
                return None
            v = float(rate)
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


    def _try_bootstrap_followon_mediation_from_init_events(
        self,
        prev_session_id: str,
        events: List[dict],
        ts_hint: float,
    ) -> bool:
        """
        If a MEDIATION_INIT bucket is being asked to run for a session that is no longer active
        (because the prior mediation accepted/rejected while in-flight), we can bootstrap
        a NEW mediation session from the latest buffered MediationTurn that includes plan_update.

        Returns True if a new session was created and scheduled, else False.
        """
        # Find the latest turn with plan_update
        last_turn = None
        for ev in reversed(events or []):
            try:
                payload = ev.get("payload") or {}
                t = payload.get("turn")
                if t is None:
                    continue
                meta = getattr(t, "meta", None) or {}
                if isinstance(meta, dict) and meta.get("plan_update"):
                    last_turn = t
                    break
            except Exception:
                continue

        if last_turn is None:
            return False

        meta = getattr(last_turn, "meta", None) or {}
        plan_update = meta.get("plan_update") if isinstance(meta, dict) else None
        if not isinstance(plan_update, dict):
            return False

        proposer_id = getattr(last_turn, "role", None) or "unknown"
        now_ts = float(meta.get("ts") or ts_hint or time.time())

        # Extract plans from plan_update
        prefix_plan = plan_update.get("prefix_plan")
        candidate_plan = plan_update.get("candidate_plan")

        if not isinstance(prefix_plan, dict) or not prefix_plan:
            return False

        # If candidate missing, try to build it (same path as normal)
        if not isinstance(candidate_plan, dict) or not candidate_plan:
            try:
                agents = self._build_agents_for_optimizer()
                boxes = getattr(self, "_last_boxes_for_optimizer", None)
                if boxes is None:
                    boxes, _ = self._build_boxes_for_optimizer(getattr(self, "_last_boxes_state", []) or [])

                current_time = getattr(self, "_last_server_time", None) or time.time()
                box_positions = getattr(self, "_last_box_positions", {}) or {}
                agent_positions = self._snapshot_agent_positions()
                travel_time_fn = self._make_travel_time_fn(agent_positions, box_positions)
                horizon = self.optimizer_horizon_sec
                weights = PlannerWeights()

                candidate_plan = extend_plan_with_prefix(
                    prefix_plan=prefix_plan,
                    agents=agents,
                    boxes=boxes,
                    current_time=current_time,
                    horizon=horizon,
                    travel_time_fn=travel_time_fn,
                    weights=weights,
                )
            except Exception:
                candidate_plan = None

        if not isinstance(candidate_plan, dict) or not candidate_plan:
            return False

        # Baseline is whatever is committed NOW (post-finalize)
        current_plan = getattr(self, "_committed_plan", None) or {}

        # Only treat as changes what differs from committed
        prefix_changes = self._subtract_committed_prefix_in_order_all_agents(
            requested=prefix_plan,
            committed=current_plan,
            agents=["robot", "human_a", "human_b"],
            lookahead=12,
        )
        if not self._plan_has_any_actions(prefix_changes):
            # Nothing new relative to committed; don’t spin up a session
            self._confirm_already_committed(proposer_id, prefix_plan)
            return True

        # Build objective (use whatever plan_update already computed; leave unknowns as None/0)
        suboptimal_pct = 0.0
        objective = MediationObjectiveMetrics(
            suboptimality_pct=suboptimal_pct,
            baseline_score=None,
            candidate_score=None,
            deadline_risk=plan_update.get("deadline_risk"),
            imbalance_XY=plan_update.get("imbalance_XY"),
            fulfillment_history_ok=True,
            notes="\n\n".join([x for x in [
                plan_update.get("parse_issues_note"),
                plan_update.get("impossible_note"),
            ] if x]) or None,
        )

        # Social + interaction context
        (
            proposer_success_rate,
            conflict_index,
            override_frequency,
            leadership_contestation,
        ) = self._compute_social_context_for_proposer(
            proposer_id=proposer_id,
            window_sec=600.0,
        )
        social = MediationSocialContext(
            proposer_id=proposer_id,
            proposer_success_rate=proposer_success_rate,
            conflict_index=conflict_index,
            override_frequency=override_frequency,
            leadership_contestation=leadership_contestation,
        )

        recent_chat = self._get_recent_chat_turns(limit=12)
        recent_utterances: List[MediationTurn] = []
        for t in recent_chat:
            try:
                role = (t.get("role") or "unknown")
                text = (t.get("text") or "").strip()
                if not text:
                    continue
                recent_utterances.append(
                    MediationTurn(role=role, text=text, meta={"ts": float(t.get("ts") or now_ts)})
                )
            except Exception:
                continue

        interaction = MediationInteractionContext(
            event_summary=self._event_summary_text,
            robot_role_description="Bob is a cooperative teammate that balances safety, efficiency, and human preferences.",
            session_notes="Follow-on mediation bootstrapped from buffered human input that arrived during a previous in-flight mediation.",
            human_profiles=self._load_current_human_profiles() or {},
            recent_utterances=recent_utterances,
        )

        # New session id (do NOT reuse old id)
        new_req_id = meta.get("req_id") or f"followon:{int(now_ts * 1000)}"
        new_session_id = f"mediation:{new_req_id}"

        # Seed with the last human turn (so mediator sees the updated request immediately)
        initial_turns = [last_turn]

        state = MediationState(
            session_id=new_session_id,
            baseline_plan=current_plan,
            candidate_plan=candidate_plan,
            objective=objective,
            social=social,
            interaction=interaction,
            turns=initial_turns,
            prefix_plan=prefix_changes,
            human_ids=list(self.human_agent_ids),
            baseline_provenance=(getattr(self, "_committed_plan_provenance", None) or getattr(self, "_last_plan_provenance", None)),
        )

        # Register & activate
        self._mediation_sessions[new_session_id] = state
        self._active_mediation_id = new_session_id

        if not self.sim_mode:
            self._robot_say("Okay—new request. Let me think.")

        # Schedule INIT for the new session
        self._llm_submit(
            channel="MEDIATION_INIT",
            key=new_session_id,
            ev_type="init",
            payload={"ts": float(now_ts)},
            ts=float(now_ts),
        )

        return True


    def _qualitative_objective_summary_for_reflection(self, objective: Any) -> dict:
        """
        Convert MediationObjectiveMetrics (numeric-heavy) into qualitative-only fields
        for reflection prompts/payloads.
        """
        try:
            subopt_lvl = self._pct_to_level(getattr(objective, "suboptimality_pct", None))
            deadline_risk = getattr(objective, "deadline_risk", None)
            imbalance_xy = getattr(objective, "imbalance_XY", None)
        except Exception:
            subopt_lvl = None
            deadline_risk = None
            imbalance_xy = None

        return {
            "suboptimality_level": subopt_lvl,                 # very_low..very_high
            "deadline_risk_level": self._level_5(deadline_risk),  # very_low..very_high
            "imbalance_xy_level": self._level_5(imbalance_xy),    # very_low..very_high
            "fulfillment_history_ok": bool(getattr(objective, "fulfillment_history_ok", True)),
            "notes": getattr(objective, "notes", None),
        }


    def _qualitative_social_summary_for_reflection(self, social: Any, conflict_index: Optional[float] = None) -> dict:
        """
        Convert social context to qualitative-only fields for reflection prompts/payloads.
        """
        proposer_id = getattr(social, "proposer_id", None)

        # If you have numeric proposer_success_rate / conflict_index, bucket them.
        psr = getattr(social, "proposer_success_rate", None)
        ci = conflict_index if conflict_index is not None else getattr(social, "conflict_index", None)

        return {
            "proposer_id": proposer_id,
            "proposer_success_level": self._rate_to_trust_level(psr),
            "conflict_level": self._level_3(ci),
            "override_frequency": getattr(social, "override_frequency", None),        # already low/medium/high in your code
            "leadership_contestation": getattr(social, "leadership_contestation", None),  # already none/emerging/strong
        }


    # ---------- Per-human periodic reflection ----------

    def _run_periodic_profile_reflection_for(self, human_id: str, ts: float):
        """
        Lightweight reflection that runs every N utterances for a given human,
        based on GLOBAL chat history (not only mediation sessions).
        """
        # Recent conversation focused on this human
        recent_turns = self._get_human_turns_with_prev_context(
            human_id=human_id,
            limit=50,
            ctx_prev=1,
        )

        recent_turns_display = []
        for t in recent_turns:
            try:
                tt = dict(t)
                tt["role"] = (tt.get("role") or "unknown")  # keep canonical ids
                recent_turns_display.append(tt)
            except Exception:
                continue

        if len(recent_turns) < 2:
            # Not enough signal to reflect
            return

        current_profiles = self._load_current_human_profiles() or {}
        cur_prof = current_profiles.get(human_id) or {}

        payload = {
            "mode": "periodic",
            "human_id": human_id,
            "timestamp": ts,
            "recent_utterances": recent_turns_display,
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
                "for this ONE human.\n\n"
                "IMPORTANT:\n"
                "- For each trait update object, output ONLY: new_value (required), "
                "evidence_level (optional), evidence_notes (optional).\n"
                "- Do NOT output confidence_delta or numeric fields.\n\n"
                "Return STRICT JSON of the form:\n"
                "{\n"
                f'  "humans": {{ "{human_id}": {{\n'
                '      "summary_delta": "optional short summary",\n'
                '      "trait_updates": {\n'
                '        "leadership_preference": {"new_value": "emerging", "evidence_level": "medium"},\n'
                '        "trust_in_robot_plans": {"new_value": "low", "evidence_level": "high", "evidence_notes": "…"}\n'
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

        if self._always_accept():
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

        # Buffer a reflection request (coalesces if reflection already running)
        self._llm_submit(
            channel="REFLECTION",
            key=speaker_id,  # human_id
            ev_type="trigger",
            payload={"mode": "periodic"},
            ts=float(ts),
        )

        # Reset counter after triggering
        self._profile_msg_counts[speaker_id] = 0



    def _to_display_text(self, s: str) -> str:
        """
        Convert canonical ids appearing in human-facing strings to display names.
        E.g., "human_a" -> "Sam", "human_b" -> "Jacob", "robot" -> "Bob".
        """
        try:
            mapping = getattr(self, "agent_id_to_human_name", None) or {}
            if not isinstance(s, str) or not s.strip() or not mapping:
                return s
            return self._swap_str(s, mapping)  # uses your existing word-boundary swapper
        except Exception:
            return s


    def _robot_say(self, text: str):
        """
        Send one-shot TTS utterance to SkillsAgent.
        Applies display-name filter so humans never hear canonical ids.
        """
        text = (text or "").strip()
        if not text:
            return

        # Human-facing output: canonical -> display
        say_text = self._to_display_text(text)

        try:
            self.pub_tts_immediate.publish(StringMsg(data=say_text))
        except Exception as e:
            self.get_logger().warn(f"[mediation][tts] failed to publish utterance: {e}")

        # Log robot utterance in global history
        # IMPORTANT: keep CANONICAL text internally so LLM-boundary translation stays consistent.
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
            "additionalProperties": True,
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

    # ---------- Feasibility filtering (drop-only, not whole-plan reject) ----------

    def _action_key(self, aid: str, box_id: int, prop: str, kind: str) -> Tuple[str, int, str, str]:
        return (str(aid), int(box_id), str(prop), str(kind))

    def _index_provenance(self, prov: Optional[dict]) -> Dict[Tuple[str, int, str, str], dict]:
        """
        Build a lookup: (aid, box_id, prop, kind) -> provenance entry.
        Expected provenance entry keys: origin, proposed_by, etc.
        """
        idx: Dict[Tuple[str, int, str, str], dict] = {}
        if not isinstance(prov, dict):
            return idx
        for aid, entries in prov.items():
            for e in (entries or []):
                try:
                    k = self._action_key(aid, int(e["box_id"]), e["property"], e["kind"])
                    idx[k] = dict(e)
                except Exception:
                    continue
        return idx

    def _plan_to_keyset(self, plan: Plan) -> set:
        """
        Convert Plan dict -> set of (aid, box_id, prop, kind) for fast membership.
        """
        s = set()
        for aid, actions in (plan or {}).items():
            for (box_id, prop, kind) in (actions or []):
                try:
                    s.add(self._action_key(str(aid), int(box_id), str(prop), str(kind)))
                except Exception:
                    pass
        return s


    def _prune_human_agreed_plan(self, plan: Plan, boxes, current_time: float) -> Plan:
        """
        Keep agreed actions until they are truly fulfilled or infeasible.
        DO NOT use membership in self._last_plan as a completion signal,
        because optimizer replanning can drop/reorder actions that are still agreed.
        """
        plan = copy.deepcopy(plan or {})
        if not plan:
            return {}

        box_by_id = {b.box_id: b for b in (boxes or [])}

        out: Plan = {}
        for aid, actions in plan.items():
            kept = []
            for (box_id, prop, kind) in (actions or []):
                try:
                    ok, _ = self._check_action_feasible(
                        aid=str(aid),
                        box_id=int(box_id),
                        prop=str(prop),
                        kind=str(kind),
                        box_by_id=box_by_id,
                        current_time=float(current_time),
                    )
                    if not ok:
                        continue

                    # Also drop if it is already fulfilled (stronger than "feasible")
                    b = box_by_id.get(int(box_id))
                    if b is None:
                        continue

                    if kind == "dispose":
                        if prop == "X" and bool(getattr(b, "disposed_X", False)):
                            continue
                        if prop == "Y" and bool(getattr(b, "disposed_Y", False)):
                            continue

                    if kind == "sense":
                        already = (getattr(b, "already_sensed", None) or {}).get(str(aid), {}).get(str(prop), False)
                        if already:
                            continue

                    kept.append((int(box_id), str(prop), str(kind)))
                except Exception:
                    continue

            if kept:
                out[str(aid)] = kept

        return out


    def _check_action_feasible(
        self,
        aid: str,
        box_id: int,
        prop: str,
        kind: str,
        box_by_id: dict,
        current_time: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Returns (ok, reason_if_not_ok)
        """
        b = box_by_id.get(int(box_id))
        if b is None:
            return (False, "unknown box")

        # Deadline passed blocks both sense and dispose
        if b.deadline is not None and float(current_time) >= float(b.deadline):
            return (False, "deadline already passed")

        if kind == "dispose":
            if prop == "X" and bool(getattr(b, "disposed_X", False)):
                return (False, "already disposed for X")
            if prop == "Y" and bool(getattr(b, "disposed_Y", False)):
                return (False, "already disposed for Y")

        if kind == "sense":
            already = (getattr(b, "already_sensed", None) or {}).get(aid, {}).get(prop, False)
            if already:
                return (False, "already sensed by this agent")

        return (True, None)

    def _filter_plan_for_feasibility(
        self,
        plan: Plan,
        boxes,
        current_time: float,
        provenance: Optional[dict] = None,
        prefix_plan: Optional[Plan] = None,
        default_proposer: str = "unknown",
    ) -> Tuple[Plan, List[dict]]:
        """
        Drop ONLY infeasible actions from plan.

        Returns:
          - filtered_plan (same structure as Plan)
          - dropped: list of dicts describing dropped actions:
              {
                "aid": ..., "box_id": ..., "property": ..., "kind": ...,
                "reason": "...",
                "origin": "human|robot|optimizer|unknown",
                "proposed_by": "human_a|human_b|unknown|None"
              }
        """
        filtered: Plan = {}
        dropped: List[dict] = []

        if not plan:
            return {}, []

        box_by_id = {b.box_id: b for b in (boxes or [])}

        prov_idx = self._index_provenance(provenance)
        prefix_set = set()
        if isinstance(prefix_plan, dict):
            for aid, actions in (prefix_plan or {}).items():
                for (box_id, prop, kind) in (actions or []):
                    prefix_set.add(self._action_key(aid, int(box_id), prop, kind))

        for aid, actions in (plan or {}).items():
            kept = []
            for (box_id, prop, kind) in (actions or []):
                ok, reason = self._check_action_feasible(
                    aid=str(aid),
                    box_id=int(box_id),
                    prop=str(prop),
                    kind=str(kind),
                    box_by_id=box_by_id,
                    current_time=current_time,
                )
                if ok:
                    kept.append((int(box_id), str(prop), str(kind)))
                    continue

                k = self._action_key(aid, int(box_id), prop, kind)
                prov = prov_idx.get(k, {})
                origin = (prov.get("origin") or "unknown")
                proposed_by = prov.get("proposed_by")

                # If provenance missing, infer "human" if it was explicitly in prefix_plan
                if origin == "unknown" and k in prefix_set:
                    origin = "human"
                    proposed_by = default_proposer

                dropped.append(
                    {
                        "aid": str(aid),
                        "box_id": int(box_id),
                        "property": str(prop),
                        "kind": str(kind),
                        "reason": reason or "not feasible",
                        "origin": origin,
                        "proposed_by": proposed_by,
                    }
                )

            filtered[str(aid)] = kept

        # Normalize: remove empty agent lists? keep them for schema consistency if you prefer
        return filtered, dropped

    def _summarize_dropped_actions_for_humans(self, dropped: List[dict]) -> Optional[str]:
        if not dropped:
            return None

        human_drops = [d for d in dropped if (d.get("origin") in ("human", "robot"))]
        if human_drops:
            d = human_drops[0]
            who = d.get("proposed_by") or "a human"
            #who = self._to_display_human(str(who))  # <-- NEW (humans only)
            return (
                f"I couldn't include {who}'s request to {d['kind']} box {d['box_id']} ({d['property']}) "
                f"because {d['reason']}."
            )

        d = dropped[0]
        return (
            f"I dropped an infeasible action ({d['aid']} {d['kind']} box {d['box_id']} {d['property']}) "
            f"because {d['reason']}."
        )


    def _compose_committed_plan_utterance(
        self,
        committed_plan: Plan,
        dropped: List[dict],
        fallback: str = "Okay.",
    ) -> str:
        """
        Keep this short; consistent with your “1–2 sentences” norm.
        """
        # If we dropped something, explain drop + confirm remaining robot action if any
        drop_note = self._summarize_dropped_actions_for_humans(dropped)
        robot_actions = committed_plan.get("robot") or []

        if drop_note and robot_actions:
            (box_id, prop, kind) = robot_actions[0]
            return f"{drop_note} I will {kind} box {box_id} ({prop}) now."

        if drop_note and not robot_actions:
            return drop_note

        # No drops: say what robot will do (first action)
        if robot_actions:
            (box_id, prop, kind) = robot_actions[0]
            return f"I will {kind} box {box_id} ({prop}) now."

        return fallback

    def _build_committed_provenance(
        self,
        committed_plan: Plan,
        session: MediationState,
        baseline_provenance: Optional[dict] = None,
    ) -> dict:
        """
        Provenance rules:
          - default origin=optimizer
          - upgrade to origin=human for actions in session.prefix_plan
          - PRESERVE baseline human/robot origins for actions that were already human-agreed
        """
        # Start from optimizer default
        prov = self._build_optimizer_provenance(committed_plan)

        # Build prefix_set (explicit new human request)
        prefix = getattr(session, "prefix_plan", None) or {}
        prefix_set = set()
        for aid, actions in (prefix or {}).items():
            for (box_id, prop, kind) in (actions or []):
                prefix_set.add(self._action_key(aid, int(box_id), prop, kind))

        proposer = (getattr(session, "social", None) and session.social.proposer_id) or "unknown"

        # Index baseline provenance so we can preserve prior human-agreed entries
        base_idx = self._index_provenance(baseline_provenance)

        for aid, actions in (committed_plan or {}).items():
            updated = []
            for (box_id, prop, kind) in (actions or []):
                k = self._action_key(aid, int(box_id), prop, kind)

                origin = "optimizer"
                proposed_by = None

                # 1) Explicit new human request wins
                if k in prefix_set:
                    origin = "human"
                    proposed_by = proposer

                # 2) Otherwise preserve baseline if it was already human/robot origin
                else:
                    prev = base_idx.get(k)
                    if isinstance(prev, dict):
                        prev_origin = (prev.get("origin") or "").strip()
                        if prev_origin in ("human", "robot"):
                            origin = prev_origin
                            proposed_by = prev.get("proposed_by")

                updated.append(
                    {
                        "box_id": int(box_id),
                        "property": str(prop),
                        "kind": str(kind),
                        "origin": origin,
                        "proposed_by": proposed_by,
                    }
                )
            prov[str(aid)] = updated

        return prov



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
                        f"- {aid} {kind} box {box_id} ({prop}) -> {reason_str}"
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
            if not self.sim_mode:
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

        current_plan = getattr(self, "_committed_plan", None) or {}# or getattr(self, "_last_plan", {}) or {}


        committed = current_plan

        # ✅ Only treat as "human_proposed_changes" what differs from committed plan (per-agent, in-order)
        prefix_changes = self._subtract_committed_prefix_in_order_all_agents(
            requested=prefix_plan,
            committed=committed,
            agents=["robot", "human_a", "human_b"],
            lookahead=12,
        )

        # after prefix_plan is built
        original_prefix_plan = copy.deepcopy(prefix_plan)


        # If nothing remains, there's nothing to mediate.
        if not self._plan_has_any_actions(prefix_changes):
            self.get_logger().info("[llm-plan] requested actions already match committed plan prefix; skipping mediation_init")

            self._confirm_already_committed(proposer_id, prefix_plan)  # confirm original request text
            return



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
        if prefix_plan and parse_issues_note and not self.sim_mode:
            self._robot_say(parse_issues_note)


        # 5) First, extend the prefix plan into a full candidate via optimizer
        try:
            candidate_plan = extend_plan_with_prefix(
                prefix_plan=prefix_changes,
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

        # ✅ EARLY BYPASS: if this request is already covered by the committed plan, do NOT start mediation
        try:
            # Prefer checking the explicit request (prefix), fall back to candidate if needed
            if self._is_prefix_plan_already_committed_in_order_all_agents(prefix_plan, committed, agents=["robot","human_a","human_b"]):
                self.get_logger().info("[llm-plan] requested robot actions already in committed plan; skipping mediation_init")

                # confirm using the most direct representation of what they asked
                self._confirm_already_committed(proposer_id, prefix_plan or candidate_plan)
                return
        except Exception as e:
            self.get_logger().warn(f"[llm-plan] early already-committed bypass failed: {e}")
            # fall through to normal behavior


        if self._always_accept():
            # Build session with objective metrics (optional) but do NOT compute `better/adopt`
            better = True
            suboptimal_pct = 0.0
            score_optimal = None
            score_candidate = None
        else:
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

        base_notes = base_notes + "\n\n"# + plan_diff_note

        current_profiles = self._load_current_human_profiles()


        # Build recent dialogue from GLOBAL chat history (last N turns)
        recent_chat = self._get_recent_chat_turns(limit=12)
        recent_utterances: List[MediationTurn] = []
        for i, t in enumerate(recent_chat):
            try:
                # IMPORTANT: keep canonical agent ids here: human_a / human_b / robot
                role = (t.get("role") or "unknown")
                text = (t.get("text") or "").strip()
                if not text:
                    continue
                ts_chat = float(t.get("ts") or ts)
                recent_utterances.append(
                    MediationTurn(
                        role=role,              # <-- DO NOT map to display name here
                        text=text,
                        meta={"ts": ts_chat},
                    )
                )
            except Exception:
                continue

        seen_turn_keys = set()
        for u in recent_utterances:
            try:
                seen_turn_keys.add((u.role, u.text.strip()))
            except Exception:
                pass


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
            ht = human_text.strip()

            already_in_recent = any(
                (t.role == proposer_id and (t.text or "").strip() == ht)
                for t in (recent_utterances or [])
            )

            if not already_in_recent:
                initial_turns.append(
                    MediationTurn(
                        role=proposer_id,
                        text=ht,
                        meta={"ts": ts, "req_id": req_id, "source_rule": trace_entry.get("rule")},
                    )
                )



        human_ids = list(self.human_agent_ids)  # ["human_a","human_b"]


        # ---- Human-agreed baseline carryover (pruned) ----
        baseline_agreed = copy.deepcopy(getattr(self, "_baseline_human_agreed_plan", {}) or {})
        baseline_agreed_prov = copy.deepcopy(getattr(self, "_baseline_human_agreed_provenance", {}) or {})

        # Prune completed/expired agreed actions
        baseline_agreed = self._prune_human_agreed_plan(
            plan=baseline_agreed,
            boxes=boxes,
            current_time=current_time,
        )


        self.get_logger().warn(f"[DEBUG] baseline_agreed AFTER prune keys={list((baseline_agreed or {}).keys())}")

        # If nothing left, we don't "wait" on anything; baseline agreed becomes empty naturally.
        # (No special waiting behavior needed: it only affects planning_view.)


        self.get_logger().info(f'To comitt {candidate_plan}, {current_plan}, {prefix_plan}')

        state = MediationState(
            session_id=session_id,
            baseline_plan=current_plan,
            candidate_plan=candidate_plan,
            objective=objective,
            social=social,
            interaction=interaction,
            turns=initial_turns,
            prefix_plan=prefix_changes,
            human_ids=human_ids,
            baseline_provenance=(getattr(self, "_committed_plan_provenance", None) or getattr(self, "_last_plan_provenance", None)),
        )
        
        self.get_logger().warn(
            f"[DEBUG] baseline_agreed_override keys={list((baseline_agreed or {}).keys())}, "
            f"last_plan_provenance keys={list((getattr(self,'_last_plan_provenance', {}) or {}).keys())}"
        )


        setattr(state, "original_prefix_plan", original_prefix_plan)
        setattr(state, "_seen_turn_keys", seen_turn_keys)
        setattr(state, "baseline_human_agreed_override", baseline_agreed)
        setattr(state, "baseline_human_agreed_override_provenance", baseline_agreed_prov)
        setattr(state, "committed_plan_snapshot", copy.deepcopy(current_plan))
        setattr(state, "committed_plan_provenance_snapshot", copy.deepcopy(getattr(self, "_committed_plan_provenance", None)))


        if self._always_accept():
            # Detect whether this always-accept request overrides a recent one
            new_robot_action = self._first_robot_action(candidate_plan)
            override_info = self._detect_always_accept_override(
                proposer_id=proposer_id,
                ts=float(ts),
                new_robot_action=new_robot_action,
            )
            setattr(state, "_always_accept_override_info", override_info)
            setattr(state, "_always_accept_proposer_id", proposer_id)

            # raw robot_utterance is not trusted later; finalize() will compose the final utterance
            raw = {
                "decision": "accept",
                "planner_action": {"kind": "adopt_candidate", "notes": "forced by always_accept policy"},
                "robot_utterance": "",  # will be composed from committed plan in finalize
                "log_tags": {"strategy": "always_accept", "rationale": "policy"},
            }
            self._finalize_mediation_session(session_id=session_id, session=state, raw_decision=raw, ts=float(ts))
            return



        self._mediation_sessions[session_id] = state
        self._active_mediation_id = session_id

        # 7) Kick off first mediation step ASYNC (never block ROS thread)
        if not self.sim_mode:
            self._robot_say("Let me think.")

        # Important: store initial state + mark active BEFORE enqueueing
        self._mediation_sessions[session_id] = state
        self._active_mediation_id = session_id

        self.get_logger().warn(f"[DEBUG] baseline_human_agreed_plan keys={list(getattr(self,'_baseline_human_agreed_plan',{}).keys())}")
        try:
            self.get_logger().warn(f"[DEBUG] human_ids={getattr(sess, 'human_ids', None)}")
            self.get_logger().warn(f"[DEBUG] proposer_id={(getattr(sess, 'social', None) and sess.social.proposer_id) or None}")
        except Exception:
            pass

        self.get_logger().warn(f"[DEBUG] chat_recent_roles={[t.get('role') for t in self._get_recent_chat_turns(6)]}")

        # Run initial mediator.step() via buffering system
        self._llm_submit(
            channel="MEDIATION_INIT",
            key=session_id,
            ev_type="init",
            payload={"ts": float(ts)},
            ts=float(ts),
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

    def _has_committed_plan(self) -> bool:
        cp = getattr(self, "_committed_plan", None)
        return isinstance(cp, dict) and any((cp.get(a) or []) for a in ("robot", "human_a", "human_b"))




    def _first_robot_action(self, plan: Plan) -> Optional[Tuple[int, str, str]]:
        """
        Returns (box_id, prop, kind) for robot's first action, or None.
        """
        try:
            ra = (plan or {}).get("robot") or []
            if not ra:
                return None
            box_id, prop, kind = ra[0]
            return (int(box_id), str(prop), str(kind))
        except Exception:
            return None

    def _detect_always_accept_override(
        self,
        proposer_id: str,
        ts: float,
        new_robot_action: Optional[Tuple[int, str, str]],
    ) -> Optional[dict]:
        """
        If the new always-accept request overrides a recent prior always-accept request
        from a different human, return info about the overridden request.
        """
        prev = getattr(self, "_last_always_accept_commit", None)
        if not prev or not new_robot_action:
            return None

        prev_ts = float(prev.get("ts") or 0.0)
        if ts - prev_ts > float(getattr(self, "always_accept_override_window_sec", 25.0)):
            return None

        prev_proposer = (prev.get("proposer_id") or "unknown")
        if prev_proposer == proposer_id:
            return None

        prev_action = prev.get("robot_action")
        if not prev_action:
            return None

        # If action is identical, it's not really an override.
        if tuple(prev_action) == tuple(new_robot_action):
            return None

        # Only treat real humans as override targets (optional but usually desired)
        if not (isinstance(prev_proposer, str) and prev_proposer.startswith("human_")):
            return None
        if not (isinstance(proposer_id, str) and proposer_id.startswith("human_")):
            return None

        return {
            "prev_proposer_id": prev_proposer,
            "prev_robot_action": tuple(prev_action),
            "prev_ts": prev_ts,
        }

    def _compose_always_accept_utterance(
        self,
        committed_plan: Plan,
        dropped: List[dict],
        proposer_id: str,
        override_info: Optional[dict],
    ) -> str:
        """
        Always-accept speech:
          - If override happened: tell the previous human we won't do their action anymore.
          - Confirm to the new proposer what we *will* do (grounded in first robot action).
          - Preserve your dropped-action explanation if relevant.
        """
        drop_note = self._summarize_dropped_actions_for_humans(dropped)
        new_act = self._first_robot_action(committed_plan)

        # Build the "switch" sentence if override
        switch_sentence = None
        if override_info:
            prev_id = override_info.get("prev_proposer_id") or "unknown"
            prev_name = self._to_display_text(prev_id)
            new_name = self._to_display_text(proposer_id)

            try:
                (p_box, p_prop, p_kind) = override_info.get("prev_robot_action")
                switch_sentence = (
                    f"Hey {prev_name}, I'm switching to {new_name}'s request, "
                    f"so I won't {p_kind} box {p_box} ({p_prop}) anymore."
                )
            except Exception:
                switch_sentence = (
                    f"Hey {prev_name}, I'm switching to {new_name}'s request, "
                    f"so I won't follow your last request anymore."
                )

        # Build the "accept + action" sentence
        new_name = self._to_display_text(proposer_id)
        if new_act:
            (box_id, prop, kind) = new_act
            accept_sentence = f"Hey {new_name}, I accept—I'll {kind} box {box_id} ({prop}) now."
        else:
            accept_sentence = f"Hey {new_name}, I accept your plan."

        parts = []
        if switch_sentence:
            parts.append(switch_sentence)

        # If we dropped something, include it but keep it short
        if drop_note:
            parts.append(drop_note)

        parts.append(accept_sentence)

        self.get_logger().info(f'accept why {committed_plan} {override_info} {dropped} {new_act}')

        # Keep to ~1–3 short sentences
        return " ".join([p.strip() for p in parts if p and p.strip()])


    # ---------- Finalize mediation session ----------

    def _finalize_mediation_session(
        self,
        session_id: str,
        session: MediationState,
        raw_decision: dict,
        ts: float,
    ):
        """
        Terminal mediation handler with PARTIAL feasibility filtering:

        - If LLM chooses adopt_candidate/merge_plans, we filter out infeasible actions
          (deadline passed, already disposed, already sensed) instead of rejecting everything.
        - If the dropped action was human-origin, we explicitly say so.
        - Robot utterance is forced to match the committed plan (post-filter).
        """
        
        self.get_logger().info(f'Raw {raw_decision}')
        
        try:
            raw_decision = self._normalize_mediation_decision(raw_decision or {})
            pa = (raw_decision or {}).get("planner_action") or {}
            kind = (pa.get("kind") or "keep_baseline").strip()

            delta = (pa.get("candidate_plan_delta") or {}) if isinstance(pa, dict) else {}


            baseline_plan = session.baseline_plan or {}
            candidate_plan = session.candidate_plan or {}
            prefix_plan = getattr(session, "prefix_plan", None) or {}





            # Pull boxes snapshot for feasibility checks
            boxes = getattr(self, "_last_boxes_for_optimizer", None)
            if boxes is None:
                boxes, _ = self._build_boxes_for_optimizer(getattr(self, "_last_boxes_state", []) or [])

            current_time = getattr(self, "_last_server_time", None) or time.time()
            box_positions = getattr(self, "_last_box_positions", {}) or {}

            # Baseline provenance (what we had before this session)
            baseline_prov = (
                getattr(session, "baseline_provenance", None)
                or getattr(self, "_committed_plan_provenance", None)
                or getattr(self, "_baseline_human_agreed_provenance", None)
                or getattr(self, "_last_plan_provenance", None)
            )



            # Decide the intended plan BEFORE filtering
            intended_plan: Plan

            if self._always_accept():
                # if always_accept was triggered by a specific human request, you probably want to
                # commit ONLY that request, not a full optimizer plan
                intended_plan = self._plan_mediator._apply_candidate_delta(session, baseline_plan, delta) if delta else session.prefix_plan

            else:
                if kind == "keep_baseline":
                    intended_plan = baseline_plan

                elif kind == "adopt_candidate":
                    # IMPORTANT: adopt_candidate should *adopt exactly what the LLM decided*.
                    # In your schema, that is candidate_plan_delta (agreements-only).
                    intended_plan = self._plan_mediator._apply_candidate_delta(session, baseline_plan, delta)

                elif kind == "merge_plans":
                    # merge == patch baseline with delta
                    intended_plan = self._plan_mediator._apply_candidate_delta(session, baseline_plan, delta)

                else:
                    intended_plan = baseline_plan

            self.get_logger().info(f'Committed 1 plan {raw_decision}, {delta}, {intended_plan}, {candidate_plan}, {baseline_plan}')
            # Feasibility filtering:
            # - If we’re keeping baseline, we can still filter baseline (optional), but usually baseline should already be feasible.
            # - If adopting/merging candidate, filter candidate and keep remaining feasible actions.
            proposer = (getattr(session, "social", None) and session.social.proposer_id) or "unknown"

            filtered_plan, dropped = self._filter_plan_for_feasibility(
                plan=intended_plan,
                boxes=boxes,
                current_time=current_time,
                provenance=baseline_prov,      # helps attribute drops if action existed previously
                prefix_plan=prefix_plan,       # helps attribute drops for newly proposed actions
                default_proposer=proposer,
            )
            self.get_logger().info(f"[filter] {filtered_plan}, {dropped}")
            # If LLM wanted candidate/merge but filtering removed everything new and candidate becomes empty,
            # fall back to baseline (also filtered lightly).
            if kind in ("adopt_candidate", "merge_plans") and not self._always_accept():
                # If filtered_plan is empty across all agents, it's not useful.
                any_actions = any((filtered_plan.get(a) or []) for a in ("robot", "human_a", "human_b"))
                if not any_actions:
                    self.get_logger().warn(
                        f"[mediation] candidate became empty after feasibility filtering; keeping baseline."
                    )
                    kind = "keep_baseline"
                    filtered_plan, dropped = self._filter_plan_for_feasibility(
                        plan=baseline_plan,
                        boxes=boxes,
                        current_time=current_time,
                        provenance=baseline_prov,
                        prefix_plan={},  # baseline isn't “newly proposed” here
                        default_proposer="unknown",
                    )

            committed_plan = filtered_plan or {}

            self._commit_plan_with_provenance(
                plan=committed_plan,
                session=session,
                baseline_provenance=baseline_prov,
            )

            self.get_logger().info(f'Committed plan {committed_plan}, {intended_plan}, {candidate_plan}, {baseline_plan}')
            # ✅ Update committed plan (single source of truth)
            self._committed_plan = copy.deepcopy(committed_plan)
            self._committed_plan_provenance = copy.deepcopy(getattr(self, "_last_plan_provenance", None))


            # After committing (and after self._last_plan_provenance is set)
            prov = getattr(self, "_last_plan_provenance", None) or {}

            new_agreed: Plan = {}

            for aid, entries in (prov or {}).items():
                for e in (entries or []):
                    try:
                        origin = (e.get("origin") or "unknown")
                        if origin not in ("human", "robot"):
                            continue  # ignore optimizer-only tasks

                        new_agreed.setdefault(str(aid), []).append(
                            (int(e["box_id"]), str(e["property"]), str(e["kind"]))
                        )
                    except Exception:
                        continue

            # Merge with previous agreed baseline, then prune by world state
            prev_agreed = copy.deepcopy(getattr(self, "_baseline_human_agreed_plan", {}) or {})
            merged = prev_agreed
            for aid, acts in new_agreed.items():
                merged.setdefault(aid, [])
                for a in acts:
                    if a not in merged[aid]:
                        merged[aid].append(a)

            merged = self._prune_human_agreed_plan(plan=merged, boxes=boxes, current_time=current_time)

            self._baseline_human_agreed_plan = merged
            self._baseline_human_agreed_provenance = copy.deepcopy(prov)







            # Publish committed plan once
            if committed_plan:
                self._publish_optimizer_plan(committed_plan, current_time, box_positions)

            # If we dropped something important, log it clearly (with attribution)
            if dropped:
                human_drops = [d for d in dropped if d.get("origin") in ("human", "robot")]
                if human_drops:
                    d = human_drops[0]
                    who = d.get("proposed_by") or d.get("origin")
                    self.get_logger().warn(
                        f"[mediation] dropped human-origin action: {who} requested "
                        f"{d['aid']} {d['kind']} box {d['box_id']} ({d['property']}), reason={d['reason']}"
                    )
                else:
                    self.get_logger().warn(f"[mediation] dropped infeasible actions: {dropped}")

            # Speak: always_accept gets a special "override-aware" acceptance message.
            if self._always_accept() or ((raw_decision.get("log_tags") or {}).get("strategy") == "always_accept"):
                proposer_id = getattr(session, "_always_accept_proposer_id", None) or (
                    (getattr(session, "social", None) and session.social.proposer_id) or "unknown"
                )
                override_info = getattr(session, "_always_accept_override_info", None)

                final_utt = self._compose_always_accept_utterance(
                    committed_plan=committed_plan,
                    dropped=dropped,
                    proposer_id=proposer_id,
                    override_info=override_info,
                )
            else:
                final_utt = self._compose_committed_plan_utterance(
                    committed_plan=committed_plan,
                    dropped=dropped,
                    fallback=(raw_decision.get("robot_utterance") or "Okay."),
                )

            if (final_utt and not self.sim_mode) or (self.sim_mode and final_utt and self._always_accept()):
                self._robot_say(final_utt)

            # Update override-tracking memory after committing (only if we actually have a robot action)
            try:
                act = self._first_robot_action(committed_plan)
                if act:
                    proposer_for_mem = (
                        getattr(session, "_always_accept_proposer_id", None)
                        or ((getattr(session, "social", None) and session.social.proposer_id) or "unknown")
                    )
                    self._last_always_accept_commit = {
                        "ts": float(ts),
                        "proposer_id": str(proposer_for_mem),
                        "robot_action": tuple(act),
                    }
            except Exception:
                pass


            self.get_logger().info(
                f"[mediation] finalized session {session_id} with planner_action={kind} "
                f"(dropped={len(dropped)})"
            )

            # Kick off reflection asynchronously
            if not self._always_accept():
                try:
                    self._kickoff_async_reflection(session)
                except Exception as e:
                    self.get_logger().warn(f"[mediation] failed to start async reflection: {e}")

        except Exception as e:
            self.get_logger().warn(f"[mediation] finalize session failed: {e}")
        finally:
            self._active_mediation_id = None
            self._mediation_pending_deadline_ts = None

            final_status = session.status or "idle"
            if final_status not in ("accept", "reject", "cancelled"):
                final_status = "idle"

            try:
                self._publish_mediation_status(session_id, final_status)
            except Exception as e:
                self.get_logger().warn(f"[mediation] failed to publish final status: {e}")

    def _commit_plan_with_provenance(
        self,
        plan: Plan,
        session: Optional[MediationState] = None,
        baseline_provenance: Optional[dict] = None,
    ):
        """
        Update both self._last_plan and self._last_plan_provenance consistently.

        - If session is provided, preserve baseline human/robot origins and mark prefix actions as human.
        - Otherwise, default all actions to optimizer provenance.
        """
        plan = plan or {}
        self._last_plan = plan

        if session is not None:
            try:
                self._last_plan_provenance = self._build_committed_provenance(
                    committed_plan=plan,
                    session=session,
                    baseline_provenance=baseline_provenance,
                )
                return
            except Exception as e:
                self.get_logger().warn(f"[provenance] failed to build committed provenance; defaulting: {e}")

        self._last_plan_provenance = self._build_optimizer_provenance(plan)


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
        session_id = getattr(session, "session_id", "unknown")
        ts = time.time()
        self._llm_submit(
            channel="REFLECTION",
            key=f"session:{session_id}",
            ev_type="trigger",
            payload={"mode": "mediation_session", "session_id": session_id},
            ts=ts,
        )



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
        transcript = []
        include_idx = set()

        turns = list(session.turns or [])
        for i, t in enumerate(turns):
            role = getattr(t, "role", None)
            if isinstance(role, str) and role.startswith("human_"):
                include_idx.add(i)
                if i - 1 >= 0:
                    include_idx.add(i - 1)

        for i in sorted(include_idx):
            tt = turns[i]
            transcript.append(
                {
                    "role": tt.role,  # keep canonical ids
                    "text": tt.text,
                    "meta": tt.meta,
                }
            )

        # 2) Optional: get current profiles from HDT / DB
        current_profiles = self._load_current_human_profiles()  # {"human_a": {...}, ...}

        # NEW: load realized disposal outcomes and compute planning accuracy per human
        disposal_outcomes = self._load_disposal_outcomes_for_session(session)
        from .optimizer_client import compute_planning_accuracy_by_human
        planning_stats = compute_planning_accuracy_by_human(disposal_outcomes)

        # Convert planning outcomes to qualitative-only summary
        planning_outcomes_qual = {}
        try:
            for hid, st in (planning_stats or {}).items():
                # st may include numeric correct_rate; do NOT forward numbers to the LLM
                cr = None
                try:
                    cr = st.get("correct_rate")
                except Exception:
                    cr = None

                planning_outcomes_qual[hid] = {
                    "planning_correctness_level": self._rate_to_trust_level(cr),
                    "summary": (
                        "often correct"
                        if self._rate_to_trust_level(cr) in ("high", "very_high")
                        else "mixed or often incorrect"
                    ),
                }
        except Exception:
            planning_outcomes_qual = {}


        objective_qual = self._qualitative_objective_summary_for_reflection(session.objective)
        social_qual = self._qualitative_social_summary_for_reflection(session.social)

        payload = {
            "session_id": session.session_id,
            "status": session.status,

            # Keep plans as text summaries (already non-numeric)
            "baseline_plan": self._plan_mediator._summarize_plan(session.baseline_plan),
            "final_candidate_plan": self._plan_mediator._summarize_plan(session.candidate_plan),

            # ✅ qualitative-only
            "objective_summary": objective_qual,
            "social_summary": social_qual,

            "interaction_context": {
                "event_summary": session.interaction.event_summary,
                "session_notes": session.interaction.session_notes,
            },
            "dialogue": transcript,
            "current_profiles": current_profiles or {},

            # ✅ qualitative-only
            "planning_outcomes_summary": planning_outcomes_qual,
        }


        system_msg = {
            "role": "system",
            "content": (
                "You are Bob's reflection module.\n"
                "- You see a complete planning conversation between Bob and the humans.\n"
                "- You also see statistics about how often each human's proposed disposals "
                "ended up being correct or incorrect.\n"
                "- Infer how each human prefers to plan and interact with Bob.\n"
                "- Use planning_outcomes_summary planning_correctness_level "
                "(very_low/low/medium/high/very_high) to calibrate trust-related traits.\n"

                "- Update only traits you have evidence for. Be conservative.\n"
            ),
        }


        user_msg = {
            "role": "user",
            "content": (
                "Given this mediation session, propose updates to each human's profile.\n"
                "Focus on leadership preference, trust in robot plans, need for explanations,\n"
                "risk aversion or deadline focus, and tolerance for disagreement.\n\n"
                "IMPORTANT:\n"
                "- For each trait, output ONLY:\n"
                "  - new_value (required)\n"
                "  - evidence_level (optional: very_low|low|medium|high|very_high)\n"
                "  - evidence_notes (optional short string)\n"
                "- Do NOT output confidence_delta or any numeric fields.\n\n"
                "Return STRICT JSON of the form:\n"
                "{\n"
                '  "humans": {\n'
                '    "human_a": {\n'
                '      "summary_delta": "optional short summary of what changed",\n'
                '      "trait_updates": {\n'
                '        "leadership_preference": {\n'
                '          "new_value": "emerging",\n'
                '          "evidence_level": "medium",\n'
                '          "evidence_notes": "brief justification"\n'
                "        },\n"
                '        "trust_in_robot_plans": {\n'
                '          "new_value": "low",\n'
                '          "evidence_level": "high"\n'
                "        }\n"
                "      }\n"
                "    },\n"
                '    "human_b": { "summary_delta": "...", "trait_updates": { ... } }\n'
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

            # 0) Extract summary_delta (optional)
            summary_delta = upd.get("summary_delta")
            if isinstance(summary_delta, str):
                summary_delta = summary_delta.strip()
                if not summary_delta:
                    summary_delta = None
            else:
                summary_delta = None

            # 1) Normalize into trait_updates
            trait_updates = upd.get("trait_updates")
            if not isinstance(trait_updates, dict):
                trait_updates = {}
                for key, val in upd.items():
                    if key in ("summary_delta", "trait_updates"):
                        continue
                    trait_updates[key] = {"new_value": val, "evidence_level": "very_low"}

            # 2) Get or init current profile
            cur = current_profiles.get(human_id) or {
                "id": human_id,
                "summary": "",
                "traits": {},
                "last_updated_ts": now,
            }
            traits = cur.setdefault("traits", {})

            # 2.5) Apply summary_delta -> summary (append)
            if summary_delta:
                prev = (cur.get("summary") or "").strip()
                # simple concatenation; you can swap for bulleting or timestamped logs
                cur["summary"] = summary_delta



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

    def _process_mediation_init_events(self, session_id: str, events: List[dict]):
        """
        Process init events for a newly created session.
        Expects drained buffered events (type="init").
        """
        if not events:
            return

        # Session-level single-flight
        if not self._try_acquire_mediation_session_llm(session_id, "init"):
            # If busy, just re-buffer by pushing next_run_ts slightly; dispatcher will retry
            with self._llm_lock:
                b = self._llm_bucket("MEDIATION_INIT", session_id)
                for ev in events:
                    self._llm_buf[b].append(ev)
                self._llm_next_run_ts[b] = time.time() + 0.05
            return

        try:
            sid = getattr(self, "_active_mediation_id", None)
            sess = self._mediation_sessions.get(session_id)


            # If this init bucket is for a session that's no longer active/missing,
            # try to bootstrap a follow-on mediation from these events.
            if sid != session_id or not sess:
                last_ts = None
                for ev in events:
                    last_ts = ev.get("ts") or last_ts
                ts = float(last_ts or time.time())

                if self._try_bootstrap_followon_mediation_from_init_events(
                    prev_session_id=session_id,
                    events=events,
                    ts_hint=ts,
                ):
                    return
                return


            if getattr(sess, "turns_used", 0) not in (None, 0):
                return
            if getattr(sess, "status", None) in ("accept", "reject", "cancelled"):
                return

            last_ts = None
            for ev in events:
                last_ts = ev.get("ts") or last_ts
            ts = float(last_ts or time.time())

            try:
                sess2, raw = self._plan_mediator.step(sess)
            except Exception as e:
                self.get_logger().warn(f"[mediation] init step failed: {e}")

                raw = {
                    "decision": "reject",
                    "planner_action": {"kind": "keep_baseline", "notes": f"init step failed: {e}"},
                    "robot_utterance": "Something went wrong while thinking—I'll keep the current plan for now.",
                    "log_tags": {"strategy": "init_error_fallback", "rationale": "LLM/step failed"},
                }
                raw = self._normalize_mediation_decision(raw)
                sess.status = "reject"
                self._mediation_sessions[session_id] = sess
                self._finalize_mediation_session(session_id=session_id, session=sess, raw_decision=raw, ts=ts)
                return

            self.get_logger().info(f'Raw 1 {raw}')
            self._mediation_sessions[session_id] = sess2

            robot_utt = (raw.get("robot_utterance") or "").strip()
            if robot_utt:
                self.get_logger().info(f"[mediation] Bob says: {robot_utt}")
                self._robot_say(robot_utt)

            # Start human response window only after we output something and remain pending
            if getattr(sess2, "status", None) == "pending":
                self._start_pending_deadlines(ts)
                setattr(sess2, "_waiting_for_human", True)
            else:
                setattr(sess2, "_waiting_for_human", False)

            
            # If init returned a terminal decision, we must check whether any human turns arrived
            # while init was in-flight. If so, run ONE follow-up step with the latest turn
            # before finalizing, otherwise we discard that turn.
            if getattr(sess2, "status", None) in ("accept", "reject"):

                '''
                late_turns = self._drain_buffered_mediation_turns(session_id)

                if late_turns:
                    last_turn = late_turns[-1]
                    try:
                        # Make sure the session has the last human turn recorded
                        try:
                            sess2.turns.append(last_turn)
                            sess2.interaction.recent_utterances.append(last_turn)
                        except Exception:
                            pass

                        # Convert this “init accepted too fast” case into an immediate follow-up turn-step
                        # so the late human input is actually considered before commit.
                        sess3, raw2 = self._plan_mediator.step(sess2, new_human_turn=last_turn)
                        self._mediation_sessions[session_id] = sess3

                        robot_utt2 = (raw2.get("robot_utterance") or "").strip()
                        if robot_utt2:
                            self._robot_say(robot_utt2)

                        if getattr(sess3, "status", None) == "pending":
                            self._start_pending_deadlines(ts)
                            setattr(sess3, "_waiting_for_human", True)
                            return  # do NOT finalize; we’re pending again
                        self.get_logger().info(f'Raw 2 {raw2}')
                        # Terminal after the follow-up step
                        self._finalize_mediation_session(
                            session_id=session_id,
                            session=sess3,
                            raw_decision=raw2,
                            ts=ts,
                        )
                        return

                    except Exception as e:
                        self.get_logger().warn(f"[mediation] late-turn follow-up after init failed: {e}")
                        # Fall through to finalize using the init decision.
                '''
                self.get_logger().info(f'Raw 3 {raw}')
                # No late turns => safe to finalize init decision
                self._finalize_mediation_session(
                    session_id=session_id,
                    session=sess2,
                    raw_decision=raw,
                    ts=ts,
                )
                return


        finally:
            self._release_mediation_session_llm(session_id)


    def _extract_plan_update_from_intent_event(self, trace_entry: dict) -> Optional[dict]:
        """
        If this trace_entry (speech_intent_inferred) contains agents_plan, rebuild:
          - prefix_plan (from agents_plan)
          - candidate_plan (optimizer extend)
          - objective metrics (deadline risk / imbalance / suboptimality if desired)

        Returns a dict you can stash into MediationTurn.meta["plan_update"], or None.
        """
        data = trace_entry.get("data") or {}

        # 1) Parse the JSON payload produced by your intent LLM
        plan_json = data.get("json")
        if not isinstance(plan_json, dict):
            jt = data.get("json_text") or data.get("text")
            if isinstance(jt, str) and jt.strip():
                try:
                    plan_json = json.loads(jt)
                except Exception:
                    plan_json = None

        if not isinstance(plan_json, dict):
            return None

        agents_plan = plan_json.get("agents_plan")
        if not isinstance(agents_plan, dict):
            return None

        # 2) Build prefix from agents_plan
        try:
            new_prefix, parse_issues = build_plan_from_llm_agents_plan(
                agents_plan,
                allowed_agents=["robot", "human_a", "human_b"],
                collect_issues=True,
            )
        except Exception:
            return None

        if not new_prefix:
            # No robot actions -> not a meaningful plan update for mediation
            return None

        parse_issues_note = summarize_plan_parse_issues(parse_issues) if parse_issues else None

        # 3) Build boxes/agents/current_time the same way you do in _handle_llm_speech_plan
        try:
            agents = self._build_agents_for_optimizer()
            boxes = getattr(self, "_last_boxes_for_optimizer", None)
            if boxes is None:
                boxes, _ = self._build_boxes_for_optimizer(getattr(self, "_last_boxes_state", []) or [])
        except Exception:
            return {"prefix_plan": new_prefix, "parse_issues_note": parse_issues_note}

        current_time = getattr(self, "_last_server_time", None) or time.time()
        box_positions = getattr(self, "_last_box_positions", {}) or {}
        agent_positions = self._snapshot_agent_positions()
        travel_time_fn = self._make_travel_time_fn(agent_positions, box_positions)
        horizon = self.optimizer_horizon_sec
        weights = PlannerWeights()

        impossible_note = self._summarize_impossible_actions(
            plan=new_prefix,
            boxes=boxes,
            current_time=current_time,
        )

        # 4) Extend prefix -> candidate
        try:
            new_candidate = extend_plan_with_prefix(
                prefix_plan=new_prefix,
                agents=agents,
                boxes=boxes,
                current_time=current_time,
                horizon=horizon,
                travel_time_fn=travel_time_fn,
                weights=weights,
            )
        except Exception:
            new_candidate = None

        out = {
            "prefix_plan": new_prefix,
            "candidate_plan": new_candidate,
            "parse_issues_note": parse_issues_note,
            "impossible_note": impossible_note,
            "current_time": float(current_time),
        }

        # Optional: recompute objective summaries for *this updated candidate*
        if new_candidate:
            try:
                xy_metrics = compute_xy_imbalance_for_plan(plan=new_candidate, boxes=boxes)
                out["imbalance_XY"] = xy_metrics.get("imbalance")
            except Exception:
                pass

            try:
                out["deadline_risk"] = estimate_deadline_risk_for_plan(
                    plan=new_candidate,
                    boxes=boxes,
                    current_time=current_time,
                    travel_time_fn=travel_time_fn,
                    horizon=horizon,
                )
            except Exception:
                pass

        return out


    def _process_mediation_events(self, session_id: str, events: List[dict]):
        """
        Process buffered mediation turns for a session_id.
        Expects drained events (type="turn") with payload {"turn": MediationTurn}.
        Routes last-call to autoresolve if (turns_used+1 >= max).
        """
        

        
        if not events:
            return

        if not self._try_acquire_mediation_session_llm(session_id, "turn"):
            # Re-buffer and retry soon via dispatcher
            with self._llm_lock:
                b = self._llm_bucket("MEDIATION", session_id)
                for ev in events:
                    self._llm_buf[b].append(ev)
                self._llm_next_run_ts[b] = time.time() + 0.02
            return

        self.get_logger().info(
            f"MEDIATION initiatied {events}"
        )

        try:
            active_sid = getattr(self, "_active_mediation_id", None)
            sess = self._mediation_sessions.get(session_id)
            if active_sid != session_id or not sess:
                return
            if getattr(sess, "status", None) != "pending":
                return

            turns = []
            last_ts = None
            for ev in events:
                if ev.get("type") != "turn":
                    continue
                t = (ev.get("payload") or {}).get("turn")
                if t is not None:
                    turns.append(t)
                last_ts = ev.get("ts") or last_ts

            if not turns:
                return

            now_ts = float(last_ts or time.time())

            # Append all but last directly (coalesce)
            for t in turns[:-1]:
                try:
                    sess.turns.append(t)
                    sess.interaction.recent_utterances.append(t)
                except Exception:
                    pass

            self._mediation_sessions[session_id] = sess


            last_turn = turns[-1]
                        
            try:
                pu = (getattr(last_turn, "meta", None) or {}).get("plan_update")
                if pu:
                    sess = self._apply_plan_update_to_session(sess, pu)
                    self._mediation_sessions[session_id] = sess
            except Exception:
                pass
            
            turns_used = int(getattr(sess, "turns_used", 0) or 0)
            max_turns = int(getattr(self, "mediation_max_turns", 3) or 3)

            # LAST-CALL policy -> autoresolve instead of another normal step
            if (turns_used + 1) >= max_turns:
                self.get_logger().warn(
                    f"[mediation] last allowed call (turns_used={turns_used}, max={max_turns}); "
                    f"routing to autoresolve for {session_id}"
                )
                try:
                    setattr(sess, "_waiting_for_human", False)
                except Exception:
                    pass

                try:
                    sess.turns.append(last_turn)
                    sess.interaction.recent_utterances.append(last_turn)
                except Exception:
                    pass
                self._mediation_sessions[session_id] = sess

                # enqueue autoresolve (buffered)
                self._llm_submit(
                    channel="MEDIATION_AUTORESOLVE",
                    key=session_id,
                    ev_type="autoresolve",
                    payload={"reason": "max_turns_last_call", "now_ts": now_ts},
                    ts=now_ts,
                )
                return

            
            
            try:
                sess2, raw = self._plan_mediator.step(sess, new_human_turn=last_turn)
            except Exception as e:
                self.get_logger().warn(f"[mediation] buffered step failed: {e}")
                return

            self._mediation_sessions[session_id] = sess2

            robot_utt = (raw.get("robot_utterance") or "").strip()
            if robot_utt:
                self._robot_say(robot_utt)

            if getattr(sess2, "status", None) == "pending":
                self._start_pending_deadlines(now_ts)
                setattr(sess2, "_waiting_for_human", True)
            else:
                setattr(sess2, "_waiting_for_human", False)

            if getattr(sess2, "status", None) in ("accept", "reject"):
                self._finalize_mediation_session(
                    session_id=session_id,
                    session=sess2,
                    raw_decision=raw,
                    ts=now_ts,
                )

        finally:
            self._release_mediation_session_llm(session_id)


    def _drain_buffered_mediation_turns(self, session_id: str) -> List[MediationTurn]:
        """
        Drain any queued MEDIATION turns that are waiting for this session_id.
        This is used to avoid losing turns that arrive while MEDIATION_INIT is in-flight.
        """
        b = self._llm_bucket("MEDIATION", session_id)
        turns: List[MediationTurn] = []

        with self._llm_lock:
            q = list(self._llm_buf.get(b, []))
            if q:
                self._llm_buf[b].clear()

        for ev in q:
            try:
                if ev.get("type") != "turn":
                    continue
                t = (ev.get("payload") or {}).get("turn")
                if t is not None:
                    turns.append(t)
            except Exception:
                continue

        return turns


    def _apply_plan_update_to_session(self, sess: MediationState, plan_update: dict) -> MediationState:
        """
        Mutate the session so the planning_view reflects the latest human request.
        """
        if not isinstance(plan_update, dict):
            return sess

        committed = getattr(self, "_committed_plan", None) or {}

        new_prefix = plan_update.get("prefix_plan")
        if isinstance(new_prefix, dict) and new_prefix:
            new_prefix_changes = self._subtract_committed_prefix_in_order_all_agents(
                requested=new_prefix,
                committed=committed,
                agents=["robot", "human_a", "human_b"],
                lookahead=12,
            )

            sess.prefix_plan = new_prefix_changes  # may become {}



        new_candidate = plan_update.get("candidate_plan")
        if isinstance(new_candidate, dict) and new_candidate:
            sess.candidate_plan = new_candidate

        # Keep objective in sync (so mediator sees the real metrics for the updated candidate)
        try:
            if getattr(sess, "objective", None):
                if "deadline_risk" in plan_update:
                    sess.objective.deadline_risk = plan_update.get("deadline_risk")
                if "imbalance_XY" in plan_update:
                    sess.objective.imbalance_XY = plan_update.get("imbalance_XY")

                # Also carry notes so the model can explain parse failures / impossible actions
                notes_parts = []
                if plan_update.get("parse_issues_note"):
                    notes_parts.append(plan_update["parse_issues_note"])
                if plan_update.get("impossible_note"):
                    notes_parts.append(plan_update["impossible_note"])
                if notes_parts:
                    extra = "\n\n".join([p for p in notes_parts if p])
                    prev = getattr(sess.objective, "notes", None) or ""
                    sess.objective.notes = (prev + "\n\n" + extra).strip()
        except Exception:
            pass

        return sess


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

        plan_update = None
        if rule == "speech_intent_inferred":
            try:
                try:
                    plan_update = self._extract_plan_update_from_intent_event(trace_entry)
                except Exception as e:
                    self.get_logger().warn(f"Error in here 1")
                try:
                    prefix_plan, speaker_id2 = self._prefix_plan_from_speech_intent_inferred(trace_entry)
                except Exception as e:
                    self.get_logger().warn(f"Error in here 2")
                if prefix_plan and self._prefix_plan_robot_already_fulfilled_server_truth(prefix_plan):
                    try:
                        self._say_already_done_for_prefix(speaker_id2, prefix_plan)
                    except Exception as e:
                        self.get_logger().warn(f"Error in here 3")
                    return True
            except Exception as e:
                self.get_logger().warn(f"[routing] error in routing: {e}, {trace_entry}")


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

        dedup_key = (speaker_id, text.strip())
        seen = getattr(sess, "_seen_turn_keys", None)
        if seen is None:
            seen = set()
            setattr(sess, "_seen_turn_keys", seen)
        if dedup_key in seen:
            pass
            #return True
        seen.add(dedup_key)

        self._mediation_sessions[sid] = sess  # persist mutation if you want

        meta = {"ts": ts, "rule": rule}
        if plan_update:
            meta["plan_update"] = plan_update

        human_turn = MediationTurn(
            role=speaker_id,
            text=text,
            meta=meta,
        )


        # ✅ We got a human turn; we are no longer in 'waiting for human' mode.
        try:
            setattr(sess, "_waiting_for_human", False)
        except Exception:
            pass



        # Buffer the turn into the MEDIATION channel for this session.
        # If a mediation LLM call is in-flight, this will simply queue it.
        # Otherwise, this will debounce and then do ONE coalesced step.
        self._llm_submit(
            channel="MEDIATION",
            key=sid,
            ev_type="turn",
            payload={"turn": human_turn},
            ts=float(ts),
        )

        return True


    def _prefix_plan_from_speech_intent_inferred(self, trace_entry: dict) -> tuple[dict, str]:
        """
        Returns (prefix_plan, speaker_id).
        prefix_plan is a Plan dict like {"robot":[(9,"Y","sense")], ...}
        speaker_id like "human_b" (or "unknown").
        """
        data = trace_entry.get("data") or {}

        speaker_id = (
            (data.get("request") or {}).get("speaker_id")
            or data.get("speaker_id")
            or "unknown"
        )

        # json_text is the structured intent payload (stringified JSON)
        jt = data.get("json_text") or data.get("raw_text") or data.get("text")
        if not isinstance(jt, str) or not jt.strip():
            return {}, speaker_id

        try:
            obj = json.loads(jt)
        except Exception:
            return {}, speaker_id

        agents_plan = obj.get("agents_plan")
        if not isinstance(agents_plan, dict):
            return {}, speaker_id

        try:
            prefix_plan, _issues = build_plan_from_llm_agents_plan(
                agents_plan,
                allowed_agents=["robot", "human_a", "human_b"],
                collect_issues=False,
            )
        except Exception:
            return {}, speaker_id

        return (prefix_plan or {}), speaker_id


    def _say_already_done_for_prefix(self, speaker_id: str, prefix_plan: Plan):
        ra = (prefix_plan or {}).get("robot") or []
        if not ra:
            self._robot_say("That is already done.")
            return
        (box_id, prop, kind) = ra[0]
        who = self._to_display_text(speaker_id)
        verb = "sensed" if kind == "sense" else "disposed"
        self._robot_say(f"Hey {who}, I already {verb} box {box_id} for {prop}.")


    def _prefix_plan_robot_already_fulfilled_server_truth(self, prefix_plan: Plan) -> bool:
        """
        True if the robot actions in prefix_plan are already fulfilled per box-server truth.
        """
        boxes_state = getattr(self, "_last_boxes_state", None)
        if not isinstance(boxes_state, list) or not prefix_plan:
            return False

        for (box_id, prop, kind) in (prefix_plan.get("robot") or []):
            try:
                box_id = int(box_id)
                prop = str(prop)
                kind = str(kind)

                done = self._server_action_fulfilled(
                    boxes_state,
                    agent_id="robot" if kind == "sense" else None,
                    box_id=box_id,
                    prop=prop,
                    kind=kind,
                )
                if done:
                    return True
            except Exception:
                continue

        return False


    def _process_mediation_autoresolve_events(self, session_id: str, events: List[dict]):
        """
        Async autoresolve processor: performs the autoresolve LLM call and finalizes.
        Expects drained events (type="autoresolve") payload includes {"reason","now_ts"}.
        """
        if not events:
            return

        if not self._try_acquire_mediation_session_llm(session_id, "autoresolve"):
            with self._llm_lock:
                b = self._llm_bucket("MEDIATION_AUTORESOLVE", session_id)
                for ev in events:
                    self._llm_buf[b].append(ev)
                self._llm_next_run_ts[b] = time.time() + 0.05
            return

        try:
            last = events[-1]
            payload = last.get("payload") or {}
            reason = (payload.get("reason") or "unknown")
            now_ts = float(payload.get("now_ts") or last.get("ts") or time.time())

            sid = getattr(self, "_active_mediation_id", None)
            sess = self._mediation_sessions.get(session_id)
            if sid != session_id or not sess or sess.status != "pending":
                return

            if getattr(sess, "_autoresolving", False):
                return
            setattr(sess, "_autoresolving", True)
            self._mediation_sessions[session_id] = sess

            try:
                try:
                    messages = self._plan_mediator.build_messages_for_autoresolve(sess, reason=reason)
                    raw = self._mediate_llm_call(messages)
                except Exception as e:
                    self.get_logger().warn(f"[mediation] autoresolve LLM failed: {e}; falling back to keep_baseline")
                    raw = {
                        "decision": "reject",
                        "planner_action": {"kind": "keep_baseline", "notes": f"autoresolve fallback: {e}"},
                        "robot_utterance": "No response in time—I'll keep the current plan for now.",
                        "log_tags": {"strategy": "timeout_fallback", "rationale": "LLM autoresolve failed"},
                    }

                raw = self._normalize_mediation_decision(raw)

                if (raw.get("decision") or "").strip() == "pending":
                    raw["decision"] = "reject"
                    raw["planner_action"] = {"kind": "keep_baseline", "notes": "autoresolve forced non-pending"}
                    raw["robot_utterance"] = "No response in time—I'll keep the current plan for now."
                    raw["log_tags"] = {"strategy": "timeout_forced", "rationale": "autoresolve must finalize"}

                robot_utt = (raw.get("robot_utterance") or "").strip()
                if robot_utt:
                    self.get_logger().info(f"[mediation] Bob says: {robot_utt}")
                    self._robot_say(robot_utt)

                sess.status = "accept" if raw.get("decision") == "accept" else "reject"
                self._mediation_sessions[session_id] = sess

                # finalize() will compose forced utterance if needed
                self._finalize_mediation_session(session_id=session_id, session=sess, raw_decision=raw, ts=now_ts)

            finally:
                s2 = self._mediation_sessions.get(session_id)
                if s2:
                    try:
                        setattr(s2, "_autoresolving", False)
                    except Exception:
                        pass

        finally:
            self._release_mediation_session_llm(session_id)

    def _process_reflection_events(self, reflection_key: str, events: List[dict]):
        if not events:
            return

        last = events[-1]
        payload = last.get("payload") or {}
        mode = payload.get("mode") or "periodic"
        ts = float(last.get("ts") or time.time())

        if mode == "periodic" and isinstance(reflection_key, str) and reflection_key.startswith("human_"):
            self._run_periodic_profile_reflection_for(reflection_key, ts)
            return

        if mode == "mediation_session":
            sid = payload.get("session_id")
            sess = self._mediation_sessions.get(sid) if sid else None
            if sess:
                snap = copy.deepcopy(sess)
                if self._should_run_reflection_for_session(snap):
                    self._reflect_on_mediation_session(snap)
                    self._reset_profile_msg_counts_for_session(snap)
            return



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

        # ✅ Do not trigger human-turn timeout while any mediation LLM call is running
        if self._mediation_llm_busy.get(str(sid), False):
            return

        # ✅ Only count down when we are actually waiting for humans
        waiting = bool(getattr(sess, "_waiting_for_human", False))
        if not waiting:
            return


        # Turn-wait timeout (no new utterance)
        dl = getattr(self, "_mediation_pending_deadline_ts", None)
        if dl is not None and now_ts >= float(dl):
            self.get_logger().warn(
                f"[mediation] turn timeout hit (no utterance); auto-resolving {sid}"
            )
            # NEW (enqueue bucketed autoresolve; single-flight + coalesced)
            self.get_logger().warn(
                f"[mediation] turn timeout hit (no utterance); enqueue autoresolve {sid}"
            )
            
            self.get_logger().warn(f"[DEBUG] baseline_human_agreed_plan keys={list(getattr(self,'_baseline_human_agreed_plan',{}).keys())}")
            try:
                human_ids = getattr(sess, "human_ids", None)
                proposer_id = (getattr(sess, "social", None) and sess.social.proposer_id) or None
                self.get_logger().warn(f"[DEBUG] human_ids={human_ids}")
                self.get_logger().warn(f"[DEBUG] proposer_id={proposer_id}")
            except Exception:
                pass

            self.get_logger().warn(f"[DEBUG] chat_recent_roles={[t.get('role') for t in self._get_recent_chat_turns(6)]}")

            
            self._llm_submit(
                channel="MEDIATION_AUTORESOLVE",
                key=sid,
                ev_type="autoresolve",
                payload={"reason": "turn_timeout", "now_ts": float(now_ts)},
                ts=float(now_ts),
            )
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

        self.get_logger().info(
            f"autoresolve"
        )
        
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

