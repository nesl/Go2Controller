#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple

from collections import deque
import uuid


import requests

import random

import rclpy
from rclpy.node import Node
from std_msgs.msg import String as StringMsg

from architecture.optimizer_client import (
    AgentState, BoxInfo, PlannerWeights, plan_assignments_gurobi, info_level_from_p
)

from .sim_human_policy import (
    BasePolicy,
    LLMPolicy,
    PolicyAction,
    BoxSummary,
    Pose2D,
)


Property = Literal["X", "Y"]

RED = "\033[31m"
RESET = "\033[0m"
CYAN = "\033[96m"
YELLOW = "\033[93m"

# ---------------------------
# Sim human node
# ---------------------------

class SimHumanAgent(Node):
    def __init__(self):
        super().__init__("sim_human_agent")

        # ---- basic params ----
        self.declare_parameter("agent_id", "human_a")
        self.declare_parameter("goal_property", "X")
        self.declare_parameter("server_base_url", "http://URL:8080")
        self.declare_parameter("stt_topic", "/audio/stt_text")

        # motion + timing
        self.declare_parameter("speed_mps", 1.0)
        self.declare_parameter("decision_period_sec", 1.0)
        self.declare_parameter("request_timeout_sec", 300.0)

        # policy knobs
        self.declare_parameter("policy_type", "scripted")  # scripted | llm
        self.declare_parameter("dispose_threshold", 0.80)
        self.declare_parameter("giveup_threshold", 0.20)
        self.declare_parameter("help_wait_sec", 20.0)
        self.declare_parameter("help_target_speaker", "robot")
        self.declare_parameter("dist_weight", 2.0)

        # participants + trust
        self.declare_parameter("humans_json", "[]")  # JSON list of {id,name}
        self.declare_parameter("robots_json", "[]")  # JSON list of {id,name}

        self.declare_parameter("trust_overrides_json", "{}")  # JSON string


        # LLM params
        self.declare_parameter("llm_provider", "none")  # openai | none
        self.declare_parameter("llm_model", "gpt-4.1-mini")
        self.declare_parameter("llm_temperature", 0.2)
        self.declare_parameter("llm_max_tokens", 250)
        self.declare_parameter("llm_top_k_boxes", 100)
        self.declare_parameter("llm_timeout_sec", 30.0)

        # human model traits (LLM prompt conditioning)
        self.declare_parameter("risk_aversion", 0.7)
        self.declare_parameter("stubbornness", 0.5)
        self.declare_parameter("fairness_sensitivity", 0.3)

        # logging
        self.declare_parameter("log_actions", True)
        
        self.declare_parameter("llm_jitter_enable", True)
        self.declare_parameter("llm_jitter_min_sec", 1.0)
        self.declare_parameter("llm_jitter_max_sec", 2.0)

        # ---------------------------
        # Thinking simulation (delay before deciding next action)
        # ---------------------------
        self.declare_parameter("think_sim_enable", True)
        self.declare_parameter("think_min_delay_sec", 1)
        self.declare_parameter("think_max_delay_sec", 2)
        self.declare_parameter("think_router_enable", False)  # optional: also delay in message-router decisions


        # ---------------------------
        # Speech simulation (delay before publishing utterances)
        # ---------------------------
        self.declare_parameter("speech_sim_enable", True)
        self.declare_parameter("speech_rate_wpm", 150.0)          # typical conversational ~130-170 wpm
        self.declare_parameter("speech_min_delay_sec", 0.15)      # minimum "start speaking" delay
        self.declare_parameter("speech_max_delay_sec", 2.0)       # cap so long messages don't stall forever
        self.declare_parameter("speech_punct_pause_sec", 0.06)    # extra pause per punctuation mark
        self.declare_parameter("speech_queue_max", 30)            # prevent unbounded queue growth


        self.declare_parameter("waiting_mode", "strict")  # strict | soft
        
        self.declare_parameter("prior_default_X", 0.5)
        self.declare_parameter("prior_default_Y", 0.5)
        self.declare_parameter("prior_field_json", '{"X": [{"cx": 5.0,  "cy": -4.5, "sigma": 2.5, "target_p": 0.70, "strength": 1.0}],"Y": [{"cx": -1.0, "cy":  2.0, "sigma": 2.5, "target_p": 0.7, "strength": 1.0}]}')
        self.declare_parameter("prior_temperature", 1.0)
        self.declare_parameter("prior_clip_min", 0.02)
        self.declare_parameter("prior_clip_max", 0.98)
        self.declare_parameter("no_target_speaker", True)


        # ---------------------------
        # Belief-change detection (robot-style) -> replan + ego order
        # ---------------------------
        self.declare_parameter("belief_replan_enabled", True)
        self.declare_parameter("belief_replan_min_delta_p", 0.20)
        self.declare_parameter("belief_replan_min_delta_info", 0.20)
        self.declare_parameter("belief_replan_cooldown_sec", 0.0)          # per-box cooldown (wall)
        self.declare_parameter("belief_replan_global_cooldown_sec", 1.5)   # global cooldown (sim)
        self.declare_parameter("belief_replan_max_updates", 3)

        self.belief_replan_enabled = bool(self.get_parameter("belief_replan_enabled").value)
        self.belief_replan_min_delta_p = float(self.get_parameter("belief_replan_min_delta_p").value)
        self.belief_replan_min_delta_info = float(self.get_parameter("belief_replan_min_delta_info").value)
        self.belief_replan_cooldown_sec = float(self.get_parameter("belief_replan_cooldown_sec").value)
        self.belief_replan_global_cooldown_sec = float(self.get_parameter("belief_replan_global_cooldown_sec").value)
        self.belief_replan_max_updates = int(self.get_parameter("belief_replan_max_updates").value)

        self._last_box_beliefs: Dict[int, Dict[str, float]] = {}
        self._last_box_announce_ts: Dict[int, float] = {}
        self._last_replan_sim: float = -1e9
        self._last_ego_plan_fp: Optional[str] = None

        self._seen_box_ids: set[int] = set()


        self._stop = False
        
        
        self.prior_default_X = float(self.get_parameter("prior_default_X").value)
        self.prior_default_Y = float(self.get_parameter("prior_default_Y").value)
        self.prior_temperature = float(self.get_parameter("prior_temperature").value)
        self.prior_clip_min = float(self.get_parameter("prior_clip_min").value)
        self.prior_clip_max = float(self.get_parameter("prior_clip_max").value)

        raw = str(self.get_parameter("prior_field_json").value)
        try:
            self.prior_field = json.loads(raw)
        except Exception:
            self.prior_field = {}

        self.no_target_speaker = bool(self.get_parameter("no_target_speaker").value)
        
        # ---- communication master switch ----
        self.declare_parameter("comm_enable", True)
        self.comm_enable = bool(self.get_parameter("comm_enable").value)
        self._op_remaining_cache = None
        
        self.think_sim_enable = bool(self.get_parameter("think_sim_enable").value)
        self.think_min_delay_sec = float(self.get_parameter("think_min_delay_sec").value)
        self.think_max_delay_sec = float(self.get_parameter("think_max_delay_sec").value)
        self.think_router_enable = bool(self.get_parameter("think_router_enable").value)

        
        self.speech_sim_enable = bool(self.get_parameter("speech_sim_enable").value)
        self.speech_rate_wpm = float(self.get_parameter("speech_rate_wpm").value)
        self.speech_min_delay_sec = float(self.get_parameter("speech_min_delay_sec").value)
        self.speech_max_delay_sec = float(self.get_parameter("speech_max_delay_sec").value)
        self.speech_punct_pause_sec = float(self.get_parameter("speech_punct_pause_sec").value)
        self.speech_queue_max = int(self.get_parameter("speech_queue_max").value)

        
        self.waiting_mode = str(self.get_parameter("waiting_mode").value).lower()


        self.llm_jitter_enable = bool(self.get_parameter("llm_jitter_enable").value)
        self.llm_jitter_min_sec = float(self.get_parameter("llm_jitter_min_sec").value)
        self.llm_jitter_max_sec = float(self.get_parameter("llm_jitter_max_sec").value)

        self._op_lock = threading.Lock()
        self._current_op: Optional[Dict[str, Any]] = None
        # example: {"kind":"dispose","box_id":10,"prop":"Y","started_sim":123.4}

        self._cancel_lock = threading.Lock()
        self._cancel_evt: Optional[threading.Event] = None



        self.declare_parameter("infer_target_use_llm", False)
        self.declare_parameter("infer_target_max_history", 8)

        self.infer_target_use_llm = bool(self.get_parameter("infer_target_use_llm").value)
        self.infer_target_max_history = int(self.get_parameter("infer_target_max_history").value)

        # ---- transcript: all bus rx + all tx (omniscient log, used for context) ----
        self.declare_parameter("collect_all_messages", True)
        self.declare_parameter("collect_all_messages_max", 10)
        self.collect_all_messages = bool(self.get_parameter("collect_all_messages").value)
        self.collect_all_messages_max = int(self.get_parameter("collect_all_messages_max").value)

        self._transcript_lock = threading.Lock()
        self.transcript = deque()  # each item: {dir, t_wall, speaker_id, target_speaker, text, ...}


        self.declare_parameter("prior_X", 0.5)
        self.declare_parameter("prior_Y", 0.5)
        self.prior_X = float(self.get_parameter("prior_X").value)
        self.prior_Y = float(self.get_parameter("prior_Y").value)



        self.agent_id: str = str(self.get_parameter("agent_id").value)
        self.goal_property: Property = str(self.get_parameter("goal_property").value)  # type: ignore
        self.base_url: str = str(self.get_parameter("server_base_url").value).rstrip("/")
        self.stt_topic: str = str(self.get_parameter("stt_topic").value)

        self.speed_mps: float = float(self.get_parameter("speed_mps").value)
        self.decision_period: float = float(self.get_parameter("decision_period_sec").value)
        self.timeout: float = float(self.get_parameter("request_timeout_sec").value)

        self.policy_type: str = str(self.get_parameter("policy_type").value).lower()
        self.dispose_threshold: float = float(self.get_parameter("dispose_threshold").value)
        self.giveup_threshold: float = float(self.get_parameter("giveup_threshold").value)
        self.help_wait_sec: float = float(self.get_parameter("help_wait_sec").value)
        self.help_target_speaker: str = str(self.get_parameter("help_target_speaker").value)
        self.dist_weight: float = float(self.get_parameter("dist_weight").value)



        # LLM config
        self.llm_provider = str(self.get_parameter("llm_provider").value).lower()
        self.llm_model = str(self.get_parameter("llm_model").value)
        self.llm_temperature = float(self.get_parameter("llm_temperature").value)
        self.llm_max_tokens = int(self.get_parameter("llm_max_tokens").value)
        self.llm_top_k_boxes = int(self.get_parameter("llm_top_k_boxes").value)
        self.llm_timeout_sec = float(self.get_parameter("llm_timeout_sec").value)

        # traits
        self.risk_aversion = float(self.get_parameter("risk_aversion").value)
        self.stubbornness = float(self.get_parameter("stubbornness").value)
        self.fairness_sensitivity = float(self.get_parameter("fairness_sensitivity").value)

        self.log_actions: bool = bool(self.get_parameter("log_actions").value)

        # ✅ router cadence (separate from action loop)
        self.declare_parameter("router_period_sec", 0.2)
        self.router_period_sec = float(self.get_parameter("router_period_sec").value)

        # ---- ROS pub/sub (only if comm enabled) ----
        self.pub_stt = None
        self.sub_stt = None

        # ---- speech primitives (init ONCE) ----
        self._speech_lock = threading.Lock()
        self._speech_cv = threading.Condition(self._speech_lock)
        self._speech_queue = deque()
        self._speech_stop_evt = threading.Event()
        self._speech_thread = None
        
        self._inbox_lock = threading.Lock()


        if self.comm_enable:
            self.pub_stt = self.create_publisher(StringMsg, self.stt_topic, 10)
            self.sub_stt = self.create_subscription(StringMsg, self.stt_topic, self._on_stt_text, 10)

            self._speech_thread = threading.Thread(target=self._speech_worker_main, daemon=True)
            self._speech_thread.start()
        else:
            self.pub_stt = None
            self.sub_stt = None
            self.speech_sim_enable = False

        self._router_wakeup = threading.Event()
        self._router_thread = threading.Thread(target=self._router_worker_main, daemon=True)
        self._router_thread.start()

        # timers: action always; router only if comm enabled
        self._action_timer = self.create_timer(self.decision_period, self._tick)
        self._router_timer = None
        if self.comm_enable:
            self._router_timer = self.create_timer(self.router_period_sec, self._router_tick)



        self._speech_busy_lock = threading.Lock()
        self._speech_busy = False




        # ---- internal state ----
        self.pose = Pose2D(0.0, 0.0)
        self.last_msgs: List[Dict[str, Any]] = []
        self._mem: Dict[Tuple[int, str], Dict[str, Any]] = {}

        # ✅ Use deque so router can consume safely and efficiently
        self.inbox = deque()  # type: ignore[var-annotated]
        self.declare_parameter("max_inbox_per_tick", 2)
        self.max_inbox_per_tick = int(self.get_parameter("max_inbox_per_tick").value)



        # ✅ busy flag (true while doing travel/sense/dispose)
        self._busy_lock = threading.Lock()
        self._busy = False

        # router threading
        self._router_lock = threading.Lock()
        self._router_thread: Optional[threading.Thread] = None

        # ---------------------------
        # Action journal + graceful shutdown
        # ---------------------------
        self._journal_lock = threading.Lock()
        self._action_journal: List[Dict[str, Any]] = []
        self._shutdown_lock = threading.Lock()
        self._shutdown_requested = False

        # keep timer handles so we can cancel them on shutdown
        self._action_timer = self.create_timer(self.decision_period, self._tick)


        self.plan_state: Dict[str, Any] = {
            "focus_box_id": None,
            "focus_prop": self.goal_property,
            "phase": "explore",
            "last_commitment": "",
            # ✅ new
            "commitments": [],   # list[dict]
            "next_intent": None, # optional: (box_id, prop, kind) if you want
            "active_commitment_id": None,

        }

        # --- Router -> Action immediate handoff ---
        self._plan_lock = threading.Lock()
        self.plan_state.setdefault("pending_action", None)  # dict or None


        # threading
        self._action_lock = threading.Lock()
        self._action_thread: Optional[threading.Thread] = None


        # participants/profiles
        self.participants: Dict[str, Dict[str, Any]] = {}
        self.human_ids: List[str] = []
        self.robot_ids: List[str] = []
        self.sensor_params: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.trust_map: Dict[str, float] = {}

        # ---- help / social state ----
        self.inbox_requests: List[Dict[str, Any]] = []     # [{from, box_id, prop, t_sim, t_wall}]
        self.help_history: Dict[str, int] = {}             # helper_id -> count
        self.ignore_history: Dict[str, int] = {}           # requester_id -> count
        self.last_helped_at_sim: Dict[str, float] = {}     # requester_id -> last time we helped
        self.help_cooldown_sec: float = 10.0               # don't help same person too frequently

        self.declare_parameter("help_cooldown_sec", 10.0)
        self.help_cooldown_sec = float(self.get_parameter("help_cooldown_sec").value)


        self._build_participant_registry()
        self._init_profiles_from_server()

        # policies

        self.llm_policy = LLMPolicy()
        self.policy: BasePolicy = self.llm_policy 

        self.get_logger().info(
            f"SimHumanAgent up agent_id={self.agent_id} goal={self.goal_property} "
            f"server={self.base_url} topic={self.stt_topic} policy={self.policy_type} "
            f"dispose_th={self.dispose_threshold} giveup_th={self.giveup_threshold} "
            f"help_wait={self.help_wait_sec}s speed={self.speed_mps} timeout={self.timeout}s "
            f"llm_provider={self.llm_provider} llm_model={self.llm_model}"
        )

    def _can_sense(self, b: BoxSummary, prop: Property, *, agent_id: Optional[str] = None) -> bool:
        who = agent_id or self.agent_id

        # ✅ role constraint first
        if not self._agent_can_sense_prop(str(who), prop):
            return False

        # global senseability gate
        if isinstance(b.senseable, dict):
            if not bool(b.senseable.get(prop, True)):
                return False

        # optional per-agent gate
        if b.senseable_by and isinstance(b.senseable_by, dict):
            allowed = b.senseable_by.get(prop, [])
            return (str(who) in allowed)

        return True


    def _agent_can_sense_prop(self, agent_id: str, prop: Property) -> bool:
        aid = str(agent_id)
        p = str(prop).upper()
        if aid == "robot":
            return p in ("X", "Y")
        if aid == "human_a":
            return p == "X"
        if aid == "human_b":
            return p == "Y"
        # fallback: default to ONLY its own goal if unknown
        # (or return False if you want strict)
        return p == str(self.goal_property)


    def _prior_for(self, prop: Property) -> float:
        return self._safe_prob(self.prior_X if prop == "X" else self.prior_Y)

    def _logit(self, p: float) -> float:
        p = self._safe_prob(p)
        return math.log(p / (1.0 - p))

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _prior_for_box(self, b: BoxSummary, prop: Property) -> float:
        base = self.prior_default_X if prop == "X" else self.prior_default_Y
        base = self._safe_prob(base)
        
        '''

        L0 = self._logit(base)
        L = L0

        if isinstance(b.senseable, dict) and not bool(b.senseable.get(prop, True)):
            # can't sense this prop at all -> don't let spatial prior bias it
            return float(base)


        blobs = (self.prior_field or {}).get(prop, [])
        if isinstance(blobs, list):
            for r in blobs:
                try:
                    cx = float(r.get("cx"))
                    cy = float(r.get("cy"))
                    sigma = max(1e-6, float(r.get("sigma", 3.0)))
                    target_p = self._safe_prob(float(r.get("target_p", base)))
                    strength = float(r.get("strength", 1.0))  # >=0
                except Exception:
                    continue

                dx = float(b.x) - cx
                dy = float(b.y) - cy
                d2 = dx*dx + dy*dy
                w = math.exp(-0.5 * d2 / (sigma*sigma))      # spatial weight 0..1

                # Pull log-odds toward target log-odds, scaled by (strength * w)
                Lt = self._logit(target_p)
                L += (strength * w) * (Lt - L0)

        # Global punchiness (temperature in log-odds space)
        temp = max(0.1, float(self.prior_temperature))
        L = L0 + temp * (L - L0)

        p = self._sigmoid(L)
        p = max(self.prior_clip_min, min(self.prior_clip_max, p))
        return float(p)
        '''
        return base


    def _eligible_agents_for_prop(self, b: BoxSummary, prop: Property, *, assume_robot_can_sense: bool = False) -> List[str]:
        eligible: List[str] = []
        for pid in (self.participants.keys() if isinstance(self.participants, dict) else []):
            try:
                pid = str(pid)
                if assume_robot_can_sense and pid == "robot":
                    eligible.append(pid)
                    continue
                if self._can_sense(b, prop, agent_id=pid):
                    eligible.append(pid)
            except Exception:
                continue
        return eligible



    def _optimistic_dispose_time_sec(self, b: BoxSummary, prop: Property, *, assume_robot_can_sense: bool = False) -> float:
        base = float(getattr(b, f"dispose_time_{prop}", 0.0))
        if base <= 0.0:
            return 0.0

        eligible = self._eligible_agents_for_prop(b, prop, assume_robot_can_sense=assume_robot_can_sense)
        n = max(0, len(eligible))
        if n <= 0:
            return float("inf")

        return base / (2.0 ** n)



    def _deadline_passed_by_feasibility(self, b: BoxSummary, prop: Property, now_sim: float) -> bool:
        return not self._feasible_sense_then_dispose(
            b, prop, now_sim,
            assume_travel_zero=True,
            assume_robot_can_sense=True,   # ✅ “just for the sake of it”
        )


    def _maybe_detect_belief_changes(self, boxes: List[BoxSummary], now_sim: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Robot-style semantic change detection:
          - new box appears (no prev)
          - abs(delta p) >= min_dp OR abs(delta info) >= min_dinfo (per property)
        Uses THIS agent's belief fusion pipeline (_belief_present_from_box) so it's consistent.
        Returns (changed, details).
        """
        details = {
            "updates": [],  # list of dicts
            "new_box_ids": [],
            "updated_box_ids": [],
        }

        if not self.belief_replan_enabled:
            return False, details

        if not boxes:
            return False, details

        # global sim cooldown to avoid thrashing replans
        if (float(now_sim) - float(self._last_replan_sim)) < float(self.belief_replan_global_cooldown_sec):
            return False, details

        min_dp = float(self.belief_replan_min_delta_p)
        min_dinfo = float(self.belief_replan_min_delta_info)
        cooldown_wall = float(self.belief_replan_cooldown_sec)
        max_updates = max(1, int(self.belief_replan_max_updates))

        now_wall = time.time()

        updates = []
        for b in boxes:
            bid = int(b.box_id)

            # compute semantic beliefs + info (same space planner uses)
            pX = float(self._belief_present_from_box(b, "X"))
            pY = float(self._belief_present_from_box(b, "Y"))
            infoX = float(info_level_from_p(pX))
            infoY = float(info_level_from_p(pY))

            prev = self._last_box_beliefs.get(bid)
            is_new = (bid not in self._seen_box_ids)


            changed_prop = {"X": False, "Y": False}
            changed = False

            if prev is not None:
                if abs(pX - prev["pX"]) >= min_dp: # or abs(infoX - prev["infoX"]) >= min_dinfo:
                    changed = True
                    changed_prop["X"] = True
                if abs(pY - prev["pY"]) >= min_dp: # or abs(infoY - prev["infoY"]) >= min_dinfo:
                    changed = True
                    changed_prop["Y"] = True

            if not is_new and not changed:
                continue

            # per-box wall cooldown
            last_t = self._last_box_announce_ts.get(bid)
            if last_t is not None and (now_wall - float(last_t)) < cooldown_wall:
                self._last_box_beliefs[bid] = {"pX": pX, "pY": pY, "infoX": infoX, "infoY": infoY}
                self._seen_box_ids.add(bid)
                continue


            self._last_box_announce_ts[bid] = now_wall

            upd = {
                "box_id": bid,
                "is_new": bool(is_new),
                "changed_prop": dict(changed_prop),
                "pX": pX, "infoX": infoX,
                "pY": pY, "infoY": infoY,
            }
            updates.append(upd)

            self._seen_box_ids.add(bid)


            # update snapshot immediately (prevents multiple triggers in one tick)
            self._last_box_beliefs[bid] = {"pX": pX, "pY": pY, "infoX": infoX, "infoY": infoY}

        if not updates:
            return False, details

        # keep it bounded + deterministic
        updates.sort(key=lambda u: int(u["box_id"]))
        updates = updates[:max_updates]

        details["updates"] = updates
        details["new_box_ids"] = [u["box_id"] for u in updates if u["is_new"]]
        details["updated_box_ids"] = [u["box_id"] for u in updates if (not u["is_new"])]

        return True, details


    def _maybe_replan_and_enqueue_ego_order(self, boxes: List[BoxSummary], now_sim: float) -> None:
    
        self.get_logger().info(f"[REPLAN] {boxes}")
        changed, det = self._maybe_detect_belief_changes(boxes, now_sim)
        self.get_logger().info(f"[REPLAN] {changed} {det}")
        if not changed:
            return

        # compute new plans
        try:
            cand = self._compute_candidate_plans(boxes, now_sim)
        except Exception as e:
            self.get_logger().warn(f"[REPLAN] failed to compute candidate plans: {e}")
            return

        ego_plan = (cand or {}).get("egoistic_team_plan", {}) or {}
        ego_fp = json.dumps(ego_plan, sort_keys=True, separators=(",", ":"))
        if self._last_ego_plan_fp == ego_fp:
            # plan didn't change; don't spam
            self._last_replan_sim = float(now_sim)
            return
        self._last_ego_plan_fp = ego_fp

        # commit global replan timestamp *now* (so we don't thrash)
        self._last_replan_sim = float(now_sim)

        # format a directive
        msg = self._format_ego_plan_order(ego_plan, now_sim)
        # optional: append tiny change hint (short)
        #msg += f" (new={det.get('new_box_ids')} updated={det.get('updated_box_ids')})"

        # enqueue as immediate pending action (so it is uttered automatically)
        with self._plan_lock:
            self.plan_state["pending_action"] = PolicyAction(
                kind="say",
                text=msg,
                target_speaker="all",
                reason="belief_change_replan_ego_order",
            )


    def _format_ego_plan_order(self, ego_plan: Dict[str, List[Dict[str, Any]]], now_sim: float) -> str:
        def fmt_step(s: Dict[str, Any]) -> str:
            kind = str(s.get("kind", "")).lower()
            box_id = s.get("box_id", None)
            prop = str(s.get("prop", "")).upper()
            if box_id is None:
                return kind
            if kind in ("sense", "sense_self"):
                return f"sense box {box_id} for {prop}"
            if kind == "dispose":
                return f"dispose box {box_id} for {prop}"
            if kind in ("goto", "goto_only", "move", "travel"):
                return f"go to box {box_id}"
            return f"{kind} box {box_id} {prop}".strip()

        lines = []
        my_steps = (ego_plan or {}).get(self.agent_id, []) or []
        if my_steps:
            lines.append("I will " + "; then ".join(fmt_step(s) for s in my_steps[:2]) + ".")
        for aid, steps in (ego_plan or {}).items():
            if aid == self.agent_id or not steps:
                continue
            lines.append(f"{self._display_name(aid)}: {fmt_step(steps[0])}.")
        if not lines:
            return "Update — new info. I don’t have a useful action right now."
        return "Update — new info. Follow this plan: " + " ".join(lines)


    def _compute_candidate_plans(self, boxes: List[BoxSummary], now_sim: float, k: int = 6) -> Dict[str, Any]:
        include_all = bool(getattr(self, "candidate_plans_include_all_agents", True))

        # 1) Build MILP inputs (AgentState/BoxInfo) from current world
        agents, box_infos, travel_time_fn, horizon = self._build_milp_inputs_for_candidates(
            boxes=boxes,
            now_sim=now_sim,
            k=k,
            include_all_agents=include_all,
        )

        

        # 2) Two weight profiles
        my_goal = str(self.goal_property).upper()
        #self.get_logger().info(f"{agents} {box_infos} {my_goal} {boxes}")
        
        ego_w = PlannerWeights(
            reward_correct_X=1.0 if my_goal == "X" else 0.0,
            reward_correct_Y=1.0 if my_goal == "Y" else 0.0,
            lambda_balance=0.0,         # egoistic: no X/Y balancing pressure
            weight_info=0.2,           # optional: egoist explores less
            # keep your existing gates:
            info_threshold_for_dispose=0.4,
            pmin_for_dispose=0.7,
            egoistic_goal_property=my_goal,
        )

        non_goal = "Y" if str(my_goal).upper() == "X" else "X"

        pro_w = PlannerWeights(
            reward_correct_X=1.0 if not my_goal == "X" else 0.0,
            reward_correct_Y=1.0 if not my_goal == "Y" else 0.0,
            lambda_balance=0.5,         # prosocial: balanced team progress
            weight_info=0.2,
            info_threshold_for_dispose=0.4,
            pmin_for_dispose=0.7,
            egoistic_goal_property=non_goal
        )

        # 3) Solve twice
        ego_plan = plan_assignments_gurobi(
            agents=agents,
            boxes=box_infos,
            current_time=float(now_sim),
            horizon=float(horizon),
            travel_time_fn=travel_time_fn,
            weights=ego_w,

        )
        
        
        pro_plan = plan_assignments_gurobi(
            agents=agents,
            boxes=box_infos,
            current_time=float(now_sim),
            horizon=float(horizon),
            travel_time_fn=travel_time_fn,
            weights=pro_w,
        )

        # 4) Convert optimizer Plan tuples -> your dict-of-dicts schema
        ego = self._plan_tuples_to_candidate_schema(ego_plan)
        pro = self._plan_tuples_to_candidate_schema(pro_plan)

        max_actions = int(getattr(self, "candidate_plans_max_actions_per_agent", 2))  # e.g., 2
        ego = self._limit_actions_per_agent(ego, max_actions_per_agent=max_actions)
        pro = self._limit_actions_per_agent(pro, max_actions_per_agent=max_actions)


        # 5) dilemma signal (unchanged)
        def top_self_step(tp: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
            steps = (tp or {}).get(self.agent_id, []) or []
            return steps[0] if steps else None

        ego_top = top_self_step(ego)
        pro_top = top_self_step(pro)
        conflict = (ego_top != pro_top) and (ego_top is not None) and (pro_top is not None)

        return {
            "egoistic_team_plan": ego,
            "prosocial_team_plan": pro,
            "dilemma": {
                "conflict": bool(conflict),
                "egoistic_top": {"agent_id": self.agent_id, "step": ego_top} if ego_top else None,
                "prosocial_top": {"agent_id": self.agent_id, "step": pro_top} if pro_top else None,
                "why": "egoistic optimizes my goal-property disposals; prosocial optimizes team expected hazard disposals",
            },
        }

    def _limit_actions_per_agent(
        self,
        plan: Dict[str, List[Dict[str, Any]]],
        *,
        max_actions_per_agent: int,
        always_keep_self_top: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        if max_actions_per_agent <= 0:
            return {aid: [] for aid in (plan or {}).keys()}

        out: Dict[str, List[Dict[str, Any]]] = {}

        for aid, steps in (plan or {}).items():
            steps = list(steps or [])
            if not steps:
                out[aid] = []
                continue

            if always_keep_self_top and aid == self.agent_id:
                # keep top step + next N-1
                out[aid] = steps[:max_actions_per_agent]
            else:
                out[aid] = steps[:max_actions_per_agent]

        return out


    def _build_milp_inputs_for_candidates(self, boxes: List[BoxSummary], now_sim: float, k: int, include_all_agents: bool):
        # pick candidate boxes similar to your current deadline filtering
        cand = []
        for b in boxes:
            if self._disposed_any(b):
                continue
            if float(now_sim) > float(b.deadline):
                continue
            cand.append(b)
        cand = sorted(cand, key=lambda b: float(b.deadline))[: max(1, int(k))]

        # agents to include
        agent_ids = list(self.participants.keys()) if include_all_agents else [self.agent_id]

        # ---- build AgentState list ----
        agents: List[AgentState] = []
        for aid in agent_ids:
            # TODO: plug your real max_time/horizon choices
            max_time = float(getattr(self, "planner_horizon_sec", 300.0))

            # mirror your sense constraints
            can_sense_X = (aid == "robot") or (aid == "human_a")
            can_sense_Y = (aid == "robot") or (aid == "human_b")

            agents.append(AgentState(
                agent_id=aid,
                max_time=max_time,
                can_sense_X=can_sense_X,
                can_sense_Y=can_sense_Y,
                can_dispose_X=True,   # per your rule: anyone can dispose any prop
                can_dispose_Y=True,
            ))

        # ---- build BoxInfo list ----
        box_infos: List[BoxInfo] = []
        for b in cand:
            # You already have belief + info functions in your policy code
            pX = float(self._belief_present_from_box(b, "X"))
            pY = float(self._belief_present_from_box(b, "Y"))
            infoX = float(self._info_level_for_box(b, "X"))  # implement if you don’t already have
            infoY = float(self._info_level_for_box(b, "Y"))

            already_sensed = {}
            for a in agents:
                already_sensed[a.agent_id] = {
                    "X": self._already_sensed(b, "X", a.agent_id),
                    "Y": self._already_sensed(b, "Y", a.agent_id),
                }

            box_infos.append(BoxInfo(
                box_id=int(b.box_id),
                deadline=float(b.deadline),
                sense_time_X=float(b.sense_time_X),
                sense_time_Y=float(b.sense_time_Y),
                dispose_time_X=float(b.dispose_time_X),
                dispose_time_Y=float(b.dispose_time_Y),
                p_true_X=pX,
                p_true_Y=pY,
                disposed_X=bool(b.disposed_X),
                disposed_Y=bool(b.disposed_Y),
                info_X=infoX,
                info_Y=infoY,
                already_sensed=already_sensed,
                # if you use your halving rule, keep these:
                min_disposal_team=int(getattr(b, "min_disposal_team", 1)),
                max_disposal_team=int(getattr(b, "max_disposal_team", len(agents))),
                senseable_X=bool(b.senseable["X"]),
                senseable_Y=bool(b.senseable["Y"]),
            ))

        box_map = {int(b.box_id): b for b in cand}

        def travel_time_fn(agent_id: str, box_id: int) -> float:
            b = box_map.get(int(box_id))
            if b is None:
                return 0.0
            dist = float(self._dist_to(float(b.x), float(b.y)))
            speed = float(getattr(self, "travel_speed_mps", 1.0))
            return dist / max(0.05, speed)


        horizon = float(getattr(self, "planner_horizon_sec", 60.0))
        return agents, box_infos, travel_time_fn, horizon


    def _plan_tuples_to_candidate_schema(self, plan_tuples):
        out: Dict[str, List[Dict[str, Any]]] = {}
        for aid, acts in (plan_tuples or {}).items():
            out[aid] = []
            for (box_id, prop, kind) in acts:
                out[aid].append({"kind": kind, "box_id": int(box_id), "prop": str(prop)})
        return out


    def _info_level_for_box(self, b: BoxSummary, prop: str) -> float:
        prop = str(prop).upper()

        # Use your current belief fusion (same as _score_step uses)
        p = float(self._belief_present_from_box(b, prop))

        # Convert belief -> info/confidence in [0,1]
        return float(info_level_from_p(p))

    def _already_sensed(self, b: BoxSummary, prop: str, agent_id: str) -> bool:
        """
        Returns True if this agent has already completed a sense(prop) on this box.
        """
        prop = str(prop).upper()
        aid = str(agent_id)

        for sr in (b.sense_results or []):
            if str(sr.get("property", "")).upper() != prop:
                continue

            # server uses either explicit status or completed_at timestamp
            status = sr.get("status")
            completed = (status == "completed") or ("completed_at" in sr)

            if not completed:
                continue

            if str(sr.get("agent_id", "")) == aid:
                return True

        return False



    def _sense_time_sec(self, b: BoxSummary, prop: Property) -> float:
        return float(getattr(b, f"sense_time_{prop}", 0.0))

    def _can_sense_for_feasibility(self, b: BoxSummary, prop: Property, *, agent_id: str) -> bool:
        """
        Feasibility-only sensing gate.
        If agent_id == 'robot', assume it can sense ANY property on ANY box (even if senseable says no).
        Everyone else uses normal _can_sense().
        """
        if str(agent_id) == "robot":
            return True
        return self._can_sense(b, prop, agent_id=agent_id)


    def _feasible_sense_then_dispose(self, b: BoxSummary, prop: Property, now_sim: float, *, assume_travel_zero: bool,
                                     assume_robot_can_sense: bool = False) -> bool:
        if self._disposed_any(b):
            return False

        already_sensed = any(
            sr.get("status") == "completed" and sr.get("property") == prop
            for sr in (b.sense_results or [])
        )

        travel = 0.0
        if not assume_travel_zero:
            travel = self._dist_to(b.x, b.y) / max(1e-6, float(self.speed_mps))

        t_dispose = self._optimistic_dispose_time_sec(b, prop, assume_robot_can_sense=assume_robot_can_sense)
        if not math.isfinite(t_dispose):
            return False

        if already_sensed:
            return (float(now_sim) + travel + float(t_dispose)) <= float(b.deadline)

        # need sensing; if we’re in “robot can sense anything” mode, allow it even if box says no.
        if assume_robot_can_sense:
            # if there is no robot participant, fall back to normal feasibility
            if "robot" not in self.participants:
                return False
        else:
            # normal world: require that *someone* can sense it (using real gating)
            if not any(self._can_sense(b, prop, agent_id=str(pid)) for pid in self.participants.keys()):
                return False

        t_sense = self._sense_time_sec(b, prop)
        t_finish = float(now_sim) + travel + float(t_sense) + float(t_dispose)
        return t_finish <= float(b.deadline)




    @staticmethod
    def _clamp01(x: float) -> float:
        try:
            x = float(x)
        except Exception:
            return 0.5
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    @staticmethod
    def _safe_prob(x: float, eps: float = 1e-4) -> float:
        """Clamp probability into (eps, 1-eps) to avoid log(0)."""
        x = SimHumanAgent._clamp01(x)
        if x < eps:
            return eps
        if x > 1.0 - eps:
            return 1.0 - eps
        return x

    def _latest_completed_senses_by_agent(
        self,
        sense_results: List[Dict[str, Any]],
        prop: Property,
    ) -> List[Dict[str, Any]]:
        """
        Pick latest completed sense per agent for a property to reduce
        overconfidence from repeated senses by same agent.
        """
        best: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        for sr in (sense_results or []):
            if sr.get("status") != "completed":
                continue
            if sr.get("property") != prop:
                continue

            aid = str(sr.get("agent_id") or "")
            if not aid:
                continue

            t = sr.get("completed_at")
            tv = float(t) if isinstance(t, (int, float)) else 0.0

            prev = best.get(aid)
            if prev is None or tv > prev[0]:
                best[aid] = (tv, sr)

        # return in time order (oldest->newest) for nice debug traces
        out = [pair[1] for pair in sorted(best.values(), key=lambda x: x[0])]
        return out

    def _bayes_fuse_present(
        self,
        evidence: List[Dict[str, Any]],
        prop: Property,
        prior: float = 0.5,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Bayesian fusion using detection params:
          present := P(det=1 | present)
          absent  := P(det=1 | absent)

        Updates log-odds with each (agent, detected) observation.
        Ignores sr["probability"] by design.
        """
        prior = self._safe_prob(prior)
        L = math.log(prior / (1.0 - prior))  # log-odds

        trace = []
        used = 0

        for sr in evidence:
            aid = str(sr.get("agent_id") or "")
            detected = sr.get("detected", None)
            if detected is None:
                continue

            sp = self.sensor_params.get(aid, {}).get(prop, None)
            if not isinstance(sp, dict):
                continue

            p_det_given_present = self._safe_prob(sp.get("present", 0.5))
            p_det_given_absent = self._safe_prob(sp.get("absent", 0.5))

            if bool(detected):
                lr = p_det_given_present / p_det_given_absent
                llr = math.log(lr)
                L += llr
                used += 1
                trace.append({
                    "agent": aid,
                    "det": True,
                    "present": round(float(p_det_given_present), 3),
                    "absent": round(float(p_det_given_absent), 3),
                    "llr": round(float(llr), 3),
                    "server_prob_ignored": sr.get("probability", None),
                })
            else:
                # det=False -> use (1-p_det | present) / (1-p_det | absent)
                lr = (1.0 - p_det_given_present) / (1.0 - p_det_given_absent)
                llr = math.log(lr)
                L += llr
                used += 1
                trace.append({
                    "agent": aid,
                    "det": False,
                    "present": round(float(p_det_given_present), 3),
                    "absent": round(float(p_det_given_absent), 3),
                    "llr": round(float(llr), 3),
                    "server_prob_ignored": sr.get("probability", None),
                })

        p = 1.0 / (1.0 + math.exp(-L))
        p = self._clamp01(p)

        details = {
            "prop": prop,
            "prior": round(float(prior), 3),
            "n_used": int(used),
            "p_posterior": round(float(p), 3),
            "trace": trace[-8:],  # keep it short
        }
        return p, details

    def _format_fusion_details(self, det: Dict[str, Any]) -> str:
        """
        Compact single-line debug string.
        """
        if not isinstance(det, dict):
            return ""
        parts = [
            f"prior={det.get('prior')}",
            f"n={det.get('n_used')}",
            f"post={det.get('p_posterior')}",
        ]
        tr = det.get("trace", [])
        if isinstance(tr, list) and tr:
            # e.g., human_a:+(llr=0.60) robot:-(llr=-1.2)
            evs = []
            for e in tr:
                if not isinstance(e, dict):
                    continue
                a = e.get("agent", "?")
                s = "+" if e.get("det") is True else "-"
                llr = e.get("llr", None)
                if llr is None:
                    evs.append(f"{a}:{s}")
                else:
                    evs.append(f"{a}:{s}(llr={llr})")
            parts.append("ev=[" + " ".join(evs) + "]")
        return " ".join(parts)


    def _disposed_any(self, b: BoxSummary) -> bool:
        # Your semantics: disposing either property disposes the whole object
        return bool(b.disposed_X) or bool(b.disposed_Y)

    def _op_matches(self, kind: str, box_id: int, prop: str) -> bool:
        op = self._get_current_op()
        if not op:
            return False
        return (
            str(op.get("kind")) == str(kind)
            and int(op.get("box_id")) == int(box_id)
            and str(op.get("prop")).upper() == str(prop).upper()
        )


    def _maybe_think(self, where: str = "") -> None:
        """
        Optional "thinking" delay before deciding what to do.
        Sleeps in small increments so shutdown stays responsive.
        """
        if not getattr(self, "think_sim_enable", False):
            return

        lo = max(0.0, float(getattr(self, "think_min_delay_sec", 0.0)))
        hi = max(lo, float(getattr(self, "think_max_delay_sec", lo)))

        if hi <= 0.0:
            return

        dt = random.uniform(lo, hi)

        # Log sparingly (you can remove if too chatty)
        self._log("THINK", f"{where} pause {dt:.2f}s")

        end = time.time() + dt
        while time.time() < end:
            if self._stop:
                break
            time.sleep(0.05)


    def _request_preempt(self, why: str = "") -> None:
        self._log("PREEMPT", why)

        # 1) cancel server op if we are in sense/dispose
        self._cancel_current_server_op()

        # 2) also cancel travel sleep / local work (your own cancel event)
        with self._cancel_lock:
            if self._cancel_evt is not None:
                self._cancel_evt.set()


    def _new_cancel_evt(self) -> threading.Event:
        with self._cancel_lock:
            self._cancel_evt = threading.Event()
            return self._cancel_evt


    def _set_current_op(self, kind: str, box_id: int, prop: str, now_sim: float) -> None:
        with self._op_lock:
            self._current_op = {"kind": kind, "box_id": int(box_id), "prop": str(prop), "started_sim": float(now_sim)}

    def _clear_current_op(self) -> None:
        with self._op_lock:
            self._current_op = None

    def _get_current_op(self) -> Optional[Dict[str, Any]]:
        with self._op_lock:
            return dict(self._current_op) if self._current_op else None


    def _cancel_current_server_op(self) -> bool:
        op = self._get_current_op()
        if not op:
            return False

        kind = op["kind"]
        box_id = int(op["box_id"])
        prop = str(op["prop"])

        try:
            if kind == "sense":
                r = self._http("POST", "/sense/cancel", json_body={"agent_id": self.agent_id, "box_id": box_id, "property": prop})
            elif kind == "dispose":
                r = self._http("POST", "/dispose/cancel", json_body={"agent_id": self.agent_id, "box_id": box_id, "property": prop})
            else:
                return False

            # if cancel succeeded or already done, treat as “we no longer own it”
            if r.status_code == 200:
                self._log("CANCEL", f"{kind} box={box_id} prop={prop} -> {r.json().get('status')}")
                if self._op_matches(kind, box_id, prop):
                    self._clear_current_op()
                return True
        except Exception as e:
            self.get_logger().warn(f"[CANCEL] failed: {e}")
        return False


    def _record_transcript(self, evt: Dict[str, Any]) -> None:
        if not self.collect_all_messages:
            return
        with self._transcript_lock:
            self.transcript.append(evt)
            while len(self.transcript) > self.collect_all_messages_max:
                self.transcript.popleft()

    def _get_transcript_tail(self, n: int) -> List[Dict[str, Any]]:
        with self._transcript_lock:
            tail = list(self.transcript)[-max(0, int(n)):]
        # keep it compact for prompts
        out = []
        for e in tail:
            out.append({
                "speaker_id": e.get("speaker_id"),
                "target_speaker": e.get("target_speaker"),
                "text": e.get("text"),
                "t_sim": e.get("t_sim"),
                "t_wall": e.get("t_wall"),
            })
        return out


    def _clear_waiting_help(self, why: str = "") -> None:
        if self.plan_state.get("phase") == "waiting_help":
            self._log("MEM", f"clear waiting_help {why}".strip())
        self.plan_state["phase"] = "explore"
        self.plan_state["waiting_help_box_id"] = None
        self.plan_state["waiting_help_prop"] = None
        self.plan_state["waiting_on"] = None
        self.plan_state["waiting_started_sim"] = None

    def _waiting_help_matches(self, box_id: int, prop: Property) -> bool:
        return (
            self.plan_state.get("phase") == "waiting_help"
            and self.plan_state.get("waiting_help_box_id") == int(box_id)
            and str(self.plan_state.get("waiting_help_prop")) == str(prop)
        )

    def _waiting_help_block_same_task(self, box_id: int, prop: Property, now_sim: float) -> bool:
        """True only if we're actively waiting AND it's for THIS exact (box_id, prop)."""
        return self._waiting_help_matches(box_id, prop) and self._waiting_help_active(now_sim)


    def _dbg_llm(self, tag: str, txt: str, max_chars: int = 100000) -> None:
        # keep logs readable
        s = txt if len(txt) <= max_chars else (txt[:max_chars] + f"...[trunc {len(txt)-max_chars} chars]")
        self._log("LLM_PROMPT", f"{tag}={s}")


    def _commitments(self) -> List[Dict[str, Any]]:
        self.plan_state.setdefault("commitments", [])
        return self.plan_state["commitments"]

    def _find_active_commitment(self, *, requester: str, box_id: int, prop: str) -> Optional[Dict[str, Any]]:
        for c in reversed(self._commitments()):
            if c.get("status") != "active":
                continue
            if c.get("from") == requester and int(c.get("box_id")) == int(box_id) and str(c.get("prop")) == str(prop):
                return c
        return None

    def _current_commitments(self, now_sim: Optional[float] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Return only CURRENT commitments (status == 'active').
        If now_sim is provided, also hide ones already expired by time.
        """
        out: List[Dict[str, Any]] = []
        for c in (self.plan_state.get("commitments", []) or []):
            if not isinstance(c, dict):
                continue
            if c.get("status") in ["done","expired","cancelled"]:
                continue
            if now_sim is not None:
                exp = c.get("expires_at", None)
                if isinstance(exp, (int, float)) and float(now_sim) > float(exp):
                    continue
            out.append(c)

        # newest last -> keep the most recent ones
        if limit > 0:
            out = out[-int(limit):]
        return out

    def _add_or_update_commitment(
        self,
        *,
        requester: str,
        box_id: Optional[int],
        prop: Optional[str],
        decision: str,
        now_sim: float,
        requested_kind: Optional[str] = None,
        due_after: Optional[float] = None,
        notes: str = "",
    ) -> None:
        if box_id is None or prop is None:
            return

        prop = str(prop).upper()
        if prop not in ("X", "Y"):
            return

        existing = self._find_active_commitment(requester=requester, box_id=int(box_id), prop=prop)
        if existing is None:
            cid = f"{requester}:{int(box_id)}:{prop}:{now_sim:.2f}"
            existing = {
                "id": cid,
                "from": requester,
                "box_id": int(box_id),
                "prop": prop,
                "requested_kind": requested_kind or "sense_self",
                "decision": decision,
                "status": "active",
                "created_at": float(now_sim),
                "due_after": float(due_after) if due_after is not None else float(now_sim),
                "expires_at": float(now_sim) + 60.0,
                "notes": notes,

                # ✅ scheduling knobs
                "priority": 10,            # lower = sooner
                "urgent_override": False,  # if True, can preempt while busy
                "blocked_on_busy": True,   # default: do after current action
            }

            self._commitments().append(existing)
            # cap list size
            self.plan_state["commitments"] = self.plan_state["commitments"][-30:]
        else:
            existing["decision"] = decision
            if requested_kind:
                existing["requested_kind"] = requested_kind
            if due_after is not None:
                existing["due_after"] = float(due_after)
            if notes:
                existing["notes"] = notes
            existing["expires_at"] = float(now_sim) + 60.0

    def _expire_old_commitments(self, now_sim: float) -> None:
        for c in self._commitments():
            if c.get("status") != "active":
                continue
            exp = c.get("expires_at", None)
            if isinstance(exp, (int, float)) and float(now_sim) > float(exp):
                c["status"] = "expired"

    def _next_executable_commitment(self, now_sim: float) -> Optional[Dict[str, Any]]:
        self._expire_old_commitments(now_sim)

        candidates = []
        busy = self._is_busy()# or self._is_speaking()


        for c in self._commitments():
            if c.get("status") != "active":
                continue
            if c.get("decision") not in ("accept", "defer", "negotiate"):
                continue
            if float(now_sim) < float(c.get("due_after", now_sim)):
                continue

            # ✅ if we're busy, only allow urgent_override commitments
            if busy and c.get("blocked_on_busy", True) and not bool(c.get("urgent_override", False)):
                continue

            candidates.append(c)

        if not candidates:
            return None

        # ✅ priority first, then oldest
        candidates.sort(key=lambda c: (int(c.get("priority", 10)), float(c.get("created_at", 0.0))))
        return candidates[0]


    def _complete_commitment(self, c: Dict[str, Any], status: str = "done") -> None:
        c["status"] = status


    def _memory_brief(self, limit: int = 30) -> List[Dict[str, Any]]:
        out = []
        for (box_id, prop), st in list(self._mem.items())[-limit:]:
            out.append({
                "box_id": box_id,
                "prop": prop,
                "status": st.get("status"),
                "asked_help_at_sim": st.get("asked_help_at_sim"),
                "asked_help_to": st.get("asked_help_to"),
                "ask_count": st.get("ask_count", 0),
                "self_sensed": st.get("self_sensed", False),
            })
        return out


    def _should_help_request(self, req: Dict[str, Any], now_sim: float, boxes: List[BoxSummary]) -> bool:
        """
        Decide whether to help based on trust, fairness, stubbornness, cooldown, urgency.
        """
        requester = str(req.get("from", ""))
        if not requester or requester == self.agent_id:
            return False

        # cooldown against repeated asks
        last = self.last_helped_at_sim.get(requester, None)
        if last is not None and (now_sim - float(last)) < float(self.help_cooldown_sec):
            return False

        trust = float(self.trust_map.get(requester, 0.5))

        # urgency for OUR goal (if we're near missing a deadline, help less)
        # quick proxy: best candidate deadline for our goal minus now
        slack = 9999.0
        
        for b in boxes:
            if self._is_done_or_abandoned(b.box_id, self.goal_property):
                continue
            if self._is_disposed_for_goal(b, self.goal_property):
                continue
            slack = min(slack, float(b.deadline) - now_sim)

        # social utility
        # - more fairness_sensitivity => more likely to help
        # - more stubbornness => less likely to help
        score = (
            0.55 * trust +
            0.30 * float(self.fairness_sensitivity) -
            0.25 * float(self.stubbornness)
        )

        # if we are very urgent (slack small), reduce willingness
        if slack < 20.0:
            score -= 0.25
        if slack < 10.0:
            score -= 0.25

        return score >= 0.45

    def _pop_help_request_action(self, boxes: List[BoxSummary], now_sim: float) -> Optional[PolicyAction]:
        """
        If there's a pending request, choose to help (sense) or ignore (say/idle).
        """
        if not self.inbox_requests:
            return None

        # newest first (or oldest first—pick one; newest tends to feel responsive)
        req = self.inbox_requests.pop(0)
        requester = str(req["from"])
        box_id = int(req["box_id"])
        prop = str(req["prop"]).upper()
        if prop not in ("X", "Y"):
            return None

        # ignore if box already has a completed sense for that prop (anyone)
        b = next((bb for bb in boxes if bb.box_id == box_id), None)
        if b is None:
            return None
        p = self._belief_present_from_box(b, prop)  # uses most recent completed
        already_sensed = any(
            sr.get("status") == "completed" and sr.get("property") == prop
            for sr in b.sense_results
        )
        if already_sensed:
            return PolicyAction(
                kind="say",
                text=f"{self._display_name(requester)}, box {box_id} already has a recent sense for {prop}.",
                reason="help_request_already_sensed",
            )

        if self._should_help_request(req, now_sim, boxes):
            self.help_history[requester] = self.help_history.get(requester, 0) + 1
            self.last_helped_at_sim[requester] = float(now_sim)
            return PolicyAction(
                kind="sense_self",
                box_id=box_id,
                prop=prop,  # note: may be non-goal property; this is “helping”
                text=f"Okay {self._display_name(requester)}, I’ll sense box {box_id} for {prop}.",
                reason=f"help_request_accept trust={self.trust_map.get(requester,0.5):.2f}",
            )

        # ignore / defer
        self.ignore_history[requester] = self.ignore_history.get(requester, 0) + 1
        return PolicyAction(
            kind="say",
            text=f"Sorry {self._display_name(requester)}, I can’t help right now.",
            reason="help_request_decline",
        )


    def _load_people_json(self, param_name: str) -> List[Dict[str, str]]:
        raw = str(self.get_parameter(param_name).value)
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                return []
            out = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                pid = str(item.get("id", "")).strip()
                if not pid:
                    continue
                name = str(item.get("name", pid)).strip()
                out.append({"id": pid, "name": name})
            return out
        except Exception:
            return []


    # ---------------------------
    # Logging
    # ---------------------------
    def _log(self, tag: str, msg: str) -> None:
        if not self.log_actions:
            return
        if tag in {"START", "HTTP"}:
            return
        self.get_logger().info(f"[{tag}] {msg}")


    def _journal_add(self, evt: Dict[str, Any]) -> int:
        """Append an action event and return its index (so we can update outcomes later)."""
        with self._journal_lock:
            self._action_journal.append(evt)
            return len(self._action_journal) - 1

    def _journal_update(self, idx: int, patch: Dict[str, Any]) -> None:
        with self._journal_lock:
            if 0 <= idx < len(self._action_journal):
                self._action_journal[idx].update(patch)

    def _print_action_summary(self, *, now_sim: float, time_limit: float) -> None:
        with self._journal_lock:
            rows = list(self._action_journal)

        self.get_logger().info("")
        self.get_logger().info("========== ACTION SUMMARY ==========")
        self.get_logger().info(f"agent_id={self.agent_id} goal={self.goal_property}  end_t={now_sim:.2f}/{time_limit:.2f}")
        if not rows:
            self.get_logger().info("(no actions recorded)")
            self.get_logger().info("====================================")
            self.get_logger().info("")
            return

        for i, e in enumerate(rows, start=1):
            t = e.get("t_sim", None)
            kind = e.get("kind", "")
            box_id = e.get("box_id", None)
            prop = e.get("prop", None)
            tgt = e.get("target_speaker", None)
            reason = e.get("reason", "")
            text = e.get("text", None)

            parts = [f"{i:03d}"]
            if isinstance(t, (int, float)):
                parts.append(f"t={float(t):.2f}")
            parts.append(f"kind={kind}")

            if box_id is not None:
                parts.append(f"box={box_id}")
            if prop is not None:
                parts.append(f"prop={prop}")
            if tgt:
                parts.append(f"to={tgt}")

            # outcomes (optional)
            if "status" in e:
                parts.append(f"status={e.get('status')}")
            if "success" in e:
                parts.append(f"success={e.get('success')}")
            if "detected" in e:
                parts.append(f"detected={e.get('detected')}")
            if "probability" in e:
                parts.append(f"p={e.get('probability')}")

            line = " | ".join(parts)

            # keep text/reason short to avoid spam
            if isinstance(text, str) and text.strip():
                line += f" | text={text.strip()[:160]!r}"
            if isinstance(reason, str) and reason.strip():
                line += f" | reason={reason.strip()[:160]!r}"

            self.get_logger().info(line)

        self.get_logger().info("====================================")
        self.get_logger().info("")

    def _request_shutdown_with_summary(self, *, now_sim: float, time_limit: float, why: str) -> None:
        """Print summary once, stop timers/threads, and shutdown ROS so the program exits."""
        with self._shutdown_lock:
            if self._shutdown_requested:
                return
            self._shutdown_requested = True

        self._log("TIME", f"Shutting down: {why}")

        # stop future work
        self._stop = True

        # cancel timers so no more callbacks fire
        try:
            if hasattr(self, "_action_timer") and self._action_timer is not None:
                self._action_timer.cancel()
        except Exception:
            pass
        try:
            if hasattr(self, "_router_timer") and self._router_timer is not None:
                self._router_timer.cancel()
        except Exception:
            pass

        # best-effort cancel any in-flight ops
        try:
            self._cancel_current_server_op()
        except Exception:
            pass
        try:
            with self._cancel_lock:
                if self._cancel_evt is not None:
                    self._cancel_evt.set()
        except Exception:
            pass

        # print step-by-step summary
        self._print_action_summary(now_sim=now_sim, time_limit=time_limit)

        # shutdown ROS -> rclpy.spin() returns -> program ends
        try:
            rclpy.shutdown()
        except Exception:
            pass


    # ---------------------------
    # Participant registry + profiles
    # ---------------------------
    def _safe_get_param(self, name: str, default):
        try:
            return self.get_parameter(name).value
        except Exception:
            return default

    def _display_name(self, pid: str) -> str:
        info = self.participants.get(pid)
        if info and info.get("name"):
            return str(info["name"])
        return pid


    def _build_participant_registry(self) -> None:
        # ✅ read JSON-string params (ROS-safe)
        humans = self._load_people_json("humans_json")
        robots = self._load_people_json("robots_json")

        regs: List[Dict[str, Any]] = []
        for h in humans:
            regs.append({"id": h["id"], "name": h["name"], "type": "human"})
        for r in robots:
            regs.append({"id": r["id"], "name": r["name"], "type": "robot"})

        # fallback if none provided: at least include self
        if not regs:
            regs = [{"id": self.agent_id, "name": self.agent_id, "type": "human"}]

        self.participants = {p["id"]: p for p in regs}

        # ensure self is included
        if self.agent_id not in self.participants:
            self.participants[self.agent_id] = {
                "id": self.agent_id,
                "name": self.agent_id,
                "type": "human",
            }

        # recompute id lists from participants (not regs, since we may have added self)
        self.human_ids = [pid for pid, p in self.participants.items() if p.get("type") == "human"]
        self.robot_ids = [pid for pid, p in self.participants.items() if p.get("type") == "robot"]

        # If default help target isn't listed, still allow asking it
        if self.help_target_speaker and self.help_target_speaker not in self.participants:
            self.participants[self.help_target_speaker] = {
                "id": self.help_target_speaker,
                "name": self.help_target_speaker,
                "type": "robot" if self.help_target_speaker == "robot" else "unknown",
            }
            # update lists
            if self.participants[self.help_target_speaker]["type"] == "robot":
                if self.help_target_speaker not in self.robot_ids:
                    self.robot_ids.append(self.help_target_speaker)

        self._log("PROFILE", f"participants={list(self.participants.values())}")


    @staticmethod
    def _sensor_skill_from_params(present: float, absent: float) -> float:
        present = max(0.0, min(1.0, float(present)))
        absent = max(0.0, min(1.0, float(absent)))
        return max(0.0, min(1.0, 0.6 * present + 0.4 * (1.0 - absent)))

    def _fetch_agent_params(self) -> Optional[Dict[str, Any]]:
        try:
            r = self._http("GET", "/agents/params")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            self.get_logger().warn(f"[WARN] failed to fetch /agents/params: {e}")
            return None

    def _init_profiles_from_server(self) -> None:
        self.sensor_params = {}
        self.trust_map = {}

        params = self._fetch_agent_params()
        agents = (params or {}).get("agents", {})
        default = (params or {}).get("default", None)

        for pid in self.participants.keys():
            raw = agents.get(pid, default)
            if not raw:
                continue
            self.sensor_params[pid] = {}
            for prop in ("X", "Y"):
                try:
                    present = float(raw[prop]["present"])
                    absent = float(raw[prop]["absent"])
                except Exception:
                    continue
                skill = self._sensor_skill_from_params(present, absent)
                self.sensor_params[pid][prop] = {"present": present, "absent": absent, "skill": skill}

        # Parse trust overrides ONCE
        trust_overrides_raw = str(self.get_parameter("trust_overrides_json").value)
        try:
            trust_overrides = json.loads(trust_overrides_raw)
            if not isinstance(trust_overrides, dict):
                trust_overrides = {}
        except Exception:
            trust_overrides = {}

        for pid in self.participants.keys():
            if pid == self.agent_id:
                continue
            sp = self.sensor_params.get(pid, {}).get(self.goal_property)
            base = float(sp["skill"]) if sp else 0.5
            if self.participants.get(pid, {}).get("type") == "robot":
                base = min(1.0, base + 0.1)

            # apply override if provided
            if pid in trust_overrides:
                try:
                    base = float(trust_overrides[pid])
                except Exception:
                    pass

            self.trust_map[pid] = base




        self._log("PROFILE", f"trust_map={json.dumps({k: round(v,2) for k,v in self.trust_map.items()})}")

    def _choose_best_helper(self, goal_prop: Property) -> Optional[str]:
        """
        Choose helper with max (0.65*trust + 0.35*sensor_skill_goal).
        """
        best_id = None
        best_score = -1.0
        for pid in self.participants.keys():
            if pid == self.agent_id:
                continue
            trust = float(self.trust_map.get(pid, 0.5))
            skill = float(self.sensor_params.get(pid, {}).get(goal_prop, {}).get("skill", 0.5))
            score = 0.65 * trust + 0.35 * skill
            if score > best_score:
                best_score = score
                best_id = pid
        return best_id

    def _set_busy(self, v: bool) -> None:
        with self._busy_lock:
            self._busy = bool(v)

    def _is_busy(self) -> bool:
        with self._busy_lock:
            return bool(self._busy)


    def _infer_target_llm(self, speaker_id: str, text: str, now_sim: float) -> Optional[str]:
        if not self.infer_target_use_llm or self.llm_provider != "openai":
            return None

        client = self.llm_policy._get_client(self)  # reuse existing OpenAI client init
        if client is None:
            return None

        # Build short context: last few dialogue turns + participant roster
        tail = self._get_transcript_tail(self.infer_target_max_history)
        hist = [{
            "speaker_id": e.get("speaker_id"),
            "target_speaker": e.get("target_speaker"),
            "text": e.get("text"),
        } for e in tail]


        roster = [{"id": pid, "name": self._display_name(pid), "type": self.participants.get(pid, {}).get("type", "unknown")}
                  for pid in self.participants.keys()]

        sys_msg = (
            "You are a message recipient classifier in a multi-agent chat.\n"
            "Given a new message that omitted an explicit target_speaker, infer who it is addressed to.\n"
            "Return ONLY JSON with keys: target_speaker and confidence.\n"
            "target_speaker must be one of the participant ids, or \"all\".\n"
            "If ambiguous, choose \"all\".\n"
        )

        user_obj = {
            "time": round(now_sim, 2),
            "you_are": self.agent_id,
            "participants": roster,
            "recent_dialogue": hist,
            "incoming": {"speaker_id": speaker_id, "text": text},
            "output_schema": {"target_speaker": "string", "confidence": "number(0..1)"},
        }

        # ✅ NEW: log the prompt we send to the infer-target LLM
        #self.get_logger().info(f"INFER_TARGET_SYSTEM={sys_msg}")
        #self.get_logger().info(f"INFER_TARGET_USER={json.dumps(user_obj)}")


        try:
            resp = client.responses.create(
                model=self.llm_model,
                input=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": json.dumps(user_obj)},
                ],
                max_output_tokens=80,
            )
            raw = resp.output_text
        except Exception as e:
            self.get_logger().warn(f"[LLM] infer_target call failed: {e}")
            return None

        try:
            data = json.loads(raw)
        except Exception:
            return None

        tgt = str(data.get("target_speaker", "")).strip()
        conf = data.get("confidence", 0.0)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0

        allowed = set(self.participants.keys()) | {"all"}
        if tgt not in allowed:
            return None

        # If it’s low confidence, treat as broadcast (prevents accidental ignoring)
        if conf < 0.55:
            return "all"

        return tgt

    def _infer_target_speaker(self, speaker_id: str, text: str, now_sim: float) -> str:

        # 2) LLM (may return self / all / someone else)
        llm = self._infer_target_llm(speaker_id, text, now_sim)
        if llm is None:
            return "all"  # safe fallback
        return llm


    # ---------------------------
    # ROS bus I/O
    # ---------------------------
    def _on_stt_text(self, msg: StringMsg) -> None:
    
        if not getattr(self, "comm_enable", False):
            return

    
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                return

            speaker = str(payload.get("speaker_id", "")).strip()
            text = payload.get("text")
            target = payload.get("target_speaker", None)


            if not speaker or not isinstance(text, str):
                return

            # ✅ record everything seen on the bus
            self._record_transcript({
                "t_wall": time.time(),
                "t_sim": None,  # you can fill this in router thread where you know now_sim
                "speaker_id": speaker,
                "target_speaker": (str(target) if target is not None else None),
                "text": text,
                "raw": payload,
            })

            # ----- existing routing filters -----
            if target is not None and target and str(target) not in (self.agent_id, "all"):
                return

            if speaker == self.agent_id:
                return

        except Exception:
            self._log("whatt??","")
            return

        event = {"speaker_id": speaker, "text": text, "t_wall": time.time()}
        self.last_msgs.append(event)
        self.last_msgs = self.last_msgs[-100:]

        with self._inbox_lock:
            self.inbox.append(event)
            while len(self.inbox) > 50:
                self.inbox.popleft()


        self._log("HEAR", f"{RED}from={speaker} text={text!r}{RESET}")

    def _set_speech_busy(self, v: bool) -> None:
        with self._speech_busy_lock:
            self._speech_busy = bool(v)

    def _is_speaking(self) -> bool:
        with self._speech_busy_lock:
            return bool(self._speech_busy)


    def _publish_utterance(self, text: str, target_speaker: Optional[str] = None) -> None:
        """
        Blocking utterance: do not return until the speech worker has finished
        "speaking" and the message has been published.
        """
        if not getattr(self, "comm_enable", False):
            return

        
        # If we're somehow calling from the speech thread itself, don't deadlock.
        if getattr(self, "_speech_thread", None) is not None and threading.current_thread() is self._speech_thread:
            # publish immediately (no simulated delay here)
            out = StringMsg()

            prefix = ""
            if target_speaker and target_speaker not in ("all", ""):
                prefix = self._display_name(str(target_speaker))

            final_text = text.strip() if isinstance(text, str) else ""
            if not final_text:
                return

            if prefix and (prefix not in final_text):
                final_text = "Hey " + prefix + ", " + final_text

            payload = {"text": final_text, "speaker_id": self.agent_id}
            if target_speaker:
                payload["target_speaker"] = str(target_speaker)

            if self.no_target_speaker:
                payload["target_speaker"] = ""
            out.data = json.dumps(payload)
            self.pub_stt.publish(out)
            self._log("SAY", f"{YELLOW}{final_text}{RESET}")
            return

        # Normal path: block until done
        self._speak_and_wait(text, target_speaker=target_speaker)




    def _enqueue_utterance(self, text: str, target_speaker: Optional[str], *, done_evt: Optional[threading.Event] = None) -> None:
        if not isinstance(text, str):
            if done_evt:
                done_evt.set()
            return
        text = text.strip()
        if not text:
            if done_evt:
                done_evt.set()
            return

        with self._speech_cv:
            if len(self._speech_queue) >= int(self.speech_queue_max):
                # drop oldest; also unblock whoever was waiting on it
                dropped = self._speech_queue.popleft()
                ev = dropped.get("done_evt")
                if isinstance(ev, threading.Event):
                    ev.set()

            self._speech_queue.append({
                "text": text,
                "target_speaker": target_speaker,
                "t_enq": time.time(),
                "done_evt": done_evt,  # ✅ let caller wait until finished
            })
            self._speech_cv.notify()

    def _speak_and_wait(self, text: str, target_speaker: Optional[str] = None, *, timeout: Optional[float] = None) -> None:
        if not getattr(self, "comm_enable", False):
            return

        th = getattr(self, "_speech_thread", None)
        if th is None or (hasattr(th, "is_alive") and not th.is_alive()):
            self.get_logger().warn("[SPEECH] worker thread not alive -> publishing immediately")
            self._publish_utterance_now(text, target_speaker)
            return


        ev = threading.Event()
        self._enqueue_utterance(text, target_speaker, done_evt=ev)

        if timeout is None:
            timeout = max(5.0, float(self.speech_max_delay_sec) + 5.0)

        ok = ev.wait(timeout=timeout)
        if not ok:
            self.get_logger().warn(
                f"[SPEECH] timeout waiting publish (qlen={len(self._speech_queue)}) "
                f"text={str(text)[:120]!r}"
            )
            # Optional safety net: publish immediately so we never go silent
            self._publish_utterance_now(text, target_speaker)



    def _publish_utterance_now(self, text: str, target_speaker: Optional[str] = None) -> None:
        if not getattr(self, "comm_enable", False):
            return
        if self.pub_stt is None:
            return

        final_text = text.strip() if isinstance(text, str) else ""
        if not final_text:
            return

        prefix = ""
        if target_speaker and target_speaker not in ("all", ""):
            prefix = self._display_name(str(target_speaker))
        if prefix and (prefix not in final_text):
            final_text = "Hey " + prefix + ", " + final_text

        payload = {"text": final_text, "speaker_id": self.agent_id}
        if target_speaker:
            payload["target_speaker"] = str(target_speaker)
        if self.no_target_speaker:
            payload["target_speaker"] = ""

        out = StringMsg()
        out.data = json.dumps(payload)
        self.pub_stt.publish(out)
        self._log("SAY", f"{YELLOW}{final_text}{RESET}")



    def _speech_worker_main(self) -> None:
        while not self._speech_stop_evt.is_set():
            item = None
            with self._speech_cv:
                while not self._speech_queue and not self._speech_stop_evt.is_set():
                    self._speech_cv.wait(timeout=0.2)
                if self._speech_stop_evt.is_set():
                    break
                try:
                    item = self._speech_queue.popleft()
                except Exception as e:
                    self.get_logger().warn(f"[SPEECH] pop failed: {e}")
                    item = None

            if not item:
                continue

            done_evt = item.get("done_evt", None)
            if done_evt is not None and not isinstance(done_evt, threading.Event):
                done_evt = None

            try:
                self._set_speech_busy(True)

                text = str(item.get("text", "")).strip()
                target_speaker = item.get("target_speaker", None)
                if target_speaker is not None:
                    target_speaker = str(target_speaker)

                if not text:
                    return

                # ---- simulated speaking ----
                if self.speech_sim_enable:
                    dt = self._estimate_speech_delay_sec(text)
                    end = time.time() + dt
                    while time.time() < end:
                        if self._speech_stop_evt.is_set() or self._stop:
                            break
                        time.sleep(0.05)
                    if self._speech_stop_evt.is_set() or self._stop:
                        return

                # ---- build message ----
                out = StringMsg()

                prefix = ""
                if target_speaker and target_speaker not in ("all", ""):
                    prefix = self._display_name(target_speaker)

                final_text = text
                if prefix and (prefix not in final_text):
                    final_text = "Hey " + prefix + ", " + final_text

                payload = {"text": final_text, "speaker_id": self.agent_id}
                if target_speaker:
                    payload["target_speaker"] = target_speaker
                if self.no_target_speaker:
                    payload["target_speaker"] = ""

                out.data = json.dumps(payload)

                # ---- publish ----
                if self.pub_stt is None:
                    raise RuntimeError("pub_stt is None (comm_enable?)")

                self.pub_stt.publish(out)
                self._log("SAY", f"{YELLOW}{final_text}{RESET}")

            except Exception as e:
                # THIS is what you’re missing right now
                self.get_logger().warn(f"[SPEECH] worker item failed: {type(e).__name__}: {e}")

            finally:
                self._set_speech_busy(False)
                if isinstance(done_evt, threading.Event):
                    done_evt.set()


            
    def _estimate_speech_delay_sec(self, text: str) -> float:
        """
        Estimate speaking duration from text length.
        - Base is words / WPM
        - Add small pauses for punctuation
        - Clamp to [min, max]
        """
        if not text:
            return 0.0

        # word count (simple + robust)
        words = re.findall(r"\b\w+\b", text)
        n_words = max(1, len(words))

        wpm = max(60.0, float(self.speech_rate_wpm))  # guardrail
        sec_words = (n_words / wpm) * 60.0

        # punctuation micro-pauses
        punct = re.findall(r"[,.!?;:]", text)
        sec_punct = float(len(punct)) * float(self.speech_punct_pause_sec)

        dt = sec_words + sec_punct
        dt = max(float(self.speech_min_delay_sec), dt)
        dt = min(float(self.speech_max_delay_sec), dt)
        return float(dt)

    # ---------------------------
    # Server HTTP helpers
    # ---------------------------
    def _http(self, method: str, path: str, *, json_body: Optional[dict] = None) -> requests.Response:
        url = self.base_url + path
        t0 = time.time()
        self._log("HTTP", f"{method} {path} body={json_body} timeout={self.timeout}s")
        if method == "GET":
            r = requests.get(url, timeout=self.timeout)
        elif method == "POST":
            r = requests.post(url, json=json_body, timeout=self.timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")
        dt = time.time() - t0
        self._log("HTTP", f"done {method} {path} status={r.status_code} dt={dt:.3f}s")
        return r

    def _time(self) -> Dict[str, Any]:
        r = self._http("GET", "/time")
        r.raise_for_status()
        return r.json()

    def _boxes_state(self) -> List[BoxSummary]:
        r = self._http("GET", "/boxes/state")
        r.raise_for_status()
        raw = r.json()
        out: List[BoxSummary] = []
        for b in raw:
        
            # Backward/forward compatible senseable parsing
            if "senseable" in b and isinstance(b["senseable"], dict):
                senseable = dict(b["senseable"])
            else:
                senseable = {
                    "X": bool(b.get("senseable_X", True)),
                    "Y": bool(b.get("senseable_Y", True)),
                }
        
            sense_time_X = float(b.get("sense_time_X", 0.0))
            sense_time_Y = float(b.get("sense_time_Y", 0.0))
            dispose_time_X = float(b.get("dispose_time_X", 0.0))
            dispose_time_Y = float(b.get("dispose_time_Y", 0.0))


        
            out.append(
                BoxSummary(
                    box_id=int(b["box_id"]),
                    x=float(b["x"]),
                    y=float(b["y"]),
                    deadline=float(b["deadline"]),
                    disposed_X=bool(b["disposed_X"]),
                    disposed_Y=bool(b["disposed_Y"]),
                    sense_results=list(b.get("sense_results", [])),
                    senseable=senseable,
                    senseable_by=b.get("senseable_by", None),
                    sense_time_X=sense_time_X,
                    sense_time_Y=sense_time_Y,
                    dispose_time_X=dispose_time_X,
                    dispose_time_Y=dispose_time_Y,
                )
            )
        return out


    def _sense(self, box_id: int, prop: Property) -> Dict[str, Any]:
        r = self._http("POST", "/sense", json_body={"agent_id": self.agent_id, "box_id": box_id, "property": prop})
        r.raise_for_status()
        js = r.json()
        self._log("SENSE", f"box={box_id} prop={prop} status={js.get('status')} detected={js.get('detected')} prob={js.get('probability')}")
        return js

    def _dispose(self, box_id: int, prop: Property) -> Dict[str, Any]:
        r = self._http("POST", "/dispose", json_body={"agent_id": self.agent_id, "box_id": box_id, "property": prop})
        r.raise_for_status()
        js = r.json()
        
        self._log("DISPOSE", f"box={box_id} prop={prop} status={js.get('status')} success={js.get('success')}")
        return js

    # ---------------------------
    # Belief / state helpers
    # ---------------------------
    def _dist_to(self, x: float, y: float) -> float:
        return math.hypot(x - self.pose.x, y - self.pose.y)

    def _is_disposed_for_goal(self, b: BoxSummary, prop: Property) -> bool:
        # ✅ semantic change: disposed in either property means fully disposed
        return self._disposed_any(b)


    def _box_key(self, box_id: int, prop: Property) -> Tuple[int, str]:
        return (int(box_id), str(prop))

    def _box_state(self, box_id: int, prop: Property) -> Dict[str, Any]:
        k = self._box_key(box_id, prop)
        if k not in self._mem:
            self._mem[k] = {"status": "unknown"}  # unknown|done|abandoned
        return self._mem[k]

    def _is_done_or_abandoned(self, box_id: int, prop: Property) -> bool:
        st = self._box_state(box_id, prop)
        return st.get("status") in ("done", "abandoned")

    def _mark_done(self, box_id: int, prop: Property, why: str = "") -> None:
        st = self._box_state(box_id, prop)
        st["status"] = "done"
        st["done_why"] = why
        self._log("MEM", f"done box={box_id} prop={prop} why={why}")

    def _mark_abandoned(self, box_id: int, prop: Property, why: str = "") -> None:
        st = self._box_state(box_id, prop)
        st["status"] = "abandoned"
        st["abandoned_why"] = why
        self._log("MEM", f"abandon box={box_id} prop={prop} why={why}")

    def _belief_present_from_box(self, box: BoxSummary, prop: Property) -> float:
        """
        Fuse completed sense results using *known* sensor parameters (self.sensor_params)
        and the detected boolean. Ignores sr['probability'].

        Implementation:
          - use latest completed sense per agent for this prop (reduces double counting)
          - Bayes log-odds fusion
          - cache fusion details in per-(box,prop) memory so DECIDE logs can print it
        """
        # collect evidence
        evidence = self._latest_completed_senses_by_agent(box.sense_results, prop)

        prior = prior = self._prior_for_box(box, prop) #self._prior_for(prop)

        if not evidence:
            st = self._box_state(box.box_id, prop)
            st["fusion_details"] = {"prop": prop, "prior": round(prior,3), "n_used": 0, "p_posterior": round(prior,3), "trace": []}
            return prior

        p, details = self._bayes_fuse_present(evidence, prop, prior=prior)

        # cache details so you can print/report later
        st = self._box_state(box.box_id, prop)
        st["fusion_details"] = details
        return float(p)


    def _waiting_help_active(self, now_sim: float) -> bool:
        if self.plan_state.get("phase") != "waiting_help":
            return False
        started = self.plan_state.get("waiting_started_sim", None)
        if not isinstance(started, (int, float)):
            return False
        waited = float(now_sim) - float(started)
        return waited < float(self.help_wait_sec)


    # ---------------------------
    # Movement / execution
    # ---------------------------
    def _travel_to(self, box: BoxSummary) -> None:
        dist = self._dist_to(box.x, box.y)
        travel_sec = dist / max(1e-6, self.speed_mps)
        self._log("TRAVEL", f"start box={box.box_id} from=({self.pose.x:.2f},{self.pose.y:.2f}) to=({box.x:.2f},{box.y:.2f}) dist={dist:.2f}m t={travel_sec:.2f}s")
        time.sleep(travel_sec)
        self.pose = Pose2D(box.x, box.y)
        self._log("TRAVEL", f"done  box={box.box_id} now=({self.pose.x:.2f},{self.pose.y:.2f})")

    def _finish_action_phase(self) -> None:
        # Don’t stomp waiting_help state
        if self.plan_state.get("phase") == "waiting_help":
            return
        self.plan_state["phase"] = "explore"
        self.plan_state["focus_box_id"] = None
        self.plan_state["focus_prop"] = self.goal_property

    def _current_op_remaining(self, now_sim: float, box_lookup: Dict[int, BoxSummary]) -> Optional[Dict[str, Any]]:
        op = self._get_current_op()
        if not op:
            return None

        kind = str(op.get("kind"))
        box_id = int(op.get("box_id"))
        prop = str(op.get("prop")).upper()
        started = float(op.get("started_sim", now_sim))

        b = box_lookup.get(box_id)
        if b is None:
            return None

        if kind == "sense":
            dur = float(getattr(b, f"sense_time_{prop}", 0.0))
        elif kind == "dispose":
            dur = float(getattr(b, f"dispose_time_{prop}", 0.0))
        else:
            return None

        # If dur is 0, we can’t estimate; treat as unknown
        if dur <= 0.0:
            return {"kind": kind, "box_id": box_id, "prop": prop, "remaining": None}

        remaining = max(0.0, (started + dur) - float(now_sim))

        return {
            "kind": kind,
            "box_id": box_id,
            "prop": prop,
            "started_sim": started,
            "duration": dur,
            "remaining": remaining,
        }



    def _execute(self, action: PolicyAction, box_lookup: Dict[int, BoxSummary], now_sim: float) -> None:
        self._log("ACT", f"execute kind={action.kind} box={action.box_id} prop={action.prop} reason={action.reason}")

        j_idx = None
        if action.kind != "idle":
            j_idx = self._journal_add({
                "t_sim": float(now_sim),
                "kind": action.kind,
                "box_id": action.box_id,
                "prop": action.prop,
                "target_speaker": action.target_speaker,
                "text": action.text,
                "reason": action.reason,
            })




        if self.plan_state.get("phase") == "waiting_help" and self.waiting_mode == "soft":
            w_box = self.plan_state.get("waiting_help_box_id")
            w_prop = self.plan_state.get("waiting_help_prop")
            if (
                w_box is not None and w_prop is not None
                and action.box_id is not None and action.prop is not None
                and int(action.box_id) == int(w_box)
                and str(action.prop).upper() == str(w_prop).upper()
                and action.kind in ("ask_help", "sense_self", "dispose", "goto_only")
            ):
                self._log("MEM", f"blocked by waiting_help soft same-task kind={action.kind} box={action.box_id} prop={action.prop}")
                return





        cancel_evt = None
        # update focus for real physical actions (not ask_help; we update that inside ask_help handler)
        if action.kind in ("sense_self", "dispose", "goto_only"):
            cancel_evt = self._new_cancel_evt()
            if action.box_id is not None and action.prop is not None:
                self.plan_state["focus_box_id"] = int(action.box_id)
                self.plan_state["focus_prop"] = str(action.prop)
                self.plan_state["phase"] = (
                    "sense" if action.kind == "sense_self"
                    else "dispose" if action.kind == "dispose"
                    else "goto"
                )

        if action.text:
            self.plan_state["last_commitment"] = action.text
        else:
            self.plan_state["last_commitment"] = ""

        if action.kind == "idle":
            return

        if action.kind == "say":
            if action.text:
                # ✅ default recipient if not provided
                if not action.target_speaker:
                    # if we're in a conversation context, aim it
                    if self.plan_state.get("phase") == "waiting_help":
                        action.target_speaker = str(self.plan_state.get("waiting_on") or "")
                    else:
                        # if last message exists, respond to them
                        if self.last_msgs:
                            action.target_speaker = str(self.last_msgs[-1].get("speaker_id") or "")
                    if not action.target_speaker:
                        action.target_speaker = "all"

                # ✅ block until speaking finished
                self._speak_and_wait(action.text, target_speaker=action.target_speaker)
            return




        if action.kind == "ask_help":
            if action.box_id is not None and action.prop is not None:
                st = self._box_state(action.box_id, action.prop)
                last_asked = st.get("asked_help_at_sim", None)

                # If we asked recently, suppress BOTH memory update AND speech output
                if last_asked is not None and (now_sim - float(last_asked)) < self.help_wait_sec:
                    self._log("MEM", f"suppress ask_help repeat box={action.box_id} prop={action.prop} waited={now_sim-float(last_asked):.1f}s")
                    return

                # record and speak
                st["asked_help_at_sim"] = float(now_sim)
                st["asked_help_to"] = action.target_speaker or self.help_target_speaker
                self._log("MEM", f"asked_help box={action.box_id} prop={action.prop} to={st['asked_help_to']} at_sim={now_sim:.2f}")

                # ✅ set explicit waiting phase metadata (Fix 2)
                self.plan_state["phase"] = "waiting_help"
                self.plan_state["waiting_help_box_id"] = int(action.box_id)
                self.plan_state["waiting_help_prop"] = str(action.prop)
                self.plan_state["waiting_on"] = str(st["asked_help_to"])
                self.plan_state["waiting_started_sim"] = float(now_sim)

            if not action.text or not action.text.strip():
                who = action.target_speaker or self.help_target_speaker or "someone"
                action.text = f"{self._display_name(who)}, can you sense box {action.box_id} for {action.prop}? I'm unsure."


            if action.text:
                # ✅ block until speaking finished
                self._speak_and_wait(action.text, target_speaker=action.target_speaker)
            return



        if action.box_id is None or action.box_id not in box_lookup:
            self._log("WARN", f"missing box in lookup for action: {action}")
            return

        box = box_lookup[action.box_id]

        if action.kind in ("sense_self", "dispose", "goto_only") and box is not None:
            if float(now_sim) > float(box.deadline):
                self._log("MEM", f"skip {action.kind} box={box.box_id} (deadline passed) now={now_sim:.2f} deadline={box.deadline:.2f}")
                self._complete_active_commitment_if_any(status="cancelled")
                self._finish_action_phase()
                return

        # ---------------------------
        # ✅ Rule 1: if we are already doing EXACTLY this sense/dispose, keep going.
        # ---------------------------
        if action.kind in ("sense_self", "dispose"):
            assert action.prop is not None

            # If we have an in-flight op that matches, do not cancel/restart.
            if self._op_matches("sense" if action.kind == "sense_self" else "dispose",
                                int(box.box_id), str(action.prop)):

                self._log(
                    "ACT",
                    f"dedupe: already doing {action.kind} box={box.box_id} prop={action.prop} -> keep going"
                )
                return


        if action.kind == "goto_only":
            self._set_busy(True)
            try:
                self._travel_to(box)
            finally:
                self._set_busy(False)
                self._finish_action_phase()
            return


        if action.kind == "sense_self":
            assert action.prop is not None
            st = self._box_state(box.box_id, action.prop)

            already_by_anyone = any(
                sr.get("status") == "completed" and sr.get("property") == action.prop
                for sr in box.sense_results
            )
            
            if not self._can_sense(box, action.prop, agent_id=self.agent_id):
                self._log("MEM", f"skip sense_self box={box.box_id} prop={action.prop} (not senseable by me)")
                self._complete_active_commitment_if_any(status="cancelled")
                # mark abandoned for this (box,prop) so policy doesn’t keep trying
                self._mark_abandoned(box.box_id, action.prop, why="not_senseable_by_me")
                return

            
            # ✅ Rule 2: do not sense ANY property if object already disposed (either flag true)
            if self._disposed_any(box):
                self._log("MEM", f"skip sense_self box={box.box_id} prop={action.prop} (already disposed-any)")
                self._complete_active_commitment_if_any(status="done")
                # mark both props done locally so planner stops thinking about it
                self._mark_done(box.box_id, "X", why="skip_sense_disposed_any")
                self._mark_done(box.box_id, "Y", why="skip_sense_disposed_any")
                return

            

            '''
            if already_by_anyone:
                self._log("MEM", f"skip self_sense box={box.box_id} prop={action.prop} (already sensed by someone)")
                st["self_sensed"] = True
                # if we were doing this as a commitment, fulfill it
                self._complete_active_commitment_if_any(status="done")
                return
            '''
            if st.get("self_sensed", False):
                self._log("MEM", f"skip repeat self_sense box={box.box_id} prop={action.prop} (already self_sensed)")
                self._complete_active_commitment_if_any(status="done")
                return

            already_by_me = any(
                sr.get("status") == "completed"
                and sr.get("property") == action.prop
                and str(sr.get("agent_id")) == self.agent_id
                for sr in box.sense_results
            )
            if already_by_me:
                st["self_sensed"] = True
                self._log("MEM", f"skip self_sense box={box.box_id} prop={action.prop} (server already has my completed sense)")
                self._complete_active_commitment_if_any(status="done")
                return

            self._set_busy(True)
            try:
                self._travel_to(box)
                tstart = float(self._time()["server_time"])
                self._set_current_op("sense", box.box_id, action.prop, tstart)
                js = self._sense(box.box_id, action.prop)
                
                if j_idx is not None:
                    self._journal_update(j_idx, {
                        "status": js.get("status"),
                        "detected": js.get("detected"),
                        "probability": js.get("probability"),
                    })

                
                st["self_sensed"] = True
                st["last_self_sense_status"] = js.get("status")
                # ✅ mark commitment done after success
                self._complete_active_commitment_if_any(status="done")
            finally:
                self._clear_current_op()
                self._set_busy(False)
                self._finish_action_phase()
            return


        if action.kind == "dispose":
            assert action.prop is not None

            # ✅ Rule 2: if disposed for either prop, don't dispose again
            if self._disposed_any(box):
                self._log("MEM", f"skip dispose box={box.box_id} prop={action.prop} (already disposed-any)")
                self._complete_active_commitment_if_any(status="done")
                self._mark_done(box.box_id, "X", why="already_disposed_any")
                self._mark_done(box.box_id, "Y", why="already_disposed_any")
                return

            
            self._set_busy(True)
            try:
                self._travel_to(box)
                tstart = float(self._time()["server_time"])
                self._set_current_op("dispose", box.box_id, action.prop, tstart)
                js = self._dispose(box.box_id, action.prop)
                
                if j_idx is not None:
                    self._journal_update(j_idx, {
                        "status": js.get("status"),
                        "success": js.get("success"),
                    })

                
                self._complete_active_commitment_if_any(status="done")
                
                status = js.get("status")
                success = js.get("success")

                if status == "completed" and success is True:
                    # disposing either property disposes the whole object
                    self._mark_done(box.box_id, "X", why="disposed_any")
                    self._mark_done(box.box_id, "Y", why="disposed_any")

                    # Optional: clear “ask help / self sensed” flags so you don’t keep stale state around
                    for p in ("X", "Y"):
                        st = self._box_state(box.box_id, p)
                        st.pop("asked_help_at_sim", None)
                        st.pop("asked_help_to", None)
                        st.pop("ask_count", None)
                        st.pop("self_sensed", None)
                        st.pop("fusion_details", None)

                    self._complete_active_commitment_if_any(status="done")

                else:
                    # cancelled/failed/in_progress -> do NOT mark done
                    self._complete_active_commitment_if_any(status="cancelled" if status == "cancelled" else "active")

            finally:
                self._clear_current_op()
                self._set_busy(False)
                self._finish_action_phase()
            return


    def _complete_active_commitment_if_any(self, status: str = "done") -> None:
        cid = self.plan_state.get("active_commitment_id")
        if not cid:
            return
        for cc in self.plan_state.get("commitments", []):
            if cc.get("id") == cid and cc.get("status") == "active":
                self._complete_commitment(cc, status=status)
                break
        self.plan_state["active_commitment_id"] = None


    # ---------------------------
    # Thread runner and tick
    # ---------------------------
    def _run_one_cycle(self) -> None:
        t = self._time()
        now_sim = float(t["server_time"])
        time_limit = float(t["time_limit_sec"])
        if now_sim >= time_limit:
            self._log("TIME", f"limit reached server_time={now_sim:.2f} >= {time_limit:.2f}")
            self._request_shutdown_with_summary(
                now_sim=now_sim,
                time_limit=time_limit,
                why="time_limit_reached (action loop)",
            )
            return


        boxes = self._boxes_state()
        box_lookup = {b.box_id: b for b in boxes}

        # initialize snapshots once to avoid "everything is new" at t=0
        if not self._last_box_beliefs and boxes:
            for b in boxes:
                bid = int(b.box_id)
                pX = float(self._belief_present_from_box(b, "X"))
                pY = float(self._belief_present_from_box(b, "Y"))
                self._last_box_beliefs[bid] = {
                    "pX": pX, "pY": pY,
                    "infoX": float(info_level_from_p(pX)),
                    "infoY": float(info_level_from_p(pY)),
                }

        # ✅ belief-change detection -> replan -> enqueue ego order utterance
        self._maybe_replan_and_enqueue_ego_order(boxes, now_sim)


        op_rem = self._current_op_remaining(now_sim, box_lookup)
        self._op_remaining_cache = op_rem


        # ✅ SHUTDOWN: all deadlines passed
        if boxes and all(float(now_sim) > float(b.deadline) for b in boxes):
            self._request_shutdown_with_summary(
                now_sim=now_sim,
                time_limit=time_limit,
                why="all_box_deadlines_passed",
            )
            return

        if not self.comm_enable:
            # ensure we never get stuck in waiting_help
            if self.plan_state.get("phase") == "waiting_help":
                self._clear_waiting_help(why="comm disabled")


        # expire waiting if time passed
        if self.plan_state.get("phase") == "waiting_help":
            started = self.plan_state.get("waiting_started_sim", None)
            if isinstance(started, (int, float)) and (now_sim - float(started)) >= float(self.help_wait_sec):
                self._clear_waiting_help(why=f"expired after {now_sim-float(started):.1f}s")

        # waiting behavior
        if self.plan_state.get("phase") == "waiting_help" and self.waiting_mode == "strict":
            return


        pending = None
        with self._plan_lock:
            pending = self.plan_state.get("pending_action", None)
            self.plan_state["pending_action"] = None

        if isinstance(pending, PolicyAction):
            self._execute(pending, box_lookup, now_sim)
            return


        # then normal planning action
        self._maybe_think(where="action_decide")
        action = self.policy.decide(self, boxes, now_sim)
        self._execute(action, box_lookup, now_sim)



    def _tick(self) -> None:
        if self._stop:
            return

        with self._action_lock:
            if self._action_thread is not None and self._action_thread.is_alive():
                return

            th = threading.Thread(target=self._thread_main, daemon=True)
            self._action_thread = th
            self._log("START", "spawn action thread")
            th.start()

    def _thread_main(self) -> None:
        try:
            self._run_one_cycle()
        except Exception as e:
            self.get_logger().warn(f"[FAIL] cycle failed: {e}")
        finally:
            with self._action_lock:
                self._action_thread = None


    def _router_tick(self) -> None:
        if self._stop or not getattr(self, "comm_enable", False):
            return
        # Wake router worker; don't spawn new threads
        if self.inbox:
            self._router_wakeup.set()

    def _router_worker_main(self) -> None:
        while not self._stop:
            # Sleep until someone signals new work
            self._router_wakeup.wait(timeout=0.5)
            self._router_wakeup.clear()
            if self._stop:
                break

            # Debounce window: accumulate a burst of incoming messages
            debounce_s = float(getattr(self, "router_debounce_sec", 0.10))
            if debounce_s > 0:
                time.sleep(debounce_s)

            self._router_handle_batch()


    def _router_handle_batch(self) -> None:
        try:
            t = self._time()
            now_sim = float(t["server_time"])
            time_limit = float(t["time_limit_sec"])
            if now_sim >= time_limit:
                self._request_shutdown_with_summary(
                    now_sim=now_sim,
                    time_limit=time_limit,
                    why="time_limit_reached (router worker)",
                )
                return

            boxes = self._boxes_state()

            # Drain inbox quickly (bounded)
            batch = []
            max_batch = int(getattr(self, "max_inbox_batch", 25))
            with self._inbox_lock:
                while self.inbox and len(batch) < max_batch:
                    batch.append(self.inbox.popleft())


            if not batch:
                return

            # Route + keep only those meant for me (or broadcast)
            routed = []
            for evt in batch:
                speaker = str(evt.get("speaker_id", ""))
                text = str(evt.get("text", ""))

                explicit_target = evt.get("target_speaker", None)
                if explicit_target is not None:
                    tgt = str(explicit_target)
                    route_src = "explicit"
                else:
                    tgt = self._infer_target_speaker(speaker, text, now_sim)
                    route_src = "llm_infer" if self.infer_target_use_llm else "fallback"

                #self._log("ROUTE",
                #    f"route src={route_src} from={speaker} -> target={tgt} "
                #    f"me={self.agent_id} text={text!r}"
                #)

                if tgt not in (self.agent_id, "all"):
                    self._log("ROUTE", f"ignore msg inferred_target={tgt} from={speaker} text={text!r}")
                    continue

                evt["target_speaker"] = tgt
                routed.append(evt)

            if not routed:
                return

            if self.policy_type == "llm":
                if self.think_router_enable:
                    self._maybe_think(where="router_decide_on_message_batch")

                # ✅ SINGLE call for the whole batch
                _ = self.llm_policy.decide_on_message(self, boxes, now_sim, routed)

        except Exception as e:
            self.get_logger().warn(f"[FAIL] router batch failed: {e}")



def main():
    rclpy.init()
    node = SimHumanAgent()
    try:
        rclpy.spin(node)
    finally:
        node._stop = True
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

