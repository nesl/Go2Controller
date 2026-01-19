#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple

import requests

import rclpy
from rclpy.node import Node
from std_msgs.msg import String as StringMsg

Property = Literal["X", "Y"]


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class Pose2D:
    x: float
    y: float


@dataclass
class PolicyAction:
    """
    High-level action decided by a policy.
    The node executes it (travel/sense/dispose/say).
    """
    kind: Literal["idle", "say", "ask_help", "sense_self", "dispose", "goto_only"]
    box_id: Optional[int] = None
    prop: Optional[Property] = None
    text: Optional[str] = None
    target_speaker: Optional[str] = None  # e.g., "robot" or "human_b"
    reason: str = ""


@dataclass
class BoxSummary:
    box_id: int
    x: float
    y: float
    deadline: float
    disposed_X: bool
    disposed_Y: bool
    # for each sense result: {agent_id, property, status, detected, probability, completed_at}
    sense_results: List[Dict[str, Any]]


# ---------------------------
# Policy interface
# ---------------------------

class BasePolicy:
    """
    Modular policy interface: scripted policy now, LLM policy later.
    """
    def decide(self, agent: "SimHumanAgent", boxes: List[BoxSummary], now_sim: float) -> PolicyAction:
        raise NotImplementedError


class ScriptedHelpThenDisposePolicy(BasePolicy):
    """
    Behavior:
      1) Choose a candidate box (not disposed for goal prop, not abandoned)
      2) Compute confidence p_present(goal_prop) from latest completed sense results
      3) If p_present >= dispose_threshold => dispose
      4) Else if uncertain:
           - ask best helper once for sensing help,
           - wait help_wait_sec,
           - if still uncertain, go sense self.
      5) If confidence is low (p_present <= giveup_threshold) => abandon box for this agent+prop.

    Helper choice is based on trust_map + sensor skill for goal property.
    """

    def decide(self, agent: "SimHumanAgent", boxes: List[BoxSummary], now_sim: float) -> PolicyAction:
    
        # 0) handle incoming help requests (breaks ask-loop dynamics)
        req_action = agent._pop_help_request_action(boxes, now_sim)
        if req_action is not None:
            return req_action

    
        goal = agent.goal_property

        # Build candidate list with scores
        best: Optional[Tuple[float, BoxSummary, float]] = None  # (score, box, p_present)
        for b in boxes:
            if agent._is_done_or_abandoned(b.box_id, goal):
                continue

            if agent._is_disposed_for_goal(b, goal):
                agent._mark_done(b.box_id, goal, why="already_disposed")
                continue

            p_present = agent._belief_present_from_box(b, goal)

            # give up if quite sure it's NOT present
            if p_present <= agent.giveup_threshold:
                agent._mark_abandoned(b.box_id, goal, why=f"low_confidence p_present={p_present:.2f}")
                continue

            dist = agent._dist_to(b.x, b.y)
            score = b.deadline + agent.dist_weight * dist

            # If already confident, bias strongly to execute
            if p_present >= agent.dispose_threshold:
                score -= 1e6

            if best is None or score < best[0]:
                best = (score, b, p_present)

        if best is None:
            return PolicyAction(kind="idle", reason="no_candidates")

        score, box, p_present = best
        agent._log(
            "DECIDE",
            f"candidate box={box.box_id} p_present={p_present:.2f} score={score:.2f} "
            f"deadline={box.deadline:.2f} dist={agent._dist_to(box.x, box.y):.2f}"
        )

        if p_present >= agent.dispose_threshold:
            return PolicyAction(
                kind="dispose",
                box_id=box.box_id,
                prop=goal,
                text=f"I’m confident box {box.box_id} has {goal}. I’ll dispose it.",
                reason=f"p_present={p_present:.2f} >= dispose_threshold={agent.dispose_threshold:.2f}",
            )

        st = agent._box_state(box.box_id, goal)
        asked_at = st.get("asked_help_at_sim", None)

        if asked_at is None:
            helper_id = agent._choose_best_helper(goal_prop=goal)
            if helper_id is None:
                # nobody to ask; go straight to self-sense
                return PolicyAction(
                    kind="sense_self",
                    box_id=box.box_id,
                    prop=goal,
                    text=f"No helpers available; I’ll go sense box {box.box_id} for {goal} myself.",
                    reason="uncertain_no_helpers",
                )

            return PolicyAction(
                kind="ask_help",
                box_id=box.box_id,
                prop=goal,
                target_speaker=helper_id,
                text=f"{agent._display_name(helper_id)}, can you sense box {box.box_id} for {goal}? I’m unsure.",
                reason="uncertain_and_not_asked_yet",
            )

        waited = now_sim - float(asked_at)
        if waited < agent.help_wait_sec:
            return PolicyAction(
                kind="idle",
                reason=f"waiting_for_help box={box.box_id} waited={waited:.1f}s < help_wait={agent.help_wait_sec:.1f}s",
            )

        if st.get("self_sensed", False):
            agent._mark_abandoned(box.box_id, goal, why="self_sensed_still_uncertain")
            return PolicyAction(kind="idle", reason="abandon_after_self_sense_still_uncertain")

        return PolicyAction(
            kind="sense_self",
            box_id=box.box_id,
            prop=goal,
            text=f"I’ll go sense box {box.box_id} for {goal} myself.",
            reason=f"waited_for_help {waited:.1f}s >= {agent.help_wait_sec:.1f}s",
        )


class LLMPolicy(BasePolicy):
    """
    LLM-driven policy:
      - Build a compact observation (top-k boxes + beliefs + profiles)
      - Ask LLM to output a PolicyAction JSON
      - Validate/guardrail
      - Fall back to scripted policy if anything goes wrong

    Supports asking help from robot OR another human via target_speaker.
    """

    def __init__(self, fallback: BasePolicy):
        self.fallback = fallback
        self._client = None

    def _get_client(self, agent: "SimHumanAgent"):
        if agent.llm_provider != "openai":
            return None
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
            self._client = OpenAI()
            return self._client
        except Exception as e:
            agent.get_logger().warn(f"[LLM] OpenAI client init failed: {e}")
            return None

    def _select_top_k_boxes(self, agent: "SimHumanAgent", boxes: List[BoxSummary], k: int) -> List[BoxSummary]:
        scored: List[Tuple[float, BoxSummary]] = []
        for b in boxes:
            if agent._is_disposed_for_goal(b, agent.goal_property):
                continue
            if agent._is_done_or_abandoned(b.box_id, agent.goal_property):
                continue
            dist = agent._dist_to(b.x, b.y)
            score = b.deadline + agent.dist_weight * dist
            scored.append((score, b))
        scored.sort(key=lambda x: x[0])
        return [b for _, b in scored[:k]]

    def _box_brief(self, agent: "SimHumanAgent", b: BoxSummary) -> Dict[str, Any]:
        goal = agent.goal_property
        p_present = agent._belief_present_from_box(b, goal)

        sensed_by: List[str] = []
        for sr in b.sense_results:
            if sr.get("status") == "completed" and sr.get("property") == goal and sr.get("agent_id"):
                sensed_by.append(str(sr.get("agent_id")))

        return {
            "box_id": b.box_id,
            "pos": [round(b.x, 2), round(b.y, 2)],
            "deadline": round(b.deadline, 2),
            "dist": round(agent._dist_to(b.x, b.y), 2),
            "disposed_goal": bool(b.disposed_X if goal == "X" else b.disposed_Y),
            "p_present_goal": round(float(p_present), 3),
            "goal_sensed_by": list(dict.fromkeys(sensed_by)),
        }

    def _system_prompt(self, agent: "SimHumanAgent") -> str:
        # Build helpers list with trust + sensor skill for goal property
        helpers = []
        for pid, info in agent.participants.items():
            if pid == agent.agent_id:
                continue
            trust = float(agent.trust_map.get(pid, 0.5))
            sp = agent.sensor_params.get(pid, {}).get(agent.goal_property, {})
            skill = float(sp.get("skill", 0.5))
            helpers.append({
                "id": pid,
                "name": info.get("name", pid),
                "type": info.get("type", "unknown"),
                "trust": round(trust, 2),
                "sensor_skill_goal": round(skill, 2),
            })

        # nudge behavior with risk_aversion
        eff_dispose_th = min(0.95, max(0.55, agent.dispose_threshold + 0.15 * (agent.risk_aversion - 0.5)))

        return (
            f"You are {agent.agent_id} ({agent._display_name(agent.agent_id)}), a simulated human.\n"
            f"Your personal objective: maximize disposals of property {agent.goal_property} before deadlines.\n\n"
            "You act in discrete steps. Each step you choose exactly ONE action.\n"
            "If you are uncertain a box has your goal property, prefer asking a helper to sense it first.\n"
            "You may ask either a robot or another human for sensing help.\n\n"
            "Helpers available (use trust and sensor_skill_goal to decide whom to ask):\n"
            "IMPORTANT: target_speaker MUST be an ID from helpers (e.g., \"human_a\"), NOT a name (e.g., \"Sam\").\n"
            "If there is an inbox request you can’t help with, choose kind=say and explain briefly.\n"

            f"{json.dumps(helpers)}\n\n"
            "Hard constraints:\n"
            "- Dispose only when confident a box has your goal property.\n"
            "- If uncertain, ask for sensing help before sensing yourself.\n"
            "- Choose exactly ONE action each step.\n\n"
            "Output FORMAT (strict): output ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "kind": "idle|ask_help|sense_self|dispose|say",\n'
            '  "box_id": number|null,\n'
            '  "prop": "X"|"Y"|null,\n'
            '  "target_speaker": string|null,\n'
            '  "text": string|null,\n'
            '  "reason": string\n'
            "}\n\n"
            f"Guideline thresholds:\n"
            f"- Dispose only if p_present_goal >= {eff_dispose_th:.2f}\n"
            f"- Give up if p_present_goal <= {agent.giveup_threshold:.2f}\n"
        )

    def _format_inbox(self, agent: "SimHumanAgent") -> List[Dict[str, Any]]:
        out = []
        for r in agent.inbox_requests[-5:]:
            out.append({
                "from": r.get("from"),
                "from_name": agent._display_name(str(r.get("from"))),
                "box_id": r.get("box_id"),
                "prop": r.get("prop"),
            })
        return out


    def _user_prompt(self, agent: "SimHumanAgent", boxes: List[BoxSummary], now_sim: float) -> str:
        top = self._select_top_k_boxes(agent, boxes, agent.llm_top_k_boxes)
        box_briefs = [self._box_brief(agent, b) for b in top]

        inbox = self._format_inbox(agent)  # implement helper below


        recent = [{"speaker_id": m.get("speaker_id"), "text": m.get("text")} for m in agent.last_msgs[-6:]]

        obs = {
            "time": round(now_sim, 2),
            "you": {
                "id": agent.agent_id,
                "name": agent._display_name(agent.agent_id),
                "goal_property": agent.goal_property,
                "pos": [round(agent.pose.x, 2), round(agent.pose.y, 2)],
                "dispose_threshold": agent.dispose_threshold,
                "giveup_threshold": agent.giveup_threshold,
                "help_wait_sec": agent.help_wait_sec,
            },
            "participants": list(agent.participants.values()),
            "trust_map": {k: round(float(v), 2) for k, v in agent.trust_map.items()},
            "sensor_params_goal": {
                pid: agent.sensor_params.get(pid, {}).get(agent.goal_property, {})
                for pid in agent.participants.keys()
            },
            "boxes": box_briefs,
            "recent_dialogue": recent,
            "notes": "p_present_goal is your current confidence from completed senses for your goal property. If no one sensed, it is ~0.5.",
            "inbox_help_requests": inbox,
            "help_history": agent.help_history,
            "ignore_history": agent.ignore_history,
            "help_cooldown_sec": agent.help_cooldown_sec,

        }
        return json.dumps(obs)

    def _parse_action(self, agent: "SimHumanAgent", txt: str) -> Optional[PolicyAction]:
        try:
            data = json.loads(txt)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None

        kind = data.get("kind")
        if kind not in ("idle", "ask_help", "sense_self", "dispose", "say"):
            return None

        box_id = data.get("box_id", None)
        if box_id is not None:
            try:
                box_id = int(box_id)
            except Exception:
                return None

        prop = data.get("prop", None)
        if prop is not None and prop not in ("X", "Y"):
            return None

        # map display names -> ids (case-insensitive)
        name_to_id = {str(v.get("name", k)).lower(): k for k, v in agent.participants.items()}

        target = data.get("target_speaker", None)
        if target is not None:
            target = str(target).strip()
            # if LLM gave a name, convert to id
            low = target.lower()
            if low in name_to_id:
                target = name_to_id[low]

        allowed = set(agent.participants.keys()) | {"robot"}
        if target is not None and target not in allowed:
            return None


        text = data.get("text", None)
        if text is not None and not isinstance(text, str):
            return None

        reason = data.get("reason", "")
        if not isinstance(reason, str):
            reason = ""

        # structural constraints
        if kind in ("ask_help", "sense_self", "dispose"):
            if box_id is None or prop is None:
                return None
        if kind == "ask_help" and not target:
            target = agent._choose_best_helper(goal_prop=prop) or agent.help_target_speaker

        return PolicyAction(
            kind=kind,
            box_id=box_id,
            prop=prop,
            target_speaker=target,
            text=text,
            reason=reason,
        )

    def decide(self, agent: "SimHumanAgent", boxes: List[BoxSummary], now_sim: float) -> PolicyAction:
    
        # handle incoming help requests first (social reflex)
        req_action = agent._pop_help_request_action(boxes, now_sim)
        if req_action is not None:
            return req_action

    
        if agent.llm_provider == "none":
            return self.fallback.decide(agent, boxes, now_sim)

        client = self._get_client(agent)
        if client is None:
            return self.fallback.decide(agent, boxes, now_sim)

        sys_msg = self._system_prompt(agent)
        user_msg = self._user_prompt(agent, boxes, now_sim)

        agent._log("LLM", f"call model={agent.llm_model} temp={agent.llm_temperature} max_tokens={agent.llm_max_tokens}")

        try:
            # Requires OpenAI Python SDK (modern). If your SDK uses chat.completions, adapt accordingly.
            resp = client.responses.create(
                model=agent.llm_model,
                input=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=agent.llm_temperature,
                max_output_tokens=agent.llm_max_tokens,
                timeout=agent.llm_timeout_sec,
            )
            raw = resp.output_text
        except Exception as e:
            agent.get_logger().warn(f"[LLM] call failed: {e}")
            return self.fallback.decide(agent, boxes, now_sim)

        agent._log("LLM", f"raw={raw!r}")

        action = self._parse_action(agent, raw)
        if action is None:
            agent.get_logger().warn("[LLM] invalid JSON action; falling back to scripted policy")
            return self.fallback.decide(agent, boxes, now_sim)

        # Guardrail: never allow disposing non-goal property
        if action.kind == "dispose" and action.prop != agent.goal_property:
            agent.get_logger().warn("[LLM] attempted dispose of non-goal property; overriding to idle")
            return PolicyAction(kind="idle", reason="guardrail_non_goal_dispose")

        # Guardrail: if disposing but confidence low -> convert to help request
        if action.kind == "dispose" and action.box_id is not None:
            b = next((bb for bb in boxes if bb.box_id == action.box_id), None)
            if b is not None:
                p = agent._belief_present_from_box(b, agent.goal_property)
                if p < agent.dispose_threshold:
                    agent.get_logger().warn(
                        f"[LLM] dispose requested but p_present={p:.2f} < dispose_threshold={agent.dispose_threshold:.2f}; overriding"
                    )
                    helper = agent._choose_best_helper(goal_prop=agent.goal_property) or agent.help_target_speaker
                    return PolicyAction(
                        kind="ask_help",
                        box_id=action.box_id,
                        prop=agent.goal_property,
                        target_speaker=helper,
                        text=f"{agent._display_name(helper)}, can you sense box {action.box_id} for {agent.goal_property}? I'm not confident yet.",
                        reason="override_low_confidence_dispose",
                    )

        return action


# ---------------------------
# Sim human node
# ---------------------------

class SimHumanAgent(Node):
    def __init__(self):
        super().__init__("sim_human_agent")

        # ---- basic params ----
        self.declare_parameter("agent_id", "human_a")
        self.declare_parameter("goal_property", "X")
        self.declare_parameter("server_base_url", "http://172.17.40.64:8080")
        self.declare_parameter("stt_topic", "/audio/stt_text")

        # motion + timing
        self.declare_parameter("speed_mps", 1.0)
        self.declare_parameter("decision_period_sec", 1.0)
        self.declare_parameter("request_timeout_sec", 120.0)

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
        self.declare_parameter("llm_top_k_boxes", 6)
        self.declare_parameter("llm_timeout_sec", 30.0)

        # human model traits (LLM prompt conditioning)
        self.declare_parameter("risk_aversion", 0.7)
        self.declare_parameter("stubbornness", 0.5)
        self.declare_parameter("fairness_sensitivity", 0.3)

        # logging
        self.declare_parameter("log_actions", True)

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

        # ---- ROS pub/sub ----
        self.pub_stt = self.create_publisher(StringMsg, self.stt_topic, 10)
        self.sub_stt = self.create_subscription(StringMsg, self.stt_topic, self._on_stt_text, 10)

        # ---- internal state ----
        self.pose = Pose2D(0.0, 0.0)
        self.last_msgs: List[Dict[str, Any]] = []
        self._mem: Dict[Tuple[int, str], Dict[str, Any]] = {}

        # threading
        self._action_lock = threading.Lock()
        self._action_thread: Optional[threading.Thread] = None
        self._stop = False

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
        self.scripted_policy = ScriptedHelpThenDisposePolicy()
        self.llm_policy = LLMPolicy(fallback=self.scripted_policy)
        self.policy: BasePolicy = self.llm_policy if self.policy_type == "llm" else self.scripted_policy

        self.create_timer(self.decision_period, self._tick)

        self.get_logger().info(
            f"SimHumanAgent up agent_id={self.agent_id} goal={self.goal_property} "
            f"server={self.base_url} topic={self.stt_topic} policy={self.policy_type} "
            f"dispose_th={self.dispose_threshold} giveup_th={self.giveup_threshold} "
            f"help_wait={self.help_wait_sec}s speed={self.speed_mps} timeout={self.timeout}s "
            f"llm_provider={self.llm_provider} llm_model={self.llm_model}"
        )

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
        if self.log_actions:
            self.get_logger().info(f"[{tag}] {msg}")

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

        # Base trust: derived from helper's sensor skill for our goal property
        for pid in self.participants.keys():
            if pid == self.agent_id:
                continue
            sp = self.sensor_params.get(pid, {}).get(self.goal_property)
            base = float(sp["skill"]) if sp else 0.5
            # mild robot bias
            if self.participants.get(pid, {}).get("type") == "robot":
                base = min(1.0, base + 0.1)
            self.trust_map[pid] = base

            # Parse trust overrides once
            trust_overrides_raw = str(self.get_parameter("trust_overrides_json").value)
            try:
                trust_overrides = json.loads(trust_overrides_raw)
                if not isinstance(trust_overrides, dict):
                    trust_overrides = {}
            except Exception:
                trust_overrides = {}

            # Base trust: derived from helper's sensor skill for our goal property
            for pid in self.participants.keys():
                if pid == self.agent_id:
                    continue

                sp = self.sensor_params.get(pid, {}).get(self.goal_property)
                base = float(sp["skill"]) if sp else 0.5

                # mild robot bias
                if self.participants.get(pid, {}).get("type") == "robot":
                    base = min(1.0, base + 0.1)

                # apply override if provided
                if pid in trust_overrides:
                    try:
                        base = float(trust_overrides[pid])
                    except Exception:
                        pass

                self.trust_map[pid] = max(0.0, min(1.0, base))



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

    # ---------------------------
    # ROS bus I/O
    # ---------------------------
    def _on_stt_text(self, msg: StringMsg) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                return
            speaker = str(payload.get("speaker_id"))
            text = str(payload.get("text", ""))
        except Exception:
            return

        if speaker == self.agent_id:
            return

        self.last_msgs.append({"speaker_id": speaker, "text": text, "t_wall": time.time()})
        self.last_msgs = self.last_msgs[-100:]
        self._log("HEAR", f"from={speaker} text={text!r}")

        # --- detect explicit sensing requests ---
        # Examples:
        #  "Sam, can you sense box 2 for Y? I’m unsure."
        #  "can you sense box 6 for X"
        m = re.search(r"\bsense\s+box\s+(\d+)\s+for\s+([XY])\b", text, re.IGNORECASE)
        if m:
            box_id = int(m.group(1))
            prop = m.group(2).upper()
            # store request (we'll decide whether to comply)
            self.inbox_requests.append({
                "from": speaker,
                "box_id": box_id,
                "prop": prop,
                "t_wall": time.time(),
            })
            self.inbox_requests = self.inbox_requests[-30:]
            self._log("INBOX", f"request from={speaker} box={box_id} prop={prop}")


    def _publish_utterance(self, text: str) -> None:
        out = StringMsg()
        out.data = json.dumps({"text": text, "speaker_id": self.agent_id})
        self.pub_stt.publish(out)
        self._log("SAY", text)

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
            out.append(
                BoxSummary(
                    box_id=int(b["box_id"]),
                    x=float(b["x"]),
                    y=float(b["y"]),
                    deadline=float(b["deadline"]),
                    disposed_X=bool(b["disposed_X"]),
                    disposed_Y=bool(b["disposed_Y"]),
                    sense_results=list(b.get("sense_results", [])),
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
        return b.disposed_X if prop == "X" else b.disposed_Y

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
        Use MOST RECENT completed sense for this property (any agent), converting server semantics:
          detected=True  -> probability ~ P(present | +)
          detected=False -> probability ~ P(absent | -) so p_present = 1 - probability
        """
        best_sr = None
        best_t = -1.0

        for sr in box.sense_results:
            if sr.get("status") != "completed":
                continue
            if sr.get("property") != prop:
                continue
            t = sr.get("completed_at")
            # if missing completed_at, still allow but prefer with time
            tv = float(t) if isinstance(t, (int, float)) else 0.0
            if best_sr is None or tv > best_t:
                best_t = tv
                best_sr = sr

        if best_sr is None:
            return 0.5

        detected = best_sr.get("detected", None)
        prob = best_sr.get("probability", None)
        if not isinstance(prob, (int, float)) or detected is None:
            return 0.5

        prob = float(prob)
        p_present = prob if detected is True else (1.0 - prob)
        return max(0.0, min(1.0, p_present))

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

    def _execute(self, action: PolicyAction, box_lookup: Dict[int, BoxSummary], now_sim: float) -> None:
        self._log("ACT", f"execute kind={action.kind} box={action.box_id} prop={action.prop} reason={action.reason}")

        if action.kind == "idle":
            return

        if action.text:
            self._publish_utterance(action.text)

        if action.kind in ("say", "ask_help"):
            if action.kind == "ask_help" and action.box_id is not None and action.prop is not None:
                st = self._box_state(action.box_id, action.prop)
                last_asked = st.get("asked_help_at_sim", None)
                if last_asked is None or (now_sim - float(last_asked)) >= self.help_wait_sec:
                    st["asked_help_at_sim"] = float(now_sim)
                    st["asked_help_to"] = action.target_speaker or self.help_target_speaker
                    self._log("MEM", f"asked_help box={action.box_id} prop={action.prop} to={st['asked_help_to']} at_sim={now_sim:.2f}")
                else:
                    # suppress re-asking
                    self._log("MEM", f"suppress ask_help repeat box={action.box_id} prop={action.prop} waited={now_sim-float(last_asked):.1f}s")

            return

        if action.box_id is None or action.box_id not in box_lookup:
            self._log("WARN", f"missing box in lookup for action: {action}")
            return

        box = box_lookup[action.box_id]

        if action.kind == "goto_only":
            self._travel_to(box)
            return

        if action.kind == "sense_self":
            assert action.prop is not None
            self._travel_to(box)
            js = self._sense(box.box_id, action.prop)
            st = self._box_state(box.box_id, action.prop)
            st["self_sensed"] = True
            st["last_self_sense_status"] = js.get("status")
            return

        if action.kind == "dispose":
            assert action.prop is not None
            self._travel_to(box)
            js = self._dispose(box.box_id, action.prop)
            self._mark_done(box.box_id, action.prop, why=f"dispose_attempt success={js.get('success')}")
            return

    # ---------------------------
    # Thread runner and tick
    # ---------------------------
    def _run_one_cycle(self) -> None:
        t = self._time()
        now_sim = float(t["server_time"])
        if now_sim >= float(t["time_limit_sec"]):
            self._log("TIME", f"limit reached server_time={now_sim:.2f} >= {t['time_limit_sec']:.2f}")
            return

        boxes = self._boxes_state()
        box_lookup = {b.box_id: b for b in boxes}

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

