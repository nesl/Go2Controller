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
    kind: Literal["idle", "say", "ask_help", "sense_self", "dispose", "goto_only", "assist_dispose"]

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
    senseable: Dict[str, bool]                 # e.g., {"X": True, "Y": False}
    senseable_by: Optional[Dict[str, List[str]]] = None
# ---------------------------
# Policy interface
# ---------------------------

class BasePolicy:
    """
    Modular policy interface: scripted policy now, LLM policy later.
    """
    def decide(self, agent: "SimHumanAgent", boxes: List[BoxSummary], now_sim: float) -> PolicyAction:
        raise NotImplementedError


class LLMPolicy(BasePolicy):
    """
    LLM-driven policy:
      - Build a compact observation (top-k boxes + beliefs + profiles)
      - Ask LLM to output a PolicyAction JSON
      - Validate/guardrail
      - Fall back to scripted policy if anything goes wrong

    Supports asking help from robot OR another human via target_speaker.
    """

    def __init__(self):
        self.fallback = None
        self._client = None

    def _should_ask_help_first(self, agent: "SimHumanAgent") -> bool:
        """
        Decide if asking help is worth it vs sensing self first, based on sensor skill and trust.
        Heuristic: ask if best helper has clearly higher expected info quality than self.
        """
        goal = agent.goal_property
        self_skill = float(agent.sensor_params.get(agent.agent_id, {}).get(goal, {}).get("skill", 0.5))

        best = -1.0
        for pid in agent.participants.keys():
            if pid == agent.agent_id:
                continue
            trust = float(agent.trust_map.get(pid, 0.5))
            helper_skill = float(agent.sensor_params.get(pid, {}).get(goal, {}).get("skill", 0.5))
            score = 0.65 * trust + 0.35 * helper_skill
            best = max(best, score)

        # If we ourselves are strong, sense first.
        # If a helper is meaningfully better, ask first.
        return (best - self_skill) >= 0.08


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
        #print(f'{boxes}, {k}', flush=True)
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

    def _box_brief(self, agent: "SimHumanAgent", b: BoxSummary, now_sim: float) -> Dict[str, Any]:
        goal = agent.goal_property
        p_present = agent._belief_present_from_box(b, goal)

        sensed_by: List[str] = []
        for sr in b.sense_results:
            if sr.get("status") == "completed" and sr.get("property") == goal and sr.get("agent_id"):
                sensed_by.append(str(sr.get("agent_id")))

        you_sensed_goal = any(
            sr.get("status") == "completed"
            and sr.get("property") == goal
            and str(sr.get("agent_id")) == agent.agent_id
            for sr in b.sense_results
        )

        deadline_passed = float(now_sim) > float(b.deadline)


        # ✅ NEW: indicate which properties are senseable for this object
        # b.senseable is already {"X": True/False, "Y": True/False}
        senseable_props = []
        if isinstance(b.senseable, dict):
            for prop in ("X", "Y"):
                if bool(b.senseable.get(prop, True)):
                    senseable_props.append(prop)
        else:
            senseable_props = ["X", "Y"]

        return {
            "box_id": b.box_id,
            "pos": [round(b.x, 2), round(b.y, 2)],
            "deadline": round(b.deadline, 2),
            "deadline_passed": bool(deadline_passed),
            "distance": round(agent._dist_to(b.x, b.y), 2),
            "disposed_goal": bool(b.disposed_X or b.disposed_Y),
            "p_present_goal": round(float(p_present), 3),
            "goal_sensed_by": list(dict.fromkeys(sensed_by)),
            "you_already_sensed_goal": bool(you_sensed_goal),

            # ✅ add this:
            "senseable_props": senseable_props,   # e.g., ["X"] or ["X","Y"] or []
        }

    def _sensor_params_for_prompt(self, agent: "SimHumanAgent") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for pid in agent.participants.keys():
            out[pid] = {}
            for prop in ("X", "Y"):
                sp = agent.sensor_params.get(pid, {}).get(prop, None)
                if not sp:
                    continue
                tpr = float(sp.get("present", 0.5))
                fpr = float(sp.get("absent", 0.5))
                lr_plus = (tpr / max(1e-6, fpr))
                lr_minus = ((1.0 - tpr) / max(1e-6, (1.0 - fpr)))
                out[pid][prop] = {
                    "tpr": round(tpr, 3),
                    "fpr": round(fpr, 3),
                    "skill": round(float(sp.get("skill", 0.5)), 3),
                    "lr_plus": round(lr_plus, 2),
                    "lr_minus": round(lr_minus, 2),
                }
        return out



    def _system_prompt(self, agent: "SimHumanAgent") -> str:
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

        ask_first = self._should_ask_help_first(agent)
        eff_dispose_th = min(0.95, max(0.55, agent.dispose_threshold + 0.15 * (agent.risk_aversion - 0.5)))

        allowed_kinds = self._allowed_kinds(agent)

        # --- comm gating ---
        comm_on = bool(getattr(agent, "comm_enable", False))

        base = (
            f"You are {agent.agent_id} ({agent._display_name(agent.agent_id)}), a simulated human.\n"
            f"Your personal objective: maximize disposals of property {agent.goal_property} before deadlines.\n\n"
            "You act in discrete steps. Each step you choose exactly ONE action.\n"
        )

        # Only include comm-related instruction if comm is enabled
        if comm_on:
            base += (
                "If uncertain, choose between (a) ask helper first or (b) sense yourself first.\n"
                "Use sensor skill: if your own sensor_skill_goal is close to or better than the best helper, sense yourself first.\n"
                "Ask for help first only when a helper’s combined (trust + sensor_skill_goal) is meaningfully better than yours.\n"
                "You may ask either a robot or another human for sensing help.\n\n"
                "Helpers available (use trust and sensor_skill_goal to decide whom to ask):\n"
                "IMPORTANT: target_speaker MUST be an ID from helpers (e.g., \"human_a\"), NOT a name (e.g., \"Sam\").\n"
                "- If you asked for help on a box recently (asked_help_at_sim exists and < help_wait_sec), do NOT ask again; choose idle or sense_self.\n"
                "- If your last_message_outcome indicates your last help request was rejected for the same box, do NOT repeat; sense_self instead.\n"
                f"\nHeuristic: ask_help_first_recommended={ask_first}\n"
                f"{json.dumps(helpers)}\n\n"
            )
        else:
            base += (
                "Communication is DISABLED. You cannot ask for help and you cannot speak.\n"
                "So if uncertain, you must decide between sensing yourself, moving, or idling.\n\n"
            )

        base += (
            "Hard constraints:\n"
            "- Dispose only when confident a box has your goal property.\n"
            "- Choose exactly ONE action each step.\n"
            "- You may sense a given (box_id, prop) at most ONCE yourself. If you already sensed it, do NOT choose sense_self again.\n\n"
            "Output FORMAT (strict): output ONLY valid JSON matching this schema:\n"
            "{\n"
            f'  "kind": "{ "|".join(allowed_kinds) }",\n'
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

        # Only include text rules if comm is enabled
        if comm_on:
            base += (
                "\nText constraints (comm enabled):\n"
                "- If kind is \"ask_help\" or \"say\", you MUST provide a non-empty \"text\".\n"
                "- For ask_help, the text must explicitly mention box_id and prop.\n"
            )

        base += (
          "\nCoordination:\n"
          "- If another agent is currently disposing a box, you may choose assist_dispose on that box to speed it up.\n"
          "- assist_dispose requires only box_id (prop must be null).\n"
        )


        return base



    def _format_inbox(self, agent: "SimHumanAgent") -> List[Dict[str, Any]]:
        tail = list(agent.inbox)[-5:]
        return [{"from": e.get("speaker_id"), "text": e.get("text")} for e in tail]




    def _user_prompt(self, agent: "SimHumanAgent", boxes: List[BoxSummary], now_sim: float) -> str:
        top = self._select_top_k_boxes(agent, boxes, agent.llm_top_k_boxes)
        box_briefs = [self._box_brief(agent, b, now_sim) for b in top]


        inbox = self._format_inbox(agent)  # implement helper below


        recent = [{
            "speaker_id": e.get("speaker_id"),
            "target_speaker": e.get("target_speaker"),
            "text": e.get("text"),
        } for e in agent._get_transcript_tail(12)]


        last_dec = {
            "last_message_decision": agent.plan_state.get("last_message_decision"),
            "last_message_from": agent.plan_state.get("last_message_from"),
            "last_message_text": agent.plan_state.get("last_message_text"),
            "last_message_time": agent.plan_state.get("last_message_time"),
            "last_message_reason": agent.plan_state.get("last_message_reason"),
        }


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
            "boxes": box_briefs,
            "recent_dialogue": recent,
            "notes": "p_present_goal is your current confidence from completed senses for your goal property. If no one sensed, it is ~0.5.",
            "inbox_help_requests": inbox,
            "help_history": agent.help_history,
            "ignore_history": agent.ignore_history,
            "help_cooldown_sec": agent.help_cooldown_sec,
            "last_message_outcome": {} #last_dec,


        }
        
        obs["sensor_params"] = self._sensor_params_for_prompt(agent)
        obs["sensor_params_notes"] = (
            "tpr=P(detected|present). fpr=P(detected|absent) (false positive rate). "
            "Higher lr_plus means a positive detection is more trustworthy."
        )

        
        tail_inbox = list(agent.inbox)[-5:]
        obs["inbox"] = [{"speaker_id": e.get("speaker_id"), "text": e.get("text")} for e in tail_inbox]

        


        obs["plan_state"] = {
            "phase": agent.plan_state.get("phase"),
            "waiting_help_box_id": agent.plan_state.get("waiting_help_box_id"),
            "waiting_help_prop": agent.plan_state.get("waiting_help_prop"),
            "waiting_on": agent.plan_state.get("waiting_on"),
            "waiting_started_sim": agent.plan_state.get("waiting_started_sim"),
            "commitments": agent.plan_state.get("commitments", [])[-5:],
        }


        comm_on = bool(getattr(agent, "comm_enable", False))
        if not comm_on:
            # Remove anything that exposes comm state / dialogue / negotiation affordances.
            obs.pop("recent_dialogue", None)
            obs.pop("inbox_help_requests", None)
            obs.pop("help_history", None)
            obs.pop("ignore_history", None)
            obs.pop("inbox", None)
            obs.pop("commitments", None)
            obs.pop("plan_state", None)
            obs.pop("last_message_outcome", None)

            # Also remove any comm-oriented notes (optional)
            # obs.pop("notes", None)


        return json.dumps(obs)

    def _parse_action(self, agent: "SimHumanAgent", txt: str) -> Optional[PolicyAction]:
        try:
            data = json.loads(txt)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None

        kind = data.get("kind")
        
        # ✅ alias
        if kind == "assist_dispose":
            kind = "dispose"
            data["kind"] = "dispose"

        
        allowed = set(self._allowed_kinds(agent))
        if kind not in allowed:
            return None
        # If comm is disabled, disallow any target_speaker/text reliance
        if not getattr(agent, "comm_enable", False):
            # If model tries to include them anyway, ignore them
            data["target_speaker"] = None
            data["text"] = None





        box_id = data.get("box_id", None)
        if box_id is not None:
            try:
                box_id = int(box_id)
            except Exception:
                return None

        if kind == "goto_only" and box_id is None:
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

    def decide_on_message(self, agent: "SimHumanAgent", boxes: List[BoxSummary], now_sim: float, msg_evt: Dict[str, Any]) -> Optional[PolicyAction]:
        if agent.llm_provider != "openai":
            return None
        client = self._get_client(agent)
        if client is None:
            return None

        # System prompt: negotiation + consistency + anti-loop
        sys_msg = self._system_prompt_message_router(agent)

        # User prompt includes: world, plan_state, trust, the message
        obs = json.loads(self._user_prompt(agent, boxes, now_sim))
        obs["mode"] = "handle_message"
        obs["incoming_message"] = {"speaker_id": msg_evt["speaker_id"], "text": msg_evt["text"]}
        obs["plan_state"] = agent.plan_state
        obs["memory"] = agent._memory_brief()

        user_msg = json.dumps(obs)

        agent._dbg_llm("SYSTEM_MSG_ROUTER", sys_msg)
        agent._dbg_llm("USER_MSG_ROUTER", user_msg)

        # ✅ stochastic desync to avoid both agents choosing same action at same sim-time
        if agent.llm_jitter_enable:
            lo = max(0.0, float(agent.llm_jitter_min_sec))
            hi = max(lo, float(agent.llm_jitter_max_sec))
            dt = random.uniform(lo, hi)
            #agent._log("LLM", f"jitter_sleep {dt:.2f}s before ACT call")
            time.sleep(dt)


        try:
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
            agent.get_logger().warn(f"[LLM] message-router call failed: {e}")
            return None

        agent._log("LLM", f"raw_router={raw!r}")

        parsed = self._parse_router_output(agent, raw)
        if parsed is None:
            return None

        act = parsed.get("action")  # ✅ define before using

        # record last decision (fine to keep)
        agent.plan_state["last_message_decision"] = parsed.get("message_decision")
        agent.plan_state["last_message_reason"] = parsed.get("reason", "")
        agent.plan_state["last_message_from"] = msg_evt.get("speaker_id")
        agent.plan_state["last_message_text"] = msg_evt.get("text")
        agent.plan_state["last_message_time"] = now_sim

        decision = parsed.get("message_decision")
        requester = str(msg_evt.get("speaker_id", ""))

        # ✅ If we were waiting for help from someone, and THIS message is from that person,
        # treat the router decision as the "response outcome" and update waiting_help.
        if agent.plan_state.get("phase") == "waiting_help" and requester == str(agent.plan_state.get("waiting_on")):
            # record in per-box memory so scripted/LLM can avoid re-asking / decide next
            w_box = agent.plan_state.get("waiting_help_box_id")
            w_prop = agent.plan_state.get("waiting_help_prop")
            if w_box is not None and w_prop in ("X", "Y"):
                st = agent._box_state(int(w_box), str(w_prop))  # type: ignore[arg-type]
                st["asked_help_outcome"] = decision
                st["asked_help_responded_at_sim"] = float(now_sim)

            if decision != "negotiate":
                # accept/reject/defer/ignore => we have an answer; exit waiting_help immediately
                agent._clear_waiting_help(why=f"got_response decision={decision} from={requester}")
            else:
                # negotiate => negotiation still unresolved; keep waiting_help alive
                # (optionally reset the timer so you get a fresh window)
                agent.plan_state["waiting_started_sim"] = float(now_sim)
                agent._log("MEM", f"keep waiting_help (negotiate) from={requester}")


        if decision in ("accept", "defer", "negotiate") and isinstance(act, dict):
            box_id = act.get("box_id", None)
            prop = act.get("prop", None)
            requested_kind = act.get("kind", None)

            # ✅ NEW: commitments should never be "dispose" by default for help
            allowed_commitment_kinds = {"sense_self", "goto_only", "say"}
            if requested_kind not in allowed_commitment_kinds:
                requested_kind = "sense_self"

            # basic defer: wait a little before helping if LLM chose defer
            due_after = now_sim + 8.0 if decision == "defer" else now_sim

            agent._add_or_update_commitment(
                requester=requester,
                box_id=box_id,
                prop=prop,
                decision=decision,
                now_sim=now_sim,
                requested_kind=requested_kind,
                due_after=due_after,
                notes=str(parsed.get("reply_text") or ""),
            )

            # Fetch the commitment we just created/updated
            if box_id is not None and prop is not None:
                c = agent._find_active_commitment(
                    requester=requester,
                    box_id=int(box_id),
                    prop=str(prop).upper(),
                )
                if c is not None:
                    # Priority: lower = sooner (default 10)
                    try:
                        c["priority"] = int(parsed.get("priority", 10))
                    except Exception:
                        c["priority"] = 10

                    # Urgent override: can preempt current action if True
                    c["urgent_override"] = bool(parsed.get("urgent_override", False))

                    # By default, do NOT interrupt current physical action
                    # (urgent_override=True bypasses this)
                    c["blocked_on_busy"] = not c["urgent_override"]

                    if c.get("urgent_override", False):
                        agent._request_preempt(why=f"urgent_accept from={requester} box={box_id} prop={prop}")


        # If reject/ignore and action refers to a previously-active commitment, cancel it
        if decision in ("reject", "ignore") and isinstance(act, dict):
            box_id = act.get("box_id", None)
            prop = act.get("prop", None)
            if box_id is not None and prop is not None:
                existing = agent._find_active_commitment(requester=requester, box_id=int(box_id), prop=str(prop).upper())
                if existing:
                    agent._complete_commitment(existing, status="cancelled")



        # If reply_text exists, publish it as "say" (without blocking action)
        if parsed.get("reply_text"):
            agent._publish_utterance(str(parsed["reply_text"]), target_speaker=requester)



        if act:
            return self._parse_action(agent, json.dumps(act))

        return PolicyAction(kind="idle", reason="handled_message_no_action")

    def _system_prompt_message_router(self, agent: "SimHumanAgent") -> str:
        return (
            f"You are the decision-making policy for {agent.agent_id} ({agent._display_name(agent.agent_id)}).\n"
            f"Goal: dispose boxes with property {agent.goal_property} before deadlines.\n"
            "You receive ONE incoming message and must decide how to respond.\n\n"
            "You must choose ONE message decision:\n"
            "- accept: you agree and will do it now or schedule it\n"
            "- reject: you refuse clearly\n"
            "- negotiate: counteroffer (different box, different timing, or reciprocal help)\n"
            "- defer: you acknowledge but postpone\n"
            "- ignore: no response\n\n"
            "Anti-loop constraints:\n"
            "- Do NOT repeatedly ask for help on the same box if you are already waiting.\n"
            "- If the incoming message is a help request, decide accept/reject/negotiate/defer based on your current plan_state.\n"
            "If the incoming message asks you to sense/dispose a box, you MUST include box_id and prop in the action.\n"
            "- Prefer ACCEPT if cost is low and trust is moderate/high.\n"
            "- Prefer REJECT or DEFER if it jeopardizes your imminent disposal opportunity.\n"
            "- NEGOTIATE if you can help later or want reciprocal help.\n\n"
            "- If the agent is waiting_help, do NOT recommend duplicating the SAME (waiting_help_box_id, waiting_help_prop) yourself.\n"
            "- If you respond accept/reject/defer/ignore to a waiting-help interaction, that resolves it; only negotiate keeps it unresolved.\n"

            "Output ONLY strict JSON of the form:\n"


            "{\n"
            '  "mode": "handle_message",\n'
            '  "message_decision": "accept|reject|negotiate|defer|ignore",\n'
            '  "reply_text": string|null,\n'
            '  "urgent_override": boolean,\n'
            '  "priority": number,\n'
            '  "action": {\n'
            '    "kind": "idle|ask_help|sense_self|dispose|assist_dispose|say|goto_only",\n'
            '    "box_id": number|null,\n'
            '    "prop": "X|Y|null",\n'
            '    "target_speaker": string|null,\n'
            '    "text": string|null,\n'
            '    "reason": string\n'
            "  }|null,\n"
            '  "reason": string\n'
            "}\n"
        )

    def _allowed_kinds(self, agent: "SimHumanAgent") -> List[str]:
        # When comm is disabled, never allow comm actions in LLM output.
        if not getattr(agent, "comm_enable", False):
            return ["idle", "sense_self", "dispose", "goto_only"]
        return ["idle", "ask_help", "sense_self", "dispose", "say", "goto_only"]



    def _parse_router_output(self, agent: "SimHumanAgent", txt: str) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(txt)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        if data.get("mode") != "handle_message":
            return None
        if data.get("message_decision") not in ("accept", "reject", "negotiate", "defer", "ignore"):
            return None
        # reply_text optional
        rt = data.get("reply_text", None)
        if rt is not None and not isinstance(rt, str):
            data["reply_text"] = None
        # action optional (validated later)
        if "action" in data and data["action"] is not None and not isinstance(data["action"], dict):
            data["action"] = None
            

        uo = data.get("urgent_override", False)
        data["urgent_override"] = bool(uo)

        pr = data.get("priority", 10)
        try:
            data["priority"] = int(pr)
        except Exception:
            data["priority"] = 10

            
        return data


    def decide(self, agent: "SimHumanAgent", boxes: List[BoxSummary], now_sim: float) -> PolicyAction:
    
    
        if not getattr(agent, "comm_enable", False):
            # no commitments or inbox-driven overrides in no-comm mode
            if agent.llm_provider == "none":
                return self.fallback.decide(agent, boxes, now_sim)
            # continue with normal ACT decision, but it will be adapter-filtered anyway

        # ✅ commitments override: if we promised to help someone, do that first
        c = agent._next_executable_commitment(now_sim)
        if c is not None:
            box_id = int(c["box_id"])
            prop = str(c["prop"]).upper()
            requester = str(c["from"])
            kind = str(c.get("requested_kind", "sense_self"))

            # If box already has a completed sense for that prop, we can "finish" the commitment with a say
            b = next((bb for bb in boxes if bb.box_id == box_id), None)
            if b is None:
                agent._complete_commitment(c, status="cancelled")
            else:
            
                # ✅ NEW: do not execute commitments on disposed targets
                disposed_for_prop = agent._disposed_any(b)
                if disposed_for_prop:
                    agent._complete_commitment(c, status="done")
                    return PolicyAction(
                        kind="say",
                        text=f"{agent._display_name(requester)}, box {box_id} is already disposed for {prop}.",
                        target_speaker=requester,
                        reason="commitment_fulfilled_already_disposed",
                    )

            
                already_sensed = any(sr.get("status") == "completed" and sr.get("property") == prop for sr in b.sense_results)
                if already_sensed and kind in ("sense_self", "goto_only"):
                    agent._complete_commitment(c, status="done")
                    return PolicyAction(
                        kind="say",
                        text=f"{agent._display_name(requester)}, box {box_id} already has a recent sense for {prop}.",
                        target_speaker=requester,
                        reason="commitment_fulfilled_already_sensed",
                    )

                # Execute the commitment action now
                # Most common: sense_self to help them
                act = PolicyAction(
                    kind="sense_self" if kind not in ("dispose", "goto_only", "say") else kind,  # safe mapping
                    box_id=box_id if kind != "say" else None,
                    prop=prop if kind != "say" else None,  # for say, no prop required
                    target_speaker=requester,
                    text=f"Okay {agent._display_name(requester)}, I’ll sense box {box_id} for {prop}." if kind != "say" else f"Okay {agent._display_name(requester)}.",
                    reason=f"execute_commitment decision={c.get('decision')}",
                )

                # Mark as done *optimistically* for now; or mark done after actual sensing.
                # Better: mark done after the sense succeeds. Easiest is: mark here, and if sense fails it’s fine.
                # keep it active; we'll mark done in _execute after successful sense/dispose
                agent.plan_state["active_commitment_id"] = c.get("id")
                return act

         
 
    
        if agent.llm_provider == "none":
            return self.fallback.decide(agent, boxes, now_sim)

        client = self._get_client(agent)
        if client is None:
            return self.fallback.decide(agent, boxes, now_sim)

        sys_msg = self._system_prompt(agent)
        user_msg = self._user_prompt(agent, boxes, now_sim)

        agent._dbg_llm("SYSTEM_MSG_ACT", sys_msg)
        agent._dbg_llm("USER_MSG_ACT", user_msg)


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

        if action.kind == "sense_self" and action.box_id is not None and action.prop is not None:
            st = agent._box_state(action.box_id, action.prop)
            if st.get("self_sensed", False):
                agent.get_logger().warn("[LLM] sense_self requested but already self_sensed; overriding to idle")
                return PolicyAction(kind="idle", reason="guardrail_repeat_sense_self")


        # Guardrail: never allow disposing non-goal property
        if action.kind == "dispose" and action.prop != agent.goal_property:
            agent.get_logger().warn("[LLM] attempted dispose of non-goal property; overriding to idle")
            return PolicyAction(kind="idle", reason="guardrail_non_goal_dispose")

        # Guardrail: while waiting_help is active, don't duplicate the same task yourself
        if (
            action.kind == "sense_self"
            and action.box_id is not None
            and action.prop is not None
            and agent._waiting_help_block_same_task(int(action.box_id), action.prop, now_sim)
        ):
            agent.get_logger().warn("[LLM] sense_self matches active waiting_help; overriding to idle")
            return PolicyAction(kind="idle", reason="guardrail_waiting_help_same_task")


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
        self.declare_parameter("think_min_delay_sec", 10)
        self.declare_parameter("think_max_delay_sec", 20)
        self.declare_parameter("think_router_enable", False)  # optional: also delay in message-router decisions


        # ---------------------------
        # Speech simulation (delay before publishing utterances)
        # ---------------------------
        self.declare_parameter("speech_sim_enable", True)
        self.declare_parameter("speech_rate_wpm", 150.0)          # typical conversational ~130-170 wpm
        self.declare_parameter("speech_min_delay_sec", 0.15)      # minimum "start speaking" delay
        self.declare_parameter("speech_max_delay_sec", 6.0)       # cap so long messages don't stall forever
        self.declare_parameter("speech_punct_pause_sec", 0.06)    # extra pause per punctuation mark
        self.declare_parameter("speech_queue_max", 30)            # prevent unbounded queue growth


        self.declare_parameter("waiting_mode", "strict")  # strict | soft
        
        
        # ---- communication master switch ----
        self.declare_parameter("comm_enable", True)
        self.comm_enable = bool(self.get_parameter("comm_enable").value)

        
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



        self.declare_parameter("infer_target_use_llm", True)
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

        if self.comm_enable:
            self.pub_stt = self.create_publisher(StringMsg, self.stt_topic, 10)
            self.sub_stt = self.create_subscription(StringMsg, self.stt_topic, self._on_stt_text, 10)

            self._speech_thread = threading.Thread(target=self._speech_worker_main, daemon=True)
            self._speech_thread.start()
        else:
            self.pub_stt = None
            self.sub_stt = None
            self.speech_sim_enable = False


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
        # global senseability gate
        if isinstance(b.senseable, dict):
            if not bool(b.senseable.get(prop, True)):
                return False

        # optional per-agent gate
        if b.senseable_by and isinstance(b.senseable_by, dict):
            who = agent_id or self.agent_id
            allowed = b.senseable_by.get(prop, [])
            return (who in allowed)

        return True






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
        busy = self._is_busy() or self._is_speaking()


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
            if target is not None and str(target) not in (self.agent_id, "all"):
                return

            if speaker == self.agent_id:
                return

        except Exception:
            return

        event = {"speaker_id": speaker, "text": text, "t_wall": time.time()}
        self.last_msgs.append(event)
        self.last_msgs = self.last_msgs[-100:]

        self.inbox.append(event)
        while len(self.inbox) > 50:
            self.inbox.popleft()

        self._log("HEAR", f"from={speaker} text={text!r}")

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

            out.data = json.dumps(payload)
            self.pub_stt.publish(out)
            self._log("SAY", final_text)
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
        """
        Block until the utterance has been 'spoken' and published.
        Safe to call from action loop OR router loop.
        """
        if not getattr(self, "comm_enable", False):
            return

        
        ev = threading.Event()
        self._enqueue_utterance(text, target_speaker, done_evt=ev)

        # Guardrail timeout to avoid deadlock if something breaks.
        if timeout is None:
            timeout = max(5.0, float(self.speech_max_delay_sec) + 5.0)

        ev.wait(timeout=timeout)



    def _speech_worker_main(self) -> None:
        """
        Dedicated worker thread:
          - pops queued utterances
          - waits proportional to message length (simulated speaking)
          - publishes to STT topic
        """
        while not self._speech_stop_evt.is_set():
            item = None
            with self._speech_cv:
                while not self._speech_queue and not self._speech_stop_evt.is_set():
                    self._speech_cv.wait(timeout=0.2)
                if self._speech_stop_evt.is_set():
                    break
                try:
                    item = self._speech_queue.popleft()
                except Exception:
                    item = None

            if not item:
                continue

            done_evt = item.get("done_evt", None)
            if done_evt is not None and not isinstance(done_evt, threading.Event):
                done_evt = None
            # ✅ mark speaking busy for the duration of this utterance
            self._set_speech_busy(True)


            text = str(item.get("text", ""))
            target_speaker = item.get("target_speaker", None)
            if target_speaker is not None:
                target_speaker = str(target_speaker)

            # Simulated speaking time (skip if disabled)
            if self.speech_sim_enable:
                dt = self._estimate_speech_delay_sec(text)

                # sleep in small increments so shutdown is responsive
                end = time.time() + dt
                while time.time() < end:
                    if self._speech_stop_evt.is_set() or self._stop:
                        break
                    time.sleep(0.05)

                if self._speech_stop_evt.is_set() or self._stop:
                    self._set_speech_busy(False)
                    if isinstance(done_evt, threading.Event):
                        done_evt.set()
                    continue


            # Build final outgoing text (your existing "Hey <name>" behavior)
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

            out.data = json.dumps(payload)

            # Publish (this is the actual "message send" moment)
            self.pub_stt.publish(out)
            self._log("SAY", final_text)
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

        # no evidence -> prior
        if not evidence:
            p = 0.5
            # cache minimal details
            st = self._box_state(box.box_id, prop)
            st["fusion_details"] = {"prop": prop, "prior": 0.5, "n_used": 0, "p_posterior": 0.5, "trace": []}
            return p

        p, details = self._bayes_fuse_present(evidence, prop, prior=0.5)

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

            

            
            if already_by_anyone:
                self._log("MEM", f"skip self_sense box={box.box_id} prop={action.prop} (already sensed by someone)")
                st["self_sensed"] = True
                # if we were doing this as a commitment, fulfill it
                self._complete_active_commitment_if_any(status="done")
                return

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
                self._set_current_op("sense", box.box_id, action.prop, now_sim)
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
                self._set_current_op("dispose", box.box_id, action.prop, now_sim)
                js = self._dispose(box.box_id, action.prop)
                
                if j_idx is not None:
                    self._journal_update(j_idx, {
                        "status": js.get("status"),
                        "success": js.get("success"),
                    })

                
                self._complete_active_commitment_if_any(status="done")
                self._mark_done(box.box_id, action.prop, why=f"dispose_attempt success={js.get('success')}")
            finally:
                self._clear_current_op()
                self._set_busy(False)
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
        # Don't spawn if already running
        with self._router_lock:
            if self._router_thread is not None and self._router_thread.is_alive():
                return
            if not self.inbox:
                return
            th = threading.Thread(target=self._router_thread_main, daemon=True)
            self._router_thread = th
            th.start()

    def _router_thread_main(self) -> None:
        """
        ✅ This thread MUST stay responsive and MUST NOT do travel/sense/dispose.
        It only:
          - runs LLM decide_on_message for up to N inbox items
          - publishes reply_text
          - adds/updates commitments
        """
        try:
            # Lightweight snapshot of time + boxes for router context
            t = self._time()
            now_sim = float(t["server_time"])
            
            
            time_limit = float(t["time_limit_sec"])
            if now_sim >= time_limit:
                self._request_shutdown_with_summary(
                    now_sim=now_sim,
                    time_limit=time_limit,
                    why="time_limit_reached (router loop)",
                )
                return

            
            boxes = self._boxes_state()

            handled = 0
            while handled < self.max_inbox_per_tick:
                try:
                    evt = self.inbox.popleft()
                except Exception:
                    break

                speaker = str(evt.get("speaker_id", ""))
                text = str(evt.get("text", ""))

                explicit_target = evt.get("target_speaker", None)

                if explicit_target is not None:
                    tgt = str(explicit_target)
                    route_src = "explicit"
                else:
                    tgt = self._infer_target_speaker(speaker, text, now_sim)
                    route_src = "llm_infer" if self.infer_target_use_llm else "fallback"

                # ✅ NEW: log what we think the recipient is
                self._log(
                    "ROUTE",
                    f"route src={route_src} from={speaker} -> target={tgt} "
                    f"me={self.agent_id} text={text!r}"
                )


                # If not for me (and not broadcast), ignore
                if tgt not in (self.agent_id, "all"):
                    self._log("ROUTE", f"ignore msg inferred_target={tgt} from={speaker} text={text!r}")
                    continue

                # If it IS for me, you can optionally annotate event so downstream LLM sees it
                evt["target_speaker"] = tgt

                if self.policy_type == "llm":
                    if self.think_router_enable:
                        self._maybe_think(where="router_decide_on_message")
                    _ = self.llm_policy.decide_on_message(self, boxes, now_sim, evt)


                handled += 1

        except Exception as e:
            self.get_logger().warn(f"[FAIL] router cycle failed: {e}")
        finally:
            with self._router_lock:
                self._router_thread = None


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

