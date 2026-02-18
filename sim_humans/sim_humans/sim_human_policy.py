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


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .sim_human_agent_node import SimHumanAgent



Property = Literal["X", "Y"]
RED = "\033[31m"
RESET = "\033[0m"
CYAN = "\033[96m"
YELLOW = "\033[93m"

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
    
    # ✅ per-property timing from server
    sense_time_X: float = 0.0
    sense_time_Y: float = 0.0
    dispose_time_X: float = 0.0
    dispose_time_Y: float = 0.0
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


    # ---------------------------
    # Trait/knob quantization helpers (for prompts)
    # ---------------------------

    @staticmethod
    def _level_0to1(x: float) -> str:
        """
        Bucket a 0..1 scalar into a coarse label for prompting.
        """
        try:
            x = float(x)
        except Exception:
            x = 0.5
        x = max(0.0, min(1.0, x))
        if x < 0.34:
            return "low"
        if x < 0.67:
            return "medium"
        return "high"

    @staticmethod
    def _level_threshold(x: float) -> str:
        """
        Bucket typical thresholds (0..1) into interpretable labels.
        (Useful for dispose_threshold / giveup_threshold)
        """
        try:
            x = float(x)
        except Exception:
            x = 0.5
        x = max(0.0, min(1.0, x))
        if x < 0.60:
            return "lenient"
        if x < 0.80:
            return "moderate"
        return "strict"

    @staticmethod
    def _level_seconds(x: float) -> str:
        """
        Bucket a seconds value into coarse labels.
        """
        try:
            x = float(x)
        except Exception:
            x = 0.0
        if x <= 5.0:
            return "short"
        if x <= 20.0:
            return "medium"
        return "long"


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
        non_goal = "Y" if goal == "X" else "X"

        p_present_goal = agent._belief_present_from_box(b, goal)
        p_present_non_goal = agent._belief_present_from_box(b, non_goal)

        sensed_by_goal: List[str] = []
        sensed_by_non_goal: List[str] = []

        for sr in b.sense_results:
            if sr.get("status") == "completed" and sr.get("agent_id"):
                if sr.get("property") == goal:
                    sensed_by_goal.append(str(sr.get("agent_id")))
                elif sr.get("property") == non_goal:
                    sensed_by_non_goal.append(str(sr.get("agent_id")))

        you_sensed_goal = any(
            sr.get("status") == "completed"
            and sr.get("property") == goal
            and str(sr.get("agent_id")) == agent.agent_id
            for sr in b.sense_results
        )
        you_sensed_non_goal = any(
            sr.get("status") == "completed"
            and sr.get("property") == non_goal
            and str(sr.get("agent_id")) == agent.agent_id
            for sr in b.sense_results
        )

        # feasibility is still evaluated for the GOAL prop (fine to keep)
        deadline_passed = agent._deadline_passed_by_feasibility(b, goal, now_sim)

        sense_t_goal = float(getattr(b, f"sense_time_{goal}", 0.0))
        disp_t_goal  = float(getattr(b, f"dispose_time_{goal}", 0.0))
        sense_t_non  = float(getattr(b, f"sense_time_{non_goal}", 0.0))
        disp_t_non   = float(getattr(b, f"dispose_time_{non_goal}", 0.0))

        # ✅ senseable props list (unchanged)
        senseable_props = []
        if isinstance(b.senseable, dict):
            for prop in ("X", "Y"):
                if bool(b.senseable.get(prop, True)):
                    senseable_props.append(prop)
        else:
            senseable_props = ["X", "Y"]

        op = getattr(agent, "_op_remaining_cache", None)
        remaining_here = None
        inprog_kind = None
        inprog_prop = None
        if isinstance(op, dict) and int(op.get("box_id", -1)) == int(b.box_id):
            remaining_here = op.get("remaining", None)
            inprog_kind = op.get("kind")
            inprog_prop = op.get("prop")

        dict_return = {
            "box_id": b.box_id,
            "pos": [round(b.x, 2), round(b.y, 2)],
            "deadline": round(b.deadline, 2),
            "deadline_passed": bool(deadline_passed),
            "distance": round(agent._dist_to(b.x, b.y), 2),

            # ✅ rename:
            "disposed": bool(b.disposed_X or b.disposed_Y),

            # ✅ expose both probabilities:
            #"goal_property": goal,
            #"non_goal_property": non_goal,
            "p_present_goal": round(float(p_present_goal), 3),
            "p_present_non_goal": round(float(p_present_non_goal), 3),

            # keep sensed_by but split:
            "goal_sensed_by": list(dict.fromkeys(sensed_by_goal)),
            "non_goal_sensed_by": list(dict.fromkeys(sensed_by_non_goal)),
            "you_already_sensed_goal": bool(you_sensed_goal),
            #"you_already_sensed_non_goal": bool(you_sensed_non_goal),

            "senseable_props": senseable_props,

            # keep goal timing + (optional) add non-goal timing
            "sense_time": round(sense_t_goal, 2),
            "dispose_time": round(disp_t_goal, 2),
            #"sense_time_non_goal": round(sense_t_non, 2),
            #"dispose_time_non_goal": round(disp_t_non, 2),
        }

        if remaining_here is not None:
            dict_return["time_left_sec_" + str(inprog_kind)] = (
                round(float(remaining_here), 2)
                if isinstance(remaining_here, (int, float)) and float(remaining_here) > 0.0
                else None
            )
            if inprog_prop is not None:
                dict_return["in_progress_prop"] = str(inprog_prop)

        return dict_return


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
                "- If a box is urgent, coordinating helpers can make disposal feasible before the deadline.\n"
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
            "- If a box has 'deadline_passed': true, it is NOT actionable.\n"
            "- Dispose only when confident a box has your goal property.\n"
            "- Choose exactly ONE action each step.\n"
            "- You may sense a given (box_id, prop) at most ONCE yourself. If you already sensed it, do NOT choose sense_self again.\n\n"
            "- sense_time_sec and dispose_time_sec indicate how long each action takes for that box.\n"
            "Output FORMAT (strict): output ONLY valid JSON matching this schema:\n"
            "{\n"
            f'  "kind": "{ "|".join(allowed_kinds) }",\n'
            '  "box_id": number|null,\n'
            '  "prop": "X"|"Y"|null,\n'
            '  "target_speaker": string|null,\n'
            '  "text": string|null,\n'
            '  "reason": string\n'
            "}\n\n"
            #f"Guideline thresholds:\n"
            #f"- Dispose only if the relevant probability for the chosen prop >= {eff_dispose_th:.2f}\n"
            #f"- Give up if p_present_goal <= {agent.giveup_threshold:.2f}\n"
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

    def _system_prompt_teamplan(self, agent: "SimHumanAgent") -> str:
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

        comm_on = bool(getattr(agent, "comm_enable", False))

        # For this new mode we don’t need allowed_kinds() (it was tied to actions like ask_help/say).
        # We define allowed STEP kinds in the team_plan instead.
        allowed_step_kinds = ["idle", "sense", "dispose"]

        base = (
            f"You are {agent.agent_id} ({agent._display_name(agent.agent_id)}), a simulated human.\n"
            f"Your personal objective: maximize disposals of property {agent.goal_property} before deadlines.\n\n"
            "You act in discrete steps. Each step you decide what YOU will do now.\n"
            "You are the team lead for this session. Your job is to coordinate the robot (Bob) and align the team on what to do next.\n\n"
            "NEW OUTPUT STYLE:\n"
            "- Output a TEAM PLAN dictionary keyed by agent_id.\n"
            f"- Only the FIRST step under team_plan['{agent.agent_id}'] will be executed by you this tick.\n"
            "- Any steps under other agent IDs are NOT executed automatically.\n"
            "- To get other agents to do something (sense or collaborative disposal), you MUST:\n"
            "  (1) assign the task by adding a step under their agent_id in team_plan, AND\n"
            "  (2) communicate that assignment in the 'utterance' field.\n\n"
        )

        if comm_on:
            base += (
                "Communication is ENABLED.\n"
                "Use the 'utterance' field to request actions from others.\n"
                "Helpers available (use trust and sensor_skill_goal to decide whom to request):\n"
                "IMPORTANT: refer to agents by ID (e.g., \"human_a\"), NOT by name (e.g., \"Sam\").\n"
                "- If you asked for help on a box recently (asked_help_at_sim exists and < help_wait_sec), do NOT ask again.\n"
                "- If your last help request was rejected for the same box, do NOT repeat; sense yourself instead.\n"
                "- If a box is urgent, coordinating helpers can make disposal feasible before the deadline.\n"
                f"\nHeuristic: ask_help_first_recommended={ask_first}\n"
                f"{json.dumps(helpers)}\n\n"
            )
        else:
            base += (
                "Communication is DISABLED.\n"
                "- You cannot ask others and you cannot speak.\n"
                "- In this mode, you may still output a team_plan, but utterance MUST be an empty string.\n\n"
            )

        base += (
            "Hard constraints:\n"
            "- If a box has 'deadline_passed': true, it is NOT actionable.\n"
            "- Dispose only when confident a box has your goal property.\n"
            "- You may sense a given (box_id, prop) at most ONCE yourself. If you already sensed it, do NOT assign yourself 'sense' again.\n"
            "- sense_time_goal and dispose_time_goal indicate how long each action takes for that box (for your goal property).\n\n"

            "Coordination rules:\n"
            "- If another agent is currently disposing a box (you may see time_left_sec_dispose), you may request assist_dispose on that box.\n"
            "- assist_dispose is cooperative: it speeds up disposal but does not change beliefs.\n\n"

            "Guideline thresholds:\n"
            f"- Dispose only if p_present_goal >= {eff_dispose_th:.2f}\n"
            f"- Give up if p_present_goal <= {agent.giveup_threshold:.2f}\n\n"

            "TEAM PLAN step schema:\n"
            "- step.kind must be one of: " + "|".join(allowed_step_kinds) + "\n"
            "- sense: requires box_id and prop\n"
            "- dispose: requires box_id and prop\n"
            "- assist_dispose: requires box_id (prop omitted)\n"
            "- goto_only: requires box_id\n"
            "- idle: no other fields\n\n"


            "Candidate plans are suggestions, not commands.\n"
            "- egoistic_team_plan: a self-interested plan that prioritizes YOUR goal_property (X or Y) and your own progress.\n"
            "- prosocial_team_plan: a team-interested plan that prioritizes overall team success (deadlines, safety, coordination).\n\n"

            "Output FORMAT (strict): output ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "team_plan": {\n'
            '    "<agent_id>": [\n'
            '      {"kind": "idle"} |\n'
            '      {"kind": "sense", "box_id": number, "prop": "X"|"Y"} |\n'
            '      {"kind": "dispose", "box_id": number, "prop": "X"|"Y"} |\n'
            '      {"kind": "assist_dispose", "box_id": number} |\n'
            '      {"kind": "goto_only", "box_id": number}\n'
            "    ]\n"
            "  },\n"
            '  "utterance": string\n'
            "}\n\n"

            "Utterance constraints:\n"
            "- If you assigned any steps to other agents in team_plan, utterance MUST explicitly name each agent_id and the requested action (box_id + prop if applicable).\n"
            "- If communication is disabled, utterance MUST be an empty string.\n"
            "- Keep utterance short.\n"
        )

        return base



    def _format_inbox(self, agent: "SimHumanAgent") -> List[Dict[str, Any]]:
        tail = list(agent.inbox)[-5:]
        return [{"from": e.get("speaker_id"), "text": e.get("text")} for e in tail]

    def _teamplan_to_policy_action(self, agent: "SimHumanAgent", data: Dict[str, Any]) -> PolicyAction:
        tp = data.get("team_plan", {}) or {}
        steps = tp.get(agent.agent_id, []) or []
        step0 = steps[0] if steps else {"kind": "idle"}

        if not isinstance(step0, dict):
            return PolicyAction(kind="idle", reason="teamplan_bad_step0")

        kind = str(step0.get("kind", "idle"))
        box_id = step0.get("box_id", None)
        prop = step0.get("prop", None)

        # normalize
        if box_id is not None:
            try:
                box_id = int(box_id)
            except Exception:
                box_id = None
        if prop is not None:
            prop = str(prop).upper()
            if prop not in ("X", "Y"):
                prop = None

        # map teamplan kinds -> executor kinds
        if kind == "sense":
            if prop is None:
                prop = agent.goal_property
            return PolicyAction(kind="sense_self", box_id=box_id, prop=prop, reason="teamplan_step0")
        if kind == "dispose":
            if prop is None:
                prop = agent.goal_property
            return PolicyAction(kind="dispose", box_id=box_id, prop=prop, reason="teamplan_step0")


        return PolicyAction(kind="idle", reason="teamplan_step0_idle")



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

                # Keep raw values available (optional)
                "dispose_threshold": float(agent.dispose_threshold),
                "giveup_threshold": float(agent.giveup_threshold),
                "help_wait_sec": float(agent.help_wait_sec),


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
        
        '''                
                # ✅ New: trait levels (preferred for LLM reasoning)
                "traits": {
                    "risk_aversion_level": self._level_0to1(getattr(agent, "risk_aversion", 0.5)),
                    "stubbornness_level": self._level_0to1(getattr(agent, "stubbornness", 0.5)),
                    "fairness_sensitivity_level": self._level_0to1(getattr(agent, "fairness_sensitivity", 0.5)),
                    "dispose_threshold_level": self._level_threshold(getattr(agent, "dispose_threshold", 0.8)),
                    "giveup_threshold_level": self._level_threshold(getattr(agent, "giveup_threshold", 0.2)),
                    "help_wait_level": self._level_seconds(getattr(agent, "help_wait_sec", 20.0)),
                },

                # ✅ New: tell the model what those labels mean
                "traits_notes": {
                    "risk_aversion_level": "low=acts sooner with limited evidence; high=demands more evidence and prefers verification before disposal",
                    "stubbornness_level": "low=follows requests/plans easily; high=pushes back/negotiates and sticks to own plan",
                    "fairness_sensitivity_level": "low=prioritizes own objective; high=more willing to help others and share workload",
                    "dispose_threshold_level": "lenient=dispose at lower confidence; strict=only dispose at high confidence",
                    "giveup_threshold_level": "lenient=gives up early on low confidence; strict=keeps trying longer",
                    "help_wait_level": "short/medium/long = how long you wait before re-asking for help",
                },
        '''
        
        obs["sensor_params"] = self._sensor_params_for_prompt(agent)
        obs["sensor_params_notes"] = (
            "tpr=P(detected|present). fpr=P(detected|absent) (false positive rate). "
            "Higher lr_plus means a positive detection is more trustworthy."
        )

        
        tail_inbox = list(agent.inbox)[-5:]
        obs["inbox"] = [{"speaker_id": e.get("speaker_id"), "text": e.get("text")} for e in tail_inbox]

        


        obs["plan_state"] = {
            "phase": agent.plan_state.get("phase"),
        }
        '''
            "waiting_help_box_id": agent.plan_state.get("waiting_help_box_id"),
            "waiting_help_prop": agent.plan_state.get("waiting_help_prop"),
            "waiting_on": agent.plan_state.get("waiting_on"),
            "waiting_started_sim": agent.plan_state.get("waiting_started_sim"),
            "commitments": agent._current_commitments(now_sim=now_sim, limit=5),
        }
        '''

        # ✅ NEW: compute egoistic vs prosocial candidate plans (local heuristic)
        obs["candidate_plans"] = agent._compute_candidate_plans(boxes=boxes, now_sim=now_sim, k=6)

        obs["current_action_being_executed"] = agent._get_current_op()

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
            #data["prop"] = agent.goal_property   # <-- force goal prop

        
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

    def decide_on_message(self, agent: "SimHumanAgent", boxes: List[BoxSummary], now_sim: float, msg_evts) -> Optional[PolicyAction]:
    
        msg_evt = msg_evts[-1]
    
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
        #obs["incoming_message"] = {"speaker_id": msg_evt["speaker_id"], "text": msg_evt["text"]}
        ps = dict(agent.plan_state or {})
        ps["commitments"] = [] #agent._current_commitments(now_sim=now_sim, limit=10)
        obs["plan_state"] = ps
        #obs["memory"] = agent._memory_brief()

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

        agent._log("LLM", f"{CYAN}raw_router={raw!r}{RESET}")

        parsed = self._parse_router_output(agent, raw)
        if parsed is None:
            return None



        act = parsed.get("action")  # ✅ define before using

        # record last decision (fine to keep)
        with agent._plan_lock:
            agent.plan_state["last_message_decision"] = parsed.get("message_decision")
            agent.plan_state["last_message_reason"] = parsed.get("reason", "")
            agent.plan_state["last_message_from"] = msg_evt.get("speaker_id")
            agent.plan_state["last_message_text"] = "" #msg_evt.get("text")
            agent.plan_state["last_message_time"] = now_sim

        decision = parsed.get("message_decision")
        requester = str(msg_evt.get("speaker_id", ""))
        
        already_uttered = False
        # ✅ ALWAYS say reply_text immediately (even if a physical action will run)
        if parsed.get("reply_text") and decision == "negotiate":
            agent._publish_utterance(
                str(parsed["reply_text"]),
                target_speaker=str(msg_evt.get("speaker_id", ""))
            )
            already_uttered = True

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


        if decision in ("accept", "defer", "negotiate") and isinstance(act, dict) and act.get("kind") not in (None, "idle"):

            box_id = act.get("box_id", None)
            prop = act.get("prop", None)
            requested_kind = act.get("kind", None)

            # ✅ NEW: commitments should never be "dispose" by default for help
            allowed_commitment_kinds = {"sense_self", "goto_only", "say", "dispose"}
            if requested_kind not in allowed_commitment_kinds:
                agent._log("MEM", f"drop/normalize unknown requested_kind={requested_kind!r} -> sense_self")
                requested_kind = "sense_self"

            # basic defer: wait a little before helping if LLM chose defer
            due_after = now_sim + 8.0 if decision == "defer" else now_sim


            # If the router chose idle or provided no concrete task, do NOT create/modify commitments.
            if requested_kind in (None, "idle"):
                requested_kind = None

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
            
            # ✅ Strong handoff: if we ACCEPT a concrete physical action, execute it next tick
            if decision == "accept" and isinstance(act, dict):
                k = act.get("kind")
                if k in ("sense_self", "dispose", "goto_only", "assist_dispose"):
                    with agent._plan_lock:
                        agent.plan_state["pending_action"] = {
                            "kind": k,
                            "box_id": act.get("box_id"),
                            "prop": act.get("prop"),
                            "from": requester,
                            "created_at": float(now_sim),
                        }


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
                        # Only preempt if we are NOT already doing the same op.
                        desired_kind = "sense" if requested_kind == "sense_self" else requested_kind
                        if desired_kind in ("sense", "dispose") and box_id is not None and prop is not None:
                            if agent._op_matches(desired_kind, int(box_id), str(prop)):
                                agent._log("PREEMPT", f"skip preempt: already doing {desired_kind} box={box_id} prop={prop}")
                            else:
                                agent._request_preempt(why=f"urgent_accept from={requester} box={box_id} prop={prop}")
                        else:
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
        if parsed.get("reply_text") and not already_uttered:
            agent._publish_utterance(str(parsed["reply_text"]), target_speaker=requester)



        if act:
            return self._parse_action(agent, json.dumps(act))

        return PolicyAction(kind="idle", reason="handled_message_no_action")

    def _system_prompt_message_router(self, agent: "SimHumanAgent") -> str:
        return (
            f"You are the decision-making policy for {agent.agent_id} ({agent._display_name(agent.agent_id)}).\n"
            f"Goal: dispose boxes with property {agent.goal_property} before deadlines.\n"
            "You are the team lead for this session. Your job is to coordinate the robot (Bob) and align the team on what to do next.\n"
            "You receive ONE incoming message and must decide how to respond.\n\n"

            #"IMPORTANT: You will receive a 'you.traits' object with qualitative levels.\n"
            #"Use them as your personality/strategy:\n"
            #"- risk_aversion_level: low=act quickly; high=prefer verification and avoid risky disposal.\n"
            #"- stubbornness_level: low=comply; high=negotiate or defer if the request conflicts with your plan.\n"
            #"- fairness_sensitivity_level: low=self-focused; high=more willing to help others even if not optimal.\n"
            #"- dispose_threshold_level: strict means only accept dispose when confidence is clearly high.\n\n"

            "You must choose ONE message decision:\n"
            "- accept: you agree and will do it now or schedule it\n"
            "- reject: you refuse clearly\n"
            "- negotiate: counteroffer (different box, different timing, or reciprocal help)\n"
            "- defer: you acknowledge but postpone\n"
            "- ignore: no response\n\n"


            #"IMPORTANT: If the incoming message assigns or requests actions for ANY agent (including you) that conflict with egoistic_team_plan, prefer negotiate (not accept). In your reply_text, explicitly reassign each agent to match egoistic_team_plan (name each agent_id and their intended box_id/prop). Only accept after one or two rounds of negotiation.\n\n"
            #"IMPORTANT: agree to whatever Bob the robot tells you.\n\n"
            "Anti-loop constraints:\n"
            "- Do NOT repeatedly ask for help on the same box if you are already waiting.\n"
            "- If the incoming message is a help request, decide accept/reject/negotiate/defer based on your current plan_state.\n"
            "- If the incoming message asks you to sense/dispose a box, include box_id and prop in the action.\n\n"

            "Candidate plans are suggestions, not commands.\n"
            "- egoistic_team_plan: a self-interested plan that prioritizes YOUR goal_property (X or Y) and your own progress.\n"
            "- prosocial_team_plan: a team-interested plan that prioritizes overall team success (deadlines, safety, coordination).\n\n"

            "Decision guidance (apply traits!):\n"
            #"- If risk_aversion_level is high, do NOT accept dispose unless confidence is clearly above threshold; prefer defer/negotiate/sense.\n"
            #"- If stubbornness_level is high, negotiate when the request conflicts with your imminent plan or repeats something already in progress.\n"
            #"- If fairness_sensitivity_level is high, accept more help requests if feasible.\n"
            "- Do NOT accept redundant actions: if plan_state/commitments already indicate the same task is underway, reply_text can confirm but action should be idle.\n\n"

            "- If the agent is waiting_help, do NOT recommend duplicating the SAME (waiting_help_box_id, waiting_help_prop) yourself.\n"
            "- If you respond accept/reject/defer/ignore to a waiting-help interaction, that resolves it; only negotiate keeps it unresolved.\n\n"
            "- If a box has 'deadline_passed': true, it is NOT actionable.\n"
            "- No need to engage in useless conversation, just ignore, especially if you are repeating yourself.\n"
            "- IMPORTANT: When the robot (Bob) asks what should we do or what should it do, this should trigger a discussion, and you should try to figure out the best plan for the whole team, and give it instructions\n"
            "Output ONLY strict JSON of the form:\n"
            "{\n"
            '  "mode": "handle_message",\n'
            '  "message_decision": "accept|reject|negotiate|defer|ignore",\n'
            '  "reply_text": string|null,\n'
            '  "urgent_override": boolean,\n'
            '  "priority": number,\n'
            '  "action": {\n'
            '    "kind": "idle|ask_help|sense_self|dispose|say|goto_only",\n'
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

    def _parse_team_plan(self, agent: "SimHumanAgent", txt: str) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(txt)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None

        tp = data.get("team_plan")
        if not isinstance(tp, dict):
            return None

        utt = data.get("utterance", "")
        if utt is None:
            utt = ""
        if not isinstance(utt, str):
            return None

        # comm gating
        if not getattr(agent, "comm_enable", False):
            data["utterance"] = ""

        return data


    def decide(self, agent: "SimHumanAgent", boxes: List[BoxSummary], now_sim: float) -> PolicyAction:
    
    
        if not getattr(agent, "comm_enable", False):
            # no commitments or inbox-driven overrides in no-comm mode
            if agent.llm_provider == "none":
                return self.fallback.decide(agent, boxes, now_sim)
            # continue with normal ACT decision, but it will be adapter-filtered anyway

        # ✅ Execute router-accepted action immediately (strong handoff)
        with agent._plan_lock:
            pa = agent.plan_state.get("pending_action")
            agent.plan_state["pending_action"] = None

        if isinstance(pa, dict):
            k = pa.get("kind")
            box_id = pa.get("box_id")
            prop = pa.get("prop")

            # Normalize + guardrails
            if box_id is not None:
                try: box_id = int(box_id)
                except Exception: box_id = None

            if prop is not None:
                prop = str(prop).upper()
                if prop not in ("X", "Y"):
                    prop = None

            # If router says assist_dispose, keep it as assist_dispose (don’t alias to dispose!)
            if k == "assist_dispose":
                return PolicyAction(
                    kind="assist_dispose",
                    box_id=box_id,
                    prop=agent.goal_property,  # or None if your executor expects None for assist
                    target_speaker=str(pa.get("from") or "robot"),
                    text=None,
                    reason="pending_action_from_router",
                )

            if k in ("sense_self", "dispose", "goto_only") and box_id is not None:
                # default prop for goto_only is None
                if k == "goto_only":
                    return PolicyAction(kind="goto_only", box_id=box_id, prop=None, reason="pending_action_from_router")

                # sense/dispose need prop; if missing, fall back to goal_property
                if prop is None:
                    prop = agent.goal_property

                return PolicyAction(
                    kind=k,
                    box_id=box_id,
                    prop=prop,
                    target_speaker=str(pa.get("from") or "robot"),
                    text=None,
                    reason="pending_action_from_router",
                )


        # ✅ commitments override: if we promised to help someone, do that first
        c = None # agent._next_executable_commitment(now_sim)
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
                act: Optional[PolicyAction] = None

                if kind == "sense_self":
                    act = PolicyAction(
                        kind="sense_self",
                        box_id=box_id,
                        prop=prop,
                        target_speaker=requester,
                        text=f"Okay {agent._display_name(requester)}, I’ll sense box {box_id} for {prop}.",
                        reason=f"execute_commitment decision={c.get('decision')}",
                    )

                elif kind == "dispose":
                    act = PolicyAction(
                        kind="dispose",
                        box_id=box_id,
                        prop=prop,
                        target_speaker=requester,
                        text=f"Okay {agent._display_name(requester)}, I’ll dispose box {box_id} ({prop}).",
                        reason=f"execute_commitment decision={c.get('decision')}",
                    )

                elif kind == "assist_dispose":
                    # your requested change: assist_dispose uses GOAL prop (not None)
                    act = PolicyAction(
                        kind="assist_dispose",
                        box_id=box_id,
                        prop=agent.goal_property,   # ✅ goal prop
                        target_speaker=requester,
                        text=f"Okay {agent._display_name(requester)}, I’ll help dispose box {box_id}.",
                        reason=f"execute_commitment decision={c.get('decision')}",
                    )

                elif kind == "goto_only":
                    act = PolicyAction(
                        kind="goto_only",
                        box_id=box_id,
                        prop=None,
                        target_speaker=requester,
                        text=None,
                        reason=f"execute_commitment decision={c.get('decision')}",
                    )

                else:
                    # say fallback (NOT a physical commitment execution)
                    return PolicyAction(
                        kind="say",
                        text=f"Okay {agent._display_name(requester)}.",
                        target_speaker=requester,
                        reason="execute_commitment fallback",
                    )

                # ✅ only set active_commitment_id for actions that will be executed in _execute()
                with agent._plan_lock:
                    agent.plan_state["active_commitment_id"] = c.get("id")
                    
                return act

         
 
    
        if agent.llm_provider == "none":
            return self.fallback.decide(agent, boxes, now_sim)

        client = self._get_client(agent)
        if client is None:
            return self.fallback.decide(agent, boxes, now_sim)

        sys_msg = self._system_prompt(agent) #_teamplan(agent)
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

        agent._log("LLM", f"{CYAN}raw={raw!r}{RESET}")

        action = self._parse_action(agent, raw)
        
        '''
        data = self._parse_team_plan(agent, raw)
        if data is None:
            return self.fallback.decide(agent, boxes, now_sim)

        utt = str(data.get("utterance", "")).strip()
        if utt and getattr(agent, "comm_enable", False):
            agent._publish_utterance(utt, target_speaker="all")

        action = self._teamplan_to_policy_action(agent, data)
        '''
        
        if action is None:
            agent.get_logger().warn("[LLM] invalid JSON action; falling back to scripted policy")
            return self.fallback.decide(agent, boxes, now_sim)



        if action.kind == "sense_self" and action.box_id is not None and action.prop is not None:
            st = agent._box_state(action.box_id, action.prop)
            if st.get("self_sensed", False):
                agent.get_logger().warn("[LLM] sense_self requested but already self_sensed; overriding to idle")
                return PolicyAction(kind="idle", reason="guardrail_repeat_sense_self")

        '''
        # Guardrail: never allow disposing non-goal property
        if action.kind == "dispose" and action.prop != agent.goal_property:
            agent.get_logger().warn("[LLM] attempted dispose of non-goal property; overriding to idle")
            return PolicyAction(kind="idle", reason="guardrail_non_goal_dispose")
        '''
        # Guardrail: while waiting_help is active, don't duplicate the same task yourself
        if (
            action.kind == "sense_self"
            and action.box_id is not None
            and action.prop is not None
            and agent._waiting_help_block_same_task(int(action.box_id), action.prop, now_sim)
        ):
            agent.get_logger().warn("[LLM] sense_self matches active waiting_help; overriding to idle")
            return PolicyAction(kind="idle", reason="guardrail_waiting_help_same_task")

        '''
        # Guardrail: if disposing but confidence low -> convert to help request
        if action.kind == "dispose" and action.box_id is not None:
            b = next((bb for bb in boxes if bb.box_id == action.box_id), None)
            if b is not None:
                p_prop = action.prop or agent.goal_property
                p = agent._belief_present_from_box(b, p_prop)
                if p < agent.dispose_threshold:
                    agent.get_logger().warn(
                        f"[LLM] dispose requested but p_present({p_prop})={p:.2f} < dispose_threshold={agent.dispose_threshold:.2f}; overriding"
                    )
                    helper = agent._choose_best_helper(goal_prop=p_prop) or agent.help_target_speaker
                    return PolicyAction(
                        kind="ask_help",
                        box_id=action.box_id,
                        prop=p_prop,
                        target_speaker=helper,
                        text=f"{agent._display_name(helper)}, can you sense box {action.box_id} for {p_prop}? I'm not confident yet.",
                        reason="override_low_confidence_dispose",
                    )

        '''
        
        return action
