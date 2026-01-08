#!/usr/bin/env python3
"""
optimizer_client.py

Gurobi-based planner for assigning sensing & disposal tasks to agents
(human_a, human_b, robot) given the current world state.

Key constraints:
- human_a can only sense property X
- human_b can only sense property Y
- robot can sense both X and Y
- Travel time is included for all tasks via travel_time_fn(agent_id, box_id)
- Each agent has a time budget 'max_time' for this planning horizon
- In a single plan, each agent must either be:
    * a sensing agent, OR
    * a disposal agent, OR
    * idle (no tasks)
  but not sensing & disposing at the same time.
- Disposal for (box, property) is only allowed if there is enough information
  about that box/property (info_X/info_Y >= threshold).
- Once a box has been sensed by an agent for a property, there is no use for
  that agent to sense that (box, property) again → no repeated sensing vars.
- Objective:
    * maximize expected correct disposals (using p_true_X/p_true_Y),
    * reward information-gathering for low-info boxes,
    * penalize imbalance between X and Y disposals,
    * optionally incorporate style / preference knobs (exploration vs exploitation,
      fairness, robot vs human load, deadline risk aversion).

To use:
- Build `AgentState` and `BoxInfo` from your API (/boxes/state, /time, DB).
- Provide a `travel_time_fn(agent_id, box_id)` that returns seconds.
- Build a `PlannerWeights` object to encode team/human preferences.
- Call `plan_assignments_gurobi(...)` to get lists of sense/dispose actions,
  then turn those into /sense and /dispose HTTP calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Literal, Callable, Tuple, Optional, Any

import gurobipy as gp
from gurobipy import GRB
import copy

Property = Literal["X", "Y"]
TravelTimeFn = Callable[[str, int], float]  # (agent_id, box_id) -> seconds

# Simple alias for plans:
#   { "robot":   [(box_id, "X"/"Y", "sense"/"dispose"), ...],
#     "human_a": [...],
#     "human_b": [...], ... }
Plan = Dict[str, List[Tuple[int, Property, str]]]


# ---------------------------------------------------------------------------
# Data structures to pass into the planner
# ---------------------------------------------------------------------------

@dataclass
class PlannerWeights:
    # core objective coefficients
    reward_correct_X: float = 1.0
    reward_correct_Y: float = 1.0
    weight_info: float = 0.2
    lambda_balance: float = 0.5
    info_threshold_for_dispose: float = 0.6

    # “style” knobs
    # how much we prefer exploration (sensing) vs exploitation (disposal)
    prefer_exploration: float = 0.0   # >0 → extra reward for senses

    # fairness / load sharing (between humans)
    lambda_load_fairness: float = 0.0  # penalize uneven time budgets

    # robot vs human load preference
    lambda_robot_overuse: float = 0.0  # penalize robot doing too much vs humans
    lambda_human_overuse: float = 0.0  # penalize humans doing too much vs robot

    # deadline / risk aversion
    lambda_deadline_risk: float = 0.0  # extra penalty for disposals close to / past deadline


@dataclass
class DisposalOutcome:
    """
    Outcome of a SINGLE disposal attempt (real execution, not just planned).

    Fields:
        agent_id: who did the disposal ("robot", "human_a", "human_b", ...)
        box_id: box id
        prop: "X" or "Y"
        completed_at: sim-time (seconds) when disposal finished
        success: whether the disposal was fulfilled (e.g., skill succeeded
                 AND server confirms the box is disposed_X/Y)
        correct: if you know ground truth, whether this disposal was correct
                 (e.g., contaminated box to contaminated_bin); otherwise None.
    """
    agent_id: str
    box_id: int
    prop: Property
    completed_at: Optional[float]
    success: bool
    correct: Optional[bool] = None
    planner_id: Optional[str] = None


@dataclass
class AgentState:
    """
    Agent state for the planner horizon.

    Fields:
        agent_id: string identifier (e.g., "human_a", "human_b", "robot").
        max_time: time budget (seconds) the agent can spend in this plan.
        can_sense_X: whether this agent can sense property X.
        can_sense_Y: whether this agent can sense property Y.

        detect_present_X / detect_absent_X:
            P(detect=True | X present/absent) for this agent.
        detect_present_Y / detect_absent_Y:
            P(detect=True | Y present/absent) for this agent.
    """
    agent_id: str
    max_time: float
    can_sense_X: bool
    can_sense_Y: bool

    # These defaults are just placeholders; you will override with real values
    detect_present_X: float = 0.8
    detect_absent_X: float = 0.2
    detect_present_Y: float = 0.8
    detect_absent_Y: float = 0.2


@dataclass
class BoxInfo:
    """
    Box information relevant for planning.

    Fields:
        box_id: integer ID (matches box_id from the server).
        deadline: absolute sim-time deadline (seconds) for any disposal.

        sense_time_X/Y: base sensing durations for each property.
        dispose_time_X/Y: base disposal durations for each property.

        p_true_X/Y: belief that this box truly has property X/Y (0..1).

        disposed_X/Y: whether this box has already been successfully disposed
                      for that property (True → we ignore further disposals).

        info_X/Y: scalar representing how much information we have about each
                  property. Typical scale: 0..1 (0 = no info, 1 = fully known).

        already_sensed:
            mapping: already_sensed[agent_id][prop] -> bool
            True if that agent has already completed a sense(X/Y) for this box.
            If True, planner will not create another sense variable for that
            (agent, box, property).
    """
    box_id: int
    deadline: float

    sense_time_X: float
    sense_time_Y: float
    dispose_time_X: float
    dispose_time_Y: float

    p_true_X: float
    p_true_Y: float

    disposed_X: bool
    disposed_Y: bool

    info_X: float
    info_Y: float

    already_sensed: Dict[str, Dict[Property, bool]]


@dataclass
class PlanConstraintMetrics:
    """
    Constraint-level diagnostics for a plan, independent of the scalar reward.

    - total_actions: total sense + dispose actions in the plan
    - num_sense_actions / num_dispose_actions: breakdown
    - num_dispose_deadline_violations:
        disposals that, even if executed immediately, would finish
        after the box deadline or outside the planning horizon.
    - num_dispose_info_violations:
        disposals scheduled when we don't yet have enough info
        (info_X/Y < info_threshold_for_dispose).
    - num_unknown_boxes:
        actions that refer to box_ids we don't have BoxInfo for.
    """
    total_actions: int
    num_sense_actions: int
    num_dispose_actions: int
    num_dispose_deadline_violations: int
    num_dispose_info_violations: int
    num_unknown_boxes: int


@dataclass
class PlanParseIssue:
    """
    Diagnostics for a single step in the LLM 'agents_plan'.

    Fields:
      agent_id: which agent this step was for ("robot", "human_a", ...)
      step_index: index in that agent's list (0-based)
      raw_step: the original dict from agents_plan[agent_id][step_index]
      problem: human-readable string explaining what's missing/invalid
    """
    agent_id: str
    step_index: int
    raw_step: Any
    problem: str


# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------

def plan_assignments_gurobi(
    agents: List[AgentState],
    boxes: List[BoxInfo],
    current_time: float,
    horizon: float,
    travel_time_fn: TravelTimeFn,
    weights: Optional[PlannerWeights] = None,
) -> Plan:
    """
    Build and solve a Gurobi MILP that assigns sensing and disposal tasks.

    Args:
        agents: list of AgentState (human_a, human_b, robot, ...).
        boxes: list of BoxInfo built from the current world state.
        current_time: current sim time (seconds).
        horizon: maximum time window (seconds) we plan over. Tasks must fit
                 within this window for each agent.
        travel_time_fn: function (agent_id, box_id) -> travel time (seconds).
        weights: PlannerWeights encoding both objective coefficients and
                 style/preferences (exploration, fairness, etc.).

    Returns:
        dict mapping each agent_id to a *sequential* list of actions:

            {
              "human_a": [(box_id, property, "sense"|"dispose"), ...],
              "human_b": [...],
              "robot":   [...],
            }

        Each list is ordered by (earliest box deadline, disposing before sensing
        when tied), so you can execute them in sequence per agent.
    """
    if weights is None:
        weights = PlannerWeights()

    info_threshold_for_dispose = weights.info_threshold_for_dispose
    reward_correct_X = weights.reward_correct_X
    reward_correct_Y = weights.reward_correct_Y
    weight_info = weights.weight_info
    lambda_balance = weights.lambda_balance

    model = gp.Model("box_planner")
    model.Params.OutputFlag = 0  # silent

    agents_by_id = {a.agent_id: a for a in agents}
    props: List[Property] = ["X", "Y"]

    # -----------------------------------------------------------------------
    # Role decision variables: each agent chooses sensing or disposal (or idle)
    # -----------------------------------------------------------------------
    z_sense: Dict[str, gp.Var] = {}
    z_dispose: Dict[str, gp.Var] = {}
    for a in agents:
        z_sense[a.agent_id] = model.addVar(vtype=GRB.BINARY, name=f"z_sense_{a.agent_id}")
        z_dispose[a.agent_id] = model.addVar(vtype=GRB.BINARY, name=f"z_disp_{a.agent_id}")
        model.addConstr(
            z_sense[a.agent_id] + z_dispose[a.agent_id] <= 1,
            name=f"role_choice_{a.agent_id}",
        )

    # -----------------------------------------------------------------------
    # Decision variables: sense and dispose
    # -----------------------------------------------------------------------
    s_vars: Dict[Tuple[str, int, Property], gp.Var] = {}
    d_vars: Dict[Tuple[str, int, Property], gp.Var] = {}

    for a in agents:
        for b in boxes:
            for p in props:

                # Skip properties already successfully disposed
                if p == "X" and b.disposed_X:
                    continue
                if p == "Y" and b.disposed_Y:
                    continue

                # ---------- SENSING VARIABLES ----------
                if p == "X" and not a.can_sense_X:
                    can_sense = False
                elif p == "Y" and not a.can_sense_Y:
                    can_sense = False
                else:
                    can_sense = True

                already = b.already_sensed.get(a.agent_id, {}).get(p, False)

                if can_sense and not already:
                    base_sense_time = b.sense_time_X if p == "X" else b.sense_time_Y
                    travel = travel_time_fn(a.agent_id, b.box_id)
                    total_sense_time = base_sense_time + travel

                    # NEW: respect horizon and deadline for sensing too
                    if (
                        total_sense_time <= horizon
                        and (b.deadline is None or current_time + total_sense_time <= b.deadline)
                    ):
                        s_vars[(a.agent_id, b.box_id, p)] = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"sense_{a.agent_id}_{b.box_id}_{p}",
                        )

                # ---------- DISPOSAL VARIABLES ----------
                if p == "X":
                    info_level = b.info_X
                    base_disp_time = b.dispose_time_X
                else:
                    info_level = b.info_Y
                    base_disp_time = b.dispose_time_Y

                if info_level < info_threshold_for_dispose:
                    continue

                travel = travel_time_fn(a.agent_id, b.box_id)
                total_disp_time = base_disp_time + travel

                # Must fit within horizon AND deadline
                if (
                    total_disp_time <= horizon
                    and current_time + total_disp_time <= b.deadline
                ):
                    d_vars[(a.agent_id, b.box_id, p)] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"disp_{a.agent_id}_{b.box_id}_{p}",
                    )

    # -----------------------------------------------------------------------
    # Constraints
    # -----------------------------------------------------------------------

    # (1) At most one disposal for each (box, property)
    for b in boxes:
        for p in props:
            vars_bp = [
                v
                for (aid, bid, pp), v in d_vars.items()
                if bid == b.box_id and pp == p
            ]
            if vars_bp:
                model.addConstr(
                    gp.quicksum(vars_bp) <= 1,
                    name=f"one_disp_box{b.box_id}_{p}",
                )

    # (2) Agent time budget (travel + base action times)
    agent_load_expr: Dict[str, gp.LinExpr] = {}
    for a in agents:
        expr = gp.LinExpr()
        for b in boxes:
            for p in props:
                s_var = s_vars.get((a.agent_id, b.box_id, p))
                d_var = d_vars.get((a.agent_id, b.box_id, p))

                if s_var is not None:
                    base_sense_time = b.sense_time_X if p == "X" else b.sense_time_Y
                    travel = travel_time_fn(a.agent_id, b.box_id)
                    total_sense_time = base_sense_time + travel
                    expr += total_sense_time * s_var

                if d_var is not None:
                    base_disp_time = b.dispose_time_X if p == "X" else b.dispose_time_Y
                    travel = travel_time_fn(a.agent_id, b.box_id)
                    total_disp_time = base_disp_time + travel
                    expr += total_disp_time * d_var

        model.addConstr(
            expr <= a.max_time,
            name=f"time_budget_{a.agent_id}",
        )
        agent_load_expr[a.agent_id] = expr

    # (3) Role coupling: if agent is sensing, it cannot dispose (and vice versa)
    BIG_M = 1000.0

    for a in agents:
        sense_vars_a = [v for (aid, _bid, _p), v in s_vars.items() if aid == a.agent_id]
        if sense_vars_a:
            model.addConstr(
                gp.quicksum(sense_vars_a) <= BIG_M * z_sense[a.agent_id],
                name=f"sense_role_{a.agent_id}",
            )

        disp_vars_a = [v for (aid, _bid, _p), v in d_vars.items() if aid == a.agent_id]
        if disp_vars_a:
            model.addConstr(
                gp.quicksum(disp_vars_a) <= BIG_M * z_dispose[a.agent_id],
                name=f"disp_role_{a.agent_id}",
            )

    # -----------------------------------------------------------------------
    # Objective: expected correct disposals + info gain + style terms
    # -----------------------------------------------------------------------

    total_reward = gp.LinExpr()
    totalX = gp.LinExpr()
    totalY = gp.LinExpr()

    # Disposal reward (expected correct) + deadline risk
    for (aid, bid, p), d_var in d_vars.items():
        b = next(bb for bb in boxes if bb.box_id == bid)

        if p == "X":
            p_true = b.p_true_X
            val = reward_correct_X
            base_disp_time = b.dispose_time_X
        else:
            p_true = b.p_true_Y
            val = reward_correct_Y
            base_disp_time = b.dispose_time_Y

        total_reward += val * p_true * d_var

        if p == "X":
            totalX += p_true * d_var
        else:
            totalY += p_true * d_var

        # deadline risk penalty (approximate)
        if weights.lambda_deadline_risk > 0.0 and b.deadline is not None:
            travel = travel_time_fn(aid, bid)
            finish_time = current_time + base_disp_time + travel
            slack = float(b.deadline) - float(finish_time)
            # Penalize negative slack (expected lateness)
            risk_coeff = max(0.0, -slack)
            if risk_coeff > 0.0:
                total_reward -= weights.lambda_deadline_risk * risk_coeff * d_var

    # Sensing reward (information gain, only for new senses)
    for (aid, bid, p), s_var in s_vars.items():
        b = next(bb for bb in boxes if bb.box_id == bid)
        a = agents_by_id[aid]

        if p == "X":
            p_true = b.p_true_X
            info_level = b.info_X
            agent_quality = max(a.detect_present_X - a.detect_absent_X, 0.0)
        else:
            p_true = b.p_true_Y
            info_level = b.info_Y
            agent_quality = max(a.detect_present_Y - a.detect_absent_Y, 0.0)

        entropy_like = 4.0 * p_true * (1.0 - p_true)  # max at p_true=0.5
        base_info_gain = (1.0 - info_level) * entropy_like
        info_gain = agent_quality * base_info_gain

        total_reward += weight_info * info_gain * s_var

        if weights.prefer_exploration != 0.0:
            total_reward += weights.prefer_exploration * s_var

    # X/Y balance penalty: penalize |totalX - totalY|
    d_imb = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="d_imbalance")
    model.addConstr(totalX - totalY <= d_imb, name="balance_pos")
    model.addConstr(totalY - totalX <= d_imb, name="balance_neg")
    total_reward -= lambda_balance * d_imb

    # Load fairness between humans
    human_ids = [a.agent_id for a in agents if a.agent_id.startswith("human_")]
    if weights.lambda_load_fairness > 0.0 and len(human_ids) >= 2:
        avg_human_load = (1.0 / len(human_ids)) * gp.quicksum(
            agent_load_expr[aid] for aid in human_ids
        )
        for aid in human_ids:
            diff = agent_load_expr[aid] - avg_human_load
            d_pos = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"fair_pos_{aid}")
            d_neg = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"fair_neg_{aid}")
            model.addConstr(diff <= d_pos)
            model.addConstr(-diff <= d_neg)
            total_reward -= weights.lambda_load_fairness * (d_pos + d_neg)

    # Robot vs human load preferences
    robot_id = "robot"
    if robot_id in agent_load_expr and human_ids:
        robot_load = agent_load_expr[robot_id]
        total_human_load = gp.quicksum(agent_load_expr[aid] for aid in human_ids)

        if weights.lambda_robot_overuse > 0.0:
            avg_human = total_human_load / len(human_ids)
            diff_robot = robot_load - avg_human
            d_robot_over = model.addVar(
                lb=0.0, vtype=GRB.CONTINUOUS, name="robot_overuse"
            )
            model.addConstr(diff_robot <= d_robot_over)
            total_reward -= weights.lambda_robot_overuse * d_robot_over

        if weights.lambda_human_overuse > 0.0:
            diff_humans = total_human_load - robot_load
            d_hum_over = model.addVar(
                lb=0.0, vtype=GRB.CONTINUOUS, name="human_overuse"
            )
            model.addConstr(diff_humans <= d_hum_over)
            total_reward -= weights.lambda_human_overuse * d_hum_over

    model.setObjective(total_reward, GRB.MAXIMIZE)

    # -----------------------------------------------------------------------
    # Solve and extract solution
    # -----------------------------------------------------------------------
    model.optimize()

    actions_by_agent: Plan = {}

    if model.Status == GRB.OPTIMAL:
        # collect chosen senses/disposals
        for (aid, bid, p), v in s_vars.items():
            if v.X > 0.5:
                actions_by_agent.setdefault(aid, []).append((bid, p, "sense"))

        for (aid, bid, p), v in d_vars.items():
            if v.X > 0.5:
                actions_by_agent.setdefault(aid, []).append((bid, p, "dispose"))

        # impose a simple per-agent sequence
        box_by_id = {b.box_id: b for b in boxes}

        for aid, actions in actions_by_agent.items():
            def sort_key(act: Tuple[int, Property, str]):
                box_id, prop, kind = act
                b = box_by_id.get(box_id)
                deadline = b.deadline if b is not None else 1e12
                kind_rank = 0 if kind == "dispose" else 1  # dispose before sense
                return (deadline, kind_rank, box_id, 0 if prop == "X" else 1)

            actions.sort(key=sort_key)

    return actions_by_agent


# ---------------------------------------------------------------------------
# Plan conversion helpers
# ---------------------------------------------------------------------------

def build_plan_from_llm_agents_plan(
    agents_plan: dict,
    allowed_agents: Optional[List[str]] = None,
    collect_issues: bool = False,
) -> Plan | Tuple[Plan, List[PlanParseIssue]]:
    """
    Convert an LLM 'agents_plan' JSON object into an optimizer-style plan:

        {
          "robot":   [{ "box_id": 1, "property": "X", "kind": "sense" }, ...],
          "human_a": [...],
          "human_b": [...],
        }

    - agents_plan is expected to match the schema in the speech rule.
    - If allowed_agents is provided, only those agent_ids are included
      (e.g., ["robot"] for robot-only overrides).
    - Steps without a valid box_id/property/kind are ignored, but if
      collect_issues=True we also return a list of PlanParseIssue to
      explain what was missing/invalid for each dropped step.

    Returns:
      If collect_issues is False (default):
        Plan
      If collect_issues is True:
        (Plan, List[PlanParseIssue])
    """
    if not isinstance(agents_plan, dict):
        return ({}, [] if collect_issues else {})

    if allowed_agents is None:
        allowed_agents = ["robot", "human_a", "human_b"]

    plan: Plan = {}
    issues: List[PlanParseIssue] = []

    for aid in allowed_agents:
        steps = agents_plan.get(aid) or []
        if not isinstance(steps, list):
            if collect_issues:
                issues.append(
                    PlanParseIssue(
                        agent_id=aid,
                        step_index=-1,
                        raw_step=steps,
                        problem="steps list is not a valid list",
                    )
                )
            continue

        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                if collect_issues:
                    issues.append(
                        PlanParseIssue(
                            agent_id=aid,
                            step_index=idx,
                            raw_step=step,
                            problem="step is not a JSON object",
                        )
                    )
                continue

            box_id = step.get("box_id")
            prop = step.get("property")
            kind = step.get("kind")

            problems: List[str] = []

            if box_id is None:
                problems.append("missing box_id")
            if prop not in ("X", "Y"):
                problems.append("property must be 'X' or 'Y'")
            if kind not in ("sense", "dispose"):
                problems.append("kind must be 'sense' or 'dispose'")

            # If any problems, record issue and skip
            if problems:
                if collect_issues:
                    issues.append(
                        PlanParseIssue(
                            agent_id=aid,
                            step_index=idx,
                            raw_step=step,
                            problem=", ".join(problems),
                        )
                    )
                continue

            # Validate box_id is an int
            try:
                box_id_int = int(box_id)
            except Exception:
                if collect_issues:
                    issues.append(
                        PlanParseIssue(
                            agent_id=aid,
                            step_index=idx,
                            raw_step=step,
                            problem=f"box_id '{box_id}' is not an integer",
                        )
                    )
                continue

            plan.setdefault(aid, []).append((box_id_int, prop, kind))

    if collect_issues:
        return plan, issues
    return plan


def summarize_plan_parse_issues(issues: List[PlanParseIssue]) -> str:
    """
    Turn a list of PlanParseIssue into a short natural-language summary
    the robot can say to the humans.
    """
    if not issues:
        return ""

    by_agent: Dict[str, List[PlanParseIssue]] = {}
    for iss in issues:
        by_agent.setdefault(iss.agent_id, []).append(iss)

    parts: List[str] = []
    for agent_id, agent_issues in by_agent.items():
        if agent_id == "robot":
            who = "my actions"
        elif agent_id.startswith("human_"):
            who = f"{agent_id.replace('_', ' ')}'s actions"
        else:
            who = f"{agent_id}'s actions"

        problems = []
        for iss in agent_issues:
            problems.append(f"step {iss.step_index + 1}: {iss.problem}")
        problems_text = "; ".join(problems)
        parts.append(f"For {who}, I couldn't use some steps ({problems_text})")

    return "I couldn't fully understand your plan. " + " ".join(parts)


# ---------------------------------------------------------------------------
# Objective scoring & constraint evaluation for arbitrary plans
# ---------------------------------------------------------------------------

def score_plan_objective(
    plan: Plan,
    agents: List[AgentState],
    boxes: List[BoxInfo],
    weights: PlannerWeights | None = None,
) -> float:
    """
    Compute a scalar score for a given plan using a simplified version of the
    MILP objective:

        total_reward
          = expected correct disposals (X/Y)
          + weight_info * info_gain_from_sensing
          + prefer_exploration * (# senses)
          - lambda_balance * |totalX - totalY|

    This does NOT re-solve the MILP; it just evaluates the plan.
    Time/horizon constraints are assumed to have been respected upstream.
    Style terms that depend on detailed timing (fairness, load, deadline risk)
    are not included here.
    """
    if weights is None:
        weights = PlannerWeights()

    reward_correct_X = weights.reward_correct_X
    reward_correct_Y = weights.reward_correct_Y
    weight_info = weights.weight_info
    lambda_balance = weights.lambda_balance
    prefer_exploration = weights.prefer_exploration

    agents_by_id = {a.agent_id: a for a in agents}
    box_by_id = {b.box_id: b for b in boxes}

    total_reward = 0.0
    totalX = 0.0
    totalY = 0.0

    for aid, actions in (plan or {}).items():
        a = agents_by_id.get(aid)
        if a is None:
            continue

        for (box_id, prop, kind) in actions:
            b = box_by_id.get(box_id)
            if b is None:
                continue

            if kind == "dispose":
                if prop == "X":
                    p_true = b.p_true_X
                    totalX += p_true
                    total_reward += reward_correct_X * p_true
                else:
                    p_true = b.p_true_Y
                    totalY += p_true
                    total_reward += reward_correct_Y * p_true

            elif kind == "sense":
                if prop == "X":
                    p_true = b.p_true_X
                    info_level = b.info_X
                    agent_quality = max(a.detect_present_X - a.detect_absent_X, 0.0)
                else:
                    p_true = b.p_true_Y
                    info_level = b.info_Y
                    agent_quality = max(a.detect_present_Y - a.detect_absent_Y, 0.0)

                entropy_like = 4.0 * p_true * (1.0 - p_true)
                base_info_gain = (1.0 - info_level) * entropy_like
                info_gain = agent_quality * base_info_gain

                total_reward += weight_info * info_gain
                if prefer_exploration != 0.0:
                    total_reward += prefer_exploration

    imbalance = abs(totalX - totalY)
    return total_reward - lambda_balance * imbalance


def compute_planning_accuracy_by_human(
    outcomes: List[DisposalOutcome],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate planning quality per human planner, based on disposal outcomes.

    We group by `planner_id` (who proposed the plan), not `agent_id` (who executed it).

    Returns:
      {
        "<planner_id>": {
          "num_planned_disposals": float,
          "num_correct": float,
          "num_incorrect": float,
          "correct_rate": float,         # correct / planned (where correct is not None)
          "success_rate": float,         # success / planned
        },
        ...
      }
    """
    # Filter only outcomes with a planner_id
    relevant = [e for e in outcomes if e.planner_id]

    by_planner: Dict[str, List[DisposalOutcome]] = {}
    for e in relevant:
        pid = e.planner_id or "unknown"
        by_planner.setdefault(pid, []).append(e)

    stats: Dict[str, Dict[str, float]] = {}

    for pid, evts in by_planner.items():
        n = float(len(evts))
        num_success = sum(1 for e in evts if e.success)
        labeled = [e for e in evts if e.correct is not None]
        num_correct = sum(1 for e in labeled if bool(e.correct))
        num_incorrect = sum(1 for e in labeled if not bool(e.correct))

        if labeled:
            correct_rate = num_correct / float(len(labeled))
        else:
            correct_rate = 0.0

        success_rate = num_success / n if n > 0.0 else 0.0

        stats[pid] = {
            "num_planned_disposals": n,
            "num_correct": float(num_correct),
            "num_incorrect": float(num_incorrect),
            "correct_rate": correct_rate,
            "success_rate": success_rate,
        }

    return stats


def compute_disposal_metrics(
    outcomes: List[DisposalOutcome],
    boxes: List[BoxInfo],
) -> Dict[str, Dict[str, float]]:
    """
    Compute disposal metrics from a list of realized disposal outcomes and
    the current box world (for deadlines).

    Returns a dict with:
        {
          "global": {...},
          "by_agent": {
            "<agent_id>": {...},
            ...
          }
        }

    Notes:
      - success_rate uses `success` field.
      - correct_rate uses only events with correct is not None.
      - deadline_miss_rate and avg_slack use BoxInfo.deadline and completed_at.
      - balance_ratio = min(num_X, num_Y) / max(num_X, num_Y) or 1.0 if no disposals.
    """
    box_by_id = {b.box_id: b for b in boxes}

    def agg_for(events: List[DisposalOutcome]) -> Dict[str, float]:
        n = len(events)
        if n == 0:
            return {
                "num_disposals": 0.0,
                "success_rate": 0.0,
                "correct_rate": 0.0,
                "deadline_miss_rate": 0.0,
                "avg_slack": 0.0,
                "num_X": 0.0,
                "num_Y": 0.0,
                "balance_ratio": 1.0,
            }

        num_success = sum(1 for e in events if e.success)
        labeled = [e for e in events if e.correct is not None]
        num_correct = sum(1 for e in labeled if bool(e.correct))

        slacks: List[float] = []
        num_missed = 0
        for e in events:
            if e.completed_at is None:
                continue
            b = box_by_id.get(e.box_id)
            if b is None or b.deadline is None:
                continue
            slack = float(b.deadline) - float(e.completed_at)
            slacks.append(slack)
            if slack < 0.0:
                num_missed += 1

        avg_slack = sum(slacks) / len(slacks) if slacks else 0.0
        deadline_miss_rate = (num_missed / len(slacks)) if slacks else 0.0

        num_X = sum(1 for e in events if e.prop == "X")
        num_Y = sum(1 for e in events if e.prop == "Y")
        if num_X == 0 and num_Y == 0:
            balance_ratio = 1.0
        else:
            balance_ratio = min(num_X, num_Y) / max(num_X, num_Y)

        return {
            "num_disposals": float(n),
            "success_rate": num_success / float(n),
            "correct_rate": (num_correct / float(len(labeled))) if labeled else 0.0,
            "deadline_miss_rate": deadline_miss_rate,
            "avg_slack": avg_slack,
            "num_X": float(num_X),
            "num_Y": float(num_Y),
            "balance_ratio": balance_ratio,
        }

    global_metrics = agg_for(outcomes)

    by_agent: Dict[str, Dict[str, float]] = {}
    agents_seen = {e.agent_id for e in outcomes}

    for aid in agents_seen:
        events_a = [e for e in outcomes if e.agent_id == aid]
        by_agent[aid] = agg_for(events_a)

    return {
        "global": global_metrics,
        "by_agent": by_agent,
    }


def evaluate_plan_constraints(
    plan: Plan,
    boxes: List[BoxInfo],
    current_time: float,
    travel_time_fn: TravelTimeFn,
    horizon: float,
    weights: PlannerWeights,
) -> PlanConstraintMetrics:
    box_by_id = {b.box_id: b for b in boxes}

    total_actions = 0
    num_sense = 0
    num_disp = 0
    num_deadline_viol = 0
    num_info_viol = 0
    num_unknown = 0

    for aid, actions in (plan or {}).items():
        for (box_id, prop, kind) in actions:
            total_actions += 1
            b = box_by_id.get(box_id)
            if b is None:
                num_unknown += 1
                continue

            # --- SENSE ---
            if kind == "sense":
                num_sense += 1

                # Use the SAME feasibility notion as plan_assignments_gurobi
                if prop == "X":
                    base_time = b.sense_time_X
                else:
                    base_time = b.sense_time_Y

                travel = travel_time_fn(aid, box_id)
                total_t = base_time + travel
                finish_time = current_time + total_t

                # Count as a deadline/horizon violation if it wouldn't fit
                if total_t > horizon or (
                    b.deadline is not None and finish_time > b.deadline
                ):
                    num_deadline_viol += 1

                continue

            # --- DISPOSE ---
            num_disp += 1

            info = b.info_X if prop == "X" else b.info_Y
            if info < weights.info_threshold_for_dispose:
                num_info_viol += 1

            base_time = b.dispose_time_X if prop == "X" else b.dispose_time_Y
            travel = travel_time_fn(aid, box_id)
            total_t = base_time + travel
            finish_time = current_time + total_t

            if total_t > horizon or (
                b.deadline is not None and finish_time > b.deadline
            ):
                num_deadline_viol += 1

    return PlanConstraintMetrics(
        total_actions=total_actions,
        num_sense_actions=num_sense,
        num_dispose_actions=num_disp,
        num_dispose_deadline_violations=num_deadline_viol,
        num_dispose_info_violations=num_info_viol,
        num_unknown_boxes=num_unknown,
    )


def evaluate_candidate_plan(
    current_plan: Plan,
    candidate_plan: Plan,
    agents: List[AgentState],
    boxes: List[BoxInfo],
    current_time: float,
    travel_time_fn: TravelTimeFn,
    horizon: float,
    weights: PlannerWeights,
    margin: float = 0.01,
) -> Dict[str, Any]:
    """
    Unified evaluation helper for broker / LLM overrides.

    Uses:
      - scalar objective (score_plan_objective) with shared PlannerWeights
      - constraint diagnostics (evaluate_plan_constraints)
    and returns:

      {
        "adopt": bool,                # should we adopt candidate?
        "score_current": float,       # scalar objective for current plan
        "score_candidate": float,     # scalar objective for candidate
        "suboptimal_pct": float,      # candidate's suboptimality vs current
        "constraints_current": PlanConstraintMetrics,
        "constraints_candidate": PlanConstraintMetrics,
      }

    Adoption rule:
      1) Candidate must NOT be strictly worse on:
         - deadline violations
         - info violations
         - unknown boxes
      2) Candidate's scalar score must beat current by at least `margin`.
    """
    # Scalar scores
    score_curr = score_plan_objective(current_plan, agents, boxes, weights=weights)
    score_cand = score_plan_objective(candidate_plan, agents, boxes, weights=weights)

    # Constraint diagnostics
    constraints_curr = evaluate_plan_constraints(
        plan=current_plan,
        boxes=boxes,
        current_time=current_time,
        travel_time_fn=travel_time_fn,
        horizon=horizon,
        weights=weights,
    )
    constraints_cand = evaluate_plan_constraints(
        plan=candidate_plan,
        boxes=boxes,
        current_time=current_time,
        travel_time_fn=travel_time_fn,
        horizon=horizon,
        weights=weights,
    )

    candidate_worse_on_constraints = (
        constraints_cand.num_dispose_deadline_violations
        > constraints_curr.num_dispose_deadline_violations
        or constraints_cand.num_dispose_info_violations
        > constraints_curr.num_dispose_info_violations
        or constraints_cand.num_unknown_boxes > constraints_curr.num_unknown_boxes
    )

    if candidate_worse_on_constraints:
        adopt = False
    else:
        adopt = (score_cand > score_curr + margin)

    if candidate_worse_on_constraints:
        suboptimal_pct = 100.0
    else:
        if score_curr <= 0.0:
            suboptimal_pct = 0.0 if score_cand >= score_curr else 100.0
        else:
            gap = max(0.0, score_curr - score_cand)
            suboptimal_pct = 100.0 * gap / abs(score_curr)

    return {
        "adopt": adopt,
        "score_current": score_curr,
        "score_candidate": score_cand,
        "suboptimal_pct": suboptimal_pct,
        "constraints_current": constraints_curr,
        "constraints_candidate": constraints_cand,
    }


def compare_plans_with_constraints(
    current_plan: Plan,
    candidate_plan: Plan,
    agents: List[AgentState],
    boxes: List[BoxInfo],
    current_time: float,
    travel_time_fn: TravelTimeFn,
    horizon: float,
    weights: PlannerWeights,
    margin: float = 0.01,
) -> Tuple[bool, float, float, PlanConstraintMetrics, PlanConstraintMetrics]:
    """
    Joint comparison of two plans, returning a compact tuple:

        (adopt, score_curr, score_cand, constraints_curr, constraints_cand)

    Uses the same logic as evaluate_candidate_plan.
    """
    result = evaluate_candidate_plan(
        current_plan=current_plan,
        candidate_plan=candidate_plan,
        agents=agents,
        boxes=boxes,
        current_time=current_time,
        travel_time_fn=travel_time_fn,
        horizon=horizon,
        weights=weights,
        margin=margin,
    )
    return (
        result["adopt"],
        result["score_current"],
        result["score_candidate"],
        result["constraints_current"],
        result["constraints_candidate"],
    )


def extend_plan_with_prefix(
    prefix_plan: Plan,
    agents: List[AgentState],
    boxes: List[BoxInfo],
    current_time: float,
    horizon: float,
    travel_time_fn: TravelTimeFn,
    weights: Optional[PlannerWeights] = None,
) -> Plan:
    """
    Take a small, human/LLM-proposed prefix_plan and ask the optimizer
    to complete it.

    Semantics:
      - prefix_plan actions are treated as *already decided*.
      - We:
          * subtract their time cost from each agent's time budget,
          * update box world as if those actions had just completed
            (disposed flags, already_sensed).
      - Then we call plan_assignments_gurobi() on the residual problem,
        and finally return:

            full_plan[agent] = prefix_actions[agent] + suffix_actions[agent]

    Notes:
      - We do NOT try to be perfect about deadlines vs. prefix time; this
        function uses the same coarse horizon/deadline model as the base
        optimizer (per-action checks).
      - Info gains from prefix senses are approximated by bumping info_X/Y.
    """
    if weights is None:
        weights = PlannerWeights()

    # Fast exits
    if not prefix_plan:
        # No prefix: just run the plain optimizer
        return plan_assignments_gurobi(
            agents=agents,
            boxes=boxes,
            current_time=current_time,
            horizon=horizon,
            travel_time_fn=travel_time_fn,
            weights=weights,
        )

    # --- 1) Build quick lookup for boxes (original) ---
    box_by_id: Dict[int, BoxInfo] = {b.box_id: b for b in boxes}

    # --- 2) Compute time already "consumed" per agent by prefix ---
    used_time: Dict[str, float] = {a.agent_id: 0.0 for a in agents}

    for aid, actions in (prefix_plan or {}).items():
        for (box_id, prop, kind) in actions:
            b = box_by_id.get(box_id)
            if b is None:
                continue
            travel = travel_time_fn(aid, box_id)
            if kind == "sense":
                base = b.sense_time_X if prop == "X" else b.sense_time_Y
            elif kind == "dispose":
                base = b.dispose_time_X if prop == "X" else b.dispose_time_Y
            else:
                continue
            used_time[aid] = used_time.get(aid, 0.0) + (base + travel)

    # --- 3) Build residual agent states with reduced max_time ---
    residual_agents: List[AgentState] = []
    for a in agents:
        remaining = max(0.0, a.max_time - used_time.get(a.agent_id, 0.0))
        residual_agents.append(
            AgentState(
                agent_id=a.agent_id,
                max_time=remaining,
                can_sense_X=a.can_sense_X,
                can_sense_Y=a.can_sense_Y,
                detect_present_X=a.detect_present_X,
                detect_absent_X=a.detect_absent_X,
                detect_present_Y=a.detect_present_Y,
                detect_absent_Y=a.detect_absent_Y,
            )
        )

    # --- 4) Build residual box world: treat prefix as if just completed ---
    residual_boxes: List[BoxInfo] = copy.deepcopy(boxes)
    box_res_by_id: Dict[int, BoxInfo] = {b.box_id: b for b in residual_boxes}

    for aid, actions in (prefix_plan or {}).items():
        for (box_id, prop, kind) in actions:
            b = box_res_by_id.get(box_id)
            if b is None:
                continue

            if kind == "sense":
                # Mark that this agent has already sensed this (box, prop)
                amap = b.already_sensed.setdefault(aid, {})
                amap[prop] = True

                # Heuristic: bump info_X/Y because we are planning *after* this sense
                if prop == "X":
                    b.info_X = min(1.0, max(b.info_X, 0.7))
                else:
                    b.info_Y = min(1.0, max(b.info_Y, 0.7))

            elif kind == "dispose":
                # Mark disposal as already done so optimizer won't schedule it again
                if prop == "X":
                    b.disposed_X = True
                else:
                    b.disposed_Y = True

    # --- 5) Run optimizer on residual problem ---
    suffix_plan = plan_assignments_gurobi(
        agents=residual_agents,
        boxes=residual_boxes,
        current_time=current_time,
        horizon=horizon,
        travel_time_fn=travel_time_fn,
        weights=weights,
    )

    # --- 6) Merge prefix + suffix into a full plan ---
    full_plan: Plan = {}
    all_agents = set(list(prefix_plan.keys()) + list(suffix_plan.keys()))

    for aid in all_agents:
        prefix_actions = prefix_plan.get(aid, [])
        suffix_actions = suffix_plan.get(aid, [])
        # Keep human/LLM-proposed steps first
        full_plan[aid] = list(prefix_actions) + list(suffix_actions)

    return full_plan


def estimate_deadline_risk_for_plan(
    plan: Plan,
    boxes: List[BoxInfo],
    current_time: float,
    travel_time_fn: TravelTimeFn,
    horizon: float,
) -> str:
    """
    Heuristic deadline risk label for a *plan* ("low" | "medium" | "high").

    Uses:
      - slack = deadline - (current_time + travel_time + dispose_time)
      - horizon violations (total_time > horizon)
    """
    box_by_id = {b.box_id: b for b in boxes}

    slacks: List[float] = []
    num_viol = 0
    num_disp = 0

    for aid, actions in (plan or {}).items():
        for (box_id, prop, kind) in actions:
            if kind != "dispose":
                continue

            b = box_by_id.get(box_id)
            if b is None or b.deadline is None:
                continue

            if prop == "X":
                base_disp_time = b.dispose_time_X
            else:
                base_disp_time = b.dispose_time_Y

            travel = travel_time_fn(aid, box_id)
            total_t = base_disp_time + travel
            finish_time = current_time + total_t

            num_disp += 1
            slack = float(b.deadline) - float(finish_time)
            slacks.append(slack)

            if total_t > horizon or slack < 0.0:
                num_viol += 1

    # No disposals → essentially no deadline pressure
    if num_disp == 0 or not slacks:
        return "low"

    min_slack = min(slacks)
    tight_5 = sum(1 for s in slacks if s < 5.0)
    tight_20 = sum(1 for s in slacks if s < 20.0)

    frac_tight_5 = tight_5 / float(len(slacks))
    frac_tight_20 = tight_20 / float(len(slacks))

    # High risk: any violations or many very-tight disposals
    if num_viol > 0:
        return "high"
    if min_slack < 0.0:
        return "high"
    if frac_tight_5 > 0.3:   # >30% within 5s of deadline
        return "high"

    # Medium risk: no violations but a fair number close to deadline
    if min_slack < 10.0:
        return "medium"
    if frac_tight_20 > 0.3:  # >30% within 20s of deadline
        return "medium"

    # Otherwise, comfortably low risk
    return "low"

def compute_xy_imbalance_for_plan(
    plan: Plan,
    boxes: List[BoxInfo],
) -> Dict[str, float]:
    """
    Compute X/Y balance metrics for a *plan*.

    We use expected disposals (probability-weighted), consistent with the
    MILP objective:

        totalX = sum p_true_X for X-disposals in plan
        totalY = sum p_true_Y for Y-disposals in plan

    Returns:
      {
        "totalX": float,
        "totalY": float,
        "imbalance": float,      # |totalX - totalY|
        "balance_ratio": float,  # min(totalX, totalY) / max(...), or 1.0 if no disposals
      }
    """
    box_by_id = {b.box_id: b for b in boxes}

    totalX = 0.0
    totalY = 0.0

    for aid, actions in (plan or {}).items():
        for (box_id, prop, kind) in actions:
            if kind != "dispose":
                continue
            b = box_by_id.get(box_id)
            if b is None:
                continue

            if prop == "X":
                totalX += float(b.p_true_X)
            else:
                totalY += float(b.p_true_Y)

    imbalance = abs(totalX - totalY)

    if totalX == 0.0 and totalY == 0.0:
        balance_ratio = 1.0
    else:
        balance_ratio = min(totalX, totalY) / max(totalX, totalY)

    return {
        "totalX": totalX,
        "totalY": totalY,
        "imbalance": imbalance,
        "balance_ratio": balance_ratio,
    }


# ---------------------------------------------------------------------------
# Example of how you might call this (for reference / testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def dummy_travel_time(agent_id: str, box_id: int) -> float:
        return 1.0

    agents = [
        AgentState(agent_id="human_a", max_time=30.0, can_sense_X=True,  can_sense_Y=False),
        AgentState(agent_id="human_b", max_time=30.0, can_sense_X=False, can_sense_Y=True),
        AgentState(agent_id="robot",   max_time=30.0, can_sense_X=True,  can_sense_Y=True),
    ]

    boxes = [
        BoxInfo(
            box_id=1,
            deadline=200.0,
            sense_time_X=3.0,
            sense_time_Y=3.0,
            dispose_time_X=4.0,
            dispose_time_Y=4.0,
            p_true_X=0.7,
            p_true_Y=0.5,
            disposed_X=False,
            disposed_Y=False,
            info_X=0.3,
            info_Y=0.2,
            already_sensed={
                "human_a": {"X": False, "Y": False},
                "human_b": {"X": False, "Y": False},
                "robot":   {"X": False, "Y": False},
            },
        )
    ]

    weights = PlannerWeights(
        reward_correct_X=1.0,
        reward_correct_Y=1.0,
        weight_info=0.2,
        lambda_balance=0.5,
        info_threshold_for_dispose=0.6,
        prefer_exploration=0.0,
        lambda_load_fairness=0.0,
        lambda_robot_overuse=0.0,
        lambda_human_overuse=0.0,
        lambda_deadline_risk=0.0,
    )

    plan = plan_assignments_gurobi(
        agents=agents,
        boxes=boxes,
        current_time=0.0,
        horizon=30.0,
        travel_time_fn=dummy_travel_time,
        weights=weights,
    )

    for aid, actions in plan.items():
        print(f"{aid}:")
        for box_id, prop, kind in actions:
            print(f"  {kind.upper()} box {box_id} property {prop}")

