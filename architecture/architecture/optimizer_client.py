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

from gurobipy import quicksum
import math

import itertools

Property = Literal["X", "Y"]
TravelTimeFn = Callable[[str, int], float]  # (agent_id, box_id) -> seconds

# Simple alias for plans:
#   { "robot":   [(box_id, "X"/"Y", "sense"/"dispose"), ...],
#     "human_a": [...],
#     "human_b": [...], ... }
Plan = Dict[str, List[Tuple[int, Property, str]]]

DEBUG_MILP = True

def dbg(msg: str):
    if DEBUG_MILP:
        print(f"[MILP-DBG] {msg}", flush=True)




def entropy(p: float) -> float:
    p = clamp(p, 1e-9, 1-1e-9)
    return -(p*math.log(p) + (1-p)*math.log(1-p))

def expected_entropy_after_one(p: float, tpr: float, fpr: float) -> float:
    p = clamp(p); tpr = clamp(tpr, 1e-6, 1-1e-6); fpr = clamp(fpr, 1e-6, 1-1e-6)
    p_det1 = p*tpr + (1-p)*fpr
    p_det0 = 1 - p_det1
    p1 = (p*tpr) / max(1e-12, p*tpr + (1-p)*fpr)
    p0 = (p*(1-tpr)) / max(1e-12, p*(1-tpr) + (1-p)*(1-fpr))
    return p_det1*entropy(p1) + p_det0*entropy(p0)

def expected_info_gain_one(p: float, tpr: float, fpr: float) -> float:
    return entropy(p) - expected_entropy_after_one(p, tpr, fpr)


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
    info_threshold_for_dispose: float = 0.4

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
    lambda_sense_slack: float = 0.005  # small like 0.001

    pmin_for_dispose: float = 0.8
    
    egoistic_goal_property: Optional[str] = None

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

    # NEW: disposal capabilities
    can_dispose_X: bool = True
    can_dispose_Y: bool = True

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
    # NEW: how many agents are needed to carry/dispose (1 for light objects)
    min_disposal_team: int = 1

    # NEW: optional cap (e.g., you might not want >2 agents on a small box)
    max_disposal_team: int = 3

    
    senseable_X: bool = True
    senseable_Y: bool = True

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


def clamp(p: float, lo: float = 1e-6, hi: float = 1.0 - 1e-6) -> float:
    return max(lo, min(hi, float(p)))

def _sr_status(sr: dict) -> str:
    # server records imply completion; accept both styles
    return sr.get("status") or ("completed" if "completed_at" in sr else "")

def _sr_prop(sr: dict) -> str:
    return sr.get("property") or sr.get("prop")

def p_present_from_sense_results_fused(sense_results: List[dict], prop: Property, prior: float = 0.5) -> float:
    prior = clamp(prior)
    L = math.log(prior / (1.0 - prior))

    for sr in sense_results or []:
        if _sr_status(sr) != "completed":
            continue
        if _sr_prop(sr) != prop:
            continue

        detected = sr.get("detected", None)
        prob = sr.get("probability", None)
        if detected is None or not isinstance(prob, (int, float)):
            continue

        q = clamp(float(prob), 1e-4, 1.0 - 1e-4)
        p_meas = q if bool(detected) else (1.0 - q)
        p_meas = clamp(p_meas, 1e-4, 1.0 - 1e-4)

        L += math.log(p_meas / (1.0 - p_meas))

    p = 1.0 / (1.0 + math.exp(-L))
    return clamp(p, 0.0, 1.0)

def p_present_from_sense_results_bayes(
    sense_results: list[dict],
    prop: Property,
    agents_by_id: dict[str, AgentState],
    prior: float = 0.5,
) -> float:
    prior = clamp(prior)
    L = math.log(prior / (1.0 - prior))  # log-odds

    for sr in sense_results or []:
        if _sr_status(sr) != "completed":
            continue
        if _sr_prop(sr) != prop:
            continue

        detected = sr.get("detected", None)
        if detected is None:
            continue

        aid = str(sr.get("agent_id") or "")
        a = agents_by_id.get(aid)
        if a is None:
            # Unknown agent: skip or assume weak default
            continue

        if prop == "X":
            p_det_given_present = clamp(a.detect_present_X, 1e-4, 1 - 1e-4)
            p_det_given_absent  = clamp(a.detect_absent_X,  1e-4, 1 - 1e-4)
        else:
            p_det_given_present = clamp(a.detect_present_Y, 1e-4, 1 - 1e-4)
            p_det_given_absent  = clamp(a.detect_absent_Y,  1e-4, 1 - 1e-4)

        if bool(detected):
            # LR for detected=True
            lr = p_det_given_present / p_det_given_absent
        else:
            # LR for detected=False
            lr = (1.0 - p_det_given_present) / (1.0 - p_det_given_absent)

        L += math.log(lr)

    p = 1.0 / (1.0 + math.exp(-L))
    return clamp(p, 0.0, 1.0)


def info_level_from_p(p_present: float) -> float:
    p = clamp(p_present)
    conf = max(p, 1.0 - p)       # [0.5, 1.0]
    info = (conf - 0.5) / 0.5    # [0, 1]
    return max(0.0, min(1.0, info))


SPEEDUP_FACTOR = {1: 1.0, 2: 0.50, 3: 0.25}
def speed_factor(k: int) -> float:
    return SPEEDUP_FACTOR.get(k, 1.0 / float(k))


import itertools

def best_case_disposal_time_rel(
    *,
    agents: List[AgentState],
    b: BoxInfo,
    prop: Property,
    travel_time_fn: TravelTimeFn,
) -> Optional[float]:
    """
    Best-case (relative) time to complete disposal of (b, prop), assuming:
      - disposal starts after sensing finishes (conservative)
      - disposal team chosen among agents who can dispose prop
      - team rendezvous arrival time = max travel among selected agents
      - execution time = base_dispose_time * speed_factor(k)

    Returns:
      minimal time (seconds) from 'start disposal' to 'finish disposal',
      or None if no feasible disposal team exists.
    """
    # eligible disposers for this prop
    eligible = []
    for a in agents:
        if prop == "X" and not getattr(a, "can_dispose_X", True):
            continue
        if prop == "Y" and not getattr(a, "can_dispose_Y", True):
            continue
        eligible.append(a)

    if not eligible:
        return None

    k_min = max(1, int(getattr(b, "min_disposal_team", 1)))
    k_max = min(int(getattr(b, "max_disposal_team", len(eligible))), len(eligible))
    if k_min > k_max:
        return None

    base = float(b.dispose_time_X if prop == "X" else b.dispose_time_Y)

    best = None
    # brute force combinations; with 3 agents this is tiny and safe
    for k in range(k_min, k_max + 1):
        for team in itertools.combinations(eligible, k):
            max_travel = 0.0
            for a in team:
                max_travel = max(max_travel, float(travel_time_fn(a.agent_id, b.box_id)))
            t = max_travel + base * float(speed_factor(k))
            if best is None or t < best:
                best = t

    return best


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
    if weights is None:
        weights = PlannerWeights()

    for b in boxes:
        dbg(
            f"BOX {b.box_id}: now={current_time:.2f} deadline={b.deadline:.2f} "
            f"infoX={b.info_X:.3f} infoY={b.info_Y:.3f} p_true_X={b.p_true_X:.3f} p_true_Y={b.p_true_Y:.3f} "
            f"disposedX={b.disposed_X} disposedY={b.disposed_Y}"
        )

    info_threshold_for_dispose = weights.info_threshold_for_dispose
    reward_correct_X = weights.reward_correct_X
    reward_correct_Y = weights.reward_correct_Y
    weight_info = weights.weight_info
    lambda_balance = weights.lambda_balance

    model = gp.Model("box_planner")
    model.Params.OutputFlag = 0  # silent

    agents_by_id = {a.agent_id: a for a in agents}
    box_by_id = {b.box_id: b for b in boxes}  # <-- needed early
    props: List[Property] = ["X", "Y"]


    # -----------------------------------------------------------------------
    # Decision variables
    # -----------------------------------------------------------------------
    s_vars: Dict[Tuple[str, int, Property], gp.Var] = {}

    # (box,prop) disposal selected
    y_disp_prop: Dict[Tuple[int, Property], gp.Var] = {}

    # (agent,box,prop) participates in disposal
    x_disp_part: Dict[Tuple[str, int, Property], gp.Var] = {}

    # (box,prop,k) chosen team size if disposing that prop
    z_team: Dict[Tuple[int, Property, int], gp.Var] = {}

    # (agent,box,prop,k) = x_part AND z_team  (linearization)
    w_part_k: Dict[Tuple[str, int, Property, int], gp.Var] = {}

    # NEW: team arrival (max travel) and finish time for (box,prop,k)
    t_arrive: Dict[Tuple[int, Property, int], gp.Var] = {}
    t_finish: Dict[Tuple[int, Property, int], gp.Var] = {}

    # NEW: linearization for per-agent budget: u_finish = t_finish * w_part_k
    u_finish: Dict[Tuple[str, int, Property, int], gp.Var] = {}


    max_team_overall = max(1, len(agents))
    BIG_M = 1000.0


    # -----------------------------------------------------------------------
    # Create vars
    # -----------------------------------------------------------------------
    for a in agents:
        for b in boxes:
            # Domain semantics: disposal removes object; if already disposed, ignore box
            if b.disposed_X or b.disposed_Y:
                continue

            for p in props:
                if p == "X" and b.disposed_X:
                    continue
                if p == "Y" and b.disposed_Y:
                    continue

                # ---------- SENSING VARS ----------
                can_sense = (a.can_sense_X if p == "X" else a.can_sense_Y)
                already = b.already_sensed.get(a.agent_id, {}).get(p, False)


                senseable = True
                if p == "X":
                    senseable = getattr(b, "senseable_X", True)
                else:
                    senseable = getattr(b, "senseable_Y", True)


                if senseable and can_sense and not already:
                    base_sense_time = b.sense_time_X if p == "X" else b.sense_time_Y
                    travel = travel_time_fn(a.agent_id, b.box_id)
                    total_sense_time = float(base_sense_time) + float(travel)

                    # --- NEW: require that disposal is still feasible afterward (best-case team) ---
                    disp_best_rel = best_case_disposal_time_rel(
                        agents=agents,
                        b=b,
                        prop=p,
                        travel_time_fn=travel_time_fn,
                    )

                    # If nobody eligible can dispose this prop with required team size, don't sense it.
                    if disp_best_rel is None:
                        continue

                    # Conservative: disposal can only start after sensing finishes.
                    total_sense_plus_best_disp = total_sense_time + float(disp_best_rel)

                    # Gate by horizon + deadline
                    if total_sense_time <= float(horizon):
                        if b.deadline is None:
                            # If no deadline, you may still want to require it fits in horizon,
                            # but your statement is about deadlines; keep horizon check as-is.
                            pass
                        else:
                            # Must be able to sense AND then still dispose before deadline
                            if float(current_time) + total_sense_plus_best_disp > float(b.deadline):
                                continue

                        # Optional: also require the combined sense+dispose fits in horizon.
                        # If you want strictly "within planning horizon" feasibility too, uncomment:
                        # if total_sense_plus_best_disp > float(horizon):
                        #     continue

                        s_vars[(a.agent_id, b.box_id, p)] = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"sense_{a.agent_id}_{b.box_id}_{p}",
                        )


                # ---------- TEAM DISPOSAL VARS ----------
                # capability gate
                if p == "X" and not getattr(a, "can_dispose_X", True):
                    continue
                if p == "Y" and not getattr(a, "can_dispose_Y", True):
                    continue

                # probability gate (box-level: any property sufficiently likely)
                p_box = max(float(b.p_true_X), float(b.p_true_Y))
                if p_box < float(weights.pmin_for_dispose):
                    continue



                # info gate
                info_level = b.info_X if p == "X" else b.info_Y
                if info_level < info_threshold_for_dispose:
                    continue

                # (box,prop) disposal var once
                if (b.box_id, p) not in y_disp_prop:
                    y_disp_prop[(b.box_id, p)] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"y_disp_{b.box_id}_{p}",
                    )

                    # team size choice vars once
                    k_min = max(1, int(getattr(b, "min_disposal_team", 1)))
                    k_max = min(int(getattr(b, "max_disposal_team", max_team_overall)), max_team_overall)
                    for k in range(k_min, k_max + 1):
                        z_team[(b.box_id, p, k)] = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"z_team_{b.box_id}_{p}_{k}",
                        )

                # (agent,box,prop) participation var once
                if (a.agent_id, b.box_id, p) not in x_disp_part:
                    x_disp_part[(a.agent_id, b.box_id, p)] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"x_part_{a.agent_id}_{b.box_id}_{p}",
                    )

                # linearization vars once per k
                k_min = max(1, int(getattr(b, "min_disposal_team", 1)))
                k_max = min(int(getattr(b, "max_disposal_team", max_team_overall)), max_team_overall)
                for k in range(k_min, k_max + 1):
                    key = (a.agent_id, b.box_id, p, k)
                    if key not in w_part_k:
                        w_part_k[key] = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"w_{a.agent_id}_{b.box_id}_{p}_{k}",
                        )




    # -----------------------------------------------------------------------
    # NEW: Create team max-travel and finish-time vars for each (bid,p,k)
    # -----------------------------------------------------------------------
    # Choose a time Big-M that safely dominates any plausible time expression.
    # This should be >= max(horizon, agent.max_time, deadline slack) + max travel + max base time.
    max_travel = 0.0
    max_disp_base = 0.0
    max_deadline_slack = 0.0
    for b in boxes:
        # travel upper bound across agents (coarse but safe)
        for a in agents:
            try:
                max_travel = max(max_travel, float(travel_time_fn(a.agent_id, b.box_id)))
            except Exception:
                pass
        max_disp_base = max(max_disp_base, float(b.dispose_time_X), float(b.dispose_time_Y))
        if b.deadline is not None:
            max_deadline_slack = max(max_deadline_slack, float(b.deadline) - float(current_time))

    max_agent_time = max([float(a.max_time) for a in agents] + [0.0])
    BIG_M_TIME = max(float(horizon), max_agent_time, max_deadline_slack, 0.0) + max_travel + max_disp_base + 10.0

    # Create t_arrive and t_finish per (bid,p,k) that exists in z_team
    for (bid, p, k), z in z_team.items():
        t_arrive[(bid, p, k)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"t_arrive_{bid}_{p}_{k}")
        t_finish[(bid, p, k)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"t_finish_{bid}_{p}_{k}")

        b = box_by_id[bid]
        base = b.dispose_time_X if p == "X" else b.dispose_time_Y

        base_k = float(base) * float(speed_factor(k))
        z = z_team[(bid, p, k)]
        tf = t_finish[(bid, p, k)]
        ta = t_arrive[(bid, p, k)]

        # Only enforce when z=1; relax when z=0
        model.addConstr(tf - ta - base_k <= BIG_M_TIME * (1.0 - z),
                        name=f"def_t_finish_ub_{bid}_{p}_{k}")
        model.addConstr(tf - ta - base_k >= -BIG_M_TIME * (1.0 - z),
                        name=f"def_t_finish_lb_{bid}_{p}_{k}")


        # If z=0, force times to 0 (prevents "free" time vars floating)
        model.addConstr(t_arrive[(bid, p, k)] <= BIG_M_TIME * z, name=f"arrive_zero_if_not_{bid}_{p}_{k}")
        model.addConstr(t_finish[(bid, p, k)] <= BIG_M_TIME * z, name=f"finish_zero_if_not_{bid}_{p}_{k}")


    # -----------------------------------------------------------------------
    # NEW: Horizon/deadline feasibility for chosen team size
    # Enforce these only when z_team(b,p,k)=1
    # -----------------------------------------------------------------------
    for (bid, p, k), z in z_team.items():
        b = box_by_id[bid]
        tf = t_finish[(bid, p, k)]

        # Within horizon (relative time)
        model.addConstr(tf <= float(horizon) + BIG_M_TIME * (1.0 - z), name=f"team_horizon_{bid}_{p}_{k}")

        # Meet absolute deadline if present: current_time + tf <= deadline
        if b.deadline is not None:
            model.addConstr(
                float(current_time) + tf <= float(b.deadline) + BIG_M_TIME * (1.0 - z),
                name=f"team_deadline_{bid}_{p}_{k}",
            )


    # -----------------------------------------------------------------------
    # Constraints
    # -----------------------------------------------------------------------

    # (A) At most one disposal total per box across props (object removed)
    for b in boxes:
        ys = [y_disp_prop.get((b.box_id, pp)) for pp in props]
        ys = [v for v in ys if v is not None]
        if ys:
            model.addConstr(gp.quicksum(ys) <= 1, name=f"one_disp_total_{b.box_id}")

    # (B) If dispose (box,prop), choose exactly one team size k
    for (bid, p), y in y_disp_prop.items():
        ks = [k for (_bid, _p, k) in z_team.keys() if _bid == bid and _p == p]
        model.addConstr(gp.quicksum(z_team[(bid, p, k)] for k in ks) == y, name=f"choose_k_{bid}_{p}")

    # (C) Team size match: sum participants == sum(k * z_k)
    for (bid, p), y in y_disp_prop.items():
        parts = [x_disp_part[(aid, bid, p)] for (aid, _bid, _p) in x_disp_part.keys() if _bid == bid and _p == p]
        ks = [k for (_bid, _p, k) in z_team.keys() if _bid == bid and _p == p]
        model.addConstr(
            gp.quicksum(parts) == gp.quicksum(k * z_team[(bid, p, k)] for k in ks),
            name=f"team_size_match_{bid}_{p}",
        )

    # (D) Linearize w = x_part AND z_team
    for (aid, bid, p, k), w in w_part_k.items():
        x = x_disp_part[(aid, bid, p)]
        z = z_team[(bid, p, k)]
        model.addConstr(w <= x, name=f"w_le_x_{aid}_{bid}_{p}_{k}")
        model.addConstr(w <= z, name=f"w_le_z_{aid}_{bid}_{p}_{k}")
        model.addConstr(w >= x + z - 1, name=f"w_ge_and_{aid}_{bid}_{p}_{k}")

    # -----------------------------------------------------------------------
    # NEW: Max-travel rendezvous constraints
    # t_arrive(b,p,k) >= travel(a,b) for all participating agents (w=1)
    # -----------------------------------------------------------------------
    for (aid, bid, p, k), w in w_part_k.items():
        T = t_arrive[(bid, p, k)]
        travel = float(travel_time_fn(aid, bid))
        # If w=1, T >= travel; if w=0, constraint becomes T >= 0
        model.addConstr(T >= travel * w, name=f"arrive_lb_{aid}_{bid}_{p}_{k}")


    # (E) Build a box-level disposal indicator y_disp_box[bid] from y_disp_prop
    y_disp_box: Dict[int, gp.Var] = {}
    for b in boxes:
        y_disp_box[b.box_id] = model.addVar(vtype=GRB.BINARY, name=f"y_disp_box_{b.box_id}")
        ys = [y_disp_prop.get((b.box_id, pp)) for pp in props]
        ys = [v for v in ys if v is not None]
        if ys:
            model.addConstr(gp.quicksum(ys) == y_disp_box[b.box_id], name=f"link_y_disp_box_{b.box_id}")
        else:
            model.addConstr(y_disp_box[b.box_id] == 0, name=f"link_y_disp_box_zero_{b.box_id}")

    # (F) Sense XOR dispose at the box level (your original semantics)
    y_sense: Dict[int, gp.Var] = {}
    for b in boxes:
        y_sense[b.box_id] = model.addVar(vtype=GRB.BINARY, name=f"y_sense_{b.box_id}")

        svs_b = [v for (aid, bid, _pp), v in s_vars.items() if bid == b.box_id]
        if svs_b:
            model.addConstr(gp.quicksum(svs_b) >= y_sense[b.box_id], name=f"link_y_sense_lb_{b.box_id}")
            model.addConstr(gp.quicksum(svs_b) <= BIG_M * y_sense[b.box_id], name=f"link_y_sense_ub_{b.box_id}")
        else:
            model.addConstr(y_sense[b.box_id] == 0, name=f"link_y_sense_zero_{b.box_id}")

        model.addConstr(
            y_disp_box[b.box_id] + y_sense[b.box_id] <= 1,
            name=f"sense_xor_dispose_{b.box_id}",
        )

    # also block any sense vars if disposing that box
    for (aid, bid, p), s_var in s_vars.items():
        model.addConstr(s_var <= 1 - y_disp_box[bid], name=f"no_sense_if_disp_{aid}_{bid}_{p}")

    # -----------------------------------------------------------------------
    # (2) Agent time budget
    # -----------------------------------------------------------------------
    agent_load_expr: Dict[str, gp.LinExpr] = {}
    for a in agents:
        expr = gp.LinExpr()

        # sensing time
        for (aid, bid, p), s_var in s_vars.items():
            if aid != a.agent_id:
                continue
            b = box_by_id[bid]
            base_sense_time = b.sense_time_X if p == "X" else b.sense_time_Y
            travel = travel_time_fn(aid, bid)
            expr += (base_sense_time + travel) * s_var

        # NEW: disposal busy time for participants = t_finish(b,p,k)
        # Need linearization u_finish = t_finish * w
        for (aid, bid, p, k), w in w_part_k.items():
            if aid != a.agent_id:
                continue

            tf = t_finish[(bid, p, k)]

            key = (aid, bid, p, k)
            if key not in u_finish:
                u_finish[key] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"u_finish_{aid}_{bid}_{p}_{k}")

                u = u_finish[key]
                # u = tf * w  (standard big-M linearization)
                model.addConstr(u <= tf, name=f"u_le_tf_{aid}_{bid}_{p}_{k}")
                model.addConstr(u <= BIG_M_TIME * w, name=f"u_le_Mw_{aid}_{bid}_{p}_{k}")
                model.addConstr(u >= tf - BIG_M_TIME * (1.0 - w), name=f"u_ge_tf_M_{aid}_{bid}_{p}_{k}")
                model.addConstr(u >= 0.0, name=f"u_ge_0_{aid}_{bid}_{p}_{k}")

            expr += u_finish[key]


        model.addConstr(expr <= a.max_time, name=f"time_budget_{a.agent_id}")
        agent_load_expr[a.agent_id] = expr

    # -----------------------------------------------------------------------
    # Objective
    # -----------------------------------------------------------------------
    total_reward = gp.LinExpr()
    totalX = gp.LinExpr()
    totalY = gp.LinExpr()

    # Disposal reward counted ONCE per (box,prop)
    for (bid, p), y in y_disp_prop.items():
        b = box_by_id[bid]
        if p == "X":
            p_true = float(b.p_true_X)
            val = float(reward_correct_X)
            totalX += p_true * y
        else:
            p_true = float(b.p_true_Y)
            val = float(reward_correct_Y)
            totalY += p_true * y

        total_reward += val * p_true * y

        if weights.lambda_deadline_risk > 0.0 and b.deadline is not None:
            base_disp_time = b.dispose_time_X if p == "X" else b.dispose_time_Y

            # conservative/best-case: smallest travel among agents that *could* participate
            feasible_travels = []
            for a in agents:
                if p == "X" and not getattr(a, "can_dispose_X", True):
                    continue
                if p == "Y" and not getattr(a, "can_dispose_Y", True):
                    continue
                feasible_travels.append(float(travel_time_fn(a.agent_id, bid)))
            travel_min = min(feasible_travels) if feasible_travels else 0.0

            # use the actually-chosen team size k via z_team[(bid,p,k)]
            ks = [k for (_bid, _p, k) in z_team.keys() if _bid == bid and _p == p]
            for k in ks:
                finish_time_k = float(current_time) + float(travel_min) + float(base_disp_time) * float(speed_factor(k))
                slack_k = float(b.deadline) - float(finish_time_k)
                risk_coeff_k = max(0.0, -slack_k)  # constant given (bid,p,k)

                if risk_coeff_k > 0.0:
                    total_reward -= float(weights.lambda_deadline_risk) * float(risk_coeff_k) * z_team[(bid, p, k)]


    # Sensing reward (unchanged)
    for (aid, bid, p), s_var in s_vars.items():
        b = box_by_id[bid]
        a = agents_by_id[aid]

        if p == "X":
            p_prior = float(b.p_true_X)
            tpr, fpr = float(a.detect_present_X), float(a.detect_absent_X)
        else:
            p_prior = float(b.p_true_Y)
            tpr, fpr = float(a.detect_present_Y), float(a.detect_absent_Y)

        # Egoistic gating: only reward sensing for my goal property
        if weights.egoistic_goal_property is not None:
            if str(p) != str(weights.egoistic_goal_property):
                continue


        ig = expected_info_gain_one(p_prior, tpr, fpr)

        total_reward += weights.weight_info * ig * s_var

        if weights.lambda_sense_slack > 0.0 and b.deadline is not None:
            travel = float(travel_time_fn(aid, bid))
            base_sense_time = float(b.sense_time_X if p == "X" else b.sense_time_Y)
            finish_time = float(current_time) + base_sense_time + travel
            slack = float(b.deadline) - float(finish_time)
            slack_norm = max(0.0, slack) / max(1.0, float(horizon))
            total_reward -= float(weights.lambda_sense_slack) * slack_norm * s_var

        if weights.prefer_exploration != 0.0:
            total_reward += float(weights.prefer_exploration) * s_var

    # X/Y balance penalty: |totalX - totalY|
    d_imb = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="d_imbalance")
    model.addConstr(totalX - totalY <= d_imb, name="balance_pos")
    model.addConstr(totalY - totalX <= d_imb, name="balance_neg")
    total_reward -= float(lambda_balance) * d_imb

    # fairness / load terms (unchanged from your original)
    human_ids = [a.agent_id for a in agents if a.agent_id.startswith("human_")]
    if weights.lambda_load_fairness > 0.0 and len(human_ids) >= 2:
        avg_human_load = (1.0 / len(human_ids)) * gp.quicksum(agent_load_expr[aid] for aid in human_ids)
        for aid in human_ids:
            diff = agent_load_expr[aid] - avg_human_load
            d_pos = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"fair_pos_{aid}")
            d_neg = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"fair_neg_{aid}")
            model.addConstr(diff <= d_pos)
            model.addConstr(-diff <= d_neg)
            total_reward -= float(weights.lambda_load_fairness) * (d_pos + d_neg)

    robot_id = "robot"
    if robot_id in agent_load_expr and human_ids:
        robot_load = agent_load_expr[robot_id]
        total_human_load = gp.quicksum(agent_load_expr[aid] for aid in human_ids)

        if weights.lambda_robot_overuse > 0.0:
            avg_human = total_human_load / len(human_ids)
            diff_robot = robot_load - avg_human
            d_robot_over = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="robot_overuse")
            model.addConstr(diff_robot <= d_robot_over)
            total_reward -= float(weights.lambda_robot_overuse) * d_robot_over

        if weights.lambda_human_overuse > 0.0:
            diff_humans = total_human_load - robot_load
            d_hum_over = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="human_overuse")
            model.addConstr(diff_humans <= d_hum_over)
            total_reward -= float(weights.lambda_human_overuse) * d_hum_over

    model.setObjective(total_reward, GRB.MAXIMIZE)

    dbg(f"counts: s_vars={len(s_vars)} y_disp_prop={len(y_disp_prop)} x_disp_part={len(x_disp_part)} z_team={len(z_team)} w_part_k={len(w_part_k)}")

    # -----------------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------------
    model.optimize()

    '''
    for (bid,p,k), z in z_team.items():
        if z.X > 0.5:
            dbg(f"CHOSEN TEAM box={bid} prop={p} k={k} t_arrive={t_arrive[(bid,p,k)].X:.2f} t_finish={t_finish[(bid,p,k)].X:.2f}")
    '''

    # -----------------------------------------------------------------------
    # Extract plan
    # -----------------------------------------------------------------------
    actions_by_agent: Plan = {}

    if model.Status == GRB.OPTIMAL:
        # senses
        for (aid, bid, p), v in s_vars.items():
            if v.X > 0.5:
                actions_by_agent.setdefault(aid, []).append((bid, p, "sense"))

        # disposals: any participating agent gets a dispose action on that (box,prop)
        for (aid, bid, p), x in x_disp_part.items():
            if x.X > 0.5:
                actions_by_agent.setdefault(aid, []).append((bid, p, "dispose"))

        # sort per agent
        for aid, actions in actions_by_agent.items():
            def sort_key(act: Tuple[int, Property, str]):
                box_id, prop, kind = act
                b = box_by_id.get(box_id)
                deadline = b.deadline if b is not None else 1e12
                kind_rank = 0 if kind == "dispose" else 1
                return (deadline, kind_rank, box_id, 0 if prop == "X" else 1)

            actions.sort(key=sort_key)
            
    dbg(f"solve status = {model.Status} ({model.Status})")
    if model.Status == GRB.INFEASIBLE:
        dbg("Model infeasible; computing IIS...")
        model.computeIIS()
        dbg("IIS constraints:")
        for c in model.getConstrs():
            if c.IISConstr:
                dbg(f"  {c.ConstrName}")
        dbg("IIS variable bounds:")
        for v in model.getVars():
            if v.IISLB or v.IISUB:
                dbg(f"  {v.VarName}  IISLB={v.IISLB} IISUB={v.IISUB}")


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
            problems.append(f"{iss.problem}")
        problems_text = "; ".join(problems)
        parts.append(f"For {who}, I couldn't use some steps: {problems_text}")

    return "I couldn't fully understand your plan. " + " ".join(parts)


# ---------------------------------------------------------------------------
# Objective scoring & constraint evaluation for arbitrary plans
# ---------------------------------------------------------------------------

def score_plan_objective(
    plan: Plan,
    agents: List[AgentState],
    boxes: List[BoxInfo],
    travel_time_fn: TravelTimeFn,
    *,
    current_time: float,
    horizon: float,
    weights: PlannerWeights | None = None,
) -> float:
    """
    Scalar score for a plan (no MILP solve). Includes:
      - expected correct disposals
      - info gain from sensing
      - exploration bonus
      - X/Y balance penalty
      - OPTIONAL: sensing slack shaping (deadline-aware sensing preference)

    Note: This is a heuristic evaluator; it does not simulate full sequencing.
    """
    if weights is None:
        weights = PlannerWeights()

    reward_correct_X = weights.reward_correct_X
    reward_correct_Y = weights.reward_correct_Y
    weight_info = weights.weight_info
    lambda_balance = weights.lambda_balance
    prefer_exploration = weights.prefer_exploration
    lambda_sense_slack = getattr(weights, "lambda_sense_slack", 0.0)

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
            b = box_by_id.get(int(box_id))
            if b is None:
                continue

            if kind == "dispose":
                if prop == "X":
                    p_true = float(b.p_true_X)
                    totalX += p_true
                    total_reward += reward_correct_X * p_true
                else:
                    p_true = float(b.p_true_Y)
                    totalY += p_true
                    total_reward += reward_correct_Y * p_true

            elif kind == "sense":
                if prop == "X":
                    p_true = float(b.p_true_X)
                    info_level = float(b.info_X)
                    agent_quality = max(float(a.detect_present_X) - float(a.detect_absent_X), 0.0)
                    base_sense_time = float(b.sense_time_X)
                else:
                    p_true = float(b.p_true_Y)
                    info_level = float(b.info_Y)
                    agent_quality = max(float(a.detect_present_Y) - float(a.detect_absent_Y), 0.0)
                    base_sense_time = float(b.sense_time_Y)

                entropy_like = 4.0 * p_true * (1.0 - p_true)
                base_info_gain = (1.0 - info_level) * entropy_like
                info_gain = agent_quality * base_info_gain

                total_reward += weight_info * info_gain

                # exploration bonus
                if prefer_exploration != 0.0:
                    total_reward += prefer_exploration

                # --- sensing slack shaping (deadline-aware) ---
                # This mirrors the MILP term conceptually, but is purely numeric.
                if lambda_sense_slack > 0.0 and getattr(b, "deadline", None) is not None:
                    travel = float(travel_time_fn(aid, int(box_id)))
                    finish_time = float(current_time) + base_sense_time + travel
                    slack = float(b.deadline) - finish_time

                    # Penalize sensing things with LOTS of slack (encourages near-deadline senses)
                    slack_norm = max(0.0, slack) / max(1.0, float(horizon))
                    total_reward -= lambda_sense_slack * slack_norm

            # else: ignore unknown kinds

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
    agents: List[AgentState],
    boxes: List[BoxInfo],
    current_time: float,
    travel_time_fn: TravelTimeFn,
    horizon: float,
    weights: PlannerWeights,
) -> PlanConstraintMetrics:
    box_by_id = {b.box_id: b for b in boxes}

    agents_by_id = {a.agent_id: a for a in agents}

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
            # NEW: capability check
            if prop == "X" and not getattr(agents_by_id.get(aid, None), "can_dispose_X", True):
                num_info_viol += 1  # or better: add a new counter "num_capability_violations"
                continue
            if prop == "Y" and not getattr(agents_by_id.get(aid, None), "can_dispose_Y", True):
                num_info_viol += 1
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
    score_curr = score_plan_objective(
        current_plan,
        agents,
        boxes,
        travel_time_fn=travel_time_fn,
        current_time=current_time,
        horizon=horizon,
        weights=weights,
    )

    score_cand = score_plan_objective(
        candidate_plan,
        agents,
        boxes,
        travel_time_fn=travel_time_fn,
        current_time=current_time,
        horizon=horizon,
        weights=weights,
    )


    # Constraint diagnostics
    constraints_curr = evaluate_plan_constraints(
        plan=current_plan,
        agents=agents, 
        boxes=boxes,
        current_time=current_time,
        travel_time_fn=travel_time_fn,
        horizon=horizon,
        weights=weights,
    )
    constraints_cand = evaluate_plan_constraints(
        plan=candidate_plan,
        agents=agents, 
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
                can_dispose_X=getattr(a, "can_dispose_X", True),
                can_dispose_Y=getattr(a, "can_dispose_Y", True),
                detect_present_X=a.detect_present_X,
                detect_absent_X=a.detect_absent_X,
                detect_present_Y=a.detect_present_Y,
                detect_absent_Y=a.detect_absent_Y,
            )
        )

    # --- 4) Build residual box world ---
    # Prefix are PROPOSED, not executed -> do NOT modify already_sensed/info/disposed.
    residual_boxes: List[BoxInfo] = copy.deepcopy(boxes)

    # --- 5) Run optimizer on residual problem ---
    suffix_plan = plan_assignments_gurobi(
        agents=residual_agents,
        boxes=residual_boxes,
        current_time=current_time,
        horizon=horizon,
        travel_time_fn=travel_time_fn,
        weights=weights,
    )

    # --- 5.5) Remove suffix actions that duplicate prefix actions ---
    prefix_set = set()
    for aid, actions in (prefix_plan or {}).items():
        for (box_id, prop, kind) in actions:
            prefix_set.add((aid, int(box_id), prop, kind))

    for aid, actions in list((suffix_plan or {}).items()):
        filtered = []
        for (box_id, prop, kind) in actions:
            if (aid, int(box_id), prop, kind) in prefix_set:
                continue
            filtered.append((int(box_id), prop, kind))
        if filtered:
            suffix_plan[aid] = filtered
        else:
            suffix_plan.pop(aid, None)


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
        AgentState(agent_id="human_a", max_time=30.0, can_sense_X=True,  can_sense_Y=False, can_dispose_X=True,  can_dispose_Y=False),
        AgentState(agent_id="human_b", max_time=30.0, can_sense_X=False, can_sense_Y=True, can_dispose_X=False, can_dispose_Y=True),
        AgentState(agent_id="robot",   max_time=30.0, can_sense_X=True,  can_sense_Y=True, can_dispose_X=True,  can_dispose_Y=True),
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
        info_threshold_for_dispose=0.4,
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
        horizon=120.0,
        travel_time_fn=dummy_travel_time,
        weights=weights,
    )

    for aid, actions in plan.items():
        print(f"{aid}:")
        for box_id, prop, kind in actions:
            print(f"  {kind.upper()} box {box_id} property {prop}")

