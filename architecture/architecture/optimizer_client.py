#!/usr/bin/env python3
"""
optimizer_client.py

Gurobi-based planner for assigning sensing & disposal tasks to agents
(human_A, human_B, robot) given the current world state.

Key constraints:
- human_A can only sense property X
- human_B can only sense property Y
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
    * penalize imbalance between X and Y disposals.

To use:
- Build `AgentState` and `BoxInfo` from your API (/boxes/state, /time, DB).
- Provide a `travel_time_fn(agent_id, box_id)` that returns seconds.
- Call `plan_assignments_gurobi(...)` to get lists of sense/dispose actions,
  then turn those into /sense and /dispose HTTP calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Literal, Callable, Tuple
import gurobipy as gp
from gurobipy import GRB


Property = Literal["X", "Y"]
TravelTimeFn = Callable[[str, int], float]  # (agent_id, box_id) -> seconds


# ---------------------------------------------------------------------------
# Data structures to pass into the planner
# ---------------------------------------------------------------------------

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
                  property. You choose the scale; typical is 0..1, where
                  0 = no info, 1 = fully known.

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


# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------

def plan_assignments_gurobi(
    agents: List[AgentState],
    boxes: List[BoxInfo],
    current_time: float,
    horizon: float,
    travel_time_fn: TravelTimeFn,
    info_threshold_for_dispose: float = 0.6,
    reward_correct_X: float = 1.0,
    reward_correct_Y: float = 1.0,
    weight_info: float = 0.2,
    lambda_balance: float = 0.5,
) -> Dict[str, List[Tuple[str, int, Property]]]:
    """
    Build and solve a Gurobi MILP that assigns sensing and disposal tasks.

    Args:
        agents: list of AgentState (human_A, human_B, robot, ...).
        boxes: list of BoxInfo built from the current world state.
        current_time: current sim time (seconds).
        horizon: maximum time window (seconds) we plan over. Tasks must fit
                 within this window for each agent.
        travel_time_fn: function (agent_id, box_id) -> travel time (seconds).
        info_threshold_for_dispose: minimum info_X/Y required to even allow
                 disposal for a given property.
        reward_correct_X/Y: reward weights for expected correct X/Y disposal.
        weight_info: weight given to information-gain reward from sensing.
        lambda_balance: weight of penalty on X vs Y disposal imbalance.

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
    model = gp.Model("box_planner")
    model.Params.OutputFlag = 0  # silent

    agents_by_id = {a.agent_id: a for a in agents}


    props: List[Property] = ["X", "Y"]

    # -----------------------------------------------------------------------
    # Role decision variables: each agent chooses sensing or disposal (or idle)
    # -----------------------------------------------------------------------
    # z_sense[a] = 1 → agent a acts as a sensing agent
    # z_dispose[a] = 1 → agent a acts as a disposal agent
    # constraint: z_sense[a] + z_dispose[a] <= 1
    z_sense = {}
    z_dispose = {}
    for a in agents:
        z_sense[a.agent_id] = model.addVar(
            vtype=GRB.BINARY,
            name=f"z_sense_{a.agent_id}",
        )
        z_dispose[a.agent_id] = model.addVar(
            vtype=GRB.BINARY,
            name=f"z_disp_{a.agent_id}",
        )
        model.addConstr(
            z_sense[a.agent_id] + z_dispose[a.agent_id] <= 1,
            name=f"role_choice_{a.agent_id}",
        )

    # -----------------------------------------------------------------------
    # Decision variables: sense and dispose
    # Respect:
    #   - capabilities (can_sense_X/Y),
    #   - no repeated sensing for same (agent, box, property),
    #   - info threshold for disposal,
    #   - deadlines & horizon (time feasibility).
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
                # Capability: human_A → only X; human_B → only Y; robot → both
                if p == "X" and not a.can_sense_X:
                    can_sense = False
                elif p == "Y" and not a.can_sense_Y:
                    can_sense = False
                else:
                    can_sense = True

                # No repeated sensing of same (box, property) by same agent
                already = b.already_sensed.get(a.agent_id, {}).get(p, False)

                if can_sense and not already:
                    s_vars[(a.agent_id, b.box_id, p)] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"sense_{a.agent_id}_{b.box_id}_{p}",
                    )

                # ---------- DISPOSAL VARIABLES ----------
                # Disposal only if enough information is available
                if p == "X":
                    info_level = b.info_X
                    base_disp_time = b.dispose_time_X
                else:
                    info_level = b.info_Y
                    base_disp_time = b.dispose_time_Y

                if info_level < info_threshold_for_dispose:
                    # Not enough info yet → we don't allow disposal on this
                    continue

                travel = travel_time_fn(a.agent_id, b.box_id)
                total_disp_time = base_disp_time + travel

                # Must fit within horizon AND deadline
                if total_disp_time <= horizon and current_time + total_disp_time <= b.deadline:
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

    # (3) Role coupling: if agent is sensing, it cannot dispose (and vice versa)
    # Use big-M constraints linking s_vars/d_vars to z_sense/z_dispose.
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
    # Objective: expected correct disposals + info gain, with X/Y balancing
    # -----------------------------------------------------------------------

    total_reward = gp.LinExpr()
    totalX = gp.LinExpr()
    totalY = gp.LinExpr()

    # Disposal reward (expected correct)
    for (aid, bid, p), d_var in d_vars.items():
        b = next(bb for bb in boxes if bb.box_id == bid)

        if p == "X":
            p_true = b.p_true_X
            val = reward_correct_X
            totalX += p_true * d_var
        else:
            p_true = b.p_true_Y
            val = reward_correct_Y
            totalY += p_true * d_var

        total_reward += val * p_true * d_var

    # Sensing reward (information gain, only for new senses)
    for (aid, bid, p), s_var in s_vars.items():
        b = next(bb for bb in boxes if bb.box_id == bid)
        a = agents_by_id[aid]

        if p == "X":
            p_true = b.p_true_X
            info_level = b.info_X
            # Simple scalar for how discriminative this agent is on X:
            # high present-detect, low false-positive → high quality
            agent_quality = max(a.detect_present_X - a.detect_absent_X, 0.0)
        else:
            p_true = b.p_true_Y
            info_level = b.info_Y
            agent_quality = max(a.detect_present_Y - a.detect_absent_Y, 0.0)

        # Base "how useful is more info here?" term:
        entropy_like = 4.0 * p_true * (1.0 - p_true)  # max at p_true=0.5
        base_info_gain = (1.0 - info_level) * entropy_like

        # Scale by agent sensing quality so robot > humans
        info_gain = agent_quality * base_info_gain

        total_reward += weight_info * info_gain * s_var


    # X/Y balance penalty: penalize |totalX - totalY|
    d_imb = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="d_imbalance")
    model.addConstr(totalX - totalY <= d_imb, name="balance_pos")
    model.addConstr(totalY - totalX <= d_imb, name="balance_neg")

    model.setObjective(total_reward - lambda_balance * d_imb, GRB.MAXIMIZE)

    # -----------------------------------------------------------------------
    # Solve and extract solution
    # -----------------------------------------------------------------------
    model.optimize()

    # -----------------------------------------------------------------------
    # Solve and extract solution as per-agent sequential action lists
    # -----------------------------------------------------------------------
    actions_by_agent: Dict[str, List[Tuple[int, Property, str]]] = {}

    if model.Status == GRB.OPTIMAL:
        # First collect all chosen senses/dispenses
        for (aid, bid, p), v in s_vars.items():
            if v.X > 0.5:
                actions_by_agent.setdefault(aid, []).append((bid, p, "sense"))

        for (aid, bid, p), v in d_vars.items():
            if v.X > 0.5:
                actions_by_agent.setdefault(aid, []).append((bid, p, "dispose"))

        # Now impose a simple *sequence* per agent:
        #   - sort by earliest box deadline
        #   - for same deadline/box, do disposal before sensing
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
# Example of how you might call this (for reference / testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def dummy_travel_time(agent_id: str, box_id: int) -> float:
        return 1.0

    agents = [
        AgentState(agent_id="human_A", max_time=30.0, can_sense_X=True,  can_sense_Y=False),
        AgentState(agent_id="human_B", max_time=30.0, can_sense_X=False, can_sense_Y=True),
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
                "human_A": {"X": False, "Y": False},
                "human_B": {"X": False, "Y": False},
                "robot":   {"X": False, "Y": False},
            },
        )
    ]

    plan = plan_assignments_gurobi(
        agents=agents,
        boxes=boxes,
        current_time=0.0,
        horizon=30.0,
        travel_time_fn=dummy_travel_time,
    )

    for aid, actions in plan.items():
        print(f"{aid}:")
        for box_id, prop, kind in actions:
            print(f"  {kind.upper()} box {box_id} property {prop}")


