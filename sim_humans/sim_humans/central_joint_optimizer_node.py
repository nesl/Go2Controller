#!/usr/bin/env python3
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import requests
import rclpy
from rclpy.node import Node
from std_msgs.msg import String as StringMsg

import gurobipy as gp
from gurobipy import GRB
import hashlib
import math

Property = Literal["X", "Y"]
ActionKind = Literal["sense", "dispose"]
TravelTimeFn = Callable[[str, int], float]  # (agent_id, box_id) -> seconds
Plan = Dict[str, List[Tuple[int, Property, ActionKind]]]


# ---------------------------
# Planner data structures
# ---------------------------

@dataclass
class PlannerWeights:
    reward_correct_X: float = 1.0
    reward_correct_Y: float = 1.0
    weight_info: float = 0.2
    lambda_balance: float = 0.5
    info_threshold_for_dispose: float = 0.8
    prefer_exploration: float = 0.0
    lambda_deadline_risk: float = 0.1
    lambda_sense_slack: float = 0.001
    lambda_travel: float = 0.0  # start 0.01 maybe

@dataclass
class AgentState:
    agent_id: str
    max_time: float
    can_sense_X: bool
    can_sense_Y: bool
    detect_present_X: float = 0.8
    detect_absent_X: float = 0.2
    detect_present_Y: float = 0.8
    detect_absent_Y: float = 0.2


@dataclass
class BoxInfo:
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


# ---------------------------
# Joint multi-agent MILP planner
# ---------------------------

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


    debug = False  # flip off later

    if debug:
        print(
            "[DBG] weights:",
            {
                "info_threshold_for_dispose": weights.info_threshold_for_dispose,
                "lambda_deadline_risk": weights.lambda_deadline_risk,
                "lambda_travel": weights.lambda_travel,
                "weight_info": weights.weight_info,
                "prefer_exploration": weights.prefer_exploration,
                "lambda_balance": weights.lambda_balance,
                "lambda_sense_slack": weights.lambda_sense_slack,
            },
            flush=True,
        )
        print(f"[DBG] current_time={current_time:.3f} horizon={horizon:.3f}", flush=True)



    props: List[Property] = ["X", "Y"]
    agents_by_id = {a.agent_id: a for a in agents}

    model = gp.Model("joint_box_planner")
    model.Params.OutputFlag = 0  # silent

    # role: sense vs dispose vs idle (optional constraint)
    z_sense: Dict[str, gp.Var] = {}
    z_dispose: Dict[str, gp.Var] = {}
    for a in agents:
        z_sense[a.agent_id] = model.addVar(vtype=GRB.BINARY, name=f"z_sense_{a.agent_id}")
        z_dispose[a.agent_id] = model.addVar(vtype=GRB.BINARY, name=f"z_disp_{a.agent_id}")
        model.addConstr(z_sense[a.agent_id] + z_dispose[a.agent_id] <= 1, name=f"role_{a.agent_id}")

    s_vars: Dict[Tuple[str, int, Property], gp.Var] = {}
    d_vars: Dict[Tuple[str, int, Property], gp.Var] = {}

    # create feasible vars
    for a in agents:
        for b in boxes:
            # (optional) once disposed for ANY property, treat as fully disposed
            if b.disposed_X or b.disposed_Y:
                continue

            for p in props:
                # ----- sense -----
                can_sense = (p == "X" and a.can_sense_X) or (p == "Y" and a.can_sense_Y)
                already = bool(b.already_sensed.get(a.agent_id, {}).get(p, False))
                if can_sense and not already:
                    base = b.sense_time_X if p == "X" else b.sense_time_Y
                    travel = travel_time_fn(a.agent_id, b.box_id)
                    total_t = base + travel
                    if total_t <= horizon and current_time + total_t <= b.deadline:
                        s_vars[(a.agent_id, b.box_id, p)] = model.addVar(
                            vtype=GRB.BINARY, name=f"sense_{a.agent_id}_{b.box_id}_{p}"
                        )

                # ----- dispose -----
                p_true = b.p_true_X if p == "X" else b.p_true_Y
                thr = float(weights.info_threshold_for_dispose)

                if debug:
                    print(
                        f"[DBG][DISP-CHECK] a={a.agent_id} box={b.box_id} prop={p} "
                        f"p_true={p_true:.3f} thr={thr:.3f} pass={p_true >= thr}",
                        flush=True,
                    )

                if p_true < thr:
                    if debug:
                        print(
                            f"[DBG][DISP-SKIP] a={a.agent_id} box={b.box_id} prop={p} "
                            f"reason=below_threshold p_true={p_true:.3f} < thr={thr:.3f}",
                            flush=True,
                        )
                    continue

                base = b.dispose_time_X if p == "X" else b.dispose_time_Y
                travel = travel_time_fn(a.agent_id, b.box_id)
                total_t = base + travel

                # horizon feasibility
                if total_t > horizon:
                    if debug:
                        print(
                            f"[DBG][DISP-SKIP] a={a.agent_id} box={b.box_id} prop={p} "
                            f"reason=over_horizon base={base:.3f} travel={travel:.3f} total={total_t:.3f} > horizon={horizon:.3f}",
                            flush=True,
                        )
                    continue

                # deadline feasibility
                finish = current_time + total_t
                if finish > b.deadline:
                    if debug:
                        print(
                            f"[DBG][DISP-SKIP] a={a.agent_id} box={b.box_id} prop={p} "
                            f"reason=miss_deadline finish={finish:.3f} > deadline={float(b.deadline):.3f} "
                            f"(base={base:.3f} travel={travel:.3f})",
                            flush=True,
                        )
                    continue

                # if we reach here, disposal var is feasible and will be created
                if debug:
                    slack = float(b.deadline) - float(finish)
                    print(
                        f"[DBG][DISP-VAR] a={a.agent_id} box={b.box_id} prop={p} "
                        f"CREATED base={base:.3f} travel={travel:.3f} total={total_t:.3f} slack={slack:.3f}",
                        flush=True,
                    )

                d_vars[(a.agent_id, b.box_id, p)] = model.addVar(
                    vtype=GRB.BINARY, name=f"disp_{a.agent_id}_{b.box_id}_{p}"
                )


                '''
                # ----- dispose -----
                p_true = b.p_true_X if p == "X" else b.p_true_Y
                if p_true < weights.info_threshold_for_dispose:
                    continue


                base = b.dispose_time_X if p == "X" else b.dispose_time_Y
                travel = travel_time_fn(a.agent_id, b.box_id)
                total_t = base + travel
                if total_t <= horizon and current_time + total_t <= b.deadline:
                    d_vars[(a.agent_id, b.box_id, p)] = model.addVar(
                        vtype=GRB.BINARY, name=f"disp_{a.agent_id}_{b.box_id}_{p}"
                    )
                '''
                
    BIG_M = 1000.0
    if debug:
        print(f"vars created: s={len(s_vars)} d={len(d_vars)}", flush=True)






    # (1) at most one disposal per box total (disposal removes object)
    for b in boxes:
        dvs = [v for (aid, bid, _p), v in d_vars.items() if bid == b.box_id]
        if dvs:
            model.addConstr(gp.quicksum(dvs) <= 1, name=f"one_dispose_total_{b.box_id}")

    # (2) if disposed in this solve, don't sense that same (box,prop) in this solve
    y_disp: Dict[int, gp.Var] = {b.box_id: model.addVar(vtype=GRB.BINARY, name=f"y_disp_{b.box_id}") for b in boxes}
    for b in boxes:
        dvs = [v for (aid, bid, _p), v in d_vars.items() if bid == b.box_id]
        if dvs:
            model.addConstr(gp.quicksum(dvs) == y_disp[b.box_id], name=f"link_y_disp_{b.box_id}")
        else:
            model.addConstr(y_disp[b.box_id] == 0, name=f"link_y_disp_zero_{b.box_id}")

    for (aid, bid, p), sv in s_vars.items():
        model.addConstr(sv <= 1 - y_disp[bid], name=f"no_sense_if_disp_{aid}_{bid}_{p}")

    # (3) time budget per agent
    agent_load: Dict[str, gp.LinExpr] = {}
    for a in agents:
        expr = gp.LinExpr()
        for b in boxes:
            for p in props:
                sv = s_vars.get((a.agent_id, b.box_id, p))
                dv = d_vars.get((a.agent_id, b.box_id, p))
                travel = travel_time_fn(a.agent_id, b.box_id)

                if sv is not None:
                    base = b.sense_time_X if p == "X" else b.sense_time_Y
                    expr += (base + travel) * sv
                if dv is not None:
                    base = b.dispose_time_X if p == "X" else b.dispose_time_Y
                    expr += (base + travel) * dv

        model.addConstr(expr <= a.max_time, name=f"time_budget_{a.agent_id}")
        agent_load[a.agent_id] = expr

    # (4) role coupling
    for a in agents:
        s_list = [v for (aid, _bid, _p), v in s_vars.items() if aid == a.agent_id]
        d_list = [v for (aid, _bid, _p), v in d_vars.items() if aid == a.agent_id]
        if s_list:
            model.addConstr(gp.quicksum(s_list) <= BIG_M * z_sense[a.agent_id], name=f"sense_role_{a.agent_id}")
        if d_list:
            model.addConstr(gp.quicksum(d_list) <= BIG_M * z_dispose[a.agent_id], name=f"disp_role_{a.agent_id}")

    # objective
    total_reward = gp.LinExpr()
    totalX = gp.LinExpr()
    totalY = gp.LinExpr()

    box_by_id = {b.box_id: b for b in boxes}

    # disposal reward (+ deadline risk)
    for (aid, bid, p), dv in d_vars.items():
        b = box_by_id[bid]
        if p == "X":
            p_true = b.p_true_X
            val = weights.reward_correct_X
            base_disp = b.dispose_time_X
            totalX += p_true * dv
        else:
            p_true = b.p_true_Y
            val = weights.reward_correct_Y
            base_disp = b.dispose_time_Y
            totalY += p_true * dv

        total_reward += val * p_true * dv

        '''
        if weights.lambda_travel > 0.0:
            travel = travel_time_fn(aid, bid)
            total_reward -= weights.lambda_travel * travel * dv
        '''
        # dispose loop
        travel = float(travel_time_fn(aid, bid)) 
        if weights.lambda_travel > 0.0:
            total_reward -= weights.lambda_travel * (travel) * dv

        if weights.lambda_deadline_risk > 0.0:
            travel = travel_time_fn(aid, bid)
            finish_time = current_time + base_disp + travel
            slack = float(b.deadline) - float(finish_time)
            risk_coeff = max(0.0, -slack)
            if risk_coeff > 0.0:
                total_reward -= weights.lambda_deadline_risk * risk_coeff * dv

    # sensing info gain (+ exploration)

    for (aid, bid, p), sv in s_vars.items():
        b = box_by_id[bid]
        a = agents_by_id[aid]

        if p == "X":
            p_true = b.p_true_X
            info_level = b.info_X
            agent_q = max(a.detect_present_X - a.detect_absent_X, 0.0)
            base_sense = b.sense_time_X
        else:
            p_true = b.p_true_Y
            info_level = b.info_Y
            agent_q = max(a.detect_present_Y - a.detect_absent_Y, 0.0)
            base_sense = b.sense_time_Y

        # your existing info gain
        entropy_like = 4.0 * p_true * (1.0 - p_true)
        base_gain = (1.0 - info_level) * entropy_like
        info_gain = agent_q * base_gain

        total_reward += weights.weight_info * info_gain * sv
        if weights.prefer_exploration != 0.0:
            total_reward += weights.prefer_exploration * sv

        '''
        if weights.lambda_travel > 0.0:
            travel = travel_time_fn(aid, bid)
            total_reward -= weights.lambda_travel * travel * sv
        '''
        travel = float(travel_time_fn(aid, bid)) 
        # sense loop
        if weights.lambda_travel > 0.0:
            total_reward -= weights.lambda_travel * (travel) * sv

        # for sensing: deadline risk, not slack penalty
        if weights.lambda_deadline_risk > 0.0:
            finish_time = current_time + base_sense + travel
            lateness = max(0.0, float(finish_time) - float(b.deadline))
            total_reward -= weights.lambda_deadline_risk * lateness * sv
            
        finish = current_time + base_sense + travel
        
        slack = max(1e-3, float(b.deadline) - float(finish))   # seconds remaining if you do it now
        urgency = 1.0 / slack                                  # larger when slack is small
        urgency = min(urgency, 1.0)                            # cap (tune cap)
        #total_reward += weights.lambda_sense_slack * urgency * sv

            
        slack = max(0.0, float(b.deadline) - float(finish))
        total_reward -= weights.lambda_sense_slack * (slack / max(1.0, horizon)) * sv




    # X/Y balance penalty
    d_imb = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="d_imb")
    model.addConstr(totalX - totalY <= d_imb)
    model.addConstr(totalY - totalX <= d_imb)
    total_reward -= weights.lambda_balance * d_imb

    model.setObjective(total_reward, GRB.MAXIMIZE)
    model.optimize()

    if debug:
        print("status:", model.Status, "obj:", getattr(model, "ObjVal", None), flush=True)

        sum_s = 0.0
        for key, sv in s_vars.items():
            sum_s += sv.X
        print("sum sv:", sum_s, "num_s_vars:", len(s_vars), flush=True)

        for a in agents:
            print("z_sense", a.agent_id, z_sense[a.agent_id].X, "z_disp", z_dispose[a.agent_id].X, flush=True)


    out: Plan = {a.agent_id: [] for a in agents}
    if model.Status != GRB.OPTIMAL:
        return out

    for (aid, bid, p), sv in s_vars.items():
        if sv.X > 0.5:
            out[aid].append((bid, p, "sense"))
    for (aid, bid, p), dv in d_vars.items():
        if dv.X > 0.5:
            out[aid].append((bid, p, "dispose"))

    # simple ordering: earliest deadline, dispose before sense
    def sort_key(act: Tuple[int, Property, ActionKind]) -> Tuple[float, int, int, int]:
        bid, prop, kind = act
        b = box_by_id.get(bid)
        deadline = float(b.deadline) if b is not None else 1e12
        kind_rank = 0 if kind == "dispose" else 1
        prop_rank = 0 if prop == "X" else 1
        return (deadline, kind_rank, bid, prop_rank)

    for aid in list(out.keys()):
        out[aid].sort(key=sort_key)

    best = 0.0
    best_key = None
    for (aid,bid,p), sv in s_vars.items():
        b = box_by_id[bid]
        a = agents_by_id[aid]
        if p == "X":
            p_true = b.p_true_X; info_level=b.info_X; agent_q=max(a.detect_present_X-a.detect_absent_X,0.0)
        else:
            p_true = b.p_true_Y; info_level=b.info_Y; agent_q=max(a.detect_present_Y-a.detect_absent_Y,0.0)
        entropy_like = 4.0*p_true*(1.0-p_true)
        base_gain = (1.0-info_level)*entropy_like
        info_gain = agent_q*base_gain
        coeff = weights.weight_info * info_gain
        if coeff > best:
            best = coeff; best_key = (aid,bid,p,p_true,info_level,agent_q)
    print("best sense coeff:", best, "best_key:", best_key, flush=True)

    return out


# ---------------------------
# Central optimizer node
# ---------------------------

class CentralJointOptimizerNode(Node):
    def __init__(self):
        super().__init__("central_joint_optimizer")

        self.declare_parameter("server_base_url", "http://URL:8080")
        self.declare_parameter("request_timeout_sec", 120.0)
        self.declare_parameter("tick_period_sec", 0.5)

        self.declare_parameter("horizon_sec", 120.0)
        self.declare_parameter("agent_ids", ["robot", "human_a", "human_b"])

        # default action times (if server doesn't include them)
        self.declare_parameter("default_sense_time_X", 3.0)
        self.declare_parameter("default_sense_time_Y", 3.0)
        self.declare_parameter("default_dispose_time_X", 4.0)
        self.declare_parameter("default_dispose_time_Y", 4.0)
        self.declare_parameter("lambda_travel", 0.0)  # start small

        # weights
        self.declare_parameter("reward_correct_X", 1.0)
        self.declare_parameter("reward_correct_Y", 1.0)
        self.declare_parameter("weight_info", 0.2)
        self.declare_parameter("lambda_balance", 0.0)
        self.declare_parameter("info_threshold_for_dispose", 0.6)
        self.declare_parameter("prefer_exploration", 0.0)
        self.declare_parameter("lambda_deadline_risk", 0.1)

        self.declare_parameter("lambda_sense_slack", 0.001)


        # behavior
        self.declare_parameter("replan_period_sec", 2.0)
        self.declare_parameter("replan_on_any_result", True)

        self.declare_parameter("default_speed_mps", 1.0)
        self.declare_parameter("pose_stale_sec", 2.0)   # if pose older than this, treat as unknown
        self.declare_parameter("unknown_pose_travel_sec", 0.0)  # 0 keeps old behavior; or set big like 999

        self.default_speed_mps = float(self.get_parameter("default_speed_mps").value)
        self.pose_stale_sec = float(self.get_parameter("pose_stale_sec").value)
        self.unknown_pose_travel_sec = float(self.get_parameter("unknown_pose_travel_sec").value)


        self.base_url = str(self.get_parameter("server_base_url").value).rstrip("/")
        self.timeout = float(self.get_parameter("request_timeout_sec").value)
        self.tick_period = float(self.get_parameter("tick_period_sec").value)

        self.horizon = float(self.get_parameter("horizon_sec").value)
        self.agent_ids: List[str] = list(self.get_parameter("agent_ids").value)

        self._last_world_fp: Optional[str] = None
        self._last_published_world_fp: Optional[str] = None


        self.default_sense_time_X = float(self.get_parameter("default_sense_time_X").value)
        self.default_sense_time_Y = float(self.get_parameter("default_sense_time_Y").value)
        self.default_dispose_time_X = float(self.get_parameter("default_dispose_time_X").value)
        self.default_dispose_time_Y = float(self.get_parameter("default_dispose_time_Y").value)

        self.replan_period_sec = float(self.get_parameter("replan_period_sec").value)
        self.replan_on_any_result = bool(self.get_parameter("replan_on_any_result").value)

        self.weights = PlannerWeights(
            reward_correct_X=float(self.get_parameter("reward_correct_X").value),
            reward_correct_Y=float(self.get_parameter("reward_correct_Y").value),
            weight_info=float(self.get_parameter("weight_info").value),
            lambda_balance=float(self.get_parameter("lambda_balance").value),
            info_threshold_for_dispose=float(self.get_parameter("info_threshold_for_dispose").value),
            prefer_exploration=float(self.get_parameter("prefer_exploration").value),
            lambda_deadline_risk=float(self.get_parameter("lambda_deadline_risk").value),
            lambda_sense_slack=float(self.get_parameter("lambda_sense_slack").value),  #
            lambda_travel=float(self.get_parameter("lambda_travel").value),
        )

        self.pub_plan = self.create_publisher(StringMsg, "/central_plan", 10)
        self.sub_res = self.create_subscription(StringMsg, "/agent_result", self._on_agent_result, 50)

        self.sub_pose = self.create_subscription(StringMsg, "/agent_pose", self._on_agent_pose, 100)

        self._pose_lock = threading.Lock()
        self._agent_pose: Dict[str, Tuple[float, float, float]] = {}  # aid -> (x,y,t_wall)

        # box_id -> {"agent_id": str, "prop": "X"|"Y", "plan_id": str, "locked_at": float}
        self._dispose_locks: Dict[int, Dict[str, Any]] = {}


        self._work_lock = threading.Lock()
        self._work_thread: Optional[threading.Thread] = None

        self._state_lock = threading.Lock()
        self._dirty = True
        self._next_periodic = 0.0
        self._active_plan_id: Optional[str] = None

        self._timer = self.create_timer(self.tick_period, self._tick)

        self.get_logger().info(
            f"CentralJointOptimizerNode up server={self.base_url} agents={self.agent_ids} horizon={self.horizon}"
        )

    def _on_agent_pose(self, msg: StringMsg) -> None:
        try:
            d = json.loads(msg.data)
        except Exception:
            return

        aid = str(d.get("agent_id", ""))
        if not aid:
            return

        try:
            x = float(d.get("x"))
            y = float(d.get("y"))
            t_wall = float(d.get("t_wall", time.time()))
        except Exception:
            return

        with self._pose_lock:
            self._agent_pose[aid] = (x, y, t_wall)


    # ---- HTTP ----
    def _http(self, method: str, path: str, *, json_body: Optional[dict] = None) -> requests.Response:
        url = self.base_url + path
        if method == "GET":
            return requests.get(url, timeout=self.timeout)
        if method == "POST":
            return requests.post(url, json=json_body, timeout=self.timeout)
        raise ValueError(method)

    def _time(self) -> Dict[str, Any]:
        r = self._http("GET", "/time")
        r.raise_for_status()
        return r.json()

    def _boxes_state(self) -> List[Dict[str, Any]]:
        r = self._http("GET", "/boxes/state")
        r.raise_for_status()
        return list(r.json())

    def _agents_params(self) -> Optional[Dict[str, Any]]:
        try:
            r = self._http("GET", "/agents/params")
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def _world_fingerprint(self, boxes_raw: List[Dict[str, Any]]) -> str:
        """
        Fingerprint only the planning-relevant *world state*.
        If this doesn't change, we don't publish anything.

        IMPORTANT: do NOT include server_time.
        """
        canonical: List[Dict[str, Any]] = []

        for b in sorted(boxes_raw, key=lambda x: int(x["box_id"])):
            box_id = int(b["box_id"])
            item = {
                "box_id": box_id,
                "deadline": float(b.get("deadline", 0.0)),
                "disposed_X": bool(b.get("disposed_X", False)),
                "disposed_Y": bool(b.get("disposed_Y", False)),
                "sense_completed": [],
            }

            srs = list(b.get("sense_results", []))
            for sr in srs:
                if sr.get("status") != "completed":
                    continue
                prop = sr.get("property")
                if prop not in ("X", "Y"):
                    continue
                item["sense_completed"].append({
                    "agent_id": str(sr.get("agent_id", "")),
                    "property": str(prop),
                    "detected": bool(sr.get("detected", False)) if sr.get("detected", None) is not None else None,
                    "probability": float(sr.get("probability")) if isinstance(sr.get("probability"), (int, float)) else None,
                    # completed_at helps when a later sense overwrites belief
                    "completed_at": float(sr.get("completed_at", 0.0)) if isinstance(sr.get("completed_at"), (int, float)) else 0.0,
                })

            # sort sense results to be stable
            item["sense_completed"].sort(key=lambda x: (
                x["property"], x["agent_id"], float(x["completed_at"] or 0.0)
            ))

            canonical.append(item)

        s = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode("utf-8")).hexdigest()


    # ---- belief helpers ----
    
    @staticmethod
    def _p_present_from_sense_results(sense_results: List[Dict[str, Any]], prop: Property) -> float:
        best_sr = None
        best_t = -1.0
        for sr in sense_results:
            if sr.get("status") != "completed":
                continue
            if sr.get("property") != prop:
                continue
            t = float(sr.get("completed_at") or 0.0)
            if best_sr is None or t > best_t:
                best_t = t
                best_sr = sr
        if best_sr is None:
            return 0.5

        detected = best_sr.get("detected", None)
        prob = best_sr.get("probability", None)
        if detected is None or not isinstance(prob, (int, float)):
            return 0.5
        prob = float(prob)
        return max(0.0, min(1.0, prob if detected is True else (1.0 - prob)))
    


    @staticmethod
    def _p_present_from_sense_results_fused(
        sense_results: List[Dict[str, Any]],
        prop: Property,
        prior: float = 0.5,
    ) -> float:
        # log-odds of prior
        prior = max(1e-6, min(1.0 - 1e-6, float(prior)))
        L = math.log(prior / (1.0 - prior))

        for sr in sense_results:
            if sr.get("status") != "completed":
                continue
            if sr.get("property") != prop:
                continue

            detected = sr.get("detected", None)
            prob = sr.get("probability", None)
            if detected is None or not isinstance(prob, (int, float)):
                continue

            q = float(prob)
            # clamp confidence to avoid infinities
            q = max(1e-4, min(1.0 - 1e-4, q))

            # interpret measurement as "present" probability
            p_meas = q if bool(detected) else (1.0 - q)
            p_meas = max(1e-4, min(1.0 - 1e-4, p_meas))

            L += math.log(p_meas / (1.0 - p_meas))

        # back to probability
        p = 1.0 / (1.0 + math.exp(-L))
        return max(0.0, min(1.0, p))

    @staticmethod
    def _info_level_from_sense_results(sense_results: List[Dict[str, Any]], prop: Property) -> float:
        for sr in sense_results:
            if sr.get("status") == "completed" and sr.get("property") == prop:
                return 1.0
        return 0.0

    @staticmethod
    def _info_level_from_p(p_present: float) -> float:
        p = max(1e-6, min(1.0 - 1e-6, float(p_present)))
        conf = max(p, 1.0 - p)          # [0.5, 1.0]
        info = (conf - 0.5) / 0.5       # [0, 1]
        return max(0.0, min(1.0, info))


    # ---- build planner inputs ----
    def _build_agent_states(self, params: Optional[Dict[str, Any]]) -> List[AgentState]:
        agents_block = (params or {}).get("agents", {}) if isinstance(params, dict) else {}
        default_block = (params or {}).get("default", None) if isinstance(params, dict) else None

        out: List[AgentState] = []
        for aid in self.agent_ids:
            if aid == "human_a":
                canX, canY = True, False
            elif aid == "human_b":
                canX, canY = False, True
            else:
                canX, canY = True, True

            present_X, absent_X, present_Y, absent_Y = 0.8, 0.2, 0.8, 0.2
            raw = agents_block.get(aid, default_block) if isinstance(agents_block, dict) else default_block
            try:
                if raw:
                    present_X = float(raw["X"]["present"])
                    absent_X = float(raw["X"]["absent"])
                    present_Y = float(raw["Y"]["present"])
                    absent_Y = float(raw["Y"]["absent"])
            except Exception:
                pass

            out.append(
                AgentState(
                    agent_id=aid,
                    max_time=float(self.horizon),
                    can_sense_X=canX,
                    can_sense_Y=canY,
                    detect_present_X=present_X,
                    detect_absent_X=absent_X,
                    detect_present_Y=present_Y,
                    detect_absent_Y=absent_Y,
                )
            )
        return out

    def _build_boxes_info(self, boxes_raw: List[Dict[str, Any]]) -> List[BoxInfo]:
        out: List[BoxInfo] = []
        for b in boxes_raw:
            box_id = int(b["box_id"])
            sense_results = list(b.get("sense_results", []))

            pX = self._p_present_from_sense_results_fused(sense_results, "X")
            pY = self._p_present_from_sense_results_fused(sense_results, "Y")

            infoX = self._info_level_from_p(pX)
            infoY = self._info_level_from_p(pY)

            already_sensed: Dict[str, Dict[Property, bool]] = {aid: {"X": False, "Y": False} for aid in self.agent_ids}
            for sr in sense_results:
                if sr.get("status") != "completed":
                    continue
                aid = str(sr.get("agent_id", ""))
                prop = sr.get("property")
                if aid in already_sensed and prop in ("X", "Y"):
                    already_sensed[aid][prop] = True

            out.append(
                BoxInfo(
                    box_id=box_id,
                    deadline=float(b["deadline"]),
                    sense_time_X=float(b.get("sense_time_X", self.default_sense_time_X)),
                    sense_time_Y=float(b.get("sense_time_Y", self.default_sense_time_Y)),
                    dispose_time_X=float(b.get("dispose_time_X", self.default_dispose_time_X)),
                    dispose_time_Y=float(b.get("dispose_time_Y", self.default_dispose_time_Y)),
                    p_true_X=float(pX),
                    p_true_Y=float(pY),
                    disposed_X=bool(b["disposed_X"]),
                    disposed_Y=bool(b["disposed_Y"]),
                    info_X=float(infoX),
                    info_Y=float(infoY),
                    already_sensed=already_sensed,
                )
            )
        return out

    # ---- ROS callbacks ----
    def _on_agent_result(self, msg: StringMsg) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        success = bool(data.get("success", False))
        if not success:
            with self._state_lock:
                self._dirty = True
            return

        if self.replan_on_any_result:
            with self._state_lock:
                self._dirty = True

    # ---- plan publish ----
    def _publish_plan(self, plan: Plan, now_sim: float, world_fp: str) -> None:
        plan_id = f"world_{world_fp[:12]}"  # stable ID per world state
        payload = {
            "plan_id": plan_id,
            "world_fingerprint": world_fp,
            "server_time": now_sim,
            "agents": {}
        }
        
        for aid in self.agent_ids:
            steps = []
            for idx, (bid, prop, kind) in enumerate(plan.get(aid, []) or []):
                steps.append({"step_idx": idx, "box_id": int(bid), "property": str(prop), "kind": str(kind)})
            payload["agents"][aid] = steps

        now_wall = time.time()
        for aid in self.agent_ids:
            for (bid, prop, kind) in (plan.get(aid, []) or []):
                if kind != "dispose":
                    continue
                bid = int(bid)

                # If already locked by someone else, keep the existing lock
                # (optional: log it)
                if bid in self._dispose_locks and self._dispose_locks[bid]["agent_id"] != aid:
                    continue

                self._dispose_locks[bid] = {
                    "agent_id": aid,
                    "prop": str(prop),
                    "plan_id": plan_id,
                    "locked_at": now_wall,
                }


        self.pub_plan.publish(StringMsg(data=json.dumps(payload)))
        self._active_plan_id = plan_id

        # log summary
        parts = []
        for aid in self.agent_ids:
            acts = plan.get(aid, []) or []
            parts.append(aid + ": " + ", ".join([f"{k}:{bid}:{p}" for (bid, p, k) in acts]))
        self.get_logger().info(f"[PLAN] {plan_id} | " + " | ".join(parts))

    # ---- tick loop ----
    def _tick(self) -> None:
        with self._work_lock:
            if self._work_thread is not None and self._work_thread.is_alive():
                return
            self._work_thread = threading.Thread(target=self._worker_main, daemon=True)
            self._work_thread.start()

    def _worker_main(self) -> None:
        try:
            t = self._time()
            now_sim = float(t["server_time"])
            time_limit = float(t["time_limit_sec"])
            if now_sim >= time_limit:
                self.get_logger().info("[TIME] limit reached; shutting down")
                rclpy.shutdown()
                return

            with self._state_lock:
                if time.time() >= self._next_periodic:
                    self._dirty = True
                    self._next_periodic = time.time() + self.replan_period_sec
                dirty = bool(self._dirty)

            if not dirty:
                return

            boxes_raw = self._boxes_state()
            
            # Release lock when server shows box disposed (either flag => fully disposed in your domain)
            disposed_now = set()
            for b in boxes_raw:
                bid = int(b["box_id"])
                if bool(b.get("disposed_X", False)) or bool(b.get("disposed_Y", False)):
                    disposed_now.add(bid)

            # Drop any locks whose boxes are now disposed
            for bid in list(self._dispose_locks.keys()):
                if bid in disposed_now:
                    self._dispose_locks.pop(bid, None)

            
            box_xy: Dict[int, Tuple[float, float]] = {}
            for b in boxes_raw:
                try:
                    bid = int(b["box_id"])
                    box_xy[bid] = (float(b["x"]), float(b["y"]))
                except Exception:
                    continue

            def get_agent_xy(aid: str) -> Optional[Tuple[float, float]]:
                with self._pose_lock:
                    rec = self._agent_pose.get(aid)
                if not rec:
                    return None
                x, y, t_wall = rec
                if (time.time() - float(t_wall)) > self.pose_stale_sec:
                    return None
                return (float(x), float(y))

            def speed_for(aid: str) -> float:
                # optional per-agent speeds
                try:
                    v = float(self.agent_speeds_mps.get(aid, self.default_speed_mps))  # if you added agent_speeds_mps
                except Exception:
                    v = self.default_speed_mps
                return max(1e-6, v)

            def travel_time_fn(agent_id: str, box_id: int) -> float:
                xy = box_xy.get(int(box_id))
                if xy is None:
                    return 1e6  # unknown box => effectively infeasible
                ax = get_agent_xy(str(agent_id))
                if ax is None:
                    return float(self.unknown_pose_travel_sec)

                (bx, by) = xy
                (x, y) = ax
                dist = ((bx - x) ** 2 + (by - y) ** 2) ** 0.5
                return 0 #dist / speed_for(str(agent_id))

            
            world_fp = self._world_fingerprint(boxes_raw)

            # if world didn't change, do nothing (no replan publish)
            if self._last_world_fp == world_fp:
                return
            self._last_world_fp = world_fp

            
            params = self._agents_params()
            agents = self._build_agent_states(params)
            boxes = self._build_boxes_info(boxes_raw)

            # Build fixed in-flight disposals
            fixed_by_agent: Dict[str, List[Tuple[int, Property, ActionKind]]] = {aid: [] for aid in self.agent_ids}
            locked_boxes = set()

            for bid, rec in self._dispose_locks.items():
                aid = rec["agent_id"]
                prop = rec["prop"]
                fixed_by_agent[aid].append((int(bid), str(prop), "dispose"))
                locked_boxes.add(int(bid))

            # Remove locked boxes from optimization universe
            boxes_for_milp = [bx for bx in boxes if bx.box_id not in locked_boxes]



            plan = plan_assignments_gurobi(
                agents=agents,
                boxes=boxes_for_milp,
                current_time=now_sim,
                horizon=self.horizon,
                travel_time_fn=travel_time_fn,
                weights=self.weights,
            )

            # Prepend fixed disposals (and also ensure MILP didn't produce any disposal for locked boxes)
            for aid in self.agent_ids:
                # keep fixed first
                fixed_by_agent[aid].sort(key=lambda t: t[0])  # stable order if multiple
                plan[aid] = fixed_by_agent[aid] + (plan.get(aid, []) or [])

            self.get_logger().info(f"[PLAN] {plan}")

            if self._last_published_world_fp == world_fp:
                return

            # ... run optimizer to compute plan ...

            self._publish_plan(plan, now_sim, world_fp=world_fp)
            self._last_published_world_fp = world_fp


            with self._state_lock:
                self._dirty = False

        except Exception as e:
            self.get_logger().warn(f"[FAIL] central cycle failed: {e}")
        finally:
            with self._work_lock:
                self._work_thread = None


def main():
    rclpy.init()
    node = CentralJointOptimizerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

