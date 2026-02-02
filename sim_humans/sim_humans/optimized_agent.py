#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import requests
import rclpy
from rclpy.node import Node

import gurobipy as gp
from gurobipy import GRB


Property = Literal["X", "Y"]
TravelTimeFn = Callable[[str, int], float]  # (agent_id, box_id) -> seconds


# ---------------------------
# Planner data structures
# ---------------------------

@dataclass
class PlannerWeights:
    reward_correct_X: float = 1.0
    reward_correct_Y: float = 1.0
    weight_info: float = 0.2
    lambda_balance: float = 0.5
    info_threshold_for_dispose: float = 0.6

    prefer_exploration: float = 0.0
    lambda_deadline_risk: float = 0.0


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


# Plan for one agent: list of (box_id, prop, "sense"/"dispose")
Plan1 = List[Tuple[int, Property, str]]


# ---------------------------
# Single-agent optimizer (MILP)
# ---------------------------

def plan_single_agent_gurobi(
    agent: AgentState,
    boxes: List[BoxInfo],
    current_time: float,
    horizon: float,
    travel_time_fn: TravelTimeFn,
    weights: Optional[PlannerWeights] = None,
) -> Plan1:
    """
    Single-agent MILP:
      - role choice (sense vs dispose vs idle)
      - time budget
      - disposal requires info >= threshold
      - disposal requires prior sense (already_sensed by ANYONE) [same as your code]
      - within this solve: if we dispose (box,prop), we do NOT also sense (box,prop)
      - objective: expected correct disposals + info gain - X/Y imbalance - deadline risk + exploration knob
    """
    if weights is None:
        weights = PlannerWeights()

    props: List[Property] = ["X", "Y"]
    aid = agent.agent_id

    model = gp.Model("single_agent_box_planner")
    model.Params.OutputFlag = 0  # silent

    # Role decision variables (kept to mirror your multi-agent structure)
    z_sense = model.addVar(vtype=GRB.BINARY, name=f"z_sense_{aid}")
    z_dispose = model.addVar(vtype=GRB.BINARY, name=f"z_disp_{aid}")
    model.addConstr(z_sense + z_dispose <= 2, name=f"role_choice_{aid}")
    model.addConstr(z_sense == 1)
    model.addConstr(z_dispose == 1)

    # Decision vars
    s_vars: Dict[Tuple[int, Property], gp.Var] = {}
    d_vars: Dict[Tuple[int, Property], gp.Var] = {}

    # Create vars with feasibility filters (horizon + deadline)
    for b in boxes:
        for p in props:
            '''
            # skip disposed
            if p == "X" and b.disposed_X:
                continue
            if p == "Y" and b.disposed_Y:
                continue
            '''
            # NEW: once disposed for ANY property, treat as disposed for ALL
            if b.disposed_X or b.disposed_Y:
                continue

            # --- SENSE VAR ---
            can_sense = (p == "X" and agent.can_sense_X) or (p == "Y" and agent.can_sense_Y)
            already = b.already_sensed.get(aid, {}).get(p, False)

            if can_sense and not already:
                base_sense_time = b.sense_time_X if p == "X" else b.sense_time_Y
                travel = travel_time_fn(aid, b.box_id)
                total_t = base_sense_time + travel
                if total_t <= horizon and current_time + total_t <= b.deadline:
                    s_vars[(b.box_id, p)] = model.addVar(vtype=GRB.BINARY, name=f"sense_{aid}_{b.box_id}_{p}")

            # --- DISPOSE VAR ---
            p_true = b.p_true_X if p == "X" else b.p_true_Y
            if p_true < weights.info_threshold_for_dispose:
                continue


            base_disp_time = b.dispose_time_X if p == "X" else b.dispose_time_Y
            travel = travel_time_fn(aid, b.box_id)
            total_t = base_disp_time + travel
            if total_t <= horizon and current_time + total_t <= b.deadline:
                d_vars[(b.box_id, p)] = model.addVar(vtype=GRB.BINARY, name=f"disp_{aid}_{b.box_id}_{p}")

    BIG_M = 1000.0

    # (1) time budget
    load_expr = gp.LinExpr()
    for b in boxes:
        for p in props:
            sv = s_vars.get((b.box_id, p))
            dv = d_vars.get((b.box_id, p))
            travel = travel_time_fn(aid, b.box_id)

            if sv is not None:
                base = b.sense_time_X if p == "X" else b.sense_time_Y
                load_expr += (base + travel) * sv
            if dv is not None:
                base = b.dispose_time_X if p == "X" else b.dispose_time_Y
                load_expr += (base + travel) * dv

    model.addConstr(load_expr <= agent.max_time, name=f"time_budget_{aid}")

    # (2) role coupling (mirror your multi-agent)
    if s_vars:
        model.addConstr(gp.quicksum(s_vars.values()) <= BIG_M * z_sense, name=f"sense_role_{aid}")
    if d_vars:
        model.addConstr(gp.quicksum(d_vars.values()) <= BIG_M * z_dispose, name=f"disp_role_{aid}")

    '''
    # (3) disposal requires prior sense (by ANYONE historically)
    sensed_before: Dict[Tuple[int, Property], int] = {}
    for b in boxes:
        for p in props:
            sb = 0
            for _aid2, amap in (b.already_sensed or {}).items():
                if isinstance(amap, dict) and amap.get(p, False):
                    sb = 1
                    break
            sensed_before[(b.box_id, p)] = sb

    for b in boxes:
        for p in props:
            dv = d_vars.get((b.box_id, p))
            if dv is not None:
                model.addConstr(dv <= sensed_before[(b.box_id, p)], name=f"disp_requires_prior_sense_{b.box_id}_{p}")
    '''
    # (4) in THIS solve: if we dispose (box,prop) then do NOT sense (box,prop)
    # (single agent => this is enough; in multi-agent you also gated across agents)
    for b in boxes:
        for p in props:
            sv = s_vars.get((b.box_id, p))
            dv = d_vars.get((b.box_id, p))
            if sv is not None and dv is not None:
                model.addConstr(sv + dv <= 1, name=f"no_sense_and_disp_same_{b.box_id}_{p}")

    # (5) NEW: a box can be disposed at most once total (disposing any prop removes the object)
    for b in boxes:
        dvs = [d_vars[(b.box_id, p)] for p in props if (b.box_id, p) in d_vars]
        if dvs:
            model.addConstr(gp.quicksum(dvs) <= 1, name=f"one_dispose_total_{b.box_id}")


    # Objective
    total_reward = gp.LinExpr()
    totalX = gp.LinExpr()
    totalY = gp.LinExpr()

    box_by_id = {b.box_id: b for b in boxes}

    # disposal reward + deadline risk
    for (bid, p), dv in d_vars.items():
        b = box_by_id[bid]
        if p == "X":
            p_true = b.p_true_X
            val = weights.reward_correct_X
            base_disp_time = b.dispose_time_X
            totalX += p_true * dv
        else:
            p_true = b.p_true_Y
            val = weights.reward_correct_Y
            base_disp_time = b.dispose_time_Y
            totalY += p_true * dv

        total_reward += val * p_true * dv

        if weights.lambda_deadline_risk > 0.0:
            travel = travel_time_fn(aid, bid)
            finish_time = current_time + base_disp_time + travel
            slack = float(b.deadline) - float(finish_time)
            risk_coeff = max(0.0, -slack)
            if risk_coeff > 0.0:
                total_reward -= weights.lambda_deadline_risk * risk_coeff * dv

    # sensing reward (info gain)
    for (bid, p), sv in s_vars.items():
        b = box_by_id[bid]
        if p == "X":
            p_true = b.p_true_X
            info_level = b.info_X
            agent_quality = max(agent.detect_present_X - agent.detect_absent_X, 0.0)
        else:
            p_true = b.p_true_Y
            info_level = b.info_Y
            agent_quality = max(agent.detect_present_Y - agent.detect_absent_Y, 0.0)

        entropy_like = 4.0 * p_true * (1.0 - p_true)
        base_info_gain = (1.0 - info_level) * entropy_like
        info_gain = agent_quality * base_info_gain

        total_reward += weights.weight_info * info_gain * sv
        if weights.prefer_exploration != 0.0:
            total_reward += weights.prefer_exploration * sv

    # X/Y balance penalty
    d_imb = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="d_imbalance")
    model.addConstr(totalX - totalY <= d_imb, name="balance_pos")
    model.addConstr(totalY - totalX <= d_imb, name="balance_neg")
    total_reward -= weights.lambda_balance * d_imb

    model.setObjective(total_reward, GRB.MAXIMIZE)
    model.optimize()

    # Extract actions: sort by deadline (dispose before sense on ties)
    out: Plan1 = []
    if model.Status == GRB.OPTIMAL:
        for (bid, p), sv in s_vars.items():
            if sv.X > 0.5:
                out.append((bid, p, "sense"))
        for (bid, p), dv in d_vars.items():
            if dv.X > 0.5:
                out.append((bid, p, "dispose"))

        def sort_key(act: Tuple[int, Property, str]):
            bid, p, kind = act
            b = box_by_id.get(bid)
            deadline = b.deadline if b is not None else 1e12
            kind_rank = 0 if kind == "dispose" else 1
            return (deadline, kind_rank, bid, 0 if p == "X" else 1)

        out.sort(key=sort_key)

    return out


# ---------------------------
# Server-facing state
# ---------------------------

@dataclass
class BoxSummary:
    box_id: int
    x: float
    y: float
    deadline: float
    disposed_X: bool
    disposed_Y: bool
    sense_results: List[Dict[str, Any]]


@dataclass
class Pose2D:
    x: float
    y: float


# ---------------------------
# Single-agent optimizer node
# ---------------------------

class SingleAgentOptimizerNode(Node):
    """
    A one-agent “optimizer controller” that:
      - polls server (/time, /boxes/state)
      - builds beliefs/info from sense_results
      - plans using Gurobi MILP
      - executes ONE planned action at a time (travel + /sense or /dispose)
      - supports /sense/cancel and /dispose/cancel, plus local cancel event
    """

    def __init__(self):
        super().__init__("single_agent_optimizer")

        # ---- params (mirror your SimHumanAgent style) ----
        self.declare_parameter("agent_id", "robot")
        self.declare_parameter("server_base_url", "http://172.17.40.64:8080")
        self.declare_parameter("request_timeout_sec", 120.0)

        self.declare_parameter("speed_mps", 1.0)
        self.declare_parameter("tick_period_sec", 0.5)

        # planning horizon / budget
        self.declare_parameter("horizon_sec", 60.0)
        self.declare_parameter("max_time_sec", 60.0)

        # capabilities
        self.declare_parameter("can_sense_X", True)
        self.declare_parameter("can_sense_Y", True)

        # default action times if your server doesn't expose them elsewhere
        self.declare_parameter("default_sense_time_X", 3.0)
        self.declare_parameter("default_sense_time_Y", 3.0)
        self.declare_parameter("default_dispose_time_X", 4.0)
        self.declare_parameter("default_dispose_time_Y", 4.0)

        # weights (PlannerWeights)
        self.declare_parameter("reward_correct_X", 1.0)
        self.declare_parameter("reward_correct_Y", 1.0)
        self.declare_parameter("weight_info", 0.2)
        self.declare_parameter("lambda_balance", 0.5)
        self.declare_parameter("info_threshold_for_dispose", 0.6)
        self.declare_parameter("prefer_exploration", 0.0)
        self.declare_parameter("lambda_deadline_risk", 0.0)

        # behavior
        self.declare_parameter("execute_dispose_only_when_sensed_before_anyone", False)
        self.declare_parameter("execute_one_action_per_tick", True)

        # ---- resolved params ----
        self.agent_id = str(self.get_parameter("agent_id").value)
        self.base_url = str(self.get_parameter("server_base_url").value).rstrip("/")
        self.timeout = float(self.get_parameter("request_timeout_sec").value)

        self.speed_mps = float(self.get_parameter("speed_mps").value)
        self.tick_period = float(self.get_parameter("tick_period_sec").value)

        self.horizon = float(self.get_parameter("horizon_sec").value)
        self.max_time = float(self.get_parameter("max_time_sec").value)

        self.can_sense_X = bool(self.get_parameter("can_sense_X").value)
        self.can_sense_Y = bool(self.get_parameter("can_sense_Y").value)

        self.default_sense_time_X = float(self.get_parameter("default_sense_time_X").value)
        self.default_sense_time_Y = float(self.get_parameter("default_sense_time_Y").value)
        self.default_dispose_time_X = float(self.get_parameter("default_dispose_time_X").value)
        self.default_dispose_time_Y = float(self.get_parameter("default_dispose_time_Y").value)

        self.execute_dispose_only_when_sensed_before_anyone = bool(
            self.get_parameter("execute_dispose_only_when_sensed_before_anyone").value
        )
        self.execute_one_action_per_tick = bool(self.get_parameter("execute_one_action_per_tick").value)

        self.weights = PlannerWeights(
            reward_correct_X=float(self.get_parameter("reward_correct_X").value),
            reward_correct_Y=float(self.get_parameter("reward_correct_Y").value),
            weight_info=float(self.get_parameter("weight_info").value),
            lambda_balance=float(self.get_parameter("lambda_balance").value),
            info_threshold_for_dispose=float(self.get_parameter("info_threshold_for_dispose").value),
            prefer_exploration=float(self.get_parameter("prefer_exploration").value),
            lambda_deadline_risk=float(self.get_parameter("lambda_deadline_risk").value),
        )

        # ---- internal state ----
        self.pose = Pose2D(0.0, 0.0)

        self._busy_lock = threading.Lock()
        self._busy = False

        self._plan_lock = threading.Lock()
        self._current_plan: Plan1 = []
        self._last_plan_at_sim: Optional[float] = None

        # current server op (for cancel)
        self._op_lock = threading.Lock()
        self._current_op: Optional[Dict[str, Any]] = None

        self._cancel_lock = threading.Lock()
        self._cancel_evt: Optional[threading.Event] = None

        # worker thread so ROS timer stays responsive
        self._work_lock = threading.Lock()
        self._work_thread: Optional[threading.Thread] = None

        self._timer = self.create_timer(self.tick_period, self._tick)

        self.get_logger().info(
            f"SingleAgentOptimizerNode up agent_id={self.agent_id} server={self.base_url} "
            f"horizon={self.horizon}s max_time={self.max_time}s can_sense_X={self.can_sense_X} can_sense_Y={self.can_sense_Y}"
        )

    # ---------------------------
    # Concurrency helpers
    # ---------------------------

    def _clear_plan(self, why: str = "") -> None:
        with self._plan_lock:
            self._current_plan = []
            self._last_plan_at_sim = None
        if why:
            self.get_logger().info(f"[PLAN] cleared ({why})")


    def _set_busy(self, v: bool) -> None:
        with self._busy_lock:
            self._busy = bool(v)

    def _is_busy(self) -> bool:
        with self._busy_lock:
            return bool(self._busy)

    def _new_cancel_evt(self) -> threading.Event:
        with self._cancel_lock:
            self._cancel_evt = threading.Event()
            return self._cancel_evt

    def _cancel_evt_get(self) -> Optional[threading.Event]:
        with self._cancel_lock:
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

    # ---------------------------
    # HTTP helpers (same pattern as SimHumanAgent)
    # ---------------------------

    def _http(self, method: str, path: str, *, json_body: Optional[dict] = None) -> requests.Response:
        url = self.base_url + path
        if method == "GET":
            return requests.get(url, timeout=self.timeout)
        if method == "POST":
            return requests.post(url, json=json_body, timeout=self.timeout)
        raise ValueError(f"Unsupported method: {method}")

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

    def _agents_params(self) -> Optional[Dict[str, Any]]:
        try:
            r = self._http("GET", "/agents/params")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            self.get_logger().warn(f"[WARN] /agents/params failed: {e}")
            return None

    def _sense(self, box_id: int, prop: Property) -> Dict[str, Any]:
        r = self._http("POST", "/sense", json_body={"agent_id": self.agent_id, "box_id": box_id, "property": prop})
        r.raise_for_status()
        return r.json()

    def _dispose(self, box_id: int, prop: Property) -> Dict[str, Any]:
        r = self._http("POST", "/dispose", json_body={"agent_id": self.agent_id, "box_id": box_id, "property": prop})
        r.raise_for_status()
        return r.json()

    def _cancel_current_server_op(self) -> bool:
        op = self._get_current_op()
        if not op:
            return False
        kind = str(op["kind"])
        box_id = int(op["box_id"])
        prop = str(op["prop"])

        try:
            if kind == "sense":
                r = self._http("POST", "/sense/cancel", json_body={"agent_id": self.agent_id, "box_id": box_id, "property": prop})
            elif kind == "dispose":
                r = self._http("POST", "/dispose/cancel", json_body={"agent_id": self.agent_id, "box_id": box_id, "property": prop})
            else:
                return False

            if r.status_code == 200:
                try:
                    st = r.json().get("status")
                except Exception:
                    st = None
                self.get_logger().info(f"[CANCEL] {kind} box={box_id} prop={prop} -> {st}")
                return True
        except Exception as e:
            self.get_logger().warn(f"[CANCEL] failed: {e}")
        return False

    def request_preempt(self, why: str = "") -> None:
        self.get_logger().info(f"[PREEMPT] {why}")
        self._cancel_current_server_op()
        ev = self._cancel_evt_get()
        if isinstance(ev, threading.Event):
            ev.set()

    # ---------------------------
    # Belief + info extraction (server semantics match your SimHumanAgent)
    # ---------------------------

    @staticmethod
    def _p_present_from_sense_results(sense_results: List[Dict[str, Any]], prop: Property) -> float:
        best_sr = None
        best_t = -1.0
        for sr in sense_results:
            if sr.get("status") != "completed":
                continue
            if sr.get("property") != prop:
                continue
            tv = sr.get("completed_at")
            t = float(tv) if isinstance(tv, (int, float)) else 0.0
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
        p_present = prob if detected is True else (1.0 - prob)
        return max(0.0, min(1.0, p_present))

    @staticmethod
    def _info_level_from_sense_results(sense_results: List[Dict[str, Any]], prop: Property) -> float:
        # Simple, robust: 0 if none completed; else 1
        for sr in sense_results:
            if sr.get("status") == "completed" and sr.get("property") == prop:
                return 1.0
        return 0.0

    def _already_sensed_map_anyone(self, boxes: List[BoxSummary]) -> Dict[int, Dict[Property, bool]]:
        out: Dict[int, Dict[Property, bool]] = {}
        for b in boxes:
            out[b.box_id] = {"X": False, "Y": False}
            for sr in b.sense_results:
                if sr.get("status") != "completed":
                    continue
                p = sr.get("property")
                if p in ("X", "Y"):
                    out[b.box_id][p] = True
        return out

    def _already_sensed_map_me(self, boxes: List[BoxSummary]) -> Dict[int, Dict[Property, bool]]:
        out: Dict[int, Dict[Property, bool]] = {}
        for b in boxes:
            out[b.box_id] = {"X": False, "Y": False}
            for sr in b.sense_results:
                if sr.get("status") != "completed":
                    continue
                if str(sr.get("agent_id", "")) != self.agent_id:
                    continue
                p = sr.get("property")
                if p in ("X", "Y"):
                    out[b.box_id][p] = True
        return out

    # ---------------------------
    # Travel model
    # ---------------------------

    def _dist_to(self, x: float, y: float) -> float:
        return math.hypot(x - self.pose.x, y - self.pose.y)

    def _travel_time_sec(self, box: BoxSummary) -> float:
        dist = self._dist_to(box.x, box.y)
        return dist / max(1e-6, self.speed_mps)

    def _travel_to(self, box: BoxSummary, cancel_evt: Optional[threading.Event]) -> bool:
        dt = self._travel_time_sec(box)
        self.get_logger().info(
            f"[TRAVEL] start box={box.box_id} from=({self.pose.x:.2f},{self.pose.y:.2f}) "
            f"to=({box.x:.2f},{box.y:.2f}) t={dt:.2f}s"
        )

        end = time.time() + dt
        while time.time() < end:
            if cancel_evt is not None and cancel_evt.is_set():
                self.get_logger().warn("[TRAVEL] cancelled")
                return False
            time.sleep(0.05)

        self.pose = Pose2D(box.x, box.y)
        self.get_logger().info(f"[TRAVEL] done  box={box.box_id} now=({self.pose.x:.2f},{self.pose.y:.2f})")
        return True

    # ---------------------------
    # Build planner inputs from server state
    # ---------------------------

    def _build_agent_state(self) -> AgentState:
        # Use /agents/params like your SimHumanAgent to get detection qualities
        present_X = 0.8
        absent_X = 0.2
        present_Y = 0.8
        absent_Y = 0.2

        params = self._agents_params()
        agents = (params or {}).get("agents", {})
        default = (params or {}).get("default", None)

        raw = agents.get(self.agent_id, default) if isinstance(agents, dict) else default
        try:
            if raw:
                present_X = float(raw["X"]["present"])
                absent_X = float(raw["X"]["absent"])
                present_Y = float(raw["Y"]["present"])
                absent_Y = float(raw["Y"]["absent"])
        except Exception:
            pass

        return AgentState(
            agent_id=self.agent_id,
            max_time=float(self.max_time),
            can_sense_X=bool(self.can_sense_X),
            can_sense_Y=bool(self.can_sense_Y),
            detect_present_X=present_X,
            detect_absent_X=absent_X,
            detect_present_Y=present_Y,
            detect_absent_Y=absent_Y,
        )

    def _build_boxes_info(self, boxes: List[BoxSummary]) -> List[BoxInfo]:
        # Historical sensed-before for the “dispose requires prior sense” constraint
        sensed_any = self._already_sensed_map_anyone(boxes)
        sensed_me = self._already_sensed_map_me(boxes)

        out: List[BoxInfo] = []
        for b in boxes:
            pX = self._p_present_from_sense_results(b.sense_results, "X")
            pY = self._p_present_from_sense_results(b.sense_results, "Y")
            infoX = self._info_level_from_sense_results(b.sense_results, "X")
            infoY = self._info_level_from_sense_results(b.sense_results, "Y")

            already_sensed: Dict[str, Dict[Property, bool]] = {
                # for the constraint, we care about “anyone sensed before”
                # so we store a synthetic “anyone” plus “me”
                "anyone": {"X": bool(sensed_any[b.box_id]["X"]), "Y": bool(sensed_any[b.box_id]["Y"])},
                self.agent_id: {"X": bool(sensed_me[b.box_id]["X"]), "Y": bool(sensed_me[b.box_id]["Y"])},
            }

            # If you want to make the prior-sense gate EXACTLY “anyone”, you can later
            # interpret it by scanning all ids, same as your original optimizer does.

            out.append(
                BoxInfo(
                    box_id=b.box_id,
                    deadline=float(b.deadline),
                    sense_time_X=float(self.default_sense_time_X),
                    sense_time_Y=float(self.default_sense_time_Y),
                    dispose_time_X=float(self.default_dispose_time_X),
                    dispose_time_Y=float(self.default_dispose_time_Y),
                    p_true_X=float(pX),
                    p_true_Y=float(pY),
                    disposed_X=bool(b.disposed_X),
                    disposed_Y=bool(b.disposed_Y),
                    info_X=float(infoX),
                    info_Y=float(infoY),
                    already_sensed=already_sensed,
                )
            )

        return out

    # ---------------------------
    # Execution
    # ---------------------------

    def _execute_action(self, act: Tuple[int, Property, str], boxes: List[BoxSummary], now_sim: float) -> None:
        box_id, prop, kind = act
        box = next((bb for bb in boxes if bb.box_id == box_id), None)
        if box is None:
            self.get_logger().warn(f"[EXEC] unknown box_id={box_id}")
            return

        cancel_evt = self._new_cancel_evt()
        self._set_busy(True)
        try:
            ok = self._travel_to(box, cancel_evt)
            if not ok:
                return

            if kind == "sense":
                # skip if already disposed for that prop
                '''
                if (prop == "X" and box.disposed_X) or (prop == "Y" and box.disposed_Y):
                    self.get_logger().info(f"[EXEC] skip sense box={box_id} prop={prop} (already disposed)")
                    return
                '''
                # NEW: once disposed for ANY property, treat as disposed for ALL
                if box.disposed_X or box.disposed_Y:
                    self.get_logger().info(f"[EXEC] skip sense box={box_id} prop={prop} (already disposed-any)")
                    return

                self._set_current_op("sense", box_id, prop, now_sim)
                js = self._sense(box_id, prop)
                self.get_logger().info(
                    f"[SENSE] box={box_id} prop={prop} status={js.get('status')} detected={js.get('detected')} prob={js.get('probability')}"
                )

            elif kind == "dispose":
                if self.execute_dispose_only_when_sensed_before_anyone:
                    sensed_any = any(
                        sr.get("status") == "completed" and sr.get("property") in ("X", "Y")
                        for sr in box.sense_results
                    )
                    if not sensed_any:
                        self.get_logger().warn(
                            f"[EXEC] block dispose box={box_id} (no prior sense of any prop by anyone)"
                        )
                        return


                '''
                if (prop == "X" and box.disposed_X) or (prop == "Y" and box.disposed_Y):
                    self.get_logger().info(f"[EXEC] skip dispose box={box_id} prop={prop} (already disposed)")
                    return
                '''
                
                # NEW: once disposed for ANY property, treat as disposed for ALL
                if box.disposed_X or box.disposed_Y:
                    self.get_logger().info(f"[EXEC] skip dispose box={box_id} prop={prop} (already disposed-any)")
                    return

                self._set_current_op("dispose", box_id, prop, now_sim)
                js = self._dispose(box_id, prop)
                self.get_logger().info(
                    f"[DISPOSE] box={box_id} prop={prop} status={js.get('status')} success={js.get('success')}"
                )

            else:
                self.get_logger().warn(f"[EXEC] unknown kind={kind}")

        finally:
            self._clear_current_op()
            self._set_busy(False)

            # NEW: always replan after finishing an action
            self._clear_plan(why=f"finished {kind} box={box_id} prop={prop}")


    # ---------------------------
    # Planning + tick loop
    # ---------------------------

    def _travel_time_fn_factory(self, boxes: List[BoxSummary]) -> TravelTimeFn:
        box_by_id = {b.box_id: b for b in boxes}

        def fn(agent_id: str, box_id: int) -> float:
            b = box_by_id.get(int(box_id))
            if b is None:
                return 1e6
            # IMPORTANT: planner travel uses CURRENT pose snapshot
            return self._travel_time_sec(b)

        return fn

    def _replan_if_needed(self, now_sim: float, boxes: List[BoxSummary]) -> None:
        with self._plan_lock:
            # if we still have queued actions, keep them
            if self._current_plan:
                return

        agent = self._build_agent_state()
        boxes_info = self._build_boxes_info(boxes)
        travel_fn = self._travel_time_fn_factory(boxes)

        plan = plan_single_agent_gurobi(
            agent=agent,
            boxes=boxes_info,
            current_time=now_sim,
            horizon=self.horizon,
            travel_time_fn=travel_fn,
            weights=self.weights,
        )

        with self._plan_lock:
            self._current_plan = list(plan)
            self._last_plan_at_sim = float(now_sim)

        if plan:
            self.get_logger().info(f"[PLAN] {len(plan)} actions: " + ", ".join([f"{k}:{bid}:{p}" for (bid, p, k) in plan]))
        else:
            self.get_logger().info("[PLAN] empty (idle)")

    def _pop_next_action(self) -> Optional[Tuple[int, Property, str]]:
        with self._plan_lock:
            if not self._current_plan:
                return None
            return self._current_plan.pop(0)

    def _tick(self) -> None:
        # keep ROS timer lightweight: spawn a worker if none running
        with self._work_lock:
            if self._work_thread is not None and self._work_thread.is_alive():
                return
            th = threading.Thread(target=self._worker_main, daemon=True)
            self._work_thread = th
            th.start()

    def _worker_main(self) -> None:
        try:
            t = self._time()
            now_sim = float(t["server_time"])
            time_limit = float(t["time_limit_sec"])
            if now_sim >= time_limit:
                self.get_logger().info(f"[TIME] limit reached {now_sim:.2f} >= {time_limit:.2f}, shutting down")
                rclpy.shutdown()
                return

            # if currently busy, do nothing (unless you want to preempt externally)
            if self._is_busy():
                return

            boxes = self._boxes_state()

            # (re)plan if needed
            self._replan_if_needed(now_sim, boxes)

            # execute next action
            act = self._pop_next_action()
            if act is None:
                return

            self._execute_action(act, boxes, now_sim)

            # If you want to execute multiple actions per tick, loop here
            if not self.execute_one_action_per_tick:
                while True:
                    if self._is_busy():
                        break
                    act2 = self._pop_next_action()
                    if act2 is None:
                        break
                    # refresh time/boxes between actions for correctness
                    t2 = self._time()
                    now2 = float(t2["server_time"])
                    boxes2 = self._boxes_state()
                    self._execute_action(act2, boxes2, now2)

        except Exception as e:
            self.get_logger().warn(f"[FAIL] worker cycle failed: {e}")
        finally:
            with self._work_lock:
                self._work_thread = None


def main():
    rclpy.init()
    node = SingleAgentOptimizerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

