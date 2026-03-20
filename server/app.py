#!/usr/bin/env python3
import os
import sys
import asyncio
import random
from datetime import datetime
from typing import Literal, Optional, List, Dict, Tuple
from threading import Lock
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Boolean,
    Integer,
    ForeignKey,
    Index,
    func,
    case
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
import json
from typing import Any

from dataclasses import dataclass, field
import contextlib
from collections import defaultdict


# ---------------------------------
# DB setup & sim time
# ---------------------------------

RESUME_DB = ("--resume" in sys.argv) or (os.getenv("BOXES_RESUME", "0") == "1")

DB_PATH = "boxes.db"

LOG_DIR = Path(os.getenv("BOXES_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

FINAL_LIVE_PRINTED: bool = False


# If not resuming, delete any existing DB file so we start fresh
if not RESUME_DB and os.path.exists(DB_PATH):
    os.remove(DB_PATH)


engine = create_engine("sqlite:///boxes.db", future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

SERVER_START = datetime.utcnow()
TIME_LIMIT_SEC = 600


# --- time freeze / scoring state ---
_TIME_LOCK = Lock()
FROZEN_TIME: Optional[float] = None   # sim-time value once time is up (typically TIME_LIMIT_SEC)
TIME_UP: bool = False
FINAL_SCORE: Optional[dict] = None    # computed once at time-up

# Optional: control any manual prints in endpoints
ENABLE_STATE_PRINTS = False

# --- dynamic end condition: last box deadline (sim-time) ---
FINAL_DEADLINE_SIM: Optional[float] = None

# --- watchdog task to guarantee final print even if no requests come in ---
_FINALIZE_WATCHDOG_TASK: Optional[asyncio.Task] = None


# ---------------------------------
# Agent detection parameters
# ---------------------------------
# Each agent has per-property probabilities:
#   present: P(detect=True | property actually present)
#   absent:  P(detect=True | property actually absent)  (false-positive rate)



# Fallback for unknown agents (if any)
DEFAULT_AGENT_PARAMS = {
    "X": {"present": 0.75, "absent": 0.25},
    "Y": {"present": 0.75, "absent": 0.25},
}


# ---------------------------------
# Hardcoded scenario
# ---------------------------------
USE_HARDCODED_SCENARIO = True
SCENARIO_ID = "hard_10_v1"

# deadlines are offsets from scenario start time (seconds)
# has_X/has_Y define the ground truth
# times are durations (seconds)


# ---------------------------------
# 2) Add release_time to SCENARIO_BOXES
#    (only adding the field; you can tune values later)
# ---------------------------------
SCENARIO_BOXES = {
    1: {"release_time": 0,   "has_X": True,  "has_Y": False, "deadline_offset": 140,
        "sense_X": 18, "sense_Y": 18, "disp_X": 220, "disp_Y": 220,
        "senseable": ["X"]},

    2: {"release_time": 0,   "has_X": False, "has_Y": True,  "deadline_offset": 190,
        "sense_X": 18, "sense_Y": 18, "disp_X": 220, "disp_Y": 220,
        "senseable": ["Y"]},

    3: {"release_time": 25,  "has_X": False, "has_Y": False, "deadline_offset": 175,
        "sense_X": 15, "sense_Y": 15, "disp_X": 120, "disp_Y": 120,
        "senseable": ["X"]},

    4: {"release_time": 35,  "has_X": False, "has_Y": False, "deadline_offset": 180,
        "sense_X": 15, "sense_Y": 15, "disp_X": 120, "disp_Y": 120,
        "senseable": ["Y"]},

    5: {"release_time": 140, "has_X": False, "has_Y": True,  "deadline_offset": 260,
        "sense_X": 20, "sense_Y": 22, "disp_X": 240, "disp_Y": 240,
        "senseable": ["Y"]},

    6: {"release_time": 140, "has_X": True,  "has_Y": False, "deadline_offset": 300,
        "sense_X": 22, "sense_Y": 20, "disp_X": 240, "disp_Y": 240,
        "senseable": ["X"]},

    7: {"release_time": 140, "has_X": False, "has_Y": False, "deadline_offset": 275,
        "sense_X": 18, "sense_Y": 18, "disp_X": 90,  "disp_Y": 90,
        "senseable": ["X", "Y"]},

    # --- DILEMMA BOX: needs BOTH X and Y, tight deadline ---
    8: {"release_time": 165, "has_X": True,  "has_Y": True,  "deadline_offset": 215,
        "sense_X": 12, "sense_Y": 12, "disp_X": 160, "disp_Y": 160,
        "senseable": ["X", "Y"]},

    9: {"release_time": 300, "has_X": True,  "has_Y": False, "deadline_offset": 430,
        "sense_X": 20, "sense_Y": 20, "disp_X": 260, "disp_Y": 260,
        "senseable": ["X"]},

    10: {"release_time": 340, "has_X": False, "has_Y": True,  "deadline_offset": 470,
         "sense_X": 20, "sense_Y": 20, "disp_X": 260, "disp_Y": 260,
         "senseable": ["Y"]},
}



AGENT_DETECTION_PARAMS = {
    "robot": {
        "X": {"present": 0.92, "absent": 0.06},  # very reliable
        "Y": {"present": 0.92, "absent": 0.06},
    },
    "human_a": {  # can sense only X
        "X": {"present": 0.82, "absent": 0.18},  # okay but noisier than robot
        "Y": {"present": 0.00, "absent": 1.00},  # unused / impossible
    },
    "human_b": {  # can sense only Y
        "X": {"present": 0.00, "absent": 1.00},  # unused / impossible
        "Y": {"present": 0.82, "absent": 0.18},
    },
}



objects_pre_data = {
    1: {"x": 3.0, "y": -1.0},
    2: {"x": 0.0, "y": -1.0},
    3: {"x": 1.0, "y": 1.0},
    4: {"x": -1.0, "y": 2.0},
    5: {"x": 3.0, "y": 1.0},
    6: {"x": 4.0, "y": 0.5},
    7: {"x": 2.0, "y": -1.0},
    8: {"x": 3.0, "y": -3.0},
    9: {"x": 4.0, "y": -5.0},
    10: {"x": 6.0, "y": -4.5},
}

def sim_time() -> float:
    global FROZEN_TIME, TIME_UP, FINAL_DEADLINE_SIM

    elapsed = (datetime.utcnow() - SERVER_START).total_seconds()

    # freeze at last box deadline if known; else fallback
    limit = float(FINAL_DEADLINE_SIM) if FINAL_DEADLINE_SIM is not None else float(TIME_LIMIT_SEC)

    should_finalize = False
    with _TIME_LOCK:
        if FROZEN_TIME is not None:
            return float(FROZEN_TIME)

        if elapsed >= limit:
            FROZEN_TIME = float(limit)
            TIME_UP = True
            should_finalize = True

    if should_finalize:
        maybe_finalize_time_and_score()

    return float(elapsed)




def compute_live_score(db) -> dict:
    """
    Live scoreboard (not just final):
      - senses completed per agent
      - sensing accuracy per agent (detected == ground truth)
      - disposals completed per agent
      - disposal correctness per agent:
          * correct_on_time  (== success True in your logic)
          * wrong_property   (disposed when property not present, even if on time)
          * late             (completed after deadline, regardless of property)
      - totals
    """

    # --------
    # Sensing
    # --------
    # Accuracy: compare SenseResult.detected to Box.has_X/has_Y
    sensed_rows = (
        db.query(
            SenseResult.agent_id.label("agent"),
            func.count().label("sensed_completed"),
            func.sum(
                case(
                    # correct if detected == ground truth presence
                    (
                        case(
                            (SenseResult.property == "X", Box.has_X),
                            else_=Box.has_Y,
                        ) == SenseResult.detected,
                        1,
                    ),
                    else_=0,
                )
            ).label("sensed_correct"),
        )
        .join(Box, Box.id == SenseResult.box_id)
        .filter(SenseResult.status == "completed")
        .group_by(SenseResult.agent_id)
        .all()
    )

    sense_by_agent = {}
    for r in sensed_rows:
        total = int(r.sensed_completed or 0)
        correct = int(r.sensed_correct or 0)
        sense_by_agent[str(r.agent)] = {
            "completed": total,
            "correct": correct,
            "accuracy": (correct / total) if total > 0 else None,
        }

    # ---------
    # Disposal (RAW + CREDITED)
    # ---------

    # RAW: counts every completed disposal record (your current behavior)
    disp_by_agent_raw: Dict[str, Dict[str, int]] = {}
    completed_disposals = (
        db.query(DisposalResult, Box)
        .join(Box, Box.id == DisposalResult.box_id)
        .filter(DisposalResult.status == "completed")
        .all()
    )

    for dr, box in completed_disposals:
        a = str(dr.agent_id)
        disp_by_agent_raw.setdefault(a, {
            "completed": 0,
            "correct_on_time": 0,
            "wrong_property": 0,
            "late": 0,
        })

        disp_by_agent_raw[a]["completed"] += 1

        prop = str(dr.property)
        prop_present = bool(box.has_X) if prop == "X" else bool(box.has_Y)
        late = (dr.completed_at is not None) and (float(dr.completed_at) > float(box.deadline))

        if late:
            disp_by_agent_raw[a]["late"] += 1
        if not prop_present:
            disp_by_agent_raw[a]["wrong_property"] += 1
        if dr.success is True:
            disp_by_agent_raw[a]["correct_on_time"] += 1

    # CREDITED: dedup scoring so each (box_id,prop) can only earn points once.
    # Rule: if multiple on-time successes exist, earliest completed_at gets the credit.
    disp_by_agent_credited: Dict[str, Dict[str, int]] = {}
    credited_pairs: set[tuple[int, str]] = set()

    # Consider only disposals that claim success, and verify on-time with box.deadline.
    success_candidates = [
        (dr, box) for (dr, box) in completed_disposals
        if dr.success is True and dr.completed_at is not None
           and float(dr.completed_at) <= float(box.deadline)
    ]
    success_candidates.sort(key=lambda pair: float(pair[0].completed_at))  # earliest wins

    for dr, box in success_candidates:
        key = (int(dr.box_id), str(dr.property))
        if key in credited_pairs:
            continue

        credited_pairs.add(key)
        a = str(dr.agent_id)
        disp_by_agent_credited.setdefault(a, {"credited_success_on_time": 0})
        disp_by_agent_credited[a]["credited_success_on_time"] += 1

    # Totals
    total_disposed_raw = sum(v["completed"] for v in disp_by_agent_raw.values())
    total_correct_raw = sum(v["correct_on_time"] for v in disp_by_agent_raw.values())
    total_correct_credited = len(credited_pairs)


    # -----------------------
    # NEW: per-agent itemized lists
    # -----------------------
    sensed_items_by_agent: Dict[str, List[Dict[str, Any]]] = {}
    sensed_items = (
        db.query(SenseResult, Box)
        .join(Box, Box.id == SenseResult.box_id)
        .filter(SenseResult.status == "completed")
        .order_by(SenseResult.completed_at.asc())
        .all()
    )
    for sr, box in sensed_items:
        a = str(sr.agent_id)
        sensed_items_by_agent.setdefault(a, []).append({
            "box_id": int(sr.box_id),
            "prop": str(sr.property),
            "detected": bool(sr.detected) if sr.detected is not None else None,
            "probability": float(sr.probability) if sr.probability is not None else None,
            "completed_at": float(sr.completed_at) if sr.completed_at is not None else None,
            # optional: include ground truth so you can inspect correctness quickly
            "truth_present": bool(box.has_X) if str(sr.property) == "X" else bool(box.has_Y),
        })

    disposed_items_by_agent: Dict[str, List[Dict[str, Any]]] = {}
    disposed_items = (
        db.query(DisposalResult, Box)
        .join(Box, Box.id == DisposalResult.box_id)
        .filter(DisposalResult.status == "completed")
        .order_by(DisposalResult.completed_at.asc())
        .all()
    )
    for dr, box in disposed_items:
        a = str(dr.agent_id)
        prop = str(dr.property)
        prop_present = bool(box.has_X) if prop == "X" else bool(box.has_Y)
        late = (dr.completed_at is not None) and (dr.completed_at > box.deadline)
        wrong_property = (not prop_present)

        disposed_items_by_agent.setdefault(a, []).append({
            "box_id": int(dr.box_id),
            "prop": prop,
            "success": bool(dr.success) if dr.success is not None else None,
            "completed_at": float(dr.completed_at) if dr.completed_at is not None else None,
            "late": bool(late),
            "wrong_property": bool(wrong_property),
            "deadline": float(box.deadline),
            # optional: include truth
            "truth_present": bool(prop_present),
        })

    # Totals (use the ones we already computed)
    total_sensed = sum(v["completed"] for v in sense_by_agent.values())

    # -----------------------
    # NEW: per-box max disposal attempt duration (any prop, any outcome)
    # Helpers now overlap across properties (same box_id).
    # -----------------------
    per_box_max_dispose_attempt: Dict[int, Dict[str, Any]] = {}
    attempts_by_box: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    disp_attempts = (
        db.query(DisposalResult, Box)
        .join(Box, Box.id == DisposalResult.box_id)
        .filter(DisposalResult.status.in_(["completed", "cancelled"]))
        .all()
    )

    for dr, box in disp_attempts:
        box_id = int(dr.box_id)
        prop = str(dr.property)

        start_t = dr.started_at if dr.started_at is not None else dr.requested_at
        end_t = dr.completed_at if dr.status == "completed" else dr.cancelled_at

        if start_t is None or end_t is None:
            continue

        dur = float(end_t) - float(start_t)
        if dur < 0:
            continue

        # store interval for helper computation (per-box, cross-prop)
        attempts_by_box[box_id].append({
            "agent_id": str(dr.agent_id),
            "prop": prop,
            "start": float(start_t),
            "end": float(end_t),
            "status": str(dr.status),
            "success": bool(dr.success) if dr.success is not None else None,
        })

        # compute useful flags (for the "max attempt" record)
        prop_present = bool(box.has_X) if prop == "X" else bool(box.has_Y)
        late = (dr.completed_at is not None) and (float(dr.completed_at) > float(box.deadline))
        wrong_property = (not prop_present)

        prev = per_box_max_dispose_attempt.get(box_id)
        if (prev is None) or (dur > float(prev["max_duration_sec"])):
            per_box_max_dispose_attempt[box_id] = {
                "max_duration_sec": float(dur),
                "agent_id": str(dr.agent_id),
                "prop": prop,
                "status": str(dr.status),  # completed/cancelled
                "success": bool(dr.success) if dr.success is not None else None,
                "started_at": float(start_t),
                "ended_at": float(end_t),
                "deadline": float(box.deadline),
                "late": bool(late),
                "wrong_property": bool(wrong_property),
                "truth_present": bool(prop_present),

                # NEW:
                # helpers: list of agents that overlapped this max-attempt window,
                # regardless of their requested property.
                "helpers": [],
                # optional richer view:
                "helpers_detailed": [],
            }

    def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
        return max(a_start, b_start) <= min(a_end, b_end)

    for box_id, info in per_box_max_dispose_attempt.items():
        a_start = float(info["started_at"])
        a_end = float(info["ended_at"])

        helpers = {}  # agent_id -> set(props)
        for rec in attempts_by_box.get(int(box_id), []):
            if overlaps(a_start, a_end, float(rec["start"]), float(rec["end"])):
                helpers.setdefault(str(rec["agent_id"]), set()).add(str(rec["prop"]))

        # If you want helpers-only, uncomment next line:
        # helpers.pop(str(info["agent_id"]), None)

        info["helpers"] = sorted(helpers.keys())
        info["helpers_detailed"] = [
            {"agent_id": aid, "props": sorted(list(props))}
            for aid, props in sorted(helpers.items(), key=lambda kv: kv[0])
        ]

    # -----------------------
    # NEW: current agent activity
    # -----------------------
    agents_activity: Dict[str, Dict[str, Any]] = {}

    # Running senses
    running_senses = (
        db.query(SenseResult, Box)
        .join(Box, Box.id == SenseResult.box_id)
        .filter(SenseResult.status == "running")
        .all()
    )

    for sr, box in running_senses:
        a = str(sr.agent_id)
        agents_activity[a] = {
            "type": "sense",
            "box_id": int(sr.box_id),
            "prop": str(sr.property),
            "started_at": float(sr.started_at) if sr.started_at is not None else None,
            "elapsed_sec": (
                sim_time() - float(sr.started_at)
                if sr.started_at is not None
                else None
            ),
            "deadline": float(box.deadline),
        }

    running_disposals = (
        db.query(DisposalResult, Box)
        .join(Box, Box.id == DisposalResult.box_id)
        .filter(DisposalResult.status == "running")
        .all()
    )

    for dr, box in running_disposals:
        a = str(dr.agent_id)
        agents_activity[a] = {
            "type": "dispose",
            "box_id": int(dr.box_id),
            "prop": str(dr.property),
            "started_at": float(dr.started_at) if dr.started_at is not None else None,
            "elapsed_sec": (
                sim_time() - float(dr.started_at)
                if dr.started_at is not None
                else None
            ),
            "deadline": float(box.deadline),
        }

    # Known agents = anyone who has ever acted
    known_agents = set(sense_by_agent.keys()) | set(disp_by_agent_raw.keys())

    for a in known_agents:
        agents_activity.setdefault(a, {
            "type": "idle"
        })



    return {
        "t": sim_time(),
        "time_up": bool(TIME_UP),
        "sense_by_agent": sense_by_agent,

        # Use RAW as the main scoreboard (matches 'completed/correct_on_time/wrong_property/late')
        "dispose_by_agent": disp_by_agent_raw,

        # Optional: keep credited view separately
        "dispose_by_agent_credited": disp_by_agent_credited,

        "sensed_items_by_agent": sensed_items_by_agent,
        "disposed_items_by_agent": disposed_items_by_agent,
        "agents_activity": agents_activity,


        "totals": {
            "sensed_completed": total_sensed,
            "disposed_completed": total_disposed_raw,
            "correct_disposals_on_time_raw": total_correct_raw,
            "correct_disposals_on_time_credited": total_correct_credited,
            "per_box_max_dispose_attempt": per_box_max_dispose_attempt,
            
        },
    }


def print_live_score(db, reason: str = "") -> None:
    score = compute_live_score(db)
    tag = f" ({reason})" if reason else ""
    print("\n========== LIVE SCORE UPDATE" + tag + " ==========")
    print(json.dumps(score, indent=2, sort_keys=True))
    print("==================================================\n")


async def _finalize_watchdog_loop():
    while True:
        await asyncio.sleep(0.5)
        try:
            limit = float(FINAL_DEADLINE_SIM) if FINAL_DEADLINE_SIM is not None else float(TIME_LIMIT_SEC)
            if sim_time() >= limit:
                maybe_finalize_time_and_score()
        except Exception as e:
            print(f"[finalize_watchdog] ERROR: {type(e).__name__}: {e}")




# ---------------------------------
# DB Models (all times as sim-time floats)
# ---------------------------------
class Box(Base):
    __tablename__ = "boxes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # We'll use "id" as box_id externally.
    has_X = Column(Boolean, nullable=False)
    has_Y = Column(Boolean, nullable=False)

    senseable_X = Column(Boolean, nullable=False, default=True)
    senseable_Y = Column(Boolean, nullable=False, default=True)

    # Deadline as simulation time (seconds since SERVER_START)
    deadline = Column(Float, nullable=False)

    # Durations in seconds
    sense_time_X = Column(Float, nullable=False)
    sense_time_Y = Column(Float, nullable=False)
    dispose_time_X = Column(Float, nullable=False)
    dispose_time_Y = Column(Float, nullable=False)

    # Location
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    release_time = Column(Float, nullable=False, default=0.0)
    sense_results = relationship(
        "SenseResult", back_populates="box", cascade="all, delete-orphan"
    )
    disposal_results = relationship(
        "DisposalResult", back_populates="box", cascade="all, delete-orphan"
    )


class SenseResult(Base):
    __tablename__ = "sense_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String, nullable=False)
    box_id = Column(Integer, ForeignKey("boxes.id"), nullable=False)
    property = Column(String, nullable=False)  # "X" or "Y"
    status = Column(String, nullable=False, default="running")  # running/completed/cancelled

    # All times as sim-time seconds
    requested_at = Column(Float, nullable=False)
    started_at = Column(Float, nullable=True)
    completed_at = Column(Float, nullable=True)
    cancelled_at = Column(Float, nullable=True)

    detected = Column(Boolean, nullable=True)   # sensor output
    probability = Column(Float, nullable=True)  # probability used to sample
    duration_sec = Column(Float, nullable=True)

    box = relationship("Box", back_populates="sense_results")

    __table_args__ = (
        Index("ix_sense_agent_box_prop", "agent_id", "box_id", "property"),
    )


class DisposalResult(Base):
    __tablename__ = "disposal_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String, nullable=False)
    box_id = Column(Integer, ForeignKey("boxes.id"), nullable=False)
    property = Column(String, nullable=False)   # "X" or "Y"
    status = Column(String, nullable=False, default="running")  # running/completed/cancelled

    # All times as sim-time seconds
    requested_at = Column(Float, nullable=False)
    started_at = Column(Float, nullable=True)
    completed_at = Column(Float, nullable=True)
    cancelled_at = Column(Float, nullable=True)

    success = Column(Boolean, nullable=True)
    duration_sec = Column(Float, nullable=True)

    box = relationship("Box", back_populates="disposal_results")

    __table_args__ = (
        Index("ix_dispose_agent_box_prop", "agent_id", "box_id", "property"),
    )

class AgentPropertyDetectionParams(BaseModel):
    present: float  # P(detect=True | property present)
    absent: float   # P(detect=True | property absent)


class AgentDetectionParams(BaseModel):
    X: AgentPropertyDetectionParams
    Y: AgentPropertyDetectionParams


class AgentDetectionParamsResponse(BaseModel):
    agents: Dict[str, AgentDetectionParams]
    default: AgentDetectionParams



Base.metadata.create_all(engine)



def probability_reading_correct(
    agent_id: str,
    prop: str,
    detected: bool,
    prior_present: float = 0.5,
) -> float:
    agent_params = AGENT_DETECTION_PARAMS.get(agent_id, DEFAULT_AGENT_PARAMS)
    t = float(agent_params[prop]["present"])  # P(+ | present)
    f = float(agent_params[prop]["absent"])   # P(+ | absent)

    pi = max(0.0, min(1.0, float(prior_present)))

    if detected:
        # P(present | +)
        num = t * pi
        den = (t * pi) + (f * (1.0 - pi))
    else:
        # P(absent | -)
        num = (1.0 - f) * (1.0 - pi)
        den = ((1.0 - t) * pi) + ((1.0 - f) * (1.0 - pi))

    if den <= 1e-12:
        return 0.5
    return num / den


# ---------------------------------
# Seeding: 20 random boxes (deadlines in sim time)
# ---------------------------------
# ---------------------------------
# 3) In seeding, store release_time on the Box
# ---------------------------------
def seed_boxes_if_empty():
    global FINAL_DEADLINE_SIM
    with SessionLocal() as db:
        count = db.query(Box).count()
        if count > 0:
            # still compute in case it wasn't set (e.g., hot reload)
            mx = db.query(func.max(Box.deadline)).scalar()
            FINAL_DEADLINE_SIM = float(mx) if mx is not None else None
            return

        boxes: List[Box] = []
        scenario_start = sim_time()  # anchor deadlines + release times to this run/reset

        for idx in range(1, 11):
            x = objects_pre_data[idx]["x"]
            y = objects_pre_data[idx]["y"]

            if USE_HARDCODED_SCENARIO:
                spec = SCENARIO_BOXES[idx]

                has_X = bool(spec["has_X"])
                has_Y = bool(spec["has_Y"])

                deadline = scenario_start + float(spec["deadline_offset"])

                # NEW: release time (defaults to 0 if omitted)
                release_time = scenario_start + float(spec.get("release_time", 0.0))

                sense_time_X = float(spec["sense_X"])
                sense_time_Y = float(spec["sense_Y"])
                dispose_time_X = float(spec["disp_X"])
                dispose_time_Y = float(spec["disp_Y"])

                if "senseable_X" in spec or "senseable_Y" in spec:
                    senseable_X = bool(spec.get("senseable_X", True))
                    senseable_Y = bool(spec.get("senseable_Y", True))
                else:
                    allowed = spec.get("senseable", ["X", "Y"])
                    allowed_set = {str(p).upper() for p in allowed}
                    senseable_X = ("X" in allowed_set)
                    senseable_Y = ("Y" in allowed_set)

            else:
                has_X = random.random() < 0.5
                has_Y = random.random() < 0.5
                deadline_offset_sec = random.uniform(120, 600)
                deadline = sim_time() + deadline_offset_sec

                # NEW: random release (optional; here keep all available immediately)
                release_time = sim_time()

                sense_time_X = random.uniform(5.0, 60.0)
                sense_time_Y = random.uniform(5.0, 60.0)
                dispose_time_X = random.uniform(5.0, 60.0)
                dispose_time_Y = random.uniform(5.0, 60.0)
                senseable_X = True
                senseable_Y = True

            b = Box(
                has_X=has_X,
                has_Y=has_Y,
                deadline=deadline,
                sense_time_X=sense_time_X,
                sense_time_Y=sense_time_Y,
                dispose_time_X=dispose_time_X,
                dispose_time_Y=dispose_time_Y,
                x=x,
                y=y,
                senseable_X=senseable_X,
                senseable_Y=senseable_Y,
                # NEW:
                release_time=float(release_time),
            )
            boxes.append(b)

        db.add_all(boxes)
        db.commit()
        mx = db.query(func.max(Box.deadline)).scalar()
        FINAL_DEADLINE_SIM = float(mx) if mx is not None else None


seed_boxes_if_empty()


# ---------------------------------
# In-memory registries for running ops (for immediate cancel)
# ---------------------------------
RUNNING_SENSE_OPS: Dict[int, asyncio.Event] = {}
RUNNING_DISPOSE_OPS: Dict[int, asyncio.Event] = {}


# ---------------------------------
# NEW: collaborative disposal sessions (per-box, cross-property)
# ---------------------------------

@dataclass
class SharedDisposalSession:
    box_id: int
    required_base_time: float  # physical dispose duration (1 agent)
    deadline_sim: float

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    participants: set[int] = field(default_factory=set)          # disposal_result ids currently helping
    participant_props: Dict[int, str] = field(default_factory=dict)  # disposal_result.id -> requested prop ("X"/"Y")

    progress_base: float = 0.0
    last_sim: Optional[float] = None

    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    outcome: Optional[str] = None        # "completed" | "deadline"
    completed_at: Optional[float] = None

    completed_participants: set[int] = field(default_factory=set)
    completed_participant_props: Dict[int, str] = field(default_factory=dict)

    task: Optional[asyncio.Task] = None


# key = box_id   (NOT (box_id, prop))
DISPOSAL_SESSIONS: Dict[int, SharedDisposalSession] = {}

# map disposal_result.id -> box_id so /dispose/cancel can remove participant quickly
DISPOSE_ID_TO_SESSION: Dict[int, int] = {}



async def interruptible_sleep(duration: float, cancel_event: asyncio.Event) -> str:
    """
    Wait for 'duration' seconds, but return early if cancel_event is set.

    Returns:
        "completed" if the full duration elapsed without cancellation.
        "cancelled" if the event was set before timeout.
    """
    try:
        await asyncio.wait_for(cancel_event.wait(), timeout=duration)
        return "cancelled"
    except asyncio.TimeoutError:
        return "completed"

async def interruptible_sleep_with_deadline(
    duration: float,
    cancel_event: asyncio.Event,
    start_sim: float,
    deadline_sim: float,
) -> str:
    """
    Wait for up to `duration` seconds, but return early if:
      - cancel_event is set  -> "cancelled"
      - deadline_sim is reached before duration elapses -> "deadline"
      - duration elapses first -> "completed"

    We compare using sim-time captured at action start:
      time_to_deadline = deadline_sim - start_sim

    This assumes sim-time progresses in real-time (your sim_time() does).
    """
    # If we're already at/past deadline at start, deadline wins immediately.
    time_to_deadline = float(deadline_sim) - float(start_sim)
    if time_to_deadline <= 0.0:
        return "deadline"

    # Deadline and duration race: whichever occurs first.
    timeout = min(float(duration), time_to_deadline)

    try:
        await asyncio.wait_for(cancel_event.wait(), timeout=timeout)
        return "cancelled"
    except asyncio.TimeoutError:
        # If we timed out because we hit the deadline first
        if time_to_deadline < float(duration):
            return "deadline"
        return "completed"


# ---------------------------------
# Pydantic Schemas (sim-time floats)
# ---------------------------------
PropertyLiteral = Literal["X", "Y"]


class SenseRequest(BaseModel):
    agent_id: str = Field(..., min_length=1)
    box_id: int = Field(..., ge=1)
    property: PropertyLiteral


class SenseResponse(BaseModel):
    agent_id: str
    box_id: int
    property: PropertyLiteral
    status: Literal["completed", "cached", "cancelled"]
    detected: Optional[bool]
    probability: Optional[float]
    deadline: float        # sim time seconds
    x: float
    y: float
    requested_at: float    # sim time seconds
    completed_at: Optional[float]  # sim time seconds


class SenseCancelRequest(BaseModel):
    agent_id: str
    box_id: int
    property: PropertyLiteral


class SenseCancelResponse(BaseModel):
    agent_id: str
    box_id: int
    property: PropertyLiteral
    status: Literal["cancelled", "not_found", "already_completed"]


class DisposeRequest(BaseModel):
    agent_id: str = Field(..., min_length=1)
    box_id: int = Field(..., ge=1)
    property: PropertyLiteral


class DisposeResponse(BaseModel):
    agent_id: str
    box_id: int
    property: PropertyLiteral
    status: Literal["completed", "cancelled"]
    success: Optional[bool]
    deadline: float        # sim time seconds
    x: float
    y: float
    requested_at: float    # sim time seconds
    completed_at: Optional[float]  # sim time seconds


class DisposeCancelRequest(BaseModel):
    agent_id: str
    box_id: int
    property: PropertyLiteral


class DisposeCancelResponse(BaseModel):
    agent_id: str
    box_id: int
    property: PropertyLiteral
    status: Literal["cancelled", "not_found", "already_completed"]


class SenseResultView(BaseModel):
    agent_id: str
    property: PropertyLiteral
    status: str
    detected: Optional[bool]
    probability: Optional[float]
    completed_at: Optional[float]  # sim time seconds


class DisposalResultView(BaseModel):
    agent_id: str
    property: PropertyLiteral
    status: str
    success: Optional[bool]
    completed_at: Optional[float]


class BoxState(BaseModel):
    box_id: int
    deadline: float         # sim time seconds
    x: float
    y: float
    sense_results: List[SenseResultView]
    disposed_X: bool
    disposed_Y: bool
    sense_time_X: float
    sense_time_Y: float
    dispose_time_X: float
    dispose_time_Y: float
    has_X: bool
    has_Y: bool
    disposal_results: List[DisposalResultView]
    senseable_X: bool
    senseable_Y: bool

class TimeResp(BaseModel):
    server_time: float
    time_limit_sec: float
    time_up: bool
    score: Optional[dict] = None


# ---------------------------------
# Helpers
# ---------------------------------

def _print_final_score_once() -> None:
    global FINAL_SCORE
    with _TIME_LOCK:
        if FINAL_SCORE is None:
            return
        if FINAL_SCORE.get("_printed", False):
            return
        FINAL_SCORE["_printed"] = True

    # Print to console
    print("\n================ FINAL SCORE ================\n")
    print(json.dumps(FINAL_SCORE, indent=2, sort_keys=True))
    print("\n============================================\n")

    # Save to logs/<timestamp>_<scenario>.json
    try:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms precision
        scenario = str(globals().get("SCENARIO_ID", "scenario"))
        fname = f"final_score_{ts}_{scenario}.json"
        path = LOG_DIR / fname

        payload = dict(FINAL_SCORE)  # shallow copy
        payload["_saved_at_utc"] = datetime.utcnow().isoformat() + "Z"
        payload["_scenario_id"] = scenario

        # Useful metadata (optional)
        payload["_frozen_time_sim"] = float(FROZEN_TIME) if FROZEN_TIME is not None else None
        payload["_time_limit_sim"] = float(FINAL_DEADLINE_SIM) if FINAL_DEADLINE_SIM is not None else float(TIME_LIMIT_SEC)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

        print(f"[final_score] saved json -> {path}")
    except Exception as e:
        print(f"[final_score] WARNING: failed to save json log: {e}")



def compute_final_breakdown(db, t_end: float) -> dict:
    """
    Final-only detailed breakdown.

    IMPORTANT:
    - We do NOT modify compute_live_score().
    - We treat any still-"running" attempt as ending at t_end for effort accounting.
    - We also count completed vs cancelled per agent.
    - We split:
        * sensing time: X vs Y
        * disposal time: hazardous boxes (has_X or has_Y) vs benign boxes (no properties)
      All times are summed across agents (person-seconds) AND per agent.
    """

    def dur(start: Optional[float], end: Optional[float]) -> float:
        if start is None or end is None:
            return 0.0
        d = float(end) - float(start)
        return d if d > 0.0 else 0.0

    def ensure_agent(m: dict, agent_id: str) -> dict:
        m.setdefault(agent_id, {
            # sensing
            "sense_time_total_sec": 0.0,
            "sense_time_X_sec": 0.0,
            "sense_time_Y_sec": 0.0,
            "sense_completed": 0,
            "sense_cancelled": 0,

            # disposal
            "dispose_time_total_sec": 0.0,
            "dispose_time_prop_X_sec": 0.0,
            "dispose_time_prop_Y_sec": 0.0,
            "dispose_time_on_hazard_sec": 0.0,   # box has_X or has_Y
            "dispose_time_on_benign_sec": 0.0,   # neither
            "dispose_completed": 0,
            "dispose_cancelled": 0,
        })
        return m[agent_id]

    effort_by_agent: Dict[str, Dict[str, Any]] = {}

    # --------------------
    # Sensing: completed/cancelled/running
    # --------------------
    sense_rows = (
        db.query(SenseResult, Box)
        .join(Box, Box.id == SenseResult.box_id)
        .filter(SenseResult.status.in_(["completed", "cancelled", "running"]))
        .all()
    )

    for sr, box in sense_rows:
        a = str(sr.agent_id)
        rec = ensure_agent(effort_by_agent, a)

        start_t = sr.started_at if sr.started_at is not None else sr.requested_at
        if sr.status == "completed":
            end_t = sr.completed_at
            rec["sense_completed"] += 1
        elif sr.status == "cancelled":
            end_t = sr.cancelled_at
            rec["sense_cancelled"] += 1
        else:
            # final time-up: treat running as ending at t_end
            end_t = float(t_end)
            rec["sense_cancelled"] += 1  # optional: count running as cancelled in final

        d = dur(start_t, end_t)
        if d <= 0.0:
            continue

        rec["sense_time_total_sec"] += d
        if str(sr.property) == "X":
            rec["sense_time_X_sec"] += d
        else:
            rec["sense_time_Y_sec"] += d

    # --------------------
    # Disposal: completed/cancelled/running
    # --------------------
    disp_rows = (
        db.query(DisposalResult, Box)
        .join(Box, Box.id == DisposalResult.box_id)
        .filter(DisposalResult.status.in_(["completed", "cancelled", "running"]))
        .all()
    )

    for dr, box in disp_rows:
        a = str(dr.agent_id)
        rec = ensure_agent(effort_by_agent, a)

        start_t = dr.started_at if dr.started_at is not None else dr.requested_at
        if dr.status == "completed":
            end_t = dr.completed_at
            rec["dispose_completed"] += 1
        elif dr.status == "cancelled":
            end_t = dr.cancelled_at
            rec["dispose_cancelled"] += 1
        else:
            end_t = float(t_end)
            rec["dispose_cancelled"] += 1  # optional: count running as cancelled in final

        d = dur(start_t, end_t)
        if d <= 0.0:
            continue

        rec["dispose_time_total_sec"] += d
        if str(dr.property) == "X":
            rec["dispose_time_prop_X_sec"] += d
        else:
            rec["dispose_time_prop_Y_sec"] += d

        is_hazard = bool(box.has_X) or bool(box.has_Y)
        if is_hazard:
            rec["dispose_time_on_hazard_sec"] += d
        else:
            rec["dispose_time_on_benign_sec"] += d


    # --------------------
    # NEW: box-id sets by status (per agent + global)
    # completed wins over cancelled for the same (agent, box_id).
    # --------------------

    def _add_status_box(m: dict, agent: str, kind: str, status: str, box_id: int):
        """
        m[agent][kind] has:
          completed: set[int]
          cancelled: set[int]
        precedence: if completed, remove from cancelled.
        """
        m.setdefault(agent, {})
        m[agent].setdefault(kind, {"completed": set(), "cancelled": set()})

        if status == "completed":
            m[agent][kind]["completed"].add(int(box_id))
            m[agent][kind]["cancelled"].discard(int(box_id))
        elif status == "cancelled":
            if int(box_id) not in m[agent][kind]["completed"]:
                m[agent][kind]["cancelled"].add(int(box_id))

    # per agent: { agent: { "sense": {...}, "dispose": {...} } }
    box_ids_by_agent: Dict[str, Dict[str, Dict[str, set]]] = {}

    # global sets
    global_ids = {
        "sense": {"completed": set(), "cancelled": set()},
        "dispose": {"completed": set(), "cancelled": set()},
    }

    # ---- sensing ----
    sense_all = (
        db.query(SenseResult)
        .filter(SenseResult.status.in_(["completed", "cancelled", "running"]))
        .all()
    )

    for sr in sense_all:
        a = str(sr.agent_id)
        box_id = int(sr.box_id)

        # finalization rule: running at time-up -> treat as cancelled for this summary
        st = str(sr.status)
        if st == "running":
            st = "cancelled"

        _add_status_box(box_ids_by_agent, a, "sense", st, box_id)

    # ---- disposal ----
    disp_all = (
        db.query(DisposalResult)
        .filter(DisposalResult.status.in_(["completed", "cancelled", "running"]))
        .all()
    )

    for dr in disp_all:
        a = str(dr.agent_id)
        box_id = int(dr.box_id)

        st = str(dr.status)
        if st == "running":
            st = "cancelled"

        _add_status_box(box_ids_by_agent, a, "dispose", st, box_id)

    # Build global sets from per-agent sets (respecting precedence)
    for a, kinds in box_ids_by_agent.items():
        for kind in ["sense", "dispose"]:
            k = kinds.get(kind)
            if not k:
                continue
            global_ids[kind]["completed"].update(k["completed"])
            # cancelled only if not globally completed
            for bid in k["cancelled"]:
                if bid not in global_ids[kind]["completed"]:
                    global_ids[kind]["cancelled"].add(bid)

    # Convert sets -> sorted lists for JSON friendliness
    box_ids_by_agent_json: Dict[str, Any] = {}
    for a, kinds in box_ids_by_agent.items():
        box_ids_by_agent_json[a] = {}
        for kind in ["sense", "dispose"]:
            if kind not in kinds:
                continue
            box_ids_by_agent_json[a][kind] = {
                "completed_box_ids": sorted(list(kinds[kind]["completed"])),
                "cancelled_box_ids": sorted(list(kinds[kind]["cancelled"])),
            }

    global_box_ids_json = {
        "sense": {
            "completed_box_ids": sorted(list(global_ids["sense"]["completed"])),
            "cancelled_box_ids": sorted(list(global_ids["sense"]["cancelled"])),
        },
        "dispose": {
            "completed_box_ids": sorted(list(global_ids["dispose"]["completed"])),
            "cancelled_box_ids": sorted(list(global_ids["dispose"]["cancelled"])),
        },
    }


    # --------------------
    # NEW: wall-clock disposal time per box
    # --------------------
    # For each box: [min(start), max(end)] over all disposal attempts on that box.
    # end = completed_at if completed, cancelled_at if cancelled, else t_end if still running.
    box_windows: Dict[int, Dict[str, float]] = {}

    disp_all = (
        db.query(DisposalResult)
        .filter(DisposalResult.status.in_(["completed", "cancelled", "running"]))
        .all()
    )

    for dr in disp_all:
        box_id = int(dr.box_id)

        start_t = dr.started_at if dr.started_at is not None else dr.requested_at
        if start_t is None:
            continue

        if dr.status == "completed":
            end_t = dr.completed_at
        elif dr.status == "cancelled":
            end_t = dr.cancelled_at
        else:  # running at time-up
            end_t = float(t_end)

        if end_t is None:
            continue

        s = float(start_t)
        e = float(end_t)
        if e < s:
            continue

        w = box_windows.get(box_id)
        if w is None:
            box_windows[box_id] = {"start": s, "end": e}
        else:
            w["start"] = min(w["start"], s)
            w["end"] = max(w["end"], e)

    wallclock_by_box_sec: Dict[str, float] = {}
    for box_id, w in box_windows.items():
        wallclock_by_box_sec[str(box_id)] = max(0.0, float(w["end"]) - float(w["start"]))

    wallclock_total_sec = sum(wallclock_by_box_sec.values())


    # totals across agents
    totals = {
        "sense_time_total_sec": sum(v["sense_time_total_sec"] for v in effort_by_agent.values()),
        "sense_time_X_sec":     sum(v["sense_time_X_sec"] for v in effort_by_agent.values()),
        "sense_time_Y_sec":     sum(v["sense_time_Y_sec"] for v in effort_by_agent.values()),
        "sense_completed":      sum(v["sense_completed"] for v in effort_by_agent.values()),
        "sense_cancelled":      sum(v["sense_cancelled"] for v in effort_by_agent.values()),

        "dispose_time_total_sec":      sum(v["dispose_time_total_sec"] for v in effort_by_agent.values()),
        "dispose_time_prop_X_sec":     sum(v["dispose_time_prop_X_sec"] for v in effort_by_agent.values()),
        "dispose_time_prop_Y_sec":     sum(v["dispose_time_prop_Y_sec"] for v in effort_by_agent.values()),
        "dispose_time_on_hazard_sec":  sum(v["dispose_time_on_hazard_sec"] for v in effort_by_agent.values()),
        "dispose_time_on_benign_sec":  sum(v["dispose_time_on_benign_sec"] for v in effort_by_agent.values()),
        "dispose_completed":           sum(v["dispose_completed"] for v in effort_by_agent.values()),
        "dispose_cancelled":           sum(v["dispose_cancelled"] for v in effort_by_agent.values()),
    }

    return {
        "final_time_breakdown": {
            "effort_by_agent": effort_by_agent,
            "effort_totals": totals,
            "dispose_wallclock_by_box_sec": wallclock_by_box_sec,
            "dispose_wallclock_total_sec": wallclock_total_sec,
            "box_ids_by_agent": box_ids_by_agent_json,
            "box_ids_global": global_box_ids_json,
        }
    }



def compute_score(db) -> dict:
    """
    Score counts each (box_id, property) at most once.

    NEW:
      - Count disposal attempts split by whether the BOX had ANY property (hazard) vs none (benign).
      - Also count successful attempts split hazard vs benign.
    """

    # -----------------------------
    # NEW: disposal attempts breakdown (hazard vs benign by box truth)
    # -----------------------------
    attempt_rows = (
        db.query(DisposalResult, Box)
        .join(Box, Box.id == DisposalResult.box_id)
        .filter(DisposalResult.status.in_(["completed", "cancelled", "running"]))
        .all()
    )

    attempts_on_hazard = 0
    attempts_on_benign = 0

    success_attempts_on_hazard = 0
    success_attempts_on_benign = 0

    for dr, box in attempt_rows:
        is_hazard = bool(box.has_X) or bool(box.has_Y)

        if is_hazard:
            attempts_on_hazard += 1
        else:
            attempts_on_benign += 1

        # successful attempt = completed + success flag true + on-time (double-check)
        if dr.status == "completed" and dr.success is True and dr.completed_at is not None:
            on_time = float(dr.completed_at) <= float(box.deadline)
            if on_time:
                if is_hazard:
                    success_attempts_on_hazard += 1
                else:
                    success_attempts_on_benign += 1

    attempts_total = attempts_on_hazard + attempts_on_benign
    success_attempts_total = success_attempts_on_hazard + success_attempts_on_benign

    # -----------------------------
    # Original: credited success pairs (dedup by (box,prop))
    # -----------------------------
    success_disposals = (
        db.query(DisposalResult, Box)
        .join(Box, Box.id == DisposalResult.box_id)
        .filter(DisposalResult.status == "completed")
        .filter(DisposalResult.success == True)  # noqa: E712
        .filter(DisposalResult.completed_at != None)
        .all()
    )

    credited: set[tuple[int, str]] = set()
    credited_by_agent: Dict[str, int] = {}
    credited_by_property: Dict[str, int] = {"X": 0, "Y": 0}

    success_disposals.sort(key=lambda pair: float(pair[0].completed_at or 0.0))

    for dr, box in success_disposals:
        if float(dr.completed_at) > float(box.deadline):
            continue

        key = (int(dr.box_id), str(dr.property))
        if key in credited:
            continue

        credited.add(key)
        credited_by_agent[dr.agent_id] = credited_by_agent.get(dr.agent_id, 0) + 1
        credited_by_property[str(dr.property)] = credited_by_property.get(str(dr.property), 0) + 1

    total_success = len(credited)

    boxes = db.query(Box).all()
    total_present_props = sum((1 if b.has_X else 0) + (1 if b.has_Y else 0) for b in boxes)
    completion_rate = (total_success / total_present_props) if total_present_props > 0 else 0.0

    return {
        "total_successful_disposals": total_success,
        "total_present_properties": total_present_props,
        "completion_rate": completion_rate,
        "per_agent_success": credited_by_agent,
        "per_property_success": credited_by_property,
        "credited_pairs": [{"box_id": b, "property": p} for (b, p) in sorted(credited)],

        # NEW: attempts split by hazard vs benign
        "disposal_attempts": {
            "attempts_total": attempts_total,
            "attempts_on_hazard_boxes": attempts_on_hazard,
            "attempts_on_benign_boxes": attempts_on_benign,
            "attempt_rate_on_hazard": (attempts_on_hazard / attempts_total) if attempts_total > 0 else None,
            "attempt_rate_on_benign": (attempts_on_benign / attempts_total) if attempts_total > 0 else None,
        },
        "disposal_success_attempts": {
            "success_attempts_total": success_attempts_total,
            "success_attempts_on_hazard_boxes": success_attempts_on_hazard,
            "success_attempts_on_benign_boxes": success_attempts_on_benign,
            "success_rate_on_hazard": (
                success_attempts_on_hazard / success_attempts_total
            ) if success_attempts_total > 0 else None,
            "success_rate_on_benign": (
                success_attempts_on_benign / success_attempts_total
            ) if success_attempts_total > 0 else None,
        },
    }




def maybe_finalize_time_and_score() -> None:
    global FINAL_SCORE, FROZEN_TIME, TIME_UP, FINAL_DEADLINE_SIM, FINAL_LIVE_PRINTED

    limit = float(FINAL_DEADLINE_SIM) if FINAL_DEADLINE_SIM is not None else float(TIME_LIMIT_SEC)

    # IMPORTANT: don't call sim_time() while holding _TIME_LOCK
    now = sim_time()
    if now < limit:
        return

    # Phase 1: lock only for tiny state transitions
    with _TIME_LOCK:
        if FINAL_SCORE is not None:
            return
        # freeze time + mark time_up
        FROZEN_TIME = float(limit)
        TIME_UP = True

    # Phase 2: do DB / printing work WITHOUT holding _TIME_LOCK
    try:
        with SessionLocal() as db:
            # Force one last live update
            if not FINAL_LIVE_PRINTED:
                try:
                    print_live_score(db, reason="FINAL LIVE UPDATE @ DEADLINE")
                finally:
                    FINAL_LIVE_PRINTED = True

            base = compute_score(db)
            breakdown = compute_final_breakdown(db, t_end=float(limit))
            final_payload = {**base, **breakdown}
    except Exception as e:
        # Don't leave the server in a weird "TIME_UP but FINAL_SCORE None" state silently
        final_payload = {"error": f"finalize failed: {type(e).__name__}: {e}"}

    # Phase 3: store FINAL_SCORE under lock
    with _TIME_LOCK:
        # another thread might have set it while we were computing
        if FINAL_SCORE is None:
            FINAL_SCORE = final_payload

    _print_final_score_once()






def get_box(db, box_id: int) -> Box:
    box = db.query(Box).filter(Box.id == box_id).one_or_none()
    if box is None:
        raise HTTPException(status_code=404, detail=f"Box {box_id} not found")
    return box


def sample_detection(box: Box, prop: str, agent_id: str) -> Tuple[bool, float]:
    """
    Sample a sensing outcome based on:
      - ground truth has_X / has_Y on the box
      - fixed per-agent detection probabilities for (agent, property)

    Returns (detected_bool, probability_used).
    """
    # Ground truth
    if prop == "X":
        has_prop = box.has_X
    else:  # "Y"
        has_prop = box.has_Y

    # Get agent parameters (fallback if unknown)
    agent_params = AGENT_DETECTION_PARAMS.get(agent_id, DEFAULT_AGENT_PARAMS)
    prop_params = agent_params[prop]

    # Pick the relevant probability
    p = prop_params["present"] if has_prop else prop_params["absent"]

    detected = random.random() < p
    return detected, p



def get_sense_duration(box: Box, prop: str) -> float:
    return box.sense_time_X if prop == "X" else box.sense_time_Y


def get_dispose_duration(box: Box, prop: str) -> float:
    return box.dispose_time_X if prop == "X" else box.dispose_time_Y


def has_property(box: Box, prop: str) -> bool:
    return box.has_X if prop == "X" else box.has_Y


# ---------------------------------
# FastAPI app
# ---------------------------------
app = FastAPI(title="Box Sensing & Disposal Server (Sim Time)", version="0.3")


@app.on_event("startup")
async def _startup():
    global _FINALIZE_WATCHDOG_TASK
    if _FINALIZE_WATCHDOG_TASK is None or _FINALIZE_WATCHDOG_TASK.done():
        _FINALIZE_WATCHDOG_TASK = asyncio.create_task(_finalize_watchdog_loop())


# -------------
# Sensing
# -------------
@app.post("/sense", response_model=SenseResponse)
async def sense(req: SenseRequest):
    """
    Sensing endpoint.

    - Each (agent_id, box_id, property) can only produce one COMPLETED result.
      Subsequent calls return the cached result (status="cached").
    - If no completed result exists, we simulate sensing time (sense_time_*),
      then sample a detection outcome based on ground truth and detection probabilities.
    - Cancellation:
        * If /sense/cancel is called while sensing is sleeping, the sleep ends
          immediately and this call returns status="cancelled".
    """
    now_sim = sim_time()
    box_id = req.box_id
    prop = req.property

    # 1) Check for cached completed result
    with SessionLocal() as db:
        box = get_box(db, box_id)

        # NEW: property-level sensing constraints
        if prop == "X" and not bool(box.senseable_X):
            raise HTTPException(status_code=400, detail=f"Box {box_id} is not senseable for property X")
        if prop == "Y" and not bool(box.senseable_Y):
            raise HTTPException(status_code=400, detail=f"Box {box_id} is not senseable for property Y")

        if now_sim < float(box.release_time):
            raise HTTPException(status_code=404, detail=f"Box {box_id} not found")


        completed = (
            db.query(SenseResult)
            .filter(
                SenseResult.agent_id == req.agent_id,
                SenseResult.box_id == box.id,
                SenseResult.property == prop,
                SenseResult.status == "completed",
            )
            .order_by(SenseResult.completed_at.desc())
            .first()
        )
        if completed is not None:
            return SenseResponse(
                agent_id=req.agent_id,
                box_id=box_id,
                property=prop,
                status="cached",
                detected=completed.detected,
                probability=completed.probability,
                deadline=box.deadline,
                x=box.x,
                y=box.y,
                requested_at=completed.requested_at,
                completed_at=completed.completed_at,
            )

        # No completed result: create a new running record
        sense_duration = get_sense_duration(box, prop)
        sr = SenseResult(
            agent_id=req.agent_id,
            box_id=box.id,
            property=prop,
            status="running",
            requested_at=now_sim,
            started_at=now_sim,
            duration_sec=sense_duration,
        )
        db.add(sr)
        db.commit()
        db.refresh(sr)
        sense_id = sr.id
        box_deadline = box.deadline
        box_x, box_y = box.x, box.y

    # 2) Simulate sensing time with interruptible sleep
    cancel_event = asyncio.Event()
    RUNNING_SENSE_OPS[sense_id] = cancel_event

    sleep_result = await interruptible_sleep_with_deadline(
        duration=sense_duration,
        cancel_event=cancel_event,
        start_sim=now_sim,           # action start sim-time
        deadline_sim=box_deadline,   # box deadline sim-time
    )

    RUNNING_SENSE_OPS.pop(sense_id, None)

    # 3) After sleep (or cancel/deadline), check DB and respond accordingly
    with SessionLocal() as db:
        sr = db.query(SenseResult).filter(SenseResult.id == sense_id).one_or_none()
        box = get_box(db, box_id)

        if sr is None:
            raise HTTPException(status_code=404, detail="Sense operation vanished")

        # Manual cancel OR deadline-triggered cancel => cancelled
        if sleep_result in ("cancelled", "deadline") or sr.status == "cancelled":
            if sr.status != "cancelled":
                sr.status = "cancelled"
                sr.cancelled_at = sim_time()
                db.commit()
                db.refresh(sr)

            return SenseResponse(
                agent_id=req.agent_id,
                box_id=box_id,
                property=prop,
                status="cancelled",
                detected=None,
                probability=None,
                deadline=box.deadline,
                x=box.x,
                y=box.y,
                requested_at=sr.requested_at,
                completed_at=None,
            )

        # Still running and not cancelled: finalize
        detected, _ = sample_detection(box, prop, req.agent_id)
        prob = probability_reading_correct(
            agent_id=req.agent_id,
            prop=prop,
            detected=detected,
            prior_present=0.5,
        )

        sr.detected = detected
        sr.probability = prob
        sr.completed_at = sim_time()
        sr.status = "completed"
        db.commit()
        db.refresh(sr)

        print_live_score(db, reason=f"sense completed: agent={req.agent_id} box={box_id} prop={prop}")

        return SenseResponse(
            agent_id=req.agent_id,
            box_id=box_id,
            property=prop,
            status="completed",
            detected=detected,
            probability=prob,
            deadline=box_deadline,
            x=box_x,
            y=box_y,
            requested_at=sr.requested_at,
            completed_at=sr.completed_at,
        )


@app.post("/sense/cancel", response_model=SenseCancelResponse)
def cancel_sense(req: SenseCancelRequest):
    """
    Cancel a running sensing request for (agent_id, box_id, property).
    If a request is already completed, report that.
    """
    with SessionLocal() as db:
        box = get_box(db, req.box_id)

        running = (
            db.query(SenseResult)
            .filter(
                SenseResult.agent_id == req.agent_id,
                SenseResult.box_id == box.id,
                SenseResult.property == req.property,
                SenseResult.status == "running",
            )
            .order_by(SenseResult.requested_at.desc())
            .first()
        )

        if running is not None:
            running.status = "cancelled"
            running.cancelled_at = sim_time()
            db.commit()

            ev = RUNNING_SENSE_OPS.get(running.id)
            if ev:
                ev.set()  # wake up sleeping /sense immediately

            return SenseCancelResponse(
                agent_id=req.agent_id,
                box_id=req.box_id,
                property=req.property,
                status="cancelled",
            )

        # Check if completed exists
        completed = (
            db.query(SenseResult)
            .filter(
                SenseResult.agent_id == req.agent_id,
                SenseResult.box_id == box.id,
                SenseResult.property == req.property,
                SenseResult.status == "completed",
            )
            .first()
        )
        if completed is not None:
            return SenseCancelResponse(
                agent_id=req.agent_id,
                box_id=req.box_id,
                property=req.property,
                status="already_completed",
            )

        print_live_score(db, reason=f"sense cancelled: agent={req.agent_id} box={req.box_id} prop={req.property}")


        return SenseCancelResponse(
            agent_id=req.agent_id,
            box_id=req.box_id,
            property=req.property,
            status="not_found",
        )


async def _run_disposal_session(box_id: int) -> None:
    """
    Background loop that advances shared disposal progress based on
    current participant count, cross-property.

    Rules:
      - Effective speed = N participants
      - If participants drops to 0 before completion, progress resets to 0
      - If deadline reached before completion, session ends with outcome "deadline"
    """
    session = DISPOSAL_SESSIONS.get(box_id)
    if session is None:
        return

    tick_sec = 0.05

    try:
        while True:
            await asyncio.sleep(tick_sec)
            now = sim_time()

            async with session.lock:
                if session.done_event.is_set():
                    return

                if now >= float(session.deadline_sim):
                    session.outcome = "deadline"
                    session.completed_at = float(session.deadline_sim)

                    session.completed_participants = set(session.participants)
                    session.completed_participant_props = dict(session.participant_props)

                    session.participants.clear()
                    session.participant_props.clear()
                    session.progress_base = 0.0
                    session.last_sim = None
                    session.done_event.set()
                    return

                n = len(session.participants)

                if n == 0:
                    session.progress_base = 0.0
                    session.last_sim = None
                    continue

                if session.last_sim is None:
                    session.last_sim = now
                    continue

                dt = max(0.0, float(now) - float(session.last_sim))
                session.last_sim = now
                session.progress_base += dt * float(n)

                if session.progress_base >= float(session.required_base_time):
                    session.outcome = "completed"
                    session.completed_at = now

                    session.completed_participants = set(session.participants)
                    session.completed_participant_props = dict(session.participant_props)

                    session.participants.clear()
                    session.participant_props.clear()
                    session.progress_base = 0.0
                    session.last_sim = None
                    session.done_event.set()
                    return

    finally:
        if session.done_event.is_set():
            DISPOSAL_SESSIONS.pop(box_id, None)


# -------------
# Disposal
# -------------
@app.post("/dispose", response_model=DisposeResponse)
async def dispose(req: DisposeRequest):
    now_sim = sim_time()
    box_id = req.box_id
    prop = req.property

    with SessionLocal() as db:
        box = get_box(db, box_id)

        if now_sim < float(box.release_time):
            raise HTTPException(status_code=404, detail=f"Box {box_id} not found")

        # Per-attempt record duration can still be prop-specific (kept for logging),
        # BUT the shared physical session uses a single required_base_time below.
        duration = get_dispose_duration(box, prop)

        dr = DisposalResult(
            agent_id=req.agent_id,
            box_id=box.id,
            property=prop,
            status="running",
            requested_at=now_sim,
            started_at=now_sim,
            duration_sec=duration,
        )
        db.add(dr)
        db.commit()
        db.refresh(dr)

        dispose_id = dr.id
        box_deadline = float(box.deadline)
        box_x, box_y = float(box.x), float(box.y)

        # PHYSICAL disposal time (single shared effort):
        physical_base_time = float(max(box.dispose_time_X, box.dispose_time_Y))

    cancel_event = asyncio.Event()
    RUNNING_DISPOSE_OPS[dispose_id] = cancel_event

    # ---- per-box shared session ----
    session = DISPOSAL_SESSIONS.get(box_id)
    if session is None:
        session = SharedDisposalSession(
            box_id=box_id,
            required_base_time=float(physical_base_time),
            deadline_sim=float(box_deadline),
        )
        DISPOSAL_SESSIONS[box_id] = session
        session.task = asyncio.create_task(_run_disposal_session(box_id))

    async with session.lock:
        session.participants.add(dispose_id)
        session.participant_props[dispose_id] = prop  # track requested prop
        DISPOSE_ID_TO_SESSION[dispose_id] = box_id

    try:
        while True:
            if cancel_event.is_set():
                break
            if session.done_event.is_set():
                break
            await asyncio.sleep(0.05)
    finally:
        RUNNING_DISPOSE_OPS.pop(dispose_id, None)

    # 5) If this agent cancelled
    if cancel_event.is_set():
        with contextlib.suppress(Exception):
            async with session.lock:
                session.participants.discard(dispose_id)
                session.participant_props.pop(dispose_id, None)
                DISPOSE_ID_TO_SESSION.pop(dispose_id, None)

        with SessionLocal() as db:
            dr = db.query(DisposalResult).filter(DisposalResult.id == dispose_id).one_or_none()
            box = get_box(db, box_id)
            if dr is None:
                raise HTTPException(status_code=404, detail="Disposal operation vanished")

            if dr.status != "cancelled":
                dr.status = "cancelled"
                dr.cancelled_at = sim_time()
                db.commit()
                db.refresh(dr)

                # (Also fixes your bug: previously you referenced `outcome` here before it existed.)
                print_live_score(
                    db,
                    reason=f"dispose ended as cancelled: agent={req.agent_id} box={box_id} prop={prop}",
                )

            return DisposeResponse(
                agent_id=req.agent_id,
                box_id=box_id,
                property=prop,
                status="cancelled",
                success=None,
                deadline=float(box.deadline),
                x=float(box.x),
                y=float(box.y),
                requested_at=float(dr.requested_at),
                completed_at=None,
            )

    # Otherwise session ended: either completed or deadline
    outcome = session.outcome
    finished_sim = float(session.completed_at or sim_time())
    participated_at_finish = (dispose_id in session.completed_participants)

    if outcome != "completed" or not participated_at_finish:
        with SessionLocal() as db:
            dr = db.query(DisposalResult).filter(DisposalResult.id == dispose_id).one_or_none()
            box = get_box(db, box_id)
            if dr is None:
                raise HTTPException(status_code=404, detail="Disposal operation vanished")

            if dr.status != "cancelled":
                dr.status = "cancelled"
                dr.cancelled_at = sim_time()
                db.commit()
                db.refresh(dr)

            return DisposeResponse(
                agent_id=req.agent_id,
                box_id=box_id,
                property=prop,
                status="cancelled",
                success=None,
                deadline=float(box.deadline),
                x=float(box.x),
                y=float(box.y),
                requested_at=float(dr.requested_at),
                completed_at=None,
            )

    # Completed: compute success for THIS agent's requested prop
    with SessionLocal() as db:
        dr = db.query(DisposalResult).filter(DisposalResult.id == dispose_id).one_or_none()
        box = get_box(db, box_id)
        if dr is None:
            raise HTTPException(status_code=404, detail="Disposal operation vanished")

        prop_present = has_property(box, prop)
        finished_before_deadline = finished_sim <= float(box.deadline)
        success = bool(prop_present and finished_before_deadline)

        dr.success = success
        dr.completed_at = finished_sim
        dr.status = "completed"
        db.commit()
        db.refresh(dr)

        print_live_score(
            db,
            reason=f"dispose completed (shared cross-prop): agent={req.agent_id} box={box_id} prop={prop} success={success}",
        )

        return DisposeResponse(
            agent_id=req.agent_id,
            box_id=box_id,
            property=prop,
            status="completed",
            success=success,
            deadline=float(box.deadline),
            x=float(box.x),
            y=float(box.y),
            requested_at=float(dr.requested_at),
            completed_at=float(dr.completed_at) if dr.completed_at is not None else None,
        )


@app.post("/dispose/cancel", response_model=DisposeCancelResponse)
def cancel_dispose(req: DisposeCancelRequest):
    with SessionLocal() as db:
        box = get_box(db, req.box_id)

        running = (
            db.query(DisposalResult)
            .filter(
                DisposalResult.agent_id == req.agent_id,
                DisposalResult.box_id == box.id,
                DisposalResult.property == req.property,
                DisposalResult.status == "running",
            )
            .order_by(DisposalResult.requested_at.desc())
            .first()
        )

        if running is not None:
            running.status = "cancelled"
            running.cancelled_at = sim_time()
            db.commit()

            ev = RUNNING_DISPOSE_OPS.get(running.id)
            if ev:
                ev.set()

            # NEW: remove from per-box shared disposal session
            box_key = DISPOSE_ID_TO_SESSION.pop(running.id, None)
            if box_key is not None:
                sess = DISPOSAL_SESSIONS.get(box_key)
                if sess is not None:
                    try:
                        sess.participants.discard(running.id)
                        sess.participant_props.pop(running.id, None)
                    except Exception:
                        pass

            return DisposeCancelResponse(
                agent_id=req.agent_id,
                box_id=req.box_id,
                property=req.property,
                status="cancelled",
            )

        completed = (
            db.query(DisposalResult)
            .filter(
                DisposalResult.agent_id == req.agent_id,
                DisposalResult.box_id == box.id,
                DisposalResult.property == req.property,
                DisposalResult.status == "completed",
            )
            .first()
        )
        if completed is not None:
            return DisposeCancelResponse(
                agent_id=req.agent_id,
                box_id=req.box_id,
                property=req.property,
                status="already_completed",
            )

        print_live_score(db, reason=f"dispose cancelled: agent={req.agent_id} box={req.box_id} prop={req.property}")

        return DisposeCancelResponse(
            agent_id=req.agent_id,
            box_id=req.box_id,
            property=req.property,
            status="not_found",
        )


# -------------
# World state
# -------------
@app.get("/boxes/state", response_model=List[BoxState])
def get_boxes_state():
    """
    Returns all boxes and all sensing results from any agent.

    - deadline is sim-time seconds
    - completed_at in sense_results is sim-time seconds
    - sense_time_* and dispose_time_* are per-box durations (seconds)
    """
    
    now = sim_time()

    limit = float(FINAL_DEADLINE_SIM) if FINAL_DEADLINE_SIM is not None else float(TIME_LIMIT_SEC)
    
    if now >= limit:
        maybe_finalize_time_and_score()
    
    with SessionLocal() as db:
        boxes = (
            db.query(Box)
            .filter(Box.release_time <= now)
            .all()
        )
        result: List[BoxState] = []
        for b in boxes:
            sr_views = [
                SenseResultView(
                    agent_id=sr.agent_id,
                    property=sr.property,  # type: ignore
                    status=sr.status,
                    detected=sr.detected,
                    probability=sr.probability,
                    completed_at=sr.completed_at,
                )
                for sr in b.sense_results
            ]

            # aggregate disposal state (success per property)
            disposed_X = any(
                (dr.property == "X" and dr.status == "completed")
                for dr in b.disposal_results
            )
            disposed_Y = any(
                (dr.property == "Y" and dr.status == "completed")
                for dr in b.disposal_results
            )


            dr_views = [
                DisposalResultView(
                    agent_id=dr.agent_id,
                    property=dr.property,  # type: ignore
                    status=dr.status,
                    success=dr.success,
                    completed_at=dr.completed_at,
                )
                for dr in b.disposal_results
            ]

            result.append(
                BoxState(
                    box_id=b.id,
                    deadline=b.deadline,
                    x=b.x,
                    y=b.y,
                    sense_results=sr_views,
                    disposed_X=disposed_X,
                    disposed_Y=disposed_Y,
                    sense_time_X=b.sense_time_X,
                    sense_time_Y=b.sense_time_Y,
                    dispose_time_X=b.dispose_time_X,
                    dispose_time_Y=b.dispose_time_Y,
                    has_X=bool(b.has_X),
                    has_Y=bool(b.has_Y), 
                    disposal_results=dr_views,
                    senseable_X=bool(b.senseable_X),
                    senseable_Y=bool(b.senseable_Y),

                )
            )
            
        if ENABLE_STATE_PRINTS and not TIME_UP:
            print(result)
        return result




# -------------
# Time endpoint
# -------------
@app.get("/time", response_model=TimeResp)
def get_time():
    now = sim_time()
    limit = float(FINAL_DEADLINE_SIM) if FINAL_DEADLINE_SIM is not None else float(TIME_LIMIT_SEC)

    if now >= limit:
        maybe_finalize_time_and_score()

    return TimeResp(
        server_time=now,
        time_limit_sec=limit,
        time_up=bool(TIME_UP),
        score=FINAL_SCORE if TIME_UP else None,
    )


# -------------
# Maintenance
@app.post("/reset_boxes")
def reset_boxes():
    global SERVER_START, FROZEN_TIME, TIME_UP, FINAL_SCORE, FINAL_DEADLINE_SIM, FINAL_LIVE_PRINTED

    SERVER_START = datetime.utcnow()

    with _TIME_LOCK:
        FROZEN_TIME = None
        TIME_UP = False
        FINAL_SCORE = None
        FINAL_LIVE_PRINTED = False
        
    FINAL_DEADLINE_SIM = None

    with SessionLocal() as db:
        db.query(SenseResult).delete()
        db.query(DisposalResult).delete()
        db.query(Box).delete()
        db.commit()

    seed_boxes_if_empty()
    return {"status": "ok", "message": "Boxes reset and clock restarted"}

    
    
@app.get("/agents/params", response_model=AgentDetectionParamsResponse)
def get_agent_detection_params():
    """
    Return the static per-agent detection parameters used by the sensing model.

    Structure:
      {
        "agents": {
          "<agent_id>": {
            "X": {"present": ..., "absent": ...},
            "Y": {"present": ..., "absent": ...}
          },
          ...
        },
        "default": {
          "X": {"present": ..., "absent": ...},
          "Y": {"present": ..., "absent": ...}
        }
      }
    """
    def to_agent_params(raw: Dict[str, Dict[str, float]]) -> AgentDetectionParams:
        return AgentDetectionParams(
            X=AgentPropertyDetectionParams(**raw["X"]),
            Y=AgentPropertyDetectionParams(**raw["Y"]),
        )

    agents = {
        agent_id: to_agent_params(params)
        for agent_id, params in AGENT_DETECTION_PARAMS.items()
    }
    default = to_agent_params(DEFAULT_AGENT_PARAMS)

    return AgentDetectionParamsResponse(agents=agents, default=default)


# ---------------------------------
# Debug helpers (instant completion)
# ---------------------------------
class DebugSenseRequest(SenseRequest):
    # Optional override for debug: you can force detected/probability.
    detected: Optional[bool] = None
    probability: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Optional probability to store/use for debugging"
    )


class DebugDisposeRequest(DisposeRequest):
    # Optional override to force success/failure for debugging
    success: Optional[bool] = None


@app.post("/debug/sense", response_model=SenseResponse)
def debug_sense(req: DebugSenseRequest):
    """
    DEBUG: instantly insert a COMPLETED sense result for (agent_id, box_id, property),
    without any waiting or cancellation logic.

    - If req.detected and/or req.probability are provided, they are used directly.
    - Otherwise we fall back to the normal sample_detection() logic.
    - duration_sec is set to 0.0.
    """
    now_sim = sim_time()
    box_id = req.box_id
    prop = req.property

    with SessionLocal() as db:
        box = get_box(db, box_id)

        # Decide detected / probability
        if req.detected is not None and req.probability is not None:
            detected = req.detected
            prob = req.probability
        elif req.probability is not None:
            # Sample using the given probability
            prob = req.probability
            detected = (random.random() < prob)
        elif req.detected is not None:
            # Force detection, assign a default probability
            detected = req.detected
            prob = 1.0
        else:
            # Default: use the normal sampling model
            detected, prob = sample_detection(box, prop, req.agent_id)

        sr = SenseResult(
            agent_id=req.agent_id,
            box_id=box.id,
            property=prop,
            status="completed",
            requested_at=now_sim,
            started_at=now_sim,
            completed_at=now_sim,
            detected=detected,
            probability=prob,
            duration_sec=0.0,
        )
        db.add(sr)
        db.commit()
        db.refresh(sr)

        return SenseResponse(
            agent_id=req.agent_id,
            box_id=box_id,
            property=prop,
            status="completed",
            detected=sr.detected,
            probability=sr.probability,
            deadline=box.deadline,
            x=box.x,
            y=box.y,
            requested_at=sr.requested_at,
            completed_at=sr.completed_at,
        )

@app.post("/debug/dispose", response_model=DisposeResponse)
def debug_dispose(req: DebugDisposeRequest):
    """
    DEBUG: instantly insert a COMPLETED disposal result for (agent_id, box_id, property),
    without any waiting or cancellation logic.

    - If req.success is provided, it is used directly.
    - Otherwise success is computed with the same rule as /dispose:
        * property must actually be present, AND
        * completion time <= deadline.
    - duration_sec is set to 0.0.
    """
    now_sim = sim_time()
    box_id = req.box_id
    prop = req.property

    with SessionLocal() as db:
        box = get_box(db, box_id)

        if req.success is not None:
            success = req.success
        else:
            finished_sim = now_sim
            prop_present = has_property(box, prop)
            finished_before_deadline = finished_sim <= box.deadline
            success = prop_present and finished_before_deadline

        dr = DisposalResult(
            agent_id=req.agent_id,
            box_id=box.id,
            property=prop,
            status="completed",
            requested_at=now_sim,
            started_at=now_sim,
            completed_at=now_sim,
            success=success,
            duration_sec=0.0,
        )
        db.add(dr)
        db.commit()
        db.refresh(dr)

        return DisposeResponse(
            agent_id=req.agent_id,
            box_id=box_id,
            property=prop,
            status="completed",
            success=dr.success,
            deadline=box.deadline,
            x=box.x,
            y=box.y,
            requested_at=dr.requested_at,
            completed_at=dr.completed_at,
        )



