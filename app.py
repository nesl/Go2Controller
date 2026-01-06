#!/usr/bin/env python3
import os
import sys
import asyncio
import random
from datetime import datetime
from typing import Literal, Optional, List, Dict, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Boolean,
    Integer,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# ---------------------------------
# DB setup & sim time
# ---------------------------------

RESUME_DB = ("--resume" in sys.argv) or (os.getenv("BOXES_RESUME", "0") == "1")

DB_PATH = "boxes.db"

# If not resuming, delete any existing DB file so we start fresh
if not RESUME_DB and os.path.exists(DB_PATH):
    os.remove(DB_PATH)


engine = create_engine("sqlite:///boxes.db", future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

SERVER_START = datetime.utcnow()


# ---------------------------------
# Agent detection parameters
# ---------------------------------
# Each agent has per-property probabilities:
#   present: P(detect=True | property actually present)
#   absent:  P(detect=True | property actually absent)  (false-positive rate)

AGENT_DETECTION_PARAMS = {
    "robot": {
        "X": {"present": 0.95, "absent": 0.05},
        "Y": {"present": 0.95, "absent": 0.05},
    },
    "human_a": {
        "X": {"present": 0.80, "absent": 0.20},
        "Y": {"present": 0.75, "absent": 0.25},
    },
    "human_b": {
        "X": {"present": 0.75, "absent": 0.25},
        "Y": {"present": 0.80, "absent": 0.20},
    },
}

# Fallback for unknown agents (if any)
DEFAULT_AGENT_PARAMS = {
    "X": {"present": 0.75, "absent": 0.25},
    "Y": {"present": 0.75, "absent": 0.25},
}


def sim_time() -> float:
    """
    Simulation time in seconds since SERVER_START.
    Resets when /reset_boxes is called.
    """
    return (datetime.utcnow() - SERVER_START).total_seconds()


# ---------------------------------
# DB Models (all times as sim-time floats)
# ---------------------------------
class Box(Base):
    __tablename__ = "boxes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # We'll use "id" as box_id externally.
    has_X = Column(Boolean, nullable=False)
    has_Y = Column(Boolean, nullable=False)

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


# ---------------------------------
# Seeding: 20 random boxes (deadlines in sim time)
# ---------------------------------
def seed_boxes_if_empty():
    with SessionLocal() as db:
        count = db.query(Box).count()
        if count > 0:
            return

        boxes: List[Box] = []
        for idx in range(10):
            # Ground truth: which boxes have X/Y (decided a priori here)
            has_X = random.random() < 0.5
            has_Y = random.random() < 0.5

            # Deadlines: between 2 and 6 minutes from current sim time
            deadline_offset_sec = random.uniform(120, 600)
            deadline = sim_time() + deadline_offset_sec

            # Sense and disposal times: 1–5 seconds
            sense_time_X = random.uniform(30.0, 120.0)
            sense_time_Y = random.uniform(30.0, 120.0)
            dispose_time_X = random.uniform(30.0, 120.0)
            dispose_time_Y = random.uniform(30.0, 120.0)

            # Locations in a simple square
            x = objects_pre_data[idx+1]["x"] #random.uniform(-5.0, 5.0)
            y = objects_pre_data[idx+1]["y"] #random.uniform(-5.0, 5.0)

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
            )
            boxes.append(b)
        db.add_all(boxes)
        db.commit()



seed_boxes_if_empty()


# ---------------------------------
# In-memory registries for running ops (for immediate cancel)
# ---------------------------------
RUNNING_SENSE_OPS: Dict[int, asyncio.Event] = {}
RUNNING_DISPOSE_OPS: Dict[int, asyncio.Event] = {}


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

class TimeResp(BaseModel):
    server_time: float      # sim time seconds


# ---------------------------------
# Helpers
# ---------------------------------
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
    sleep_result = await interruptible_sleep(sense_duration, cancel_event)
    RUNNING_SENSE_OPS.pop(sense_id, None)

    # 3) After sleep (or cancel), check DB and respond accordingly
    with SessionLocal() as db:
        sr = db.query(SenseResult).filter(SenseResult.id == sense_id).one_or_none()
        box = get_box(db, box_id)

        if sr is None:
            raise HTTPException(status_code=404, detail="Sense operation vanished")

        # If cancellation was requested, status might already be "cancelled"
        if sleep_result == "cancelled" or sr.status == "cancelled":
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
        detected, prob = sample_detection(box, prop, req.agent_id)
        sr.detected = detected
        sr.probability = prob
        sr.completed_at = sim_time()
        sr.status = "completed"
        db.commit()
        db.refresh(sr)

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

        return SenseCancelResponse(
            agent_id=req.agent_id,
            box_id=req.box_id,
            property=req.property,
            status="not_found",
        )


# -------------
# Disposal
# -------------
@app.post("/dispose", response_model=DisposeResponse)
async def dispose(req: DisposeRequest):
    """
    Disposal endpoint.

    - Takes dispose_time_* seconds for the chosen box and property.
    - Returns success=True if:
        * the property is actually present (ground truth),
        * AND disposal completes before the box deadline (in sim time).
      Otherwise success=False.
    - Can be cancelled via /dispose/cancel; if cancelled mid-sleep, the
      request returns immediately with status="cancelled".
    """
    now_sim = sim_time()
    box_id = req.box_id
    prop = req.property

    with SessionLocal() as db:
        box = get_box(db, box_id)
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
        box_deadline = box.deadline
        box_x, box_y = box.x, box.y

    cancel_event = asyncio.Event()
    RUNNING_DISPOSE_OPS[dispose_id] = cancel_event
    sleep_result = await interruptible_sleep(duration, cancel_event)
    RUNNING_DISPOSE_OPS.pop(dispose_id, None)

    with SessionLocal() as db:
        dr = db.query(DisposalResult).filter(DisposalResult.id == dispose_id).one_or_none()
        box = get_box(db, box_id)
        if dr is None:
            raise HTTPException(status_code=404, detail="Disposal operation vanished")

        if sleep_result == "cancelled" or dr.status == "cancelled":
            return DisposeResponse(
                agent_id=req.agent_id,
                box_id=box_id,
                property=prop,
                status="cancelled",
                success=None,
                deadline=box.deadline,
                x=box.x,
                y=box.y,
                requested_at=dr.requested_at,
                completed_at=None,
            )

        finished_sim = sim_time()
        # Success conditions
        prop_present = has_property(box, prop)
        finished_before_deadline = finished_sim <= box.deadline
        success = prop_present and finished_before_deadline

        dr.success = success
        dr.completed_at = finished_sim
        dr.status = "completed"
        db.commit()
        db.refresh(dr)

        return DisposeResponse(
            agent_id=req.agent_id,
            box_id=box_id,
            property=prop,
            status="completed",
            success=success,
            deadline=box_deadline,
            x=box_x,
            y=box_y,
            requested_at=dr.requested_at,
            completed_at=dr.completed_at,
        )


@app.post("/dispose/cancel", response_model=DisposeCancelResponse)
def cancel_dispose(req: DisposeCancelRequest):
    """
    Cancel a running disposal request for (agent_id, box_id, property).
    """
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
                ev.set()  # wake up sleeping /dispose immediately

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
    with SessionLocal() as db:
        boxes = db.query(Box).all()
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
                (dr.property == "X" and dr.status == "completed" and dr.success)
                for dr in b.disposal_results
            )
            disposed_Y = any(
                (dr.property == "Y" and dr.status == "completed" and dr.success)
                for dr in b.disposal_results
            )

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
                )
            )
            
        print(result)
            
        return result



# -------------
# Time endpoint
# -------------
@app.get("/time", response_model=TimeResp)
def get_time():
    """
    Returns current simulation time in seconds since last reset/server start.
    """
    print(sim_time())
    return TimeResp(server_time=sim_time())


# -------------
# Maintenance
# -------------
@app.post("/reset_boxes")
def reset_boxes():
    """
    Reset simulation clock and reseed boxes.

    - SERVER_START is reset to now.
    - All boxes, sense_results, disposal_results are cleared.
    - New 20 boxes seeded with deadlines based on new sim_time().
    """
    global SERVER_START
    SERVER_START = datetime.utcnow()   # reset simulation clock

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



