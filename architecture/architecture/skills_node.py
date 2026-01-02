#!/usr/bin/env python3
# skills_node.py
#
# Single-file skills runtime + ROS agent implementing basic actions and
# executing composite skills based on EventLayer rule hits — with:
#   • hot-reloadable skills library (YAML via skills_path)
#   • planning API (what skills are eligible right now)
#   • execution API (run a skill by name or by mapped rule id)
#
from __future__ import annotations
import json, math, os, re, sqlite3, time, inspect
from typing import Any, Dict, List, Optional

import yaml
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from std_msgs.msg import String as StringMsg, Bool
from std_srvs.srv import Trigger
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from tf2_ros import Buffer, TransformListener
from go2_interfaces.msg import WebRtcReq
import requests


from dataclasses import dataclass, field
import time

from typing import Tuple

@dataclass
class SkillInstance:
    name: str
    ctx: dict
    started_ms: int
    activated: bool = False
    done: bool = False

    # State-machine specific fields
    state_id: Optional[str] = None      # logical state id (string)
    state_idx: int = 0                  # index in states[]
    state_started_ms: int = 0           # when we entered this state (ms)

    # Active primitive handle for action states
    handle: "StepHandle" | None = None
    is_root: bool = True
    
    
# ───────────────────────────────────────────────────────────────────────────────
#                          Number → words helpers (TTS)
# ───────────────────────────────────────────────────────────────────────────────
# Integer-only token (standalone, not part of a decimal)
_INT_TOKEN_RE = re.compile(r'(?<![\w.])(-?\d+)(?![\w.])')

# Decimal token: captures "-12.34", "0.56", ".75", etc.
_DECIMAL_RE = re.compile(r'(?<![\w])(-?\d*\.\d+)(?![\w])')

# CNode pattern
_CNODE_RE = re.compile(r'\bCNode(\d+)\b', re.IGNORECASE)

def _num_to_words(n: int) -> str:
    """
    Convert small-ish integers to words. For large values, just return the digits
    so we don't blow up on indexing.
    """
    # Hard guard to avoid gigantic indices / unexpected values
    if abs(n) > 999:
        return str(n)

    nums0_19 = [
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    tens = [
        "", "", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety",
    ]

    neg = n < 0
    n = abs(n)
    parts: list[str] = []

    if n >= 100:
        parts.append(nums0_19[n // 100])
        parts.append("hundred")
        n %= 100
        if n:
            parts.append("and")

    if n >= 20:
        parts.append(tens[n // 10])
        if n % 10:
            parts.append(nums0_19[n % 10])
    elif n > 0 or not parts:
        parts.append(nums0_19[n])

    spoken = " ".join(parts)
    return f"minus {spoken}" if neg else spoken


def _fraction_to_words(frac_str: str) -> str:
    """
    Fractional part after the decimal point.
    Choose **digit-by-digit** or **full number** style.
    """

    # DIGIT-BY-DIGIT STYLE (recommended)
    # "23" -> "two three"
    digit_words = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
    }
    spoken = " ".join(digit_words[d] for d in frac_str)
    return spoken

    # FULL NUMBER STYLE (the one you asked for: "1.23" -> "one point twenty three")
    # return _num_to_words(int(frac_str))


def _normalize_tts_text(text: str) -> str:
    """
    Safe TTS normalizer:
      - CNode### → spoken if ID <= 999
      - Integers → spoken if |n| <= 999
      - Decimals → fully spoken: "1.23" → "one point two three"
      - Large numbers left untouched
    """

    # 1) CNode### (before decimal/integer replacement)
    def repl_cnode(m: re.Match) -> str:
        nid_str = m.group(1)
        try:
            nid = int(nid_str)
        except:
            return m.group(0)
        if abs(nid) > 999:
            return f"node {nid_str}"
        return f"node {_num_to_words(nid)}"

    out = _CNODE_RE.sub(repl_cnode, text)

    # 2) Decimals: handle BEFORE integer replacement
    def repl_decimal(m: re.Match) -> str:
        s = m.group(0)       # e.g. "-12.34"
        negative = s.startswith("-")
        if negative:
            s2 = s[1:]       # strip sign for processing
        else:
            s2 = s

        if "." not in s2:
            return s  # shouldn't happen due to regex

        whole_str, frac_str = s2.split(".", 1)

        # numeric safety: only speak if not crazy huge
        if whole_str.isdigit() and abs(int(whole_str)) > 999:
            return s  # keep as digits

        # whole part (if empty, treat ".5" as "zero")
        whole_val = int(whole_str) if whole_str else 0
        whole_spoken = _num_to_words(whole_val)

        # fractional part
        frac_spoken = _fraction_to_words(frac_str)

        spoken = f"{whole_spoken} point {frac_spoken}"
        if negative:
            spoken = "minus " + spoken
        return spoken

    out = _DECIMAL_RE.sub(repl_decimal, out)

    # 3) Standalone integers
    def repl_int(m: re.Match) -> str:
        s = m.group(0)
        try:
            n = int(s)
        except:
            return s
        if abs(n) > 999:
            return s
        return _num_to_words(n)

    out = _INT_TOKEN_RE.sub(repl_int, out)

    return out



def _box_id_from_node_id(node_id: str) -> Optional[int]:
    """
    Map a node name like 'CNode1##' (e.g., 'CNode107') or a bare numeric
    string to an integer box_id.

    We just extract the digits from 'CNode###' and interpret as the box_id.
    """
    s = str(node_id or "").strip()
    if not s:
        return None

    m = _CNODE_RE.search(s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None

    try:
        return int(s)
    except Exception:
        return None

# ───────────────────────────────────────────────────────────────────────────────
#                               Skills YAML (inline fallback)
# ───────────────────────────────────────────────────────────────────────────────
DEFAULT_SKILLS_V2 = r"""
version: 2
defaults:
  window_ms: 3000

skills:
  # === PRIMITIVES (wired to SkillsAgent methods) ===
  - name: tts.say
    kind: primitive
    action: tts
    params:
      text: "{{ctx.text | default:'OK'}}"

  - name: gesture.greet
    kind: primitive
    action: gesture
    params:
      kind: "greet"

  - name: nav.move_relative
    kind: primitive
    action: move_relative
    params:
      azimuth_deg: "{{rule.speech_final_any.azimuth_deg | default:0}}"
      dist_m: "{{ctx.dist_m | default:0.6}}"

  - name: nav.move_absolute
    kind: primitive
    action: move_absolute
    params:
      frame: "{{ctx.frame | default:'map'}}"
      x: "{{ctx.x | default:0}}"
      y: "{{ctx.y | default:0}}"
      yaw: "{{ctx.yaw | default:0}}"

  - name: base.turn_search
    kind: primitive
    action: turn_search
    params:
      azimuth_deg: "{{rule.speech_final_any.azimuth_deg | default:0}}"
      timeout_s: "{{ctx.timeout_s | default:8}}"

  - name: db.query_top_beacons
    kind: primitive
    action: query_beacons
    params:
      top_n: "{{ctx.top_n | default:3}}"

  # === COMPOSITES (step-level triggers on rule hits) ===
  - name: greet.with_presence
    kind: composite
    steps:
      - use: gesture.greet
        when:
          all:
            - exists: pose_present_precise
              within_ms: 2500
            - exists: speech_keyword
              within_ms: 2500
      - use: tts.say
        with:
          text: "Hi! How can I help?"
        when:
          any:
            - exists: speech_final_any
              within_ms: 3000

  - name: sense.here
    kind: composite
    steps:
      - use: base.turn_search
        with:
          timeout_s: 8
        when:
          any:
            - exists: pose_present_precise
              within_ms: 3000
            - exists: speech_final_any
              within_ms: 3000
      - use: nav.move_relative
        with:
          dist_m: 0.6
        when:
          exists: pose_present_precise
          within_ms: 3000
      - use: tts.say
        with:
          text: "Scanning this area."
        when:
          any:
            - exists: pose_present_precise
              within_ms: 3000

  - name: beacons.report_top3
    kind: composite
    steps:
      - use: db.query_top_beacons
        with: { top_n: 3 }
        when:
          any:
            - exists: speech_keyword
              within_ms: 5000
            - exists: speech_final_any
              within_ms: 5000
      - use: tts.say
        with:
          text: "{{ctx.last_query_beacons_speech | default:'Done.'}}"
"""
class StepHandle:
    """Represents an in-progress primitive. Engine polls .done() and can .cancel()."""
    def __init__(self, cancel_fn=None):
        self._done = False
        self._cancel = cancel_fn or (lambda: None)

    def mark_done(self):
        self._done = True

    def done(self) -> bool:
        return self._done

    def cancel(self):
        try:
            self._cancel()
        finally:
            self._done = True

# ───────────────────────────────────────────────────────────────────────────────
#                        Core Engine (step-level ‘when’)
# ───────────────────────────────────────────────────────────────────────────────
_PLACEHOLDER = re.compile(r"""{{\s*([^}]+?)\s*}}""")

def _get_path(root: dict, path: str) -> Any:
    cur = root
    for part in path.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

class RulesView:
    """In-memory rule hit store. In ROS, see RulesViewROS below."""
    def __init__(self, now_ms: Optional[int] = None):
        self._events: List[dict] = []
        self._now_ms = now_ms or int(time.time() * 1000)

    def _now(self) -> int:
        return int(time.time() * 1000)

    def add(self, rule_id: str, payload: dict, ts_ms: Optional[int] = None):
        self._events.append({
            'id': str(rule_id),
            'ts_ms': ts_ms if ts_ms is not None else self._now(),
            'payload': payload or {}
        })

    def exists(self, rule_id: str, within_ms: int) -> bool:
        cutoff = self._now() - within_ms
        rid = str(rule_id)
        last = None
        for e in self._events:
            if e.get('id') == rid and e.get('ts_ms', 0) >= cutoff:
                if last is None or e['ts_ms'] > last['ts_ms']:
                    last = e
        if not last:
            return False
        # default to True if no 'active' key (for backward compatibility)
        return bool(last.get('payload', {}).get('active', True))

    def latest_payload(self, rule_id: str, within_ms: int) -> Optional[dict]:
        cutoff = (self._now() - within_ms)
        cand = [e for e in self._events if e['id'] == str(rule_id) and e['ts_ms'] >= cutoff]
        if not cand:
            return None
        cand.sort(key=lambda e: e['ts_ms'], reverse=True)
        return cand[0]['payload']

def _render_scalar(expr: str, ctx: dict, rules: RulesView, defaults_window_ms: int) -> str:

    def _is_missing(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    # 1) Handle "| default:" first
    if "| default:" in expr:
        left, default_str = expr.split("| default:", 1)
        left = left.strip()
        default_str = default_str.strip()

        # parse default value: allow numbers/booleans/etc via JSON, else treat as string
        try:
            default_val = json.loads(default_str)
        except Exception:
            default_val = default_str.strip("'\"")

        # rule.<id>.<field>
        if left.startswith("rule."):
            parts = left.split(".")
            if len(parts) >= 3:
                rid = parts[1]
                field = ".".join(parts[2:])
                payload = rules.latest_payload(rid, defaults_window_ms)
                if payload is not None:
                    val = _get_path(payload, field)
                    if not _is_missing(val):
                        return str(val)
            return str(default_val)

        # ctx.<path>
        if left.startswith("ctx."):
            val = _get_path({"ctx": ctx}, left)
            return str(val) if not _is_missing(val) else str(default_val)

        # bare key → interpret as ctx.<key>
        val = _get_path({"ctx": ctx}, f"ctx.{left}")
        return str(val) if not _is_missing(val) else str(default_val)

    # 2) Legacy simple forms (no default)
    if expr.startswith("rule."):
        parts = expr.split(".")
        if len(parts) >= 3:
            rid = parts[1]
            field = ".".join(parts[2:])
            payload = rules.latest_payload(rid, defaults_window_ms)
            if payload is None:
                return ""
            val = _get_path(payload, field) if field else payload
            return "" if val is None else str(val)
        return ""

    if expr.startswith("ctx."):
        val = _get_path({"ctx": ctx}, expr)
        return "" if val is None else str(val)

    # Unknown reference → empty
    return ""


def _render_any(value: Any, ctx: dict, rules: RulesView, defaults_window_ms: int) -> Any:
    if not isinstance(value, str):
        return value
    def repl(m):
        return _render_scalar(m.group(1).strip(), ctx, rules, defaults_window_ms)
    return _PLACEHOLDER.sub(repl, value)

def _render_params(params: dict, ctx: dict, rules: RulesView, defaults_window_ms: int) -> dict:
    out = {}
    for k, v in (params or {}).items():
        if isinstance(v, dict):
            out[k] = _render_params(v, ctx, rules, defaults_window_ms)
        elif isinstance(v, list):
            out[k] = [_render_any(i, ctx, rules, defaults_window_ms) for i in v]
        else:
            out[k] = _render_any(v, ctx, rules, defaults_window_ms)
    # coerce common scalars
    for k, v in list(out.items()):
        if isinstance(v, str):
            low = v.strip().lower()
            try:
                if low in ('true', 'false'):
                    out[k] = (low == 'true')
                elif v.replace('.', '', 1).replace('-', '', 1).isdigit():
                    out[k] = float(v) if ('.' in v or '-' in v) else int(v)
            except Exception:
                pass
    return out

def _cond_pass(cond: dict, rules: RulesView, defaults_window_ms: int, empty_means=True) -> bool:
    if not cond:
        return empty_means
    if 'exists' in cond:
        rid = cond.get('exists')
        win = int(cond.get('within_ms') or defaults_window_ms)
        return rules.exists(rid, win)
    if "not_exists" in cond:
        rid = cond.get('not_exists')
        win = int(cond.get('within_ms') or defaults_window_ms)
        return not rules.exists(rid, win)
    if 'any' in cond:
        return any(_cond_pass(c, rules, defaults_window_ms) for c in (cond['any'] or []))
    if 'all' in cond:
        return all(_cond_pass(c, rules, defaults_window_ms) for c in (cond['all'] or []))
    return empty_means

class SkillEngineV2:
    """
    Executes primitive skills and state-machine skills.

    Skill kinds:
      - kind: "primitive"
          name: "tts.say"
          action: "tts"
          params: {...}

      - kind: "state_machine"
          name: "assist.carry_query_and_wait"
          when: {...}        # optional composite-level gate
          until: {...}       # optional composite-level stop
          initial_state: "ask_clarify"
          states:
            - id: "ask_clarify"
              type: "action"
              action:
                use: "tts.say"
                with: {...}
              on_complete: "wait_for_carry_event"
              on_failure: "done"   # optional

            - id: "wait_for_carry_event"
              type: "wait"
              wait_for:
                any_of:
                  - { rule_id: "carry_event", within_ms: 10000 }
              on_event: "ack_and_ready"
              on_timeout: "prompt_again"

            - id: "done"
              type: "action"
              action: {...}
              on_complete: null   # or omitted => skill finishes
    """

    def __init__(
        self,
        bindings: Dict[str, callable],
        rules_view: RulesView,
        defaults_window_ms: int = 3000,
        logger=None,
        event_cb=None,
    ):
        self.bindings = dict(bindings or {})
        self.rules = rules_view
        self.defaults_window_ms = int(defaults_window_ms)
        self.registry: Dict[str, dict] = {}
        self._loaded_path: Optional[str] = None
        self.logger = logger
        self._active: List[SkillInstance] = []
        self.event_cb = event_cb  # function(event_dict) -> None
        # NEW: skills that have thrown runtime errors in this process
        self.bad_skills: set[str] = set()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    
    def _quarantine_skill(self, inst: SkillInstance, reason: str, exc: Exception | None = None):
        name = inst.name
        self.bad_skills.add(name)
        if self.logger:
            self.logger.error(
                f"[SkillEngine] Quarantining skill '{name}' due to {reason}: {exc!r}"
            )
        # Emit an error event for observability
        self._emit_event(
            "skill_error",
            inst,
            {
                "reason": reason,
                "error": str(exc) if exc else "",
            },
        )

    
    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _emit_event(self, kind: str, inst: SkillInstance, extra: dict | None = None):
        """
        Emit an event to the callback and /skills/status.

        kind: "skill_started", "skill_finished", "state_entered"
        """
        if not self.event_cb:
            return

        payload = {
            "kind": kind,
            "skill": inst.name,
            "step_idx": int(getattr(inst, "state_idx", 0)),
            "state_id": inst.state_id,
            "ctx": inst.ctx,
            "started_ms": inst.started_ms,
            "activated": inst.activated,
            "done": inst.done,
            "is_root": getattr(inst, "is_root", True),
        }
        if extra:
            payload.update(extra)
        self.event_cb(payload)

    def _start_child_state_machine(
        self,
        parent_inst: SkillInstance,
        action_spec: dict,
        child_skill: dict,
    ) -> StepHandle:
        """
        Treat a state_machine referenced in action_spec['use'] as a nested sub-skill.

        - Build child ctx: parent ctx + rendered action_spec['with']
        - Create a child SkillInstance and add it to self._active
        - Return a StepHandle whose done() mirrors child.done
        """
        ref_name = child_skill["name"]

        # Render "with" block against parent ctx/rules
        with_overrides = _render_params(
            action_spec.get("with") or {},
            parent_inst.ctx,
            self.rules,
            self.defaults_window_ms,
        )
        child_ctx = dict(parent_inst.ctx)
        child_ctx.update(with_overrides)
        
        child = SkillInstance(
            name=ref_name,
            ctx=child_ctx,
            started_ms=self._now_ms(),
            is_root=False,          # NEW
        )
        # State-machine specific fields: start inactive, no state yet
        child.activated = False
        child.state_id = None
        child.state_idx = 0
        child.state_started_ms = 0
        child.handle = None
        child.done = False

        self._active.append(child)

        def _cancel_child():
            # cancel any running primitive in the child, then mark done
            if child.handle is not None and not child.handle.done():
                try:
                    child.handle.cancel()
                except Exception:
                    pass
            child.done = True

        h = StepHandle(cancel_fn=_cancel_child)

        # Proxy handle.done() to child.done
        def _done_proxy(self_handle=h, child_inst=child):
            return child_inst.done

        h.done = _done_proxy  # override instance method
        if self.logger:
            self.logger.info(
                f"[SkillEngine] Nested state_machine '{ref_name}' started as child of '{parent_inst.name}'"
            )
        return h


    def _exec_primitive_get_handle(self, action: str, params: dict, ctx: dict) -> StepHandle:
        """
        Call the bound primitive; normalize return to a StepHandle:
          - if primitive returns a StepHandle -> use it
          - otherwise -> create a StepHandle and mark done immediately
        """
        fn = self.bindings.get(action)
        if not callable(fn):
            if self.logger:
                self.logger.error(f"[SkillEngine] No binding for action '{action}' "
                                  f"– completing step as no-op.")
            h = StepHandle()
            h.mark_done()
            return h


        rendered = _render_params(params or {}, ctx, self.rules, self.defaults_window_ms)

        sig = inspect.signature(fn)
        ret = fn(**rendered, ctx=ctx) if "ctx" in sig.parameters else fn(**rendered)

        if isinstance(ret, StepHandle):
            return ret

        h = StepHandle()
        h.mark_done()
        return h

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_from_string(self, yaml_text: str):
        data = yaml.safe_load(yaml_text)
        self.defaults_window_ms = int(data.get("defaults", {}).get("window_ms", self.defaults_window_ms))
        self.registry = {s["name"]: s for s in data.get("skills", [])}
        self._loaded_path = None
        return self

    def load_from_path(self, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f.read())
        self.defaults_window_ms = int(data.get("defaults", {}).get("window_ms", self.defaults_window_ms))
        self.registry = {s["name"]: s for s in data.get("skills", [])}
        self._loaded_path = path
        return self

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------
    def state_machine_names(self) -> List[str]:
        return [n for n, s in self.registry.items() if s.get("kind") == "state_machine"]

    def _skill_when_passes(self, skill: dict) -> bool:
        cond = skill.get("when") or {}
        return _cond_pass(cond, self.rules, self.defaults_window_ms)

    def plan_eligible(self) -> List[dict]:
        """
        Simple planning view: which state_machine skills have their top-level
        'when' condition satisfied right now.
        """
        out = []
        for name in self.state_machine_names():
            s = self.registry.get(name) or {}
            if self._skill_when_passes(s):
                out.append({"name": name})
        return out

    # ------------------------------------------------------------------
    # State-machine operations
    # ------------------------------------------------------------------
    def arm(self, skill_name: str, ctx: dict, is_root: bool = True):
        s = self.registry.get(skill_name)
        if not s:
            raise KeyError(f"Unknown skill: {skill_name}")

        if skill_name in getattr(self, "bad_skills", set()):
            raise RuntimeError(
                f"Skill '{skill_name}' is quarantined due to previous errors; refusing to arm."
            )

        kind = s.get("kind")
        if kind != "state_machine":
            raise KeyError(f"Skill '{skill_name}' is not a state_machine (kind={kind})")

        inst = SkillInstance(
            name=skill_name,
            ctx=dict(ctx or {}),
            started_ms=self._now_ms(),
            is_root=is_root,        # ← NEW
        )
        self._active.append(inst)
        if self.logger:
            self.logger.info(f"[SkillEngine] Armed state_machine '{skill_name}' ctx={ctx}")

        return inst

        
    def _spawn_nested_child(self, parent: SkillInstance, skill_name: str, ctx: dict) -> StepHandle:
        """
        Spawn a nested state_machine/composite as a child of `parent`
        and return a StepHandle that tracks its completion.
        """
        # Arm child skill with merged ctx
        child_ctx = dict(parent.ctx)
        child_ctx.update(ctx or {})

        child_inst = self.arm(skill_name, child_ctx, is_root=False)

        h = StepHandle()

        def cancel_child():
            # Best-effort cancel: stop its primitive and mark done
            child_inst.done = True
            if child_inst.handle is not None and not child_inst.handle.done():
                try:
                    child_inst.handle.cancel()
                except Exception:
                    pass
            h.mark_done()

        h._cancel = cancel_child  # replace cancel with child-aware version

        # Make this handle's done() proxy the child's done
        def done_proxy(self):
            return child_inst.done

        h.done = done_proxy.__get__(h, StepHandle)
        return h

        
    def run(self, skill_name: str, ctx: dict):
        """
        Backwards-compatible helper: arm + tick once.
        Usually you just call arm() and let the timer tick.
        """
        self.arm(skill_name, ctx)
        self.tick()

    def active_count(self) -> int:
        return sum(1 for i in self._active if not i.done)

    # ------------------------------------------------------------------
    # State lookup and transitions
    # ------------------------------------------------------------------
    def _find_state(self, skill: dict, state_id: str | None) -> Optional[dict]:
        if not state_id:
            return None
        for idx, st in enumerate(skill.get("states", [])):
            if st.get("id") == state_id:
                st = dict(st)  # copy
                st["_idx"] = idx
                return st
        return None

    def _enter_state(self, inst: SkillInstance, skill: dict, state: dict):
        """Enter a new state: set ids, start primitive or nested skill if action, emit events."""
        now_ms = self._now_ms()
        inst.state_id = state.get("id")
        inst.state_idx = int(state.get("_idx", inst.state_idx))
        inst.state_started_ms = now_ms
        inst.handle = None

        st_type = state.get("type", "action")
        extra = {"state_type": st_type}
        self._emit_event("state_entered", inst, extra)

        if self.logger:
            self.logger.info(
                f"[SkillEngine] Skill '{inst.name}': entering state "
                f"'{inst.state_id}' (idx={inst.state_idx}, type={st_type})"
            )

        if st_type != "action":
            # wait states and others don't launch a step here
            return

        action_spec = state.get("action") or {}
        use_name = action_spec.get("use")
        with_ctx = action_spec.get("with") or {}

        if not use_name:
            if self.logger:
                self.logger.warn(f"[SkillEngine] action state '{inst.state_id}' missing 'use'")
            return

        base = self.registry.get(use_name)

        # Case 1: registry primitive → normal primitive execution
        if base and base.get("kind") == "primitive":
            params = dict(base.get("params") or {})
            params.update(with_ctx)

            if self.logger:
                self.logger.info(
                    f"[SkillEngine] state '{inst.state_id}' executing primitive '{use_name}' params={params}"
                )

            # Emit step_started with primitive field
            self._emit_event(
                "step_started",
                inst,
                {"primitive": use_name, "state_type": "action"},
            )

            inst.handle = self._exec_primitive_get_handle(base["action"], params, inst.ctx)
            return

        # Case 2: registry state_machine/composite → nested skill
        if base and base.get("kind") in ("state_machine", "composite"):
            if self.logger:
                self.logger.info(
                    f"[SkillEngine] state '{inst.state_id}' spawning nested skill '{use_name}' with ctx={with_ctx}"
                )

            # Emit step_started with composite field
            self._emit_event(
                "step_started",
                inst,
                {"composite": use_name, "state_type": "action"},
            )

            inst.handle = self._spawn_nested_child(inst, use_name, with_ctx)
            return

        # Case 3: direct binding (no registry entry, `use_name` is binding key)
        params = with_ctx
        if self.logger:
            self.logger.info(
                f"[SkillEngine] state '{inst.state_id}' executing bound action '{use_name}' params={params}"
            )

        # Emit step_started as a primitive-like bound action
        self._emit_event(
            "step_started",
            inst,
            {"primitive": use_name, "state_type": "action"},
        )

        inst.handle = self._exec_primitive_get_handle(use_name, params, inst.ctx)

    def _transition_to(self, inst: SkillInstance, skill: dict, next_id: Optional[str]):
        """Transition to next state id or finish skill if next_id is falsy."""
        cur_state = inst.state_id
        
        if not next_id:
            inst.done = True
            if self.logger:
                self.logger.info(f"[SkillEngine] Skill '{inst.name}' finished (no next state)")
            self._emit_event("skill_finished", inst, {"reason": "no_next_state"})
            return

        if self.logger:
            self.logger.info(
                f"[SkillEngine] Skill '{inst.name}': state '{cur_state}' -> '{next_id}'"
            )

        st = self._find_state(skill, next_id)
        if not st:
            # If the next state does not exist, consider the skill finished.
            inst.done = True
            if self.logger:
                self.logger.warn(
                    f"[SkillEngine] Skill '{inst.name}' next state '{next_id}' not found; finishing."
                )
            self._emit_event("skill_finished", inst, {"reason": "missing_state"})
            return

        self._enter_state(inst, skill, st)

    def _resolve_wait_next_state(self, state: dict, rule_id: str) -> Optional[str]:
        """
        Given a WAIT state and the rule_id that fired, decide next state.

        Priority:
          1) First matching entry in state['branches'] (if present)
          2) state['on_event'] (if present)
          3) None (no transition)
        """
        branches = state.get("branches") or []
        for br in branches:
            if br.get("rule_id") == rule_id:
                return br.get("next")

        # fallback: plain on_event
        return state.get("on_event")


    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------
    def tick(self):
        """
        Advance all active state_machine instances.
        Call this at 10-20 Hz from the ROS node.
        """
        now_ms = self._now_ms()

        for inst in list(self._active):
            if inst.done:
                continue

            try:
                skill = self.registry.get(inst.name) or {}
                kind = skill.get("kind")

                if kind != "state_machine":
                    # Should not happen, but guard anyway
                    inst.done = True
                    continue

                comp_when = skill.get("when") or {}
                comp_until = skill.get("until") or {}

                # If not activated yet, wait for top-level "when"
                if not inst.activated:
                    if _cond_pass(comp_when, self.rules, self.defaults_window_ms):
                        inst.activated = True
                        inst.started_ms = now_ms
                        if self.logger:
                            self.logger.info(
                                f"[SkillEngine] Activating state_machine '{inst.name}' "
                                f"with ctx={inst.ctx}"
                            )
                        self._emit_event("skill_started", inst, {})

                        if not inst.state_id:
                            # Enter initial state
                            init_id = skill.get("initial_state")
                            st = self._find_state(skill, init_id)
                            if not st and self.logger:
                                self.logger.error(
                                    f"[SkillEngine] Skill '{inst.name}' missing initial_state '{init_id}'"
                                )
                                # Quarantine this skill for future runs
                                self._quarantine_skill(inst, "missing_initial_state", None)
                                inst.done = True
                                self._emit_event(
                                    "skill_finished",
                                    inst,
                                    {"reason": "missing_initial_state"},
                                )
                                continue
                            self._enter_state(inst, skill, st)
                    else:
                        # Still not activated
                        continue

                # Composite-level until: stop skill even if current state has not finished
                if _cond_pass(comp_until, self.rules, self.defaults_window_ms, empty_means=False):
                    # cancel running primitive if any
                    if inst.handle is not None and not inst.handle.done():
                        try:
                            inst.handle.cancel()
                        except Exception:
                            pass
                    inst.done = True
                    if self.logger:
                        self.logger.info(
                            f"[SkillEngine] Finished '{inst.name}' due to composite-level 'until'"
                        )
                    self._emit_event("skill_finished", inst, {"reason": "composite_until"})
                    continue

                # No current state? Nothing to do.
                st = self._find_state(skill, inst.state_id)
                if not st:
                    inst.done = True
                    # structural problem -> quarantine
                    self._quarantine_skill(inst, "missing_state", None)
                    self._emit_event("skill_finished", inst, {"reason": "no_state"})
                    continue

                st_type = st.get("type", "action")

                # 1) Action states: wait for primitive handle to complete
                if st_type == "action":
                    if inst.handle is None:
                        # We entered state but did not start primitive (should not happen)
                        if self.logger:
                            self.logger.warn(
                                f"[SkillEngine] Skill '{inst.name}' state '{inst.state_id}' "
                                "has no active handle; starting primitive now."
                            )
                        self._enter_state(inst, skill, st)
                        continue

                    if not inst.handle.done():
                        # Still in progress
                        continue

                    if self.logger:
                        self.logger.info(
                            f"[SkillEngine] Skill '{inst.name}' action state '{inst.state_id}' "
                            "completed; selecting on_complete transition."
                        )

                    next_id = st.get("on_complete")
                    self._transition_to(inst, skill, next_id)
                    continue

                # 2) Wait states: look for rule events or timeout
                if st_type == "wait":
                    wait_spec = st.get("wait_for") or {}
                    any_of = wait_spec.get("any_of") or []

                    # a) event fired?
                    fired = False
                    fired_rule = None
                    for cond in any_of:
                        rid = cond.get("rule_id")
                        win = int(cond.get("within_ms") or self.defaults_window_ms)
                        if rid and self.rules.exists(rid, win):
                            fired = True
                            fired_rule = rid
                            break

                    if fired:
                        if self.logger:
                            self.logger.info(
                                f"[SkillEngine] Skill '{inst.name}' wait state '{inst.state_id}' "
                                f"event fired (rule='{fired_rule}'); resolving branches/on_event."
                            )
                        next_id = self._resolve_wait_next_state(st, fired_rule)
                        self._transition_to(inst, skill, next_id)
                        continue

                    # b) timeout?
                    if any_of:
                        max_wait = max(int(c.get("within_ms") or self.defaults_window_ms) for c in any_of)
                    else:
                        max_wait = self.defaults_window_ms

                    if now_ms - inst.state_started_ms >= max_wait:
                        if self.logger:
                            self.logger.info(
                                f"[SkillEngine] Skill '{inst.name}' wait state '{inst.state_id}' "
                                f"timed out after {max_wait} ms; transitioning via on_timeout."
                            )
                        next_id = st.get("on_timeout")
                        self._transition_to(inst, skill, next_id)
                        continue

                    # still waiting
                    continue

                # 3) Unknown state type -> finish + quarantine
                if self.logger:
                    self.logger.warn(
                        f"[SkillEngine] Skill '{inst.name}' state '{inst.state_id}' has unknown type '{st_type}'"
                    )
                inst.done = True
                self._quarantine_skill(inst, "bad_state_type", None)
                self._emit_event("skill_finished", inst, {"reason": "bad_state_type"})

            except Exception as e:
                # RUNTIME ERROR: quarantine for future runs,
                # but for THIS instance try to advance to a reasonable next state.
                if self.logger:
                    self.logger.error(
                        f"[SkillEngine] Runtime error in skill '{inst.name}' "
                        f"state='{inst.state_id}': {e!r}"
                    )
                self._quarantine_skill(inst, "runtime_exception", e)

                # Try to move to "next" state instead of killing the instance
                skill = self.registry.get(inst.name) or {}
                st = self._find_state(skill, inst.state_id)
                st_type = st.get("type", "action") if st else "action"

                next_id = None
                if st_type == "action":
                    # Prefer explicit on_failure, else fall back to on_complete
                    next_id = st.get("on_failure") or st.get("on_complete")
                elif st_type == "wait":
                    # Prefer timeout path, else generic event path
                    next_id = st.get("on_timeout") or st.get("on_event")

                # If we couldn't find anything, finishing is still safe
                self._transition_to(inst, skill, next_id)

        # prune finished instances
        self._active = [i for i in self._active if not i.done]



# ───────────────────────────────────────────────────────────────────────────────
#                          ROS RulesView + Orchestrator
# ───────────────────────────────────────────────────────────────────────────────
class RulesViewROS(RulesView):
    """
    Bridge EventLayerNode topics into a RulesView for the engine.
    Subscribes to:
      /events/basic     {"ts": <float>, "rule": <id>, "data": {...}}
      /events/composite {"ts": <float>, "rule": <id>, "expr": <str>}
    """
    def __init__(self, node: Node, window_ms: int = 3000):
        super().__init__(now_ms=None)
        self.node = node
        self.window_ms_default = int(window_ms)
        self.sub_basic = node.create_subscription(StringMsg, '/events/basic', self._on_basic, 100)
        self.sub_comp  = node.create_subscription(StringMsg, '/events/composite', self._on_comp, 100)

    def _now(self) -> int:
        return int(self.node.get_clock().now().nanoseconds * 1e-6)

    def add(self, rule_id: str, payload: dict, ts_ms: Optional[int] = None):
        entry = {'id': str(rule_id), 'ts_ms': ts_ms if ts_ms is not None else self._now(), 'payload': payload or {}}
        self._events.append(entry)
        cutoff = self._now() - self.window_ms_default
        self._events[:] = [e for e in self._events if e['ts_ms'] >= cutoff]

    def _on_basic(self, msg: StringMsg):
        try:
            obj = json.loads(msg.data)
            ts_ms = int(float(obj.get('ts', time.time())) * 1000)
            rid = obj.get('rule')
            payload = obj.get('data') or {}
            if rid:
                self.add(str(rid), payload, ts_ms=ts_ms)
        except Exception as e:
            self.node.get_logger().warn(f"RulesViewROS basic parse error: {e}")

    def _on_comp(self, msg: StringMsg):
        try:
            obj = json.loads(msg.data)
            ts_ms = int(float(obj.get('ts', time.time())) * 1000)
            rid = obj.get('rule')
            payload = {'expr': obj.get('expr', '')}
            if rid:
                self.add(str(rid), payload, ts_ms=ts_ms)
        except Exception as e:
            self.node.get_logger().warn(f"RulesViewROS composite parse error: {e}")


# ───────────────────────────────────────────────────────────────────────────────
#                            Low-level helpers (quaternion)
# ───────────────────────────────────────────────────────────────────────────────
def yaw_to_q(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw/2.0)
    q.w = math.cos(yaw/2.0)
    return q

# ───────────────────────────────────────────────────────────────────────────────
#                              The Agent Node
# ───────────────────────────────────────────────────────────────────────────────
class SkillsAgent(Node):
    """
    Implements:
      - Basic actions (tts, gesture, navigate abs/rel, turn search, beacon DB)
      - Rules ingestion (/events/*) and hot-reload of skills library
      - Planning + Execution APIs (services + topics)
    """
    def __init__(self):
        super().__init__("skills_agent")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("rotate_topic", "/cmd_vel")
        self.declare_parameter("approach_dist_m", 1.0)
        self.declare_parameter("bt_db_path", os.path.expanduser("~/.bt_rssi_map.sqlite"))
        self.declare_parameter("search_ang_speed", 0.6)
        self.declare_parameter("full_turn_margin_deg", 5.0)
        self.declare_parameter("turn_ref_frame", "odom")
        self.declare_parameter("smoothing_alpha", 0.4)
        self.declare_parameter("name_max_ang_speed", 1.0)

        # skills library config
        self.declare_parameter("skills_base_path", "")
        self.declare_parameter("skills_composite_path", "")
        self.declare_parameter("skills_rescan_period_s", 1.0)

        self.declare_parameter("turn_speed_rad_s", 0.6)  # was 0.25
        self.declare_parameter("fwd_speed_m_s", 0.25)
        
        
        # Box server (for calling /sense directly from skills)
        self.declare_parameter("box_server_url", "http://172.17.40.64:8080")
        self.declare_parameter("box_req_timeout", 5.0)
        self.declare_parameter("agent_id", "robot")  # logical agent id for /sense

        self.box_server_url = self.get_parameter("box_server_url").get_parameter_value().string_value
        self.box_req_timeout = float(self.get_parameter("box_req_timeout").value)
        self.agent_id = self.get_parameter("agent_id").get_parameter_value().string_value or "robot"

        
        self.turn_speed = float(self.get_parameter("turn_speed_rad_s").value)
        self.fwd_speed  = float(self.get_parameter("fwd_speed_m_s").value)

        self.rotate_topic    = self.get_parameter("rotate_topic").get_parameter_value().string_value
        self.approach_dist   = float(self.get_parameter("approach_dist_m").value)
        self.bt_db_path      = self.get_parameter("bt_db_path").get_parameter_value().string_value
        self.search_w        = float(self.get_parameter("search_ang_speed").value)
        self.full_turn_eps   = math.radians(max(0.0, 180.0 - float(self.get_parameter("full_turn_margin_deg").value)))
        self.turn_ref_frame  = self.get_parameter("turn_ref_frame").get_parameter_value().string_value
        self.alpha           = float(self.get_parameter("smoothing_alpha").value)
        self.name_max_w      = float(self.get_parameter("name_max_ang_speed").value)

        # NEW: paths + mtimes
        self.skills_base_path      = self.get_parameter("skills_base_path").get_parameter_value().string_value
        self.skills_composite_path = self.get_parameter("skills_composite_path").get_parameter_value().string_value
        self._skills_rescan  = float(self.get_parameter("skills_rescan_period_s").value)
        self._skills_base_mtime: Optional[float] = None
        self._skills_comp_mtime: Optional[float] = None

        # ── ROS I/O for basic actions ─────────────────────────────────────────
        self.tts_pub     = self.create_publisher(StringMsg, "tts", 10)
        self.webrtc_req_pub = self.create_publisher(WebRtcReq, "webrtc_req", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.rotate_topic, 10)
        self.nav_client  = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.coverage_req_pub = self.create_publisher(StringMsg, "/coverage/req", 10)
        self.coverage_status_sub = self.create_subscription(
            StringMsg, "/coverage/status", self._cb_coverage_status, 10
        )
        self._coverage_pending = {}   # id -> {"handle": StepHandle, "ctx": dict}
        self._coverage_next_id = 1


        # LLM speech_check req/resp
        self.llm_speech_req_pub = self.create_publisher(StringMsg, "/llm/speech_check_req", 10)
        self.llm_speech_resp_sub = self.create_subscription(
            StringMsg,
            "/llm/speech_check_resp",
            self._cb_llm_speech_resp,
            10,
        )
        
        self.sub_tts_immediate = self.create_subscription(
            StringMsg,
            "/skills/tts_immediate",
            self._on_tts_immediate,
            10,
        )
        
        self._llm_speech_pending = {}   # req_id -> {"handle": StepHandle, "ctx": dict}
        self._llm_speech_next_id = 1

        # VLM request/response (generic, like llm_speech_check)
        self.vlm_req_pub = self.create_publisher(
            StringMsg, "/vlm/req", 10
        )
        self.vlm_resp_sub = self.create_subscription(
            StringMsg,
            "/vlm/answer",
            self._cb_vlm_resp,
            10,
        )
        self._vlm_pending = {}   # req_id -> {"handle": StepHandle, "ctx": dict}
        self._vlm_next_id = 1


        # TF buffer for turning reference
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Turn/Spin state
        self.last_cmd = Twist()
        self._search_active = False
        self._search_dir    = 1.0
        self._search_prev_yaw = 0.0
        self._search_turned_abs = 0.0
        self._twist_timer = None
        
        self._live_timers = set()  # keep strong refs to timers so GC can't kill them

        # Control loop (20 Hz) for turn_search
        #self.create_timer(0.05, self._control_loop)

        # ── Rules + Engine + Orchestrator ─────────────────────────────────────
        self.rules_view = RulesViewROS(self, window_ms=3000)
        self.skill_engine = SkillEngineV2(self._make_bindings_for_self(), self.rules_view, logger=self.get_logger(), event_cb=self._on_skill_event)

        self._load_skills_initial()



        # ── Planning & Execution APIs ─────────────────────────────────────────
        # Topics:
        #   /skills/execute  (String JSON): {"skill":"sense.here","ctx":{...}} OR {"rule":"greet_with_presence","ctx":{...}}
        #   /skills/plan_req (String JSON or empty) -> reply on /skills/plan_result
        self.sub_execute = self.create_subscription(StringMsg, "/skills/execute", self._on_execute_msg, 20)
        self.sub_planreq = self.create_subscription(StringMsg, "/skills/plan_req", self._on_plan_req_msg, 10)
        self.pub_planres = self.create_publisher(StringMsg, "/skills/plan_result", 10)

        # Services:
        #   /skills/reload (Trigger)     : reload library from file or fallback default
        #   /skills/plan   (Trigger)     : return eligible skills as JSON in response.message
        #   /skills/run_all_eligible (Trigger) : run every eligible skill once (use with care)
        self.create_service(Trigger, "/skills/reload", self._srv_reload_skills)
        self.create_service(Trigger, "/skills/plan", self._srv_plan)
        self.create_service(Trigger, "/skills/run_all_eligible", self._srv_run_all_eligible)

        self.create_service(Trigger, "/skills/cancel_all", self._srv_cancel_all)


        # Hot-reload timer
        #self.create_timer(self._skills_rescan, self._maybe_reload_skills)

        self.skill_status_pub = self.create_publisher(StringMsg, "/skills/status", 10)

        # Track TTS playback state from TTSPlayerNode (/tts_busy)
        self._tts_busy = False          # current busy flag
        self._tts_has_busy = False      # did we ever see /tts_busy?
        self._tts_waiting: List[StepHandle] = []  # handles waiting for speech to finish

        self.create_subscription(
            Bool,
            "/tts_busy",
            self._cb_tts_busy,
            10,
        )
        
        self.get_logger().info("SkillsAgent ready.")

        # call engine.tick() ~10–20 Hz, lightweight
        self.create_timer(0.05, self._tick_engine)

    def _srv_cancel_all(self, req, resp):
        """
        Cancel all active skills: stop their current primitive (if any)
        and mark them done / prune them from the active list.
        """
        canceled = 0
        for inst in list(self.skill_engine._active):
            if not inst.done:
                # cancel running primitive if there is one
                if inst.handle is not None and not inst.handle.done():
                    try:
                        inst.handle.cancel()
                    except Exception as e:
                        self.get_logger().warn(f"cancel_all: handle cancel error: {e}")
                inst.done = True
                canceled += 1

        # prune finished instances
        self.skill_engine._active = [i for i in self.skill_engine._active if not i.done]

        resp.success = True
        resp.message = f"Canceled {canceled} active skills."
        self.get_logger().info(resp.message)
        return resp


    def _cancel_all_active(self, why: str = "") -> int:
        canceled = 0
        for inst in list(self.skill_engine._active):
            if inst.done:
                continue
            # cancel running primitive if there is one
            if inst.handle is not None and not inst.handle.done():
                try:
                    inst.handle.cancel()
                except Exception as e:
                    self.get_logger().warn(f"cancel_all({why}): handle cancel error: {e}")
            inst.done = True
            canceled += 1

        # prune finished instances
        self.skill_engine._active = [i for i in self.skill_engine._active if not i.done]
        if canceled:
            self.get_logger().info(f"[SkillsAgent] canceled {canceled} active skills ({why})")
        return canceled


    # ───────────────────────────── Skills loading (base + composite) ─────────
    
    def _on_skill_event(self, event: dict):
        """
        Event from SkillEngineV2: publish to /skills/status as JSON.
        Typical payload:
          {
            "kind": "skill_started"|"skill_finished"|"step_started",
            "skill": "sense.here",
            "step_idx": 0,
            "reason": "all_steps" | "composite_until" | ...,
            "ctx": {...},
            "started_ms": ...,
            "activated": true,
            "done": true/false
          }
        """
        # publish status as before
        try:
            msg = StringMsg()
            msg.data = json.dumps(event, ensure_ascii=False)
            self.skill_status_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish skill status: {e}")

        # NEW: say something when a high-level skill finishes
        try:
            if event.get("kind") == "skill_finished" and event.get("is_root", True):
                reason = event.get("reason", "")
                self.get_logger().info(
                    f"[SkillsAgent] High-level skill '{event.get('skill')}' finished; announcing."
                )
                self.say("Execution ended.")
        except Exception as e:
            self.get_logger().warn(f"Failed to emit final TTS on skill_finished: {e}")



    def _cb_tts_busy(self, msg: Bool):
        """
        Track speaking state from TTSPlayerNode.

        We treat each falling edge (True -> False) as 'one utterance finished',
        and complete one pending TTS StepHandle for it.
        """
        prev = self._tts_busy
        self._tts_busy = bool(msg.data)
        self._tts_has_busy = True

        if prev != self._tts_busy:
            self.get_logger().info(f"[TTS busy] {prev} -> {self._tts_busy}")

        # On falling edge (done speaking): complete one waiting handle
        if prev and not self._tts_busy:
            if self._tts_waiting:
                h = self._tts_waiting.pop(0)
                if not h.done():
                    self.get_logger().info("[TTS] marking one waiting handle done (speech finished).")
                    h.mark_done()

    
    def _read_yaml_if_exists(self, path: str) -> Optional[dict]:
        if not path:
            return None
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().error(f"Failed to read skills YAML '{path}': {e}")
            return None

    def _merged_skills_doc(self) -> dict:
        """
        Merge base + composite libraries into a single doc for SkillEngineV2.
        Base is treated as immutable; composite adds more skills on top.
        """
        base_doc = self._read_yaml_if_exists(self.skills_base_path) or {}
        comp_doc = self._read_yaml_if_exists(self.skills_composite_path) or {}

        merged = {
            "version": base_doc.get("version", 2),
            "defaults": base_doc.get("defaults", {"window_ms": 3000}),
            "skills": []
        }

        # merge base skills
        base_skills = base_doc.get("skills") or []
        if isinstance(base_skills, list):
            merged["skills"].extend(base_skills)

        # merge composite skills (append)
        comp_skills = comp_doc.get("skills") or []
        if isinstance(comp_skills, list):
            merged["skills"].extend(comp_skills)

        return merged

    def _load_skills_merged(self):
        """
        Load merged skills into the engine (base + composite).
        If base path is missing or invalid, falls back to DEFAULT_SKILLS_V2.
        """
        if self.skills_base_path:
            try:
                merged = self._merged_skills_doc()
                yaml_text = yaml.safe_dump(merged)
                self.skill_engine.load_from_string(yaml_text)

                # NEW: clear quarantines on reload (new definitions)
                if hasattr(self.skill_engine, "bad_skills"):
                    self.skill_engine.bad_skills.clear()
                    self.get_logger().info("[SkillEngine] Cleared quarantined skills on reload.")


                # update mtimes
                self._skills_base_mtime = os.path.getmtime(self.skills_base_path) if os.path.isfile(self.skills_base_path) else None
                self._skills_comp_mtime = os.path.getmtime(self.skills_composite_path) if os.path.isfile(self.skills_composite_path) else None

                self.get_logger().info(
                    f"Loaded merged skills from base='{self.skills_base_path}', composite='{self.skills_composite_path}'"
                )
                return
            except Exception as e:
                self.get_logger().error(f"Failed to load merged skills: {e}")

        # Fallback: no base file → inline default
        self.skill_engine.load_from_string(DEFAULT_SKILLS_V2)
        self._skills_base_mtime = None
        self._skills_comp_mtime = None
        self.get_logger().info("Loaded inline DEFAULT_SKILLS_V2 (no base skills file)")



    def _sum_rule_field_since(self, rule_id: str, field: str, since_ms: int) -> float:
        """Sum numeric payload[field] for hits of rule_id with ts >= since_ms."""
        total = 0.0
        try:
            for e in self.rules_view._events:
                if e.get('id') == str(rule_id) and e.get('ts_ms', 0) >= int(since_ms):
                    v = e.get('payload', {}).get(field)
                    if isinstance(v, (int, float)):
                        total += float(v)
        except Exception:
            pass
        return total



    def _tick_engine(self):
        try:
            self.skill_engine.tick()
        except Exception as e:
            self.get_logger().error(f"tick error: {e}")

    # ───────────────────────────── Planning / Execution ────────────────────────
    def _eligible_report(self) -> List[dict]:
        """
        Returns a list of {"name": <composite>, "passing_steps": [indices...]}
        """
        return self.skill_engine.plan_eligible()

    def _run_skill_name(self, name: str, ctx: dict):
        self.skill_engine.run(name, ctx or {})


    def _on_tts_immediate(self, msg: StringMsg):
        """
        One-shot immediate TTS. Does NOT touch the SkillEngine at all.
        Expected payload:
          - either plain text (msg.data is the text)
          - or JSON: {"text": "..."}
        """
        text = msg.data or ""
        try:
            # Try to parse JSON first
            obj = json.loads(msg.data)
            if isinstance(obj, dict) and "text" in obj:
                text = str(obj["text"])
        except Exception:
            # Not JSON, treat as raw text
            pass

        if not text.strip():
            self.get_logger().warn("[tts_immediate] empty text, ignoring.")
            return

        self.get_logger().info(f"[tts_immediate] saying: {text!r}")
        # This uses the existing normalization + publish to /tts
        self.say(text)


    # Topic: /skills/execute
    def _on_execute_msg(self, msg: StringMsg):
        try:
            obj = json.loads(msg.data) if msg.data else {}
        except Exception as e:
            self.get_logger().warn(f"/skills/execute bad JSON: {e}")
            return

        try:
            name = str(obj["skill"])
            ctx  = obj.get("ctx") or {}

            # NEW: cancel everything before starting the new skill
            self._cancel_all_active(why=f"before_execute:{name}")

            # NEW: orchestrator explicitly requested this skill → reset quarantine for it
            if hasattr(self.skill_engine, "bad_skills"):
                if name in self.skill_engine.bad_skills:
                    self.get_logger().info(
                        f"/skills/execute: clearing quarantine for skill '{name}' "
                        f"due to explicit orchestrator request."
                    )
                    self.skill_engine.bad_skills.discard(name)

            self.skill_engine.arm(name, ctx)   # ← arm, don’t run once
            self.get_logger().info(
                f"/skills/execute armed '{name}' "
                f"(active={self.skill_engine.active_count()})"
            )
        except KeyError:
            self.get_logger().warn("execute: expected key 'skill'")
        except Exception as e:
            self.get_logger().error(f"execute error: {e}")


    # Topic: /skills/plan_req -> reply on /skills/plan_result
    def _on_plan_req_msg(self, msg: StringMsg):
        try:
            rep = {"ts": time.time(), "eligible": self._eligible_report()}
            self.pub_planres.publish(StringMsg(data=json.dumps(rep)))
        except Exception as e:
            self.get_logger().error(f"plan_req error: {e}")

    # Service: /skills/plan (Trigger)
    def _srv_plan(self, req, resp):
        try:
            eligible = self._eligible_report()
            resp.success = True
            resp.message = json.dumps(eligible)
        except Exception as e:
            resp.success = False
            resp.message = str(e)
        return resp

    # Service: /skills/run_all_eligible (Trigger)
    def _srv_run_all_eligible(self, req, resp):
        try:
            count = 0
            for entry in self._eligible_report():
                self._run_skill_name(entry["name"], ctx={})
                count += 1
            resp.success = True
            resp.message = f"Ran {count} eligible skills."
        except Exception as e:
            resp.success = False
            resp.message = f"run_all_eligible error: {e}"
        return resp

    # Service: /skills/reload (Trigger)
    def _srv_reload_skills(self, req, resp):
        try:
            self._load_skills_merged()
            resp.success = True
            resp.message = "Reloaded merged skills (base + composite)."
            self.get_logger().info(resp.message)
        except Exception as e:
            resp.success = False
            resp.message = f"Reload failed: {e}"
            self.get_logger().error(resp.message)
        return resp


    # ───────────────────────────── Basic Actions ──────────────────────────────

    def _cb_coverage_status(self, msg: StringMsg):
        try:
            obj = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"coverage/status bad JSON: {e}")
            return

        if obj.get("client") not in (None, "skills"):
            return

        req_id = obj.get("id")
        if not req_id:
            return

        pending = self._coverage_pending.get(req_id)
        if not pending:
            return

        ctx = pending["ctx"]
        ctx["coverage"] = obj  # store latest status in ctx for debugging/introspection

        state = obj.get("state", "")
        if state in ("done", "canceled", "error"):
            handle = pending["handle"]
            self._coverage_pending.pop(req_id, None)
            handle.mark_done()


    def _cb_llm_speech_resp(self, msg: StringMsg):
        try:
            obj = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"llm_speech_check_resp bad JSON: {e}")
            return

        req_id = obj.get("id")
        if req_id is None:
            return

        pending = self._llm_speech_pending.pop(req_id, None)
        if not pending:
            return

        handle: StepHandle = pending["handle"]
        ctx = pending["ctx"]

        ctx["speech_check"] = {
            "success": bool(obj.get("success", False)),
            "raw_text": obj.get("raw_text", ""),
            "json_text": obj.get("json_text", ""),
            "model_id": obj.get("model_id", ""),
            "lat_ms": float(obj.get("lat_ms", 0.0)),
            "tag": obj.get("tag", ""),
        }

        self.get_logger().info(
            f"[llm_speech_check] id={req_id} tag={ctx['speech_check']['tag']} "
            f"success={ctx['speech_check']['success']} "
            f"lat={ctx['speech_check']['lat_ms']:.1f} ms"
        )

        handle.mark_done()

    def _cb_vlm_resp(self, msg: StringMsg):
        try:
            obj = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"vlm_resp bad JSON: {e}")
            return

        client = obj.get("client")
        if client not in (None, "skills"):
            # Ignore responses from other callers (e.g., eventlayer)
            return


        req_id = obj.get("id")
        if not req_id:
            return

        key = str(req_id)
        pending = self._vlm_pending.pop(key, None)

        if not pending:
            self.get_logger().warn(f"[VLM] response for unknown id={key!r} (client={client!r}); pending={list(self._vlm_pending.keys())[:5]}")
            return



        handle: StepHandle = pending["handle"]
        ctx = pending["ctx"]

        # Parse json_text into a Python object, if present
        raw_json = obj.get("json_text", "") or ""
        parsed = None
        if raw_json:
            try:
                parsed = json.loads(raw_json)
            except Exception as e:
                self.get_logger().warn(f"[VLM] failed to parse json_text: {e} (snippet={raw_json[:160]!r})")

        # Mirror llm_speech_check, but with parsed structure
        ctx["vlm"] = {
            "success":  bool(obj.get("success", False)),
            "raw_text": obj.get("raw_text", ""),
            "json_text": raw_json,
            "parsed":   parsed,                   # ← NEW: structured dict/list
            "model_id": obj.get("model_id", ""),
            "lat_ms":   float(obj.get("lat_ms", 0.0)),
            "tag":      obj.get("tag", ""),
        }

        self.get_logger().info(
            f"[VLM] id={key} tag={ctx['vlm']['tag']} "
            f"success={ctx['vlm']['success']} "
            f"lat={ctx['vlm']['lat_ms']:.1f} ms "
            f"has_parsed={parsed is not None}"
        )

        handle.mark_done()



    def say(self, text: str):
        # Normalize to string
        raw = str(text)
        # Automatically convert numeric tokens to words
        spoken = _normalize_tts_text(raw)
        self.get_logger().info(f"[TTS] {spoken}")
        self.tts_pub.publish(StringMsg(data=spoken))


    def _lp(self, prev: float, new: float) -> float:
        return (1.0 - self.alpha) * prev + self.alpha * new

    def _publish_smoothed(self, cmd: Twist, why: str = ""):
        sm = Twist()
        sm.linear.x  = self._lp(self.last_cmd.linear.x, cmd.linear.x)
        sm.angular.z = self._lp(self.last_cmd.angular.z, cmd.angular.z)
        self.cmd_vel_pub.publish(sm)
        self.last_cmd = sm

    def _stop_turn(self, why: str = ""):
        self._publish_smoothed(Twist(), why)

    def _current_yaw_ref_base(self) -> float:
        try:
            tf_base_to_ref = self.tf_buffer.lookup_transform(
                self.turn_ref_frame, "base_link", Time(), Duration(seconds=0.2)
            )
            q = tf_base_to_ref.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return math.atan2(siny_cosp, cosy_cosp)
        except Exception as e:
            self.get_logger().warn(f"Yaw lookup ({self.turn_ref_frame}) failed: {e}")
            return 0.0

    @staticmethod
    def _wrap_pi(a: float) -> float:
        return (a + math.pi) % (2.0*math.pi) - math.pi

    def _start_person_search(self, doa_deg: float):
        if self._search_active:
            return
        self._search_dir = 1.0 if float(doa_deg) >= 0.0 else -1.0
        yaw_now = self._current_yaw_ref_base()
        self._search_prev_yaw   = yaw_now
        self._search_turned_abs = 0.0
        self._search_active     = True
        self._stop_turn("start-person-search")
        self.get_logger().info(
            f"Person search started. dir={'left' if self._search_dir>0 else 'right'}, speed={self.search_w:.2f} rad/s"
        )

    def _stop_person_search(self, why: str = ""):
        self._search_active = False
        self._stop_turn(why or "stop-person-search")

    def _control_loop(self):
        if not self._search_active:
            return
        yaw_now = self._current_yaw_ref_base()
        dyaw = self._wrap_pi(yaw_now - self._search_prev_yaw)
        self._search_prev_yaw = yaw_now
        self._search_turned_abs += abs(dyaw)
        if self._search_turned_abs >= (2.0*math.pi - math.radians(5.0)):
            self._stop_person_search("full-turn-complete")
            self.say("I didn’t find you.")
            return
        cmd = Twist()
        cmd.angular.z = self._search_dir * max(0.05, min(self.search_w, self.name_max_w))
        self._publish_smoothed(cmd, "person-search")

    # Nav2 absolute
    def navigate_absolute(self, frame: str, x: float, y: float, yaw: float):
        if not self.nav_client.wait_for_server(timeout_sec=0.5):
            self.say("Navigation is not available.")
            return
        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.orientation = yaw_to_q(float(yaw))
        goal.pose = ps
        send_future = self.nav_client.send_goal_async(goal)
        send_future.add_done_callback(self._on_nav_goal_response_abs)

    def _on_nav_goal_response_abs(self, future):
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.say("Failed to send navigation goal.")
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_nav_result_abs)

    def _on_nav_result_abs(self, future):
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f"Nav result error (absolute): {e}")
            self.say("Navigation failed.")
            return
        if getattr(result, "status", 0) == GoalStatus.STATUS_SUCCEEDED:
            self.say("Arrived at the destination.")
        else:
            self.say("Navigation failed.")

    # Nav2 relative
    def navigate_relative(self, az_deg: float, dist_m: float):
        """
        Relative move composed of:
          1) rotate-in-place by az_deg (cmd_vel)
          2) forward nudge by dist_m (cmd_vel)
        No Nav2 goals are sent here.
        """
        # --- params / clamps ---
        az_deg = float(az_deg)
        dist_m = float(dist_m)
        turn_speed = 0.25  # rad/s (tunable: expose as a ROS param if you want)
        fwd_speed  = 0.25  # m/s   (tunable)
        min_turn_deg = 3.0
        eps_dist = 0.02

        # --- 1) rotate ---
        if abs(az_deg) >= min_turn_deg:
            target = math.radians(az_deg)
            w = turn_speed if target >= 0 else -turn_speed
            # duration = angle / speed; clamp to reasonable bounds
            rot_duration = max(0.1, min(abs(target) / max(abs(w), 1e-3), 4.0))
            self._start_twist_timer(duration_s=float(rot_duration),
                                    linear_x=0.0, angular_z=w)

            # Let the rotation finish before the forward nudge
            # Chain the forward phase by scheduling it after rot_duration
            def _after_turn():
                if dist_m > eps_dist:
                    t = dist_m / max(fwd_speed, 0.05)
                    self._start_twist_timer(duration_s=float(t),
                                            linear_x=fwd_speed, angular_z=0.0,
                                            on_complete=lambda: self.say("Moved closer."))
            # one-shot timer to start forward after turn completes
            self.create_timer(rot_duration, _after_turn)
        else:
            # No meaningful rotation; just forward if requested
            if dist_m > eps_dist:
                t = dist_m / max(fwd_speed, 0.05)
                self._start_twist_timer(duration_s=float(t),
                                        linear_x=fwd_speed, angular_z=0.0,
                                        on_complete=lambda: self.say("Moved closer."))




    # Timed Twist helper
    def _start_twist_timer(self, duration_s: float, *, linear_x=0.0, angular_z=0.0, on_complete=None):
        """
        Publishes a Twist at 20 Hz for duration_s seconds using a Timer.
        Keeps a strong reference to the timer to prevent GC.
        Returns the timer so callers can cancel it.
        """
        period = 0.05  # 20 Hz
        remaining = float(duration_s)

        tw = Twist()
        tw.linear.x = float(linear_x)
        tw.angular.z = float(angular_z)

        # publish once immediately so you don't wait for the first tick
        self.cmd_vel_pub.publish(tw)
        self.get_logger().info(f"[twist] start: vx={tw.linear.x:.3f} wz={tw.angular.z:.3f} for {remaining:.2f}s")

        state = {"remaining": remaining}

        def _tick():
            state["remaining"] -= period
            if state["remaining"] > 0.0:
                self.cmd_vel_pub.publish(tw)
            else:
                # stop and cleanup
                self.cmd_vel_pub.publish(Twist())
                self.get_logger().info("[twist] stop")
                try:
                    timer.cancel()
                except Exception:
                    pass
                # drop strong ref so GC can reclaim
                self._live_timers.discard(timer)
                if callable(on_complete):
                    try:
                        on_complete()
                    except Exception as e:
                        self.get_logger().warn(f"on_complete error: {e}")

        timer = self.create_timer(period, _tick)
        # keep the timer alive
        self._live_timers.add(timer)
        return timer


    # Beacon DB speech
    def _say_top3_beacons_from_db(self, top_n: int = 3):
        db = os.path.expanduser(self.bt_db_path)
        if not os.path.exists(db):
            self.say("I don’t have any beacon data yet.")
            return
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro&cache=shared", uri=True, timeout=0.5)
            cur = conn.cursor()
            cur.execute("""
                WITH best AS (
                  SELECT
                    om.object_id,
                    COALESCE(
                      (SELECT rssi FROM obj_measurements WHERE object_id=om.object_id AND slot='current' LIMIT 1),
                      (SELECT rssi FROM obj_measurements WHERE object_id=om.object_id AND slot='min' LIMIT 1)
                    ) AS rssi,
                    COALESCE(
                      (SELECT contaminated FROM obj_measurements WHERE object_id=om.object_id AND slot='current' LIMIT 1),
                      (SELECT contaminated FROM obj_measurements WHERE object_id=om.object_id AND slot='min' LIMIT 1)
                    ) AS contaminated_local,
                    COALESCE(
                      (SELECT probability FROM obj_measurements WHERE object_id=om.object_id AND slot='current' LIMIT 1),
                      (SELECT probability FROM obj_measurements WHERE object_id=om.object_id AND slot='min' LIMIT 1)
                    ) AS probability_local,
                    COALESCE(
                      (SELECT phone_id FROM obj_measurements WHERE object_id=om.object_id AND slot='current' LIMIT 1),
                      (SELECT phone_id FROM obj_measurements WHERE object_id=om.object_id AND slot='min' LIMIT 1)
                    ) AS phone_id
                  FROM obj_measurements om
                  GROUP BY om.object_id
                ),
                merged AS (
                  SELECT
                    b.object_id,
                    b.rssi,
                    COALESCE(b.contaminated_local, cr.contaminated) AS contaminated,
                    COALESCE(b.probability_local,  cr.probability)  AS probability
                  FROM best b
                  LEFT JOIN contamination_records cr
                    ON cr.object_id = b.object_id AND cr.phone_id = b.phone_id
                )
                SELECT object_id, rssi, contaminated, probability
                FROM merged
                ORDER BY rssi DESC
                LIMIT ?;
            """, (int(top_n),))
            rows = cur.fetchall()
            conn.close()
        except Exception as e:
            self.get_logger().warn(f"DB read failed: {e}")
            self.say("I couldn’t read the beacon map.")
            return

        if not rows:
            self.say("I don’t have any beacons detected yet.")
            return



        items_spoken = []
        for object_id, rssi, contaminated, probability in rows:
            try:
                rssi_i = int(rssi)
            except Exception:
                rssi_i = -999
            rssi_words = f"{_num_to_words(rssi_i)} decibels"
            contam_words = ""
            if contaminated is not None:
                contam_words = " contaminated" if int(contaminated) == 1 else " clean"
                if probability is not None:
                    try:
                        p = float(probability)
                        if p <= 1.0: p *= 100.0
                        p_str = f"{p:.1f}".rstrip("0").rstrip(".")
                        contam_words += f" at {p_str} percent"
                    except Exception:
                        pass
            node_id_spoken = str(object_id)
            try:
                nid = int(str(object_id).split("CNode")[1])
                node_id_spoken = f"node {_num_to_words(nid)}"
            except Exception:
                pass
            phrase = f"{node_id_spoken}: {rssi_words}"
            if contam_words:
                phrase += f", {contam_words.strip()}"
            items_spoken.append(phrase)

        if len(items_spoken) == 1:
            self.say(f"The strongest signal is {items_spoken[0]}.")
        else:
            spoken = ", ".join(items_spoken[:-1]) + ", and " + items_spoken[-1] if len(items_spoken) > 2 else " and ".join(items_spoken)
            self.say(f"The top {min(top_n, len(items_spoken))} signals are {spoken}.")

    # Bindings factory (so actions call our methods)
    def _make_bindings_for_self(self):
    
        def sense_box(node_id: str, property: str = "X", ctx: dict = None):
            """
            Sensing primitive.

            Synchronously calls the FastAPI /sense endpoint:

              POST /sense
              {
                "agent_id": <self.agent_id>,
                "box_id":   <int>,
                "property": "X" | "Y"
              }

            and records the result into ctx["box"]["sense_result"].

            On cancel(), we best-effort call /sense/cancel for the same
            (agent_id, box_id, property).
            """
            h = StepHandle()
            if ctx is None:
                ctx = {}

            node_id = str(node_id or "").strip()
            if not node_id:
                self.get_logger().warn("[box.sense] empty node_id; skipping.")
                h.mark_done()
                return h

            if not self.box_server_url:
                self.get_logger().warn("[box.sense] box_server_url is empty; skipping /sense call.")
                h.mark_done()
                return h

            # Map CNode1## → int box_id
            def _box_id_from_node_id(nid: str) -> int | None:
                s = str(nid).strip()
                if s.lower().startswith("cnode"):
                    s = s[5:]
                try:
                    return int(s)
                except Exception:
                    return None

            box_id = _box_id_from_node_id(node_id)
            if box_id is None:
                self.get_logger().warn(f"[box.sense] could not map node_id={node_id!r} to box_id; skipping.")
                h.mark_done()
                return h

            prop = str(property or "X").upper()
            if prop not in ("X", "Y"):
                self.get_logger().warn(f"[box.sense] invalid property={property!r}; forcing 'X'.")
                prop = "X"

            base_url   = self.box_server_url.rstrip("/")
            url        = base_url + "/sense"
            cancel_url = base_url + "/sense/cancel"

            payload = {
                "agent_id": self.agent_id,
                "box_id":   box_id,
                "property": prop,
            }

            self.get_logger().info(f"[box.sense] POST {url} {payload}")

            result = {
                "status": None,          # "completed" | "cached" | "cancelled"
                "detected": None,        # bool | None
                "probability": None,     # float | None
                "deadline": None,
                "x": None,
                "y": None,
                "requested_at": None,
                "completed_at": None,
                "error": None,
            }

            # --- define cancel hook before the blocking call ---
            def _cancel():
                """
                Best-effort cancellation via /sense/cancel.

                If the server still has a running sense op for this triple,
                it will mark it cancelled and wake up the sleep.
                """
                try:
                    cancel_payload = dict(payload)  # same keys
                    self.get_logger().info(f"[box.sense] POST {cancel_url} {cancel_payload} (cancel)")
                    resp_c = requests.post(cancel_url, json=cancel_payload, timeout=self.box_req_timeout)
                    self.get_logger().info(
                        f"[box.sense] cancel response code={resp_c.status_code} "
                        f"body={resp_c.text[:160]!r}"
                    )
                except Exception as e:
                    self.get_logger().warn(f"[box.sense] /sense/cancel failed: {e}")
                finally:
                    h.mark_done()

            h._cancel = _cancel

            # --- blocking call to /sense ---
            try:
                resp = requests.post(url, json=payload, timeout=self.box_req_timeout)
                if resp.status_code != 200:
                    msg = f"non-200 status {resp.status_code}"
                    self.get_logger().warn(f"[box.sense] /sense {payload} -> {msg}")
                    result["error"] = msg
                else:
                    data = resp.json()
                    # Expected fields from SenseResponse:
                    #   agent_id, box_id, property, status,
                    #   detected, probability, deadline, x, y,
                    #   requested_at, completed_at
                    result.update(
                        status=data.get("status"),
                        detected=data.get("detected"),
                        probability=data.get("probability"),
                        deadline=data.get("deadline"),
                        x=data.get("x"),
                        y=data.get("y"),
                        requested_at=data.get("requested_at"),
                        completed_at=data.get("completed_at"),
                    )
            except Exception as e:
                msg = f"/sense failed: {e}"
                self.get_logger().warn(f"[box.sense] {msg}")
                result["error"] = msg

            # Expose result via ctx
            ctx.setdefault("box", {})
            ctx["box"].update({
                "node_id": node_id,
                "box_id": box_id,
                "property": prop,
                "sense_result": result,
            })

            self.get_logger().info(
                f"[box.sense] box_id={box_id}, prop={prop}, "
                f"status={result['status']}, detected={result['detected']}, "
                f"prob={result['probability']}"
            )

            h.mark_done()
            return h


        def dispose_box(node_id: str, property: str = "X", ctx: dict = None):
            """
            Disposal primitive.

            Synchronously calls the FastAPI /dispose endpoint:

              POST /dispose
              {
                "agent_id": <self.agent_id>,
                "box_id":   <int>,
                "property": "X" | "Y"
              }

            and records the result into ctx["box"]["dispose_result"].

            On cancel(), we best-effort call /dispose/cancel for the same
            (agent_id, box_id, property).
            """
            h = StepHandle()
            if ctx is None:
                ctx = {}

            node_id = str(node_id or "").strip()
            if not node_id:
                self.get_logger().warn("[box.dispose] empty node_id; skipping.")
                h.mark_done()
                return h

            if not self.box_server_url:
                self.get_logger().warn("[box.dispose] box_server_url is empty; skipping /dispose call.")
                h.mark_done()
                return h

            box_id = _box_id_from_node_id(node_id)
            if box_id is None:
                self.get_logger().warn(f"[box.dispose] could not map node_id={node_id!r} to box_id; skipping.")
                h.mark_done()
                return h

            prop = str(property or "X").upper()
            if prop not in ("X", "Y"):
                self.get_logger().warn(f"[box.dispose] invalid property={property!r}; forcing 'X'.")
                prop = "X"

            base_url   = self.box_server_url.rstrip("/")
            url        = base_url + "/dispose"
            cancel_url = base_url + "/dispose/cancel"

            payload = {
                "agent_id": self.agent_id,
                "box_id":   box_id,
                "property": prop,
            }

            self.get_logger().info(f"[box.dispose] POST {url} {payload}")

            result = {
                "status": None,          # "completed" | "cancelled"
                "success": None,         # bool | None
                "deadline": None,
                "x": None,
                "y": None,
                "requested_at": None,
                "completed_at": None,
                "error": None,
            }

            # --- define cancel hook *before* the blocking request ---
            def _cancel():
                """
                Best-effort cancellation via /dispose/cancel.

                If the server still has a running disposal for this triple,
                it will mark it cancelled and wake up the sleep.
                """
                try:
                    cancel_payload = dict(payload)  # same keys: agent_id, box_id, property
                    self.get_logger().info(f"[box.dispose] POST {cancel_url} {cancel_payload} (cancel)")
                    resp_c = requests.post(cancel_url, json=cancel_payload, timeout=self.box_req_timeout)
                    # We don't strictly care about the status here; this is best-effort.
                    self.get_logger().info(
                        f"[box.dispose] cancel response code={resp_c.status_code} body={resp_c.text[:160]!r}"
                    )
                except Exception as e:
                    self.get_logger().warn(f"[box.dispose] /dispose/cancel failed: {e}")
                finally:
                    h.mark_done()

            h._cancel = _cancel

            # --- blocking call to /dispose ---
            try:
                resp = requests.post(url, json=payload, timeout=self.box_req_timeout)
                if resp.status_code != 200:
                    msg = f"non-200 status {resp.status_code}"
                    self.get_logger().warn(f"[box.dispose] /dispose {payload} -> {msg}")
                    result["error"] = msg
                else:
                    data = resp.json()
                    # Expected fields from DisposeResponse:
                    #   agent_id, box_id, property, status,
                    #   success, deadline, x, y, requested_at, completed_at
                    result.update(
                        status=data.get("status"),
                        success=data.get("success"),
                        deadline=data.get("deadline"),
                        x=data.get("x"),
                        y=data.get("y"),
                        requested_at=data.get("requested_at"),
                        completed_at=data.get("completed_at"),
                    )
            except Exception as e:
                msg = f"/dispose failed: {e}"
                self.get_logger().warn(f"[box.dispose] {msg}")
                result["error"] = msg

            # Make result visible via ctx
            ctx.setdefault("box", {})
            ctx["box"].update({
                "node_id": node_id,
                "box_id": box_id,
                "property": prop,
                "dispose_result": result,
            })

            self.get_logger().info(
                f"[box.dispose] box_id={box_id}, prop={prop}, "
                f"status={result['status']}, success={result['success']}"
            )

            h.mark_done()
            return h


    
        def wait_box_nearby(target_node_id: str, timeout_s: float = 10.0, ctx: dict = None):
            """
            Wait until bt_rssi_seen fires for the specified node.

            We compare the normalized node name (CNode###) from the rule payload
            with the given target_node_id.
            """
            h = StepHandle()
            if ctx is None:
                ctx = {}

            target_node_id = str(target_node_id or "").strip()
            if not target_node_id:
                self.get_logger().warn("[wait_box_nearby] empty target_node_id; finishing immediately.")
                h.mark_done()
                return h

            start_ms = int(self.get_clock().now().nanoseconds * 1e-6)
            timeout_ms = max(0.5, float(timeout_s)) * 1000.0
            period = 0.1  # 10 Hz
            canceled = {"v": False}
            timers = {"t": None}

            def _cancel():
                canceled["v"] = True
                t = timers["t"]
                if t:
                    try:
                        t.cancel()
                    except Exception:
                        pass
                    self._live_timers.discard(t)
                h.mark_done()

            h._cancel = _cancel

            def _tick():
                if canceled["v"]:
                    return

                now_ms = int(self.get_clock().now().nanoseconds * 1e-6)
                elapsed = now_ms - start_ms

                # Look at latest bt_rssi_seen payload in recent window
                payload = self.rules_view.latest_payload("bt_rssi_seen", within_ms=3000)
                if payload is not None:
                    obj_id_raw = str(payload.get("object_id", "")).strip()

                    # Normalize both to a canonical "CNode###" form for comparison
                    norm_seen = obj_id_raw
                    if not norm_seen.lower().startswith("cnode"):
                        norm_seen = f"CNode{norm_seen}"

                    norm_target = target_node_id
                    if not norm_target.lower().startswith("cnode"):
                        norm_target = f"CNode{norm_target}"

                    if norm_seen == norm_target:
                        self.get_logger().info(
                            f"[wait_box_nearby] target {norm_target} is nearby (bt_rssi_seen.object_id={obj_id_raw!r})"
                        )
                        try:
                            timers["t"].cancel()
                        except Exception:
                            pass
                        self._live_timers.discard(timers["t"])

                        # Record in ctx for downstream states
                        ctx.setdefault("box", {})
                        ctx["box"].update({
                            "node_id": norm_target,
                            "box_id": _box_id_from_node_id(norm_target),
                            "seen_nearby_ms": now_ms,
                            "last_bt_payload": payload,
                        })

                        h.mark_done()
                        return

                if elapsed >= timeout_ms:
                    self.get_logger().warn(
                        f"[wait_box_nearby] timeout waiting for {target_node_id} (~{timeout_s:.1f}s)."
                    )
                    try:
                        timers["t"].cancel()
                    except Exception:
                        pass
                    self._live_timers.discard(timers["t"])
                    h.mark_done()
                    return

            timers["t"] = self.create_timer(period, _tick)
            self._live_timers.add(timers["t"])
            return h


    
        def tts(text: str, ctx: dict = None):
            """
            TTS primitive that waits for speech playback to finish.

            Flow:
              - publish text via self.say() (downstream turns it into /tts_wav)
              - if we have a /tts_busy signal, queue a StepHandle that will
                be completed on the next 'speaking=False' edge
              - if we never see /tts_busy, fall back to immediate completion
                so the state machine doesn't deadlock
            """
            
            raw_text = "" if text is None else str(text)

            # Strip common “template produced quotes” artifacts
            candidate = raw_text.strip()
            if (candidate == '""') or (candidate == "''"):
                candidate = ""

            # If empty after cleanup: DO NOT publish, just advance state
            if candidate.strip() == "":
                self.get_logger().warn("[TTS] empty text; skipping publish and completing immediately.")
                h = StepHandle()
                h.mark_done()
                return h
            
            h = StepHandle()
            self.say(str(text))

            # If we've ever seen /tts_busy, treat it as authoritative
            if getattr(self, "_tts_has_busy", False):
                # Queue this handle; _cb_tts_busy will mark it done
                self._tts_waiting.append(h)
                self.get_logger().info(
                    f"[TTS] queued handle waiting for /tts_busy to go False "
                    f"(waiting={len(self._tts_waiting)})"
                )
            else:
                # No TTS player feedback available → don't block
                self.get_logger().warn(
                    "[TTS] /tts_busy has not been observed; "
                    "completing TTS handle immediately."
                )
                h.mark_done()

            return h

        def gesture(kind: str):
            """
            Map a logical gesture 'kind' to a Go2 sport API (api_id)
            and publish a WebRtcReq on /webrtc_req, equivalent to:

              ros2 topic pub /webrtc_req go2_interfaces/msg/WebRtcReq "
                topic: 'rt/api/sport/request'
                api_id: 1016
                parameter: ''
                id: 1" --once
            """
            h = StepHandle()
            try:
                # Map high-level gesture names to the table’s API IDs
                gesture_api_map = {
                    "greet":        1016,  # Hello
                    "hello":        1016,
                }

                api_id = gesture_api_map.get(str(kind), 1016)  # default to Hello

                msg = WebRtcReq()
                msg.topic     = "rt/api/sport/request"
                msg.api_id    = int(api_id)
                msg.parameter = ""   # same as data: '' in the JS / CLI example

                # Use a simple unique-ish ID; or hard-code 1 if you prefer
                msg.id = int(time.time() * 1000) & 0x7FFFFFFF

                self.get_logger().info(
                    f"[gesture] kind='{kind}' -> api_id={msg.api_id}, topic='{msg.topic}', id={msg.id}"
                )
                self.webrtc_req_pub.publish(msg)
            finally:
                h.mark_done()
            return h


        def move_relative(azimuth_deg: float, dist_m: float):
            # return a handle that completes when the chained timers finish
            return self._move_relative_handle(float(azimuth_deg), float(dist_m))

        def move_absolute(frame: str, x: float, y: float, yaw: float):
            h = StepHandle()

            if not self.nav_client.wait_for_server(timeout_sec=0.5):
                self.say("Navigation is not available.")
                h.mark_done()
                return h

            goal = NavigateToPose.Goal()
            ps = PoseStamped()
            ps.header.frame_id = str(frame)
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation = yaw_to_q(float(yaw))
            goal.pose = ps

            def _goal_done(fut):
                try:
                    goal_handle = fut.result()
                except Exception as e:
                    self.get_logger().error(f"Nav goal error: {e}")
                    self.say("Navigation failed.")
                    h.mark_done()
                    return

                if not goal_handle or not goal_handle.accepted:
                    self.say("Navigation goal was rejected.")
                    h.mark_done()
                    return

                def _result_done(res_fut):
                    try:
                        res = res_fut.result()
                    except Exception as e:
                        self.get_logger().error(f"Nav result error: {e}")
                        self.say("Navigation failed.")
                        h.mark_done()
                        return

                    if getattr(res, "status", 0) == GoalStatus.STATUS_SUCCEEDED:
                        self.say("Arrived at the destination.")
                    else:
                        self.say("Navigation failed.")
                    h.mark_done()

                goal_handle.get_result_async().add_done_callback(_result_done)

            self.nav_client.send_goal_async(goal).add_done_callback(_goal_done)
            return h


        def query_beacons(top_n: int, ctx: dict):
            h = StepHandle()
            self._say_top3_beacons_from_db(int(top_n))
            ctx['last_query_beacons_speech'] = "Beacon report complete."
            h.mark_done()
            return h

        def llm_speech_check(prompt: str = "",
                             output_schema: str = "",
                             text: str = "",
                             tag: str = "",
                             ctx: dict = None):
            """
            Generic LLM JSON worker.
            Caller specifies prompt + output_schema; we just relay and await response.
            Result goes into ctx["speech_check"].
            """
            h = StepHandle()
            if ctx is None:
                ctx = {}

            req_id = int(time.time() * 1000) ^ self._llm_speech_next_id
            self._llm_speech_next_id += 1

            self._llm_speech_pending[req_id] = {
                "handle": h,
                "ctx": ctx,
            }

            payload = {
                "id": req_id,
                "prompt": prompt or "",
                "output_schema": output_schema or "",
                "text": text or "",
                "tag": tag or "",
            }
            self.llm_speech_req_pub.publish(StringMsg(data=json.dumps(payload)))
            return h

        def coverage_wait(
            spacing_m: float = 1.5,
            visited_radius_m: float = 0.9,
            dwell_sec: float = 2.0,
            persist_path: str = "/tmp/coverage_wait_visited.json",
            ctx: dict = None,
        ):
            h = StepHandle()
            if ctx is None:
                ctx = {}

            ts_ms = int(time.time() * 1000)
            req_id = f"skills:coverage:{ts_ms}:{self._coverage_next_id}"
            self._coverage_next_id += 1

            self._coverage_pending[req_id] = {"handle": h, "ctx": ctx}

            payload = {
                "id": req_id,
                "client": "skills",
                "cmd": "start",
                "params": {
                    "spacing_m": float(spacing_m),
                    "visited_radius_m": float(visited_radius_m),
                    "dwell_sec": float(dwell_sec),
                    "persist_path": str(persist_path),
                }
            }
            self.coverage_req_pub.publish(StringMsg(data=json.dumps(payload)))

            # cancel hook sends cancel command
            def _cancel():
                try:
                    cancel_payload = {"id": req_id, "client": "skills", "cmd": "cancel"}
                    self.coverage_req_pub.publish(StringMsg(data=json.dumps(cancel_payload)))
                finally:
                    h.mark_done()

            h._cancel = _cancel
            return h



        def vlm_inference(prompt: str = "",
                          output_schema: str = "",
                          tag: str = "",
                          mode: str = "generic",
                          ctx: dict = None):
            """
            Generic VLM micro-service, symmetric with llm_speech_check.
            We just relay request; answer comes back on /vlm/answer.
            """
            h = StepHandle()
            if ctx is None:
                ctx = {}

            ts_ms = int(time.time() * 1000)
            tag = tag or "vlm"
            #Unique + traceable id
            req_id = f"skills:{tag}:{ts_ms}:{self._vlm_next_id}"
            self._vlm_next_id += 1

            self._vlm_pending[req_id] = {"handle": h, "ctx": ctx}

            self.get_logger().info(f"[VLM primitive] id={req_id} prompt={prompt!r}, tag={tag}, mode={mode}")

            payload = {
                "id": req_id,
                "client": "skills",                 # NEW
                "prompt": prompt or "",
                "output_schema": output_schema or "",
                "tag": tag,
                "mode": mode or "generic",
            }
            # Send to VLM node; it should look at the latest frame and respond.
            self.vlm_req_pub.publish(StringMsg(data=json.dumps(payload)))
            return h


        return {
            'tts': tts,
            'gesture': gesture,
            'move_relative': move_relative,
            'move_absolute': move_absolute,
            'query_beacons': query_beacons,
            'llm_speech_check': llm_speech_check,
            'vlm_inference': vlm_inference,
            'coverage_wait': coverage_wait,
            'wait_box_nearby': wait_box_nearby,
            'sense_box': sense_box,
            'dispose_box': dispose_box,
        }

    def _move_relative_handle(self, az_deg: float, dist_m: float) -> StepHandle:
        """
        Rotate-in-place by |az_deg| and then move forward dist_m,
        using *odometry events* to decide when to stop each phase:
          - rotation: accumulate sum(dyaw_deg) from rule 'odom_rot_delta'
          - forward : accumulate sum(dxy)      from rule 'odom_dist_delta'
        Falls back on timeouts to avoid runaway if events are missing.
        """
        h = StepHandle()
        turn_speed = self.turn_speed
        fwd_speed  = self.fwd_speed
        min_turn_deg = 3.0
        eps_dist = 0.02

        # which rules/fields to read from RulesViewROS
        ROT_RULE, ROT_FIELD = "odom_rot_delta", "dyaw_deg"
        DIST_RULE, DIST_FIELD = "odom_dist_delta", "dxy"

        timers = {"rot": None, "fwd": None}
        canceled = {"v": False}

        def stop_all():
            # hard-stop base and cancel timers
            self.cmd_vel_pub.publish(Twist())
            for k, t in list(timers.items()):
                if t:
                    try: t.cancel()
                    except Exception: pass
                    self._live_timers.discard(t)
                    timers[k] = None

        def cancel_fn():
            canceled["v"] = True
            stop_all()

        h._cancel = cancel_fn

        # ------------- Forward phase (distance via odom events) -------------
        def start_forward_phase():
            if canceled["v"]:
                h.mark_done(); return

            if dist_m <= eps_dist:
                stop_all()
                h.mark_done()
                return

            period = 0.05  # 20 Hz
            start_ms = int(self.get_clock().now().nanoseconds * 1e-6)
            target_m = max(0.0, float(dist_m))
            elapsed = 0.0

            # conservative timeout: expected time @ fwd_speed + margin
            exp_t = (target_m / max(fwd_speed, 1e-3))
            timeout_s = max(2.0, min(20.0, exp_t * 2.0))

            tw = Twist()
            tw.linear.x = max(0.05, fwd_speed)  # ensure nonzero
            self.get_logger().info(f"[move_relative] forward start target={target_m:.2f} m, vx={tw.linear.x:.2f} m/s")

            # publish once upfront
            self.cmd_vel_pub.publish(tw)

            def _tick():
                nonlocal elapsed
                if canceled["v"]:
                    stop_all(); h.mark_done(); return

                # keep moving
                self.cmd_vel_pub.publish(tw)

                # accumulate distance from events since start
                acc_m = self._sum_rule_field_since(DIST_RULE, DIST_FIELD, start_ms)

                if acc_m >= target_m:
                    stop_all()
                    h.mark_done()
                    self.get_logger().info(f"[move_relative] forward complete acc={acc_m:.3f} m")
                    return

                elapsed += period
                if elapsed >= timeout_s:
                    stop_all()
                    h.mark_done()
                    self.get_logger().warn(f"[move_relative] forward timeout acc={acc_m:.3f}/{target_m:.3f} m")
                    return

            timers["fwd"] = self.create_timer(period, _tick)
            self._live_timers.add(timers["fwd"])

        # ------------- Rotation phase (degrees via odom events) -------------
        def start_rotation_phase():
            if canceled["v"]:
                h.mark_done(); return

            if abs(az_deg) < min_turn_deg:
                # no meaningful rotation → go forward directly
                start_forward_phase()
                return

            period = 0.05  # 20 Hz
            start_ms = int(self.get_clock().now().nanoseconds * 1e-6)
            target_deg = abs(float(az_deg))
            direction = 1.0 if float(az_deg) >= 0.0 else -1.0
            elapsed = 0.0

            # conservative timeout: angle/speed + margin
            exp_t = (math.radians(target_deg) / max(abs(turn_speed), 1e-3))
            timeout_s = max(2.0, min(20.0, exp_t * 2.0))

            tw = Twist()
            tw.angular.z = direction * max(0.05, min(self.search_w, self.name_max_w))
            self.get_logger().info(f"[move_relative] turn start target={target_deg:.1f}°, wz={tw.angular.z:.2f} rad/s")

            # publish once upfront
            self.cmd_vel_pub.publish(tw)

            def _tick():
                nonlocal elapsed
                if canceled["v"]:
                    stop_all(); h.mark_done(); return

                # keep turning
                self.cmd_vel_pub.publish(tw)

                # accumulate rotation (degrees) from events since start
                acc_deg = self._sum_rule_field_since(ROT_RULE, ROT_FIELD, start_ms)

                if acc_deg >= target_deg:
                    # rotation done → stop and start forward
                    self.cmd_vel_pub.publish(Twist())
                    try:
                        timers["rot"].cancel()
                    except Exception:
                        pass
                    self._live_timers.discard(timers["rot"])
                    timers["rot"] = None
                    self.get_logger().info(f"[move_relative] turn complete acc={acc_deg:.1f}°")
                    start_forward_phase()
                    return

                elapsed += period
                if elapsed >= timeout_s:
                    # timeout but still proceed to forward to avoid deadlock
                    self.get_logger().warn(f"[move_relative] turn timeout acc={acc_deg:.1f}/{target_deg:.1f}°")
                    try:
                        timers["rot"].cancel()
                    except Exception:
                        pass
                    self._live_timers.discard(timers["rot"])
                    timers["rot"] = None
                    self.cmd_vel_pub.publish(Twist())
                    start_forward_phase()
                    return

            timers["rot"] = self.create_timer(period, _tick)
            self._live_timers.add(timers["rot"])

        # Kick it off
        start_rotation_phase()
        return h



    # ───────────────────────────── Hot Reload ─────────────────────────────────
    def _load_skills_initial(self):
        self._load_skills_merged()
        
    def _reload_skills_if_changed(self) -> bool:
        """
        Hot reload if either base or composite file changed on disk.
        Base is *expected* to be immutable at runtime, but we still
        allow reload if it did change for convenience.
        """
        changed = False
        try:
            if self.skills_base_path and os.path.isfile(self.skills_base_path):
                m = os.path.getmtime(self.skills_base_path)
                if self._skills_base_mtime is None or m > self._skills_base_mtime:
                    changed = True

            if self.skills_composite_path and os.path.isfile(self.skills_composite_path):
                m = os.path.getmtime(self.skills_composite_path)
                if self._skills_comp_mtime is None or m > self._skills_comp_mtime:
                    changed = True
        except Exception as e:
            self.get_logger().warn(f"skills reload check failed: {e}")
            return False

        if changed:
            self._load_skills_merged()
            return True
        return False

    def _maybe_reload_skills(self):
        self._reload_skills_if_changed()

# ───────────────────────────────────────────────────────────────────────────────
#                                       main
# ───────────────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = SkillsAgent()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

