#!/usr/bin/env python3
# skills_engine_v2.py
#
# Engine/runtime module extracted from the original single-file skills_node.py.
#
from __future__ import annotations

import inspect
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

# ROS types are only needed for RulesViewROS; keep them here so the bridge lives
# with the RulesView abstraction.
import rclpy
from rclpy.node import Node
from std_msgs.msg import String as StringMsg


# ───────────────────────────────────────────────────────────────────────────────
#                                  Data classes
# ───────────────────────────────────────────────────────────────────────────────

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


class StepHandle:
    """Represents an in-progress primitive. Engine polls .done() and can .cancel()."""
    def __init__(self, cancel_fn=None):
        self._done = False
        self._cancel = cancel_fn or (lambda: None)
        self.outcome = "ok"   # "ok" | "timeout" | "error" | whatever you need

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
#                          Number → words helpers (TTS)
# ───────────────────────────────────────────────────────────────────────────────

_INT_TOKEN_RE = re.compile(r'(?<![\w.])(-?\d+)(?![\w.])')
_DECIMAL_RE   = re.compile(r'(?<![\w])(-?\d*\.\d+)(?![\w])')
_CNODE_RE     = re.compile(r'\bCNode(\d+)\b', re.IGNORECASE)

_LETTER_TOKEN_RE = re.compile(r'(?<![A-Za-z0-9])([XY])(?![A-Za-z0-9])')
_LETTER_MAP = {
    "X": "ex",
    "Y": "why",
}


def _num_to_words(n: int) -> str:
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
    digit_words = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
    }
    return " ".join(digit_words[d] for d in frac_str)


def _normalize_tts_text(text: str) -> str:

    return text

    def repl_cnode(m: re.Match) -> str:
        nid_str = m.group(1)
        try:
            nid = int(nid_str)
        except Exception:
            return m.group(0)
        if abs(nid) > 999:
            return f"node {nid_str}"
        return f"node {_num_to_words(nid)}"

    out = _CNODE_RE.sub(repl_cnode, text)

    def repl_letter(m: re.Match) -> str:
        return _LETTER_MAP.get(m.group(1), m.group(1))

    out = _LETTER_TOKEN_RE.sub(repl_letter, out)

    def repl_decimal(m: re.Match) -> str:
        s = m.group(0)
        negative = s.startswith("-")
        s2 = s[1:] if negative else s

        if "." not in s2:
            return s

        whole_str, frac_str = s2.split(".", 1)

        if whole_str.isdigit() and abs(int(whole_str)) > 999:
            return s

        whole_val = int(whole_str) if whole_str else 0
        whole_spoken = _num_to_words(whole_val)
        frac_spoken = _fraction_to_words(frac_str)

        spoken = f"{whole_spoken} point {frac_spoken}"
        return "minus " + spoken if negative else spoken

    out = _DECIMAL_RE.sub(repl_decimal, out)

    def repl_int(m: re.Match) -> str:
        s = m.group(0)
        try:
            n = int(s)
        except Exception:
            return s
        if abs(n) > 999:
            return s
        return _num_to_words(n)

    out = _INT_TOKEN_RE.sub(repl_int, out)
    return out


def _box_id_from_node_id(node_id: str) -> Optional[int]:
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
#                               Skills YAML fallback
# ───────────────────────────────────────────────────────────────────────────────

DEFAULT_SKILLS_V2 = r"""
version: 2
defaults:
  window_ms: 3000

skills:
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


# ───────────────────────────────────────────────────────────────────────────────
#                        Core Engine (templating + conditions)
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

    if "| default:" in expr:
        left, default_str = expr.split("| default:", 1)
        left = left.strip()
        default_str = default_str.strip()

        try:
            default_val = json.loads(default_str)
        except Exception:
            default_val = default_str.strip("'\"")

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

        if left.startswith("ctx."):
            val = _get_path({"ctx": ctx}, left)
            return str(val) if not _is_missing(val) else str(default_val)

        val = _get_path({"ctx": ctx}, f"ctx.{left}")
        return str(val) if not _is_missing(val) else str(default_val)

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


def canonical_box_action_from_execute(engine: "SkillEngineV2", root_skill_name: str, root_ctx: dict) -> dict | None:
    """
    Extract canonical box op from a state_machine skill that was requested via /skills/execute.
    Returns:
      {"kind":"sense"|"dispose", "node_id":"CNode102", "property":"X"|"Y"}
    or None.
    """
    root_def = engine.registry.get(root_skill_name)
    if not isinstance(root_def, dict) or root_def.get("kind") != "state_machine":
        return None

    states = root_def.get("states") or []
    if not isinstance(states, list):
        return None

    for st in states:
        if not isinstance(st, dict) or st.get("type") != "action":
            continue
        action = st.get("action") or {}
        if not isinstance(action, dict):
            continue

        use = str(action.get("use") or "").strip()
        with_params = action.get("with") or {}
        if not isinstance(with_params, dict):
            with_params = {}

        # Only check these wrapper/primitive skills
        if use not in ("box.sense_nearby", "box.dispose_nearby", "box.sense", "box.dispose"):
            continue

        # Step ctx = root ctx + rendered "with"
        step_ctx = dict(root_ctx or {})
        rendered_with = _render_params(with_params, step_ctx, engine.rules, engine.defaults_window_ms)
        step_ctx.update(rendered_with)

        if use == "box.sense_nearby":
            node_id = str(step_ctx.get("target_node_id") or "").strip()
            prop = str(step_ctx.get("property") or "X").strip().upper()
            return {"kind": "sense", "node_id": node_id, "property": prop}

        if use == "box.dispose_nearby":
            node_id = str(step_ctx.get("target_node_id") or "").strip()
            prop = str(step_ctx.get("property") or "X").strip().upper()
            return {"kind": "dispose", "node_id": node_id, "property": prop}

        if use == "box.sense":
            node_id = str(step_ctx.get("node_id") or "").strip()
            prop = str(step_ctx.get("property") or "X").strip().upper()
            return {"kind": "sense", "node_id": node_id, "property": prop}

        if use == "box.dispose":
            node_id = str(step_ctx.get("node_id") or "").strip()
            prop = str(step_ctx.get("property") or "X").strip().upper()
            return {"kind": "dispose", "node_id": node_id, "property": prop}

    return None


def same_canonical_box_action(a: dict | None, b: dict | None) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if a.get("kind") not in ("sense", "dispose"):
        return False
    return (
        a.get("kind") == b.get("kind")
        and str(a.get("node_id") or "").strip() == str(b.get("node_id") or "").strip()
        and str(a.get("property") or "").strip().upper() == str(b.get("property") or "").strip().upper()
    )



# ───────────────────────────────────────────────────────────────────────────────
#                                  SkillEngineV2
# ───────────────────────────────────────────────────────────────────────────────

class SkillEngineV2:
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
        self.event_cb = event_cb
        self.bad_skills: set[str] = set()

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _emit_event(self, kind: str, inst: SkillInstance, extra: dict | None = None):
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

    def _quarantine_skill(self, inst: SkillInstance, reason: str, exc: Exception | None = None):
        name = inst.name
        self.bad_skills.add(name)
        if self.logger:
            self.logger.error(f"[SkillEngine] Quarantining skill '{name}' due to {reason}: {exc!r}")
        self._emit_event("skill_error", inst, {"reason": reason, "error": str(exc) if exc else ""})

    def _exec_primitive_get_handle(self, action: str, params: dict, ctx: dict) -> StepHandle:
        fn = self.bindings.get(action)
        if not callable(fn):
            if self.logger:
                self.logger.error(f"[SkillEngine] No binding for action '{action}' – completing step as no-op.")
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

    # ---- loading ----
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

    # ---- planning helpers ----
    def state_machine_names(self) -> List[str]:
        return [n for n, s in self.registry.items() if s.get("kind") == "state_machine"]

    def _skill_when_passes(self, skill: dict) -> bool:
        cond = skill.get("when") or {}
        return _cond_pass(cond, self.rules, self.defaults_window_ms)

    def plan_eligible(self) -> List[dict]:
        out = []
        for name in self.state_machine_names():
            s = self.registry.get(name) or {}
            if self._skill_when_passes(s):
                out.append({"name": name})
        return out

    # ---- state-machine ops ----
    def arm(self, skill_name: str, ctx: dict, is_root: bool = True):
        s = self.registry.get(skill_name)
        if not s:
            raise KeyError(f"Unknown skill: {skill_name}")
        if skill_name in getattr(self, "bad_skills", set()):
            raise RuntimeError(f"Skill '{skill_name}' is quarantined due to previous errors; refusing to arm.")
        if s.get("kind") != "state_machine":
            raise KeyError(f"Skill '{skill_name}' is not a state_machine (kind={s.get('kind')})")

        inst = SkillInstance(name=skill_name, ctx=dict(ctx or {}), started_ms=self._now_ms(), is_root=is_root)
        self._active.append(inst)
        if self.logger:
            self.logger.info(f"[SkillEngine] Armed state_machine '{skill_name}' ctx={ctx}")
        return inst

    def run(self, skill_name: str, ctx: dict):
        self.arm(skill_name, ctx)
        self.tick()

    def active_count(self) -> int:
        return sum(1 for i in self._active if not i.done)

    def _find_state(self, skill: dict, state_id: str | None) -> Optional[dict]:
        if not state_id:
            return None
        for idx, st in enumerate(skill.get("states", [])):
            if st.get("id") == state_id:
                st = dict(st)
                st["_idx"] = idx
                return st
        return None

    def _spawn_nested_child(self, parent: SkillInstance, skill_name: str, ctx: dict) -> StepHandle:
        child_ctx = dict(parent.ctx)
        child_ctx.update(ctx or {})
        child_inst = self.arm(skill_name, child_ctx, is_root=False)

        h = StepHandle()

        def cancel_child():
            child_inst.done = True
            if child_inst.handle is not None and not child_inst.handle.done():
                try:
                    child_inst.handle.cancel()
                except Exception:
                    pass
            h.mark_done()

        h._cancel = cancel_child

        def done_proxy(self):
            return child_inst.done

        h.done = done_proxy.__get__(h, StepHandle)
        return h

    def _enter_state(self, inst: SkillInstance, skill: dict, state: dict):
        now_ms = self._now_ms()
        inst.state_id = state.get("id")
        inst.state_idx = int(state.get("_idx", inst.state_idx))
        inst.state_started_ms = now_ms
        inst.handle = None

        st_type = state.get("type", "action")
        self._emit_event("state_entered", inst, {"state_type": st_type})

        if self.logger:
            self.logger.info(
                f"[SkillEngine] Skill '{inst.name}': entering state '{inst.state_id}' "
                f"(idx={inst.state_idx}, type={st_type})"
            )

        if st_type != "action":
            return

        action_spec = state.get("action") or {}
        use_name = action_spec.get("use")
        with_ctx = action_spec.get("with") or {}

        if not use_name:
            if self.logger:
                self.logger.warn(f"[SkillEngine] action state '{inst.state_id}' missing 'use'")
            return

        base = self.registry.get(use_name)

        if base and base.get("kind") == "primitive":
            params = dict(base.get("params") or {})
            params.update(with_ctx)
            self._emit_event("step_started", inst, {"primitive": use_name, "state_type": "action"})
            inst.handle = self._exec_primitive_get_handle(base["action"], params, inst.ctx)
            return

        if base and base.get("kind") in ("state_machine", "composite"):
            self._emit_event("step_started", inst, {"composite": use_name, "state_type": "action"})
            inst.handle = self._spawn_nested_child(inst, use_name, with_ctx)
            return

        self._emit_event("step_started", inst, {"primitive": use_name, "state_type": "action"})
        inst.handle = self._exec_primitive_get_handle(use_name, with_ctx, inst.ctx)

    def _transition_to(self, inst: SkillInstance, skill: dict, next_id: Optional[str]):
        cur_state = inst.state_id

        if not next_id:
            inst.done = True
            if self.logger:
                self.logger.info(f"[SkillEngine] Skill '{inst.name}' finished (no next state)")
            self._emit_event("skill_finished", inst, {"reason": "no_next_state"})
            return

        if self.logger:
            self.logger.info(f"[SkillEngine] Skill '{inst.name}': state '{cur_state}' -> '{next_id}'")

        st = self._find_state(skill, next_id)
        if not st:
            inst.done = True
            if self.logger:
                self.logger.warn(
                    f"[SkillEngine] Skill '{inst.name}' next state '{next_id}' not found; finishing."
                )
            self._emit_event("skill_finished", inst, {"reason": "missing_state"})
            return

        self._enter_state(inst, skill, st)

    def _resolve_wait_next_state(self, state: dict, rule_id: str) -> Optional[str]:
        branches = state.get("branches") or []
        for br in branches:
            if br.get("rule_id") == rule_id:
                return br.get("next")
        return state.get("on_event")

    def tick(self):
        now_ms = self._now_ms()

        for inst in list(self._active):
            if inst.done:
                continue

            try:
                skill = self.registry.get(inst.name) or {}
                if skill.get("kind") != "state_machine":
                    inst.done = True
                    continue

                comp_when = skill.get("when") or {}
                comp_until = skill.get("until") or {}

                if not inst.activated:
                    if _cond_pass(comp_when, self.rules, self.defaults_window_ms):
                        inst.activated = True
                        inst.started_ms = now_ms
                        self._emit_event("skill_started", inst, {})

                        if not inst.state_id:
                            init_id = skill.get("initial_state")
                            st = self._find_state(skill, init_id)
                            if not st:
                                if self.logger:
                                    self.logger.error(
                                        f"[SkillEngine] Skill '{inst.name}' missing initial_state '{init_id}'"
                                    )
                                self._quarantine_skill(inst, "missing_initial_state", None)
                                inst.done = True
                                self._emit_event("skill_finished", inst, {"reason": "missing_initial_state"})
                                continue
                            self._enter_state(inst, skill, st)
                    else:
                        continue

                if _cond_pass(comp_until, self.rules, self.defaults_window_ms, empty_means=False):
                    if inst.handle is not None and not inst.handle.done():
                        try:
                            inst.handle.cancel()
                        except Exception:
                            pass
                    inst.done = True
                    self._emit_event("skill_finished", inst, {"reason": "composite_until"})
                    continue

                st = self._find_state(skill, inst.state_id)
                if not st:
                    inst.done = True
                    self._quarantine_skill(inst, "missing_state", None)
                    self._emit_event("skill_finished", inst, {"reason": "no_state"})
                    continue

                st_type = st.get("type", "action")

                if st_type == "action":
                    if inst.handle is None:
                        self._enter_state(inst, skill, st)
                        continue

                    if not inst.handle.done():
                        continue

                    outcome = getattr(inst.handle, "outcome", "ok")
                    if outcome == "timeout":
                        next_id = st.get("on_timeout") if "on_timeout" in st else None
                    elif outcome == "error":
                        next_id = st.get("on_failure") or st.get("on_complete")
                    else:
                        next_id = st.get("on_complete")

                    self._transition_to(inst, skill, next_id)
                    continue

                if st_type == "wait":
                    wait_spec = st.get("wait_for") or {}
                    any_of = wait_spec.get("any_of") or []

                    fired_rule = None
                    for cond in any_of:
                        rid = cond.get("rule_id")
                        win = int(cond.get("within_ms") or self.defaults_window_ms)
                        if rid and self.rules.exists(rid, win):
                            fired_rule = rid
                            break

                    if fired_rule:
                        next_id = self._resolve_wait_next_state(st, fired_rule)
                        self._transition_to(inst, skill, next_id)
                        continue

                    max_wait = max(
                        (int(c.get("within_ms") or self.defaults_window_ms) for c in any_of),
                        default=self.defaults_window_ms
                    )
                    if now_ms - inst.state_started_ms >= max_wait:
                        next_id = st.get("on_timeout")
                        self._transition_to(inst, skill, next_id)
                        continue

                    continue

                inst.done = True
                self._quarantine_skill(inst, "bad_state_type", None)
                self._emit_event("skill_finished", inst, {"reason": "bad_state_type"})

            except Exception as e:
                if self.logger:
                    self.logger.error(f"[SkillEngine] Runtime error in skill '{inst.name}' state='{inst.state_id}': {e!r}")
                self._quarantine_skill(inst, "runtime_exception", e)

                skill = self.registry.get(inst.name) or {}
                st = self._find_state(skill, inst.state_id)

                next_id = None
                if st:
                    st_type = st.get("type", "action")
                    if st_type == "action":
                        next_id = st.get("on_failure") or st.get("on_complete")
                    elif st_type == "wait":
                        next_id = st.get("on_timeout") or st.get("on_event")

                self._transition_to(inst, skill, next_id)

        self._active = [i for i in self._active if not i.done]


# ───────────────────────────────────────────────────────────────────────────────
#                          ROS RulesView bridge
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

