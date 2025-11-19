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

from std_msgs.msg import String as StringMsg
from std_srvs.srv import Trigger
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from tf2_ros import Buffer, TransformListener
from go2_interfaces.msg import WebRtcReq


from dataclasses import dataclass, field
import time

@dataclass
class SkillInstance:
    name: str
    ctx: dict
    started_ms: int
    step_idx: int = 0
    activated: bool = False   # composite-level when satisfied
    done: bool = False
    handle: StepHandle | None = None  # ← active primitive handle
    
    
# ───────────────────────────────────────────────────────────────────────────────
#                          Number → words helpers (TTS)
# ───────────────────────────────────────────────────────────────────────────────
# Standalone integer tokens
_NUM_TOKEN_RE = re.compile(r'(?<!\w)(-?\d+)(?!\w)')
# CNode pattern like "CNode101" or "cnode7"
_CNODE_RE    = re.compile(r'\bCNode(\d+)\b', re.IGNORECASE)


def _num_to_words(n: int) -> str:
    nums0_19 = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"
    ]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    neg = n < 0
    n = abs(n)
    parts = []

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


def _normalize_tts_text(text: str) -> str:
    """
    - Map 'CNode101' -> 'node one hundred and one'
    - Then map bare integers '42' -> 'forty two'
    """
    def repl_cnode(m: re.Match) -> str:
        nid = int(m.group(1))
        # You can change this to `node number ...` if you prefer
        return f"node {_num_to_words(nid)}"

    def repl_num(m: re.Match) -> str:
        s = m.group(0)
        try:
            n = int(s)
        except ValueError:
            return s
        return _num_to_words(n)

    # First: rewrite CNode### patterns
    out = _CNODE_RE.sub(repl_cnode, text)
    # Then: generic standalone integers
    out = _NUM_TOKEN_RE.sub(repl_num, out)
    return out

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
        cutoff = (self._now() - within_ms)
        return any(e['id'] == str(rule_id) and e['ts_ms'] >= cutoff for e in self._events)

    def latest_payload(self, rule_id: str, within_ms: int) -> Optional[dict]:
        cutoff = (self._now() - within_ms)
        cand = [e for e in self._events if e['id'] == str(rule_id) and e['ts_ms'] >= cutoff]
        if not cand:
            return None
        cand.sort(key=lambda e: e['ts_ms'], reverse=True)
        return cand[0]['payload']

def _render_scalar(expr: str, ctx: dict, rules: RulesView, defaults_window_ms: int) -> str:
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
                    if val is not None:
                        return str(val)
            return str(default_val)

        # ctx.<path>
        if left.startswith("ctx."):
            val = _get_path({"ctx": ctx}, left)
            return str(val if val is not None else default_val)

        # bare key → interpret as ctx.<key>
        val = _get_path({"ctx": ctx}, f"ctx.{left}")
        return str(val if val is not None else default_val)

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
    if 'any' in cond:
        return any(_cond_pass(c, rules, defaults_window_ms) for c in (cond['any'] or []))
    if 'all' in cond:
        return all(_cond_pass(c, rules, defaults_window_ms) for c in (cond['all'] or []))
    return True

class SkillEngineV2:
    """Executes composite skills with step-level `when` using a RulesView."""
    def __init__(self, bindings: Dict[str, callable], rules_view: RulesView, defaults_window_ms: int = 3000, logger=None, event_cb=None):
        self.bindings = dict(bindings or {})
        self.rules = rules_view
        self.defaults_window_ms = int(defaults_window_ms)
        self.registry: Dict[str, dict] = {}
        self._loaded_path: Optional[str] = None
        self.logger = logger
        self._active: List[SkillInstance] = []   # ← active/armed skills live here
        self.event_cb = event_cb   # ← NEW
        
        
    def _emit_event(self, kind: str, inst: SkillInstance, extra: dict | None = None):
        if not self.event_cb:
            return
        payload = {
            "kind": kind,
            "skill": inst.name,
            "step_idx": inst.step_idx,
            "ctx": inst.ctx,
            "started_ms": inst.started_ms,
            "activated": inst.activated,
            "done": inst.done,
        }
        if extra:
            payload.update(extra)
        self.event_cb(payload)

        
    def _exec_primitive_get_handle(self, action: str, params: dict, ctx: dict) -> StepHandle:
        """
        Call the bound primitive; normalize return to a StepHandle:
          - returns StepHandle → use it
          - returns None or sync primitive → create a handle, mark_done()
        """
        fn = self.bindings.get(action)
        if not callable(fn):
            raise KeyError(f"No binding for action '{action}'")

        rendered = _render_params(params, ctx, self.rules, self.defaults_window_ms)

        import inspect
        sig = inspect.signature(fn)
        ret = fn(**rendered, ctx=ctx) if 'ctx' in sig.parameters else fn(**rendered)

        if isinstance(ret, StepHandle):
            return ret

        h = StepHandle()
        # If primitive returned a truthy “instant” signal, still mark done immediately.
        h.mark_done()
        return h


        
    def load_from_string(self, yaml_text: str):
        data = yaml.safe_load(yaml_text)
        self.defaults_window_ms = int(data.get('defaults', {}).get('window_ms', self.defaults_window_ms))
        self.registry = {s['name']: s for s in data.get('skills', [])}
        self._loaded_path = None
        return self

    def load_from_path(self, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f.read())
        self.defaults_window_ms = int(data.get('defaults', {}).get('window_ms', self.defaults_window_ms))
        self.registry = {s['name']: s for s in data.get('skills', [])}
        self._loaded_path = path
        return self

    def composite_names(self) -> List[str]:
        return [n for n, s in self.registry.items() if s.get('kind') == 'composite']

    def is_step_eligible(self, step: dict) -> bool:
        return _cond_pass(step.get('when') or {}, self.rules, self.defaults_window_ms)

    def is_composite_eligible_now(self, composite_name: str) -> bool:
        s = self.registry.get(composite_name)
        if not s or s.get('kind') != 'composite':
            return False
        # Consider eligible if at least one step is currently passable.
        return any(self.is_step_eligible(st) for st in s.get('steps', []))

    def plan_eligible(self) -> List[dict]:
        out = []
        for name in self.composite_names():
            if not self.is_composite_eligible_now(name):
                continue
            steps = self.registry[name].get('steps', [])
            passing = [i for i, st in enumerate(steps) if self.is_step_eligible(st)]
            out.append({"name": name, "passing_steps": passing})
        return out

    def run(self, composite_name: str, ctx: dict):
      
        self.arm(composite_name, ctx)
        self.tick()

    def _exec_primitive(self, action: str, params: dict, ctx: dict):
        if self.logger:
            self.logger.info(f"[SkillEngine] → Executing primitive: {action} with params={params}")
        fn = self.bindings.get(action)
        if not callable(fn):
            raise KeyError(f"No binding for action '{action}'")
        rendered = _render_params(params, ctx, self.rules, self.defaults_window_ms)
        sig = inspect.signature(fn)
        if 'ctx' in sig.parameters:
            fn(**rendered, ctx=ctx)
        else:
            fn(**rendered)

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def arm(self, composite_name: str, ctx: dict):
        s = self.registry.get(composite_name)
        if not s or s.get('kind') != 'composite':
            raise KeyError(f"Unknown composite: {composite_name}")
        inst = SkillInstance(name=composite_name, ctx=dict(ctx or {}), started_ms=self._now_ms())
        self._active.append(inst)
        if self.logger:
            self.logger.info(f"[SkillEngine] Armed skill: {composite_name} ctx={ctx}")

    def active_count(self) -> int:
        return sum(1 for i in self._active if not i.done)
        
    def _start_child_composite(self, parent_inst: SkillInstance, step: dict, composite: dict) -> StepHandle:
        """
        Treat a composite referenced in `use:` as a nested sub-skill.
        We:
          - build a child ctx from parent ctx + rendered step.with
          - create a child SkillInstance and add it to _active
          - return a StepHandle that becomes done() when the child finishes
        """
        ref_name = composite["name"]

        # Render step.with into a ctx overlay, like params for primitives
        with_overrides = _render_params(
            step.get("with") or {},
            parent_inst.ctx,
            self.rules,
            self.defaults_window_ms,
        )
        child_ctx = dict(parent_inst.ctx)
        child_ctx.update(with_overrides)

        child = SkillInstance(
            name=ref_name,
            ctx=child_ctx,
            started_ms=int(time.time() * 1000),
        )
        self._active.append(child)

        # Build a StepHandle that proxies to the child's done()
        def _cancel_child():
            child.done = True

        h = StepHandle(cancel_fn=_cancel_child)

        # override done() to reflect child.done
        def _done_proxy(self_handle=h, child_inst=child):
            return child_inst.done

        h.done = _done_proxy  # monkey-patch method on this instance
        return h
        
    def tick(self):
        """Advance armed skills. Call this on a timer and/or on every /events/* message."""
        for inst in list(self._active):
            if inst.done:
                continue

            s = self.registry.get(inst.name) or {}
            steps = s.get('steps', [])
            comp_when  = s.get('when')  or {}
            comp_until = s.get('until') or {}

            # If a primitive is running, check completion or stop-early
            if inst.handle is not None:
                # stop early if composite until is true
                if _cond_pass(comp_until, self.rules, self.defaults_window_ms, empty_means=False):
                    if not inst.handle.done():
                        inst.handle.cancel()
                    inst.done = True
                    if self.logger:
                        self.logger.info(f"[SkillEngine] Finished '{inst.name}' (composite until=true during step)")
                    continue
                # still running? keep waiting
                if not inst.handle.done():
                    continue
                # finished; clear handle and advance to next step
                inst.handle = None
                inst.step_idx += 1
                # fall-through to possibly launch next step this tick

            # Not yet activated? wait for composite when
            if not inst.activated:
                if _cond_pass(comp_when, self.rules, self.defaults_window_ms):
                    inst.activated = True
                    if self.logger:
                        self.logger.info(f"[SkillEngine] Activating '{inst.name}' (when=true)")
                    self._emit_event("skill_started", inst, {})
                else:
                    continue

            # Composite-level until gate before starting a new step
            if _cond_pass(comp_until, self.rules, self.defaults_window_ms, empty_means=False):
                inst.done = True
                if self.logger:
                    self.logger.info(f"[SkillEngine] Finished '{inst.name}' (composite until=true before step)")
                self._emit_event("skill_finished", inst, {"reason": "composite_until"})
                continue

            # Launch steps (one per tick at most)
            if inst.step_idx < len(steps):
                step = steps[inst.step_idx]
                step_when  = step.get('when')  or {}
                step_until = step.get('until') or {}

                # step-level stop-before-start
                if _cond_pass(step_until, self.rules, self.defaults_window_ms, empty_means=False):
                    inst.done = True
                    if self.logger:
                        self.logger.info(f"[SkillEngine] Finished '{inst.name}' (step until=true pre-start idx={inst.step_idx})")
                    continue

                # wait until step when becomes true
                if not _cond_pass(step_when, self.rules, self.defaults_window_ms):
                    continue

                # launch the primitive and keep its handle
                ref = step.get("use")
                base = self.registry.get(ref)
                if not base:
                    raise KeyError(f"Unknown step target '{ref}'")

                kind = base.get("kind")

                if kind == "primitive":
                    # existing behavior
                    params = dict(base.get("params") or {})
                    params.update(step.get("with") or {})

                    if self.logger:
                        self.logger.info(f"[SkillEngine] → Step {inst.step_idx} executing primitive '{ref}'")

                    self._emit_event("step_started", inst, {"primitive": ref})
                    inst.handle = self._exec_primitive_get_handle(base["action"], params, inst.ctx)

                elif kind == "composite":
                    # NEW: nested composite behavior
                    if self.logger:
                        self.logger.info(f"[SkillEngine] → Step {inst.step_idx} spawning composite '{ref}'")

                    self._emit_event("step_started", inst, {"composite": ref})
                    # attach a handle that tracks the nested skill
                    # (base must carry its own name for helper; if not, pass ref directly)
                    base_with_name = dict(base)
                    base_with_name["name"] = ref
                    inst.handle = self._start_child_composite(inst, step, base_with_name)

                else:
                    raise KeyError(f"Bad step '{ref}' (unknown kind '{kind}')")


                # If primitive finished instantly (sync), advance now
                if inst.handle is not None and inst.handle.done():
                    inst.handle = None
                    inst.step_idx += 1

            # End condition
            if inst.activated and inst.step_idx >= len(steps) and inst.handle is None:
                inst.done = True
                if self.logger:
                    self.logger.info(f"[SkillEngine] Finished '{inst.name}' (all steps)")

                self._emit_event("skill_finished", inst, {"reason": "all_steps"})

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
        try:
            msg = StringMsg()
            msg.data = json.dumps(event, ensure_ascii=False)
            self.skill_status_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish skill status: {e}")

    
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
            self.skill_engine.arm(name, ctx)   # ← arm, don’t run once
            self.get_logger().info(f"/skills/execute armed '{name}' (active={self.skill_engine.active_count()})")
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
        def tts(text: str):
            # Synchronous primitive → return a handle already done
            h = StepHandle()
            self.say(str(text))
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
                    "dance1":       1022,
                    "dance2":       1023,
                    "finger_heart": 1036,
                    "front_flip":   1030,
                    "left_flip":    1042,
                    "right_flip":   1043,
                    "back_flip":    1044,
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
            # If you want this to be async-completing on Nav2 result, wrap similarly.
            h = StepHandle()
            self.navigate_absolute(str(frame), float(x), float(y), float(yaw))
            # If you need to wait for Nav2 result, set the mark_done() in the result callback,
            # and set cancel() to cancel the goal. Otherwise, just mark_done() immediately.
            h.mark_done()
            return h

        def query_beacons(top_n: int, ctx: dict):
            h = StepHandle()
            self._say_top3_beacons_from_db(int(top_n))
            ctx['last_query_beacons_speech'] = "Beacon report complete."
            h.mark_done()
            return h



        return {
            'tts': tts,
            'gesture': gesture,
            'move_relative': move_relative,
            'move_absolute': move_absolute,
            'query_beacons': query_beacons,
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

