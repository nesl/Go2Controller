#!/usr/bin/env python3
"""
compare_sessions.py

Compare two FINAL_SCORE JSON logs produced by the contamination server.

Examples:
  python compare_sessions.py logs/final_score_A.json logs/final_score_B.json
  python compare_sessions.py A.json B.json --out compare.json
  python compare_sessions.py A.json B.json --keys total_successful_disposals completion_rate
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


# ----------------------------
# Utilities
# ----------------------------

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (isinstance(x, float) and math.isnan(x))

def safe_float(x: Any) -> Optional[float]:
    return float(x) if is_number(x) else None

def flatten_numeric(d: Any, prefix: str = "") -> Dict[str, float]:
    """
    Flatten nested dicts/lists into key paths -> numeric values.
    Lists are flattened by index.
    Only numeric leaf values are included.
    """
    out: Dict[str, float] = {}

    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_numeric(v, p))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            p = f"{prefix}[{i}]"
            out.update(flatten_numeric(v, p))
    else:
        if is_number(d):
            out[prefix] = float(d)
    return out

def get_path(d: Dict[str, Any], path: str) -> Any:
    """
    Get nested value using dot-separated keys (no list indexing support here).
    """
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur

def fmt_delta(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "n/a"
    delta = b - a
    return f"{delta:+.3f}"

def fmt_val(x: Any) -> str:
    if x is None:
        return "n/a"
    if is_number(x):
        # show ints cleanly
        if isinstance(x, int) or (isinstance(x, float) and x.is_integer()):
            return str(int(x))
        return f"{float(x):.3f}"
    return str(x)

def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def print_table(rows: List[Tuple[str, Any, Any, str]], headers: Tuple[str, str, str, str]) -> None:
    # simple fixed-width table
    colw = [max(len(headers[i]), max((len(str(r[i])) for r in rows), default=0)) for i in range(4)]
    def line(vals):
        return " | ".join(str(vals[i]).ljust(colw[i]) for i in range(4))
    print(line(headers))
    print("-+-".join("-" * w for w in colw))
    for r in rows:
        print(line(r))

def set_to_sorted_list(x: Any) -> List[int]:
    if x is None:
        return []
    if isinstance(x, list):
        return sorted(int(v) for v in x)
    if isinstance(x, set):
        return sorted(int(v) for v in x)
    return []


# ----------------------------
# Comparison logic
# ----------------------------

DEFAULT_KEY_PATHS = [
    "total_successful_disposals",
    "total_present_properties",
    "completion_rate",

    # NEW: hazard-vs-benign attempt splits
    "disposal_attempts.attempts_total",
    "disposal_attempts.attempts_on_hazard_boxes",
    "disposal_attempts.attempts_on_benign_boxes",
    "disposal_attempts.attempt_rate_on_hazard",
    "disposal_attempts.attempt_rate_on_benign",

    "disposal_success_attempts.success_attempts_total",
    "disposal_success_attempts.success_attempts_on_hazard_boxes",
    "disposal_success_attempts.success_attempts_on_benign_boxes",
    "disposal_success_attempts.success_rate_on_hazard",
    "disposal_success_attempts.success_rate_on_benign",

    # final breakdown totals (if present)
    "final_time_breakdown.effort_totals.sense_time_total_sec",
    "final_time_breakdown.effort_totals.sense_time_X_sec",
    "final_time_breakdown.effort_totals.sense_time_Y_sec",
    "final_time_breakdown.effort_totals.dispose_time_total_sec",
    "final_time_breakdown.effort_totals.dispose_time_on_hazard_sec",
    "final_time_breakdown.effort_totals.dispose_time_on_benign_sec",
    "final_time_breakdown.effort_totals.dispose_wallclock_total_sec",
]


def compare_scalar_paths(a: Dict[str, Any], b: Dict[str, Any], paths: List[str]) -> List[Tuple[str, Any, Any, str]]:
    rows: List[Tuple[str, Any, Any, str]] = []
    for p in paths:
        av = get_path(a, p)
        bv = get_path(b, p)
        delta = fmt_delta(safe_float(av), safe_float(bv))
        rows.append((p, fmt_val(av), fmt_val(bv), delta))
    return rows

def compare_per_agent_success(a: Dict[str, Any], b: Dict[str, Any]) -> List[Tuple[str, Any, Any, str]]:
    a_map = a.get("per_agent_success", {}) or {}
    b_map = b.get("per_agent_success", {}) or {}
    agents = sorted(set(map(str, a_map.keys())) | set(map(str, b_map.keys())))
    rows: List[Tuple[str, Any, Any, str]] = []
    for ag in agents:
        av = a_map.get(ag, 0)
        bv = b_map.get(ag, 0)
        rows.append((ag, fmt_val(av), fmt_val(bv), fmt_delta(safe_float(av), safe_float(bv))))
    return rows

def compare_effort_by_agent(a: Dict[str, Any], b: Dict[str, Any]) -> List[Tuple[str, Any, Any, str]]:
    a_map = (((a.get("final_time_breakdown") or {}).get("effort_by_agent")) or {})
    b_map = (((b.get("final_time_breakdown") or {}).get("effort_by_agent")) or {})
    agents = sorted(set(map(str, a_map.keys())) | set(map(str, b_map.keys())))

    # pick a few high-signal agent metrics
    metric_keys = [
        "sense_time_total_sec",
        "sense_time_X_sec",
        "sense_time_Y_sec",
        "dispose_time_total_sec",
        "dispose_time_on_hazard_sec",
        "dispose_time_on_benign_sec",
        "dispose_completed",
        "dispose_cancelled",
        "sense_completed",
        "sense_cancelled",
    ]

    rows: List[Tuple[str, Any, Any, str]] = []
    for ag in agents:
        for mk in metric_keys:
            av = (a_map.get(ag, {}) or {}).get(mk)
            bv = (b_map.get(ag, {}) or {}).get(mk)
            label = f"{ag}.{mk}"
            rows.append((label, fmt_val(av), fmt_val(bv), fmt_delta(safe_float(av), safe_float(bv))))
    return rows

def compare_box_id_lists(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare the final-only box id lists if present:
      final_time_breakdown.box_ids_global
      final_time_breakdown.box_ids_by_agent

    Returns a structured diff:
      - global added/removed for each category
      - per-agent added/removed for each category
    """
    out: Dict[str, Any] = {"global": {}, "by_agent": {}}

    a_global = (((a.get("final_time_breakdown") or {}).get("box_ids_global")) or {})
    b_global = (((b.get("final_time_breakdown") or {}).get("box_ids_global")) or {})

    def diff_list(a_list: List[int], b_list: List[int]) -> Dict[str, List[int]]:
        sa, sb = set(a_list), set(b_list)
        return {
            "removed": sorted(list(sa - sb)),
            "added": sorted(list(sb - sa)),
            "kept": sorted(list(sa & sb)),
        }

    for kind in ["sense", "dispose"]:
        a_k = a_global.get(kind, {}) or {}
        b_k = b_global.get(kind, {}) or {}
        a_comp = set_to_sorted_list(a_k.get("completed_box_ids"))
        b_comp = set_to_sorted_list(b_k.get("completed_box_ids"))
        a_canc = set_to_sorted_list(a_k.get("cancelled_box_ids"))
        b_canc = set_to_sorted_list(b_k.get("cancelled_box_ids"))

        out["global"][kind] = {
            "completed": diff_list(a_comp, b_comp),
            "cancelled": diff_list(a_canc, b_canc),
        }

    a_by_agent = (((a.get("final_time_breakdown") or {}).get("box_ids_by_agent")) or {})
    b_by_agent = (((b.get("final_time_breakdown") or {}).get("box_ids_by_agent")) or {})
    agents = sorted(set(map(str, a_by_agent.keys())) | set(map(str, b_by_agent.keys())))

    for ag in agents:
        out["by_agent"][ag] = {}
        for kind in ["sense", "dispose"]:
            a_k = (a_by_agent.get(ag, {}) or {}).get(kind, {}) or {}
            b_k = (b_by_agent.get(ag, {}) or {}).get(kind, {}) or {}
            a_comp = set_to_sorted_list(a_k.get("completed_box_ids"))
            b_comp = set_to_sorted_list(b_k.get("completed_box_ids"))
            a_canc = set_to_sorted_list(a_k.get("cancelled_box_ids"))
            b_canc = set_to_sorted_list(b_k.get("cancelled_box_ids"))

            out["by_agent"][ag][kind] = {
                "completed": diff_list(a_comp, b_comp),
                "cancelled": diff_list(a_canc, b_canc),
            }

    return out

def compare_wallclock_by_box(a: Dict[str, Any], b: Dict[str, Any]) -> List[Tuple[str, Any, Any, str]]:
    a_map = (((a.get("final_time_breakdown") or {}).get("dispose_wallclock_by_box_sec")) or {})
    b_map = (((b.get("final_time_breakdown") or {}).get("dispose_wallclock_by_box_sec")) or {})
    box_ids = sorted(set(map(str, a_map.keys())) | set(map(str, b_map.keys())), key=lambda x: int(x))
    rows: List[Tuple[str, Any, Any, str]] = []
    for bid in box_ids:
        av = a_map.get(bid)
        bv = b_map.get(bid)
        rows.append((f"box_{bid}", fmt_val(av), fmt_val(bv), fmt_delta(safe_float(av), safe_float(bv))))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_a", type=Path)
    ap.add_argument("session_b", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="Write structured comparison JSON here.")
    ap.add_argument("--keys", nargs="*", default=None, help="Specific scalar key paths to compare (dot-separated).")
    ap.add_argument("--full-numeric-diff", action="store_true",
                    help="Also compute a flattened numeric diff of ALL numeric leaf fields (can be noisy).")
    args = ap.parse_args()

    A = load_json(args.session_a)
    B = load_json(args.session_b)

    meta_a = {k: A.get(k) for k in ["_saved_at_utc", "_scenario_id", "_frozen_time_sim", "_time_limit_sim"]}
    meta_b = {k: B.get(k) for k in ["_saved_at_utc", "_scenario_id", "_frozen_time_sim", "_time_limit_sim"]}

    print_section("Session metadata")
    rows_meta = [
        ("file", str(args.session_a), str(args.session_b), ""),
        ("_saved_at_utc", fmt_val(meta_a.get("_saved_at_utc")), fmt_val(meta_b.get("_saved_at_utc")), ""),
        ("_scenario_id", fmt_val(meta_a.get("_scenario_id")), fmt_val(meta_b.get("_scenario_id")), ""),
        ("_frozen_time_sim", fmt_val(meta_a.get("_frozen_time_sim")), fmt_val(meta_b.get("_frozen_time_sim")), ""),
        ("_time_limit_sim", fmt_val(meta_a.get("_time_limit_sim")), fmt_val(meta_b.get("_time_limit_sim")), ""),
    ]
    print_table(rows_meta, headers=("field", "A", "B", "Δ"))

    # Scalars
    print_section("Key scalar metrics")
    paths = args.keys if args.keys is not None else DEFAULT_KEY_PATHS
    rows = compare_scalar_paths(A, B, paths)
    print_table(rows, headers=("metric", "A", "B", "Δ"))

    # per-agent success (credited)
    print_section("Per-agent credited successes (compute_score -> per_agent_success)")
    rows = compare_per_agent_success(A, B)
    print_table(rows, headers=("agent", "A", "B", "Δ"))

    # effort by agent
    print_section("Per-agent effort deltas (final_time_breakdown.effort_by_agent)")
    rows = compare_effort_by_agent(A, B)
    print_table(rows, headers=("agent.metric", "A", "B", "Δ"))

    # wallclock disposal by box
    print_section("Wall-clock disposal time by box (seconds)")
    rows = compare_wallclock_by_box(A, B)
    if rows:
        print_table(rows, headers=("box", "A", "B", "Δ"))
    else:
        print("(no dispose_wallclock_by_box_sec found in one or both sessions)")

    # box id list diffs (completed/cancelled)
    print_section("Box-ID list diffs (completed/cancelled) if present")
    box_id_diff = compare_box_id_lists(A, B)
    # concise print
    for kind in ["sense", "dispose"]:
        g = box_id_diff["global"].get(kind, {})
        if not g:
            continue
        print(f"\nGLOBAL {kind.upper()}:")
        for st in ["completed", "cancelled"]:
            d = g.get(st, {})
            print(f"  {st}: +{d.get('added', [])}  -{d.get('removed', [])}")

    # Optional: full numeric diff
    full_numeric = None
    if args.full_numeric_diff:
        print_section("Flattened numeric diff (all numeric leaves) — can be noisy")
        flatA = flatten_numeric(A)
        flatB = flatten_numeric(B)
        keys = sorted(set(flatA.keys()) | set(flatB.keys()))
        noisy_rows: List[Tuple[str, Any, Any, str]] = []
        for k in keys:
            av = flatA.get(k)
            bv = flatB.get(k)
            # show only changed or missing
            if av is None or bv is None or abs(bv - av) > 1e-9:
                noisy_rows.append((k, fmt_val(av), fmt_val(bv), fmt_delta(av, bv)))
        # Print first N to avoid flooding
        N = 80
        if len(noisy_rows) > N:
            print(f"(showing first {N} of {len(noisy_rows)} diffs; use --out to save full)")
        print_table(noisy_rows[:N], headers=("path", "A", "B", "Δ"))
        full_numeric = {"flatA": flatA, "flatB": flatB}

    # Output structured comparison JSON
    if args.out is not None:
        report = {
            "session_a": str(args.session_a),
            "session_b": str(args.session_b),
            "meta_a": meta_a,
            "meta_b": meta_b,
            "scalar_paths": paths,
            "scalar_comparison": [
                {"path": p, "A": get_path(A, p), "B": get_path(B, p),
                 "delta": (safe_float(get_path(B, p)) - safe_float(get_path(A, p)))
                          if (safe_float(get_path(A, p)) is not None and safe_float(get_path(B, p)) is not None)
                          else None}
                for p in paths
            ],
            "per_agent_success": {
                "A": A.get("per_agent_success", {}) or {},
                "B": B.get("per_agent_success", {}) or {},
            },
            "effort_by_agent": {
                "A": ((A.get("final_time_breakdown") or {}).get("effort_by_agent")) or {},
                "B": ((B.get("final_time_breakdown") or {}).get("effort_by_agent")) or {},
            },
            "wallclock_by_box": {
                "A": ((A.get("final_time_breakdown") or {}).get("dispose_wallclock_by_box_sec")) or {},
                "B": ((B.get("final_time_breakdown") or {}).get("dispose_wallclock_by_box_sec")) or {},
            },
            "box_id_list_diff": box_id_diff,
            "full_numeric": full_numeric,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"\n[compare] wrote report -> {args.out}")

if __name__ == "__main__":
    main()

