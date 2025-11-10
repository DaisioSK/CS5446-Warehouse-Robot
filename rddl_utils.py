"""
Utility routines for parsing and summarizing RDDL files.

Extracted from the original CS5446 warehouse notebook so they can be reused
both in notebooks and scripts without duplicating a wall of helper code.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Public regex collection so other modules can reuse them if needed.
RE = {
    "domain_decl": re.compile(r"\bdomain\s+([A-Za-z_][A-Za-z0-9_]*)\s*{", re.IGNORECASE),
    "instance_decl": re.compile(r"\binstance\s+([A-Za-z_][A-Za-z0-9_]*)\s*{", re.IGNORECASE),
    "nonfluent_decl": re.compile(r"\bnon-fluents\s+([A-Za-z_][A-Za-z0-9_]*)\s*{", re.IGNORECASE),
    "domain_ref": re.compile(r"\bdomain\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*;", re.IGNORECASE),
    "nonfluent_ref": re.compile(r"\bnon-fluents\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*;", re.IGNORECASE),
    "types_block": re.compile(r"\btypes\s*{([^}]*)}", re.IGNORECASE | re.DOTALL),
    "objects_block": re.compile(r"\bobjects\s*{([^}]*)}", re.IGNORECASE | re.DOTALL),
    "pvars_block": re.compile(r"\bpvariables\s*{", re.IGNORECASE),
    "cpfs_block": re.compile(r"\bcpfs\s*{", re.IGNORECASE),
    "reward_block": re.compile(r"\breward\s*=\s*(.*?);", re.IGNORECASE | re.DOTALL),
    "inits_block": re.compile(r"\binit-state\s*{([^}]*)}", re.IGNORECASE | re.DOTALL),
    "horizon": re.compile(r"\bhorizon\s*=\s*([0-9]+)\s*;", re.IGNORECASE),
    "discount": re.compile(r"\bdiscount\s*=\s*([0-9]*\.?[0-9]+)\s*;", re.IGNORECASE),
    "max_nondef_actions": re.compile(r"\bmax-nondef-actions\s*=\s*([0-9]+)\s*;", re.IGNORECASE),
}


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def strip_comments(text: str) -> str:
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def parse_pairs(block: str) -> Dict[str, List[str]]:
    pairs: Dict[str, List[str]] = {}
    if not block:
        return pairs
    for line in re.split(r";|\n", block):
        if ":" not in line:
            continue
        t, rest = line.split(":", 1)
        items = [x.strip() for x in re.split(r"[,\s]+", rest) if x.strip()]
        if items:
            pairs[t.strip()] = sorted(set(items))
    return pairs


def extract_pvariables_between(text_no_comments: str) -> Dict[str, List[str]]:
    """
    Extract pvariable names between the pvariables{...} and cpfs blocks.
    Returns a dict keyed by pvariable category (state/action/etc).
    """
    out = {"state": [], "action": [], "intermediate": [], "observable": [], "nonfluent": [], "others": []}
    m_start = RE["pvars_block"].search(text_no_comments)
    m_end = RE["cpfs_block"].search(text_no_comments)
    segment = text_no_comments[m_start.end(): m_end.start()] if (m_start and m_end) else text_no_comments

    style1 = re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*{\s*[^,}]+,\s*([A-Za-z-]+)\s*}", segment)
    for match in style1:
        name = match.group(1)
        kind = match.group(2).lower()
        if "state-fluent" in kind:
            out["state"].append(name)
        elif "action-fluent" in kind:
            out["action"].append(name)
        elif "intermediate" in kind or "interm-fluent" in kind:
            out["intermediate"].append(name)
        elif "observ" in kind:
            out["observable"].append(name)
        elif "non-fluent" in kind or "nonfluent" in kind:
            out["nonfluent"].append(name)
        else:
            out["others"].append(name)

    style2 = re.finditer(r"\b(state|action|intermediate|observ|nonfluent)\b[^\n;]*?\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", segment, re.IGNORECASE)
    for match in style2:
        kind = match.group(1).lower()
        if kind.startswith("observ"):
            kind = "observable"
        out.get(kind, out["others"]).append(match.group(2))

    for key in out:
        out[key] = sorted(set(out[key]))
    out["action"] = [n for n in out["action"] if n != "fluents"]
    return out


def summarize_domain(text_raw: str) -> Dict[str, object]:
    text = strip_comments(text_raw)
    info: Dict[str, object] = {
        "kind": "domain",
        "domain_name": (RE["domain_decl"].search(text).group(1) if RE["domain_decl"].search(text) else None),
        "types": sorted(parse_pairs(RE["types_block"].search(text).group(1)).keys())
        if RE["types_block"].search(text) else [],
        "objects": parse_pairs(RE["objects_block"].search(text).group(1))
        if RE["objects_block"].search(text) else {},
        "pvariables": extract_pvariables_between(text)
        if (RE["pvars_block"].search(text) and RE["cpfs_block"].search(text)) else {},
        "has_reward": bool(RE["reward_block"].search(text)),
        "cpfs_count": 0,
    }
    m_c = re.search(r"\bcpfs\s*{(.*?)}", text, flags=re.IGNORECASE | re.DOTALL)
    info["cpfs_count"] = len([s for s in re.split(r";", m_c.group(1)) if s.strip()]) if m_c else 0
    return info


def summarize_instance(text_raw: str) -> Dict[str, object]:
    text = strip_comments(text_raw)
    info: Dict[str, object] = {
        "kind": "instance",
        "instance_name": (RE["instance_decl"].search(text).group(1) if RE["instance_decl"].search(text) else None),
        "domain_ref": (RE["domain_ref"].search(text).group(1) if RE["domain_ref"].search(text) else None),
        "nonfluent_ref": (RE["nonfluent_ref"].search(text).group(1) if RE["nonfluent_ref"].search(text) else None),
        "horizon": int(RE["horizon"].search(text).group(1)) if RE["horizon"].search(text) else None,
        "discount": float(RE["discount"].search(text).group(1)) if RE["discount"].search(text) else None,
        "max_nondef_actions": int(RE["max_nondef_actions"].search(text).group(1)) if RE["max_nondef_actions"].search(text) else None,
        "objects": parse_pairs(RE["objects_block"].search(text).group(1)) if RE["objects_block"].search(text) else {},
        "init_assignments": [],
        "init_agent_xy": None,
    }
    m_init = RE["inits_block"].search(text)
    if m_init:
        assigns = [s.strip() for s in re.split(r";", m_init.group(1)) if s.strip()]
        info["init_assignments"] = assigns
        ax = re.search(r"\bagent_x\s*=\s*([0-9]+)", m_init.group(1))
        ay = re.search(r"\bagent_y\s*=\s*([0-9]+)", m_init.group(1))
        if ax and ay:
            info["init_agent_xy"] = (int(ax.group(1)), int(ay.group(1)))
    return info


def summarize_nonfluents(text_raw: str) -> Dict[str, object]:
    text = strip_comments(text_raw)
    name_m = RE["nonfluent_decl"].search(text)
    info: Dict[str, object] = {
        "kind": "non-fluents",
        "nonfluent_name": (name_m.group(1) if name_m else None),
        "domain_ref": (RE["domain_ref"].search(text).group(1) if RE["domain_ref"].search(text) else None),
        "objects": parse_pairs(RE["objects_block"].search(text).group(1)) if RE["objects_block"].search(text) else {},
        "H": None,
        "W": None,
        "GOAL_X": None,
        "GOAL_Y": None,
        "obstacles": [],
    }
    inner = re.search(r"\bnon-fluents\s*{(.*?)}", text, flags=re.IGNORECASE | re.DOTALL)
    body = inner.group(1) if inner else ""

    def grab_int(var: str) -> Optional[int]:
        match = re.search(fr"\b{var}\s*=\s*([0-9]+)", body)
        return int(match.group(1)) if match else None

    info["H"] = grab_int("H")
    info["W"] = grab_int("W")
    info["GOAL_X"] = grab_int("GOAL_X")
    info["GOAL_Y"] = grab_int("GOAL_Y")
    coords = re.findall(r"OBSTACLE\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)\s*=\s*true", body, flags=re.IGNORECASE)
    info["obstacles"] = [(int(a), int(b)) for a, b in coords]
    return info


def summarize_file(path: str) -> Tuple[str, Dict[str, object]]:
    raw = read_text(path)
    if re.search(r"\bdomain\s+[A-Za-z_][A-Za-z0-9_]*\s*{", raw, re.IGNORECASE):
        return "domain", summarize_domain(raw)
    if re.search(r"\binstance\s+[A-Za-z_][A-Za-z0-9_]*\s*{", raw, re.IGNORECASE):
        return "instance", summarize_instance(raw)
    if re.search(r"\bnon-fluents\s+[A-Za-z_][A-Za-z0-9_]*\s*{", raw, re.IGNORECASE):
        return "nonfl", summarize_nonfluents(raw)
    return "unknown", {"kind": "unknown"}


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def pretty_domain(info: Dict[str, object]) -> None:
    print(f"Domain name: {info.get('domain_name')}")
    print(f"Types: {', '.join(info.get('types', [])) or '(none)'}")
    objs = info.get("objects") or {}
    if objs:
        print("Objects:")
        for t, items in objs.items():
            print(f"  - {t}: {', '.join(items)}")
    pv = info.get("pvariables", {})
    print(f"State fluents ({len(pv.get('state', []))}): {pv.get('state', []) or '(none)'}")
    print(f"Action fluents ({len(pv.get('action', []))}): {pv.get('action', []) or '(none)'}")
    print(f"Has reward: {info.get('has_reward')} | CPFs (rough count): {info.get('cpfs_count')}")


def pretty_instance(info: Dict[str, object]) -> None:
    print(f"Instance name: {info.get('instance_name')}")
    print(f"Domain ref: {info.get('domain_ref')}")
    print(f"Non-fluents ref: {info.get('nonfluent_ref')}")
    print(f"Horizon: {info.get('horizon')}  Discount: {info.get('discount')}  MaxNondefActions: {info.get('max_nondef_actions')}")
    objs = info.get("objects") or {}
    if objs:
        print("Objects:")
        for t, items in objs.items():
            print(f"  - {t}: {', '.join(items)}")
    inits = info.get("init_assignments") or []
    if inits:
        print("Init-state:")
        for a in inits:
            print(f"  - {a}")
    if info.get("init_agent_xy"):
        print(f"Parsed start (agent_x, agent_y): {info['init_agent_xy']}")


def pretty_nonfluents(info: Dict[str, object], tag: str) -> None:
    print(f"Non-fluents name: {info.get('nonfluent_name')} ({tag})")
    print(f"Domain ref: {info.get('domain_ref')}")
    objs = info.get("objects") or {}
    if objs:
        print("Objects:")
        for t, items in objs.items():
            print(f"  - {t}: {', '.join(items)}")
    H, W = info.get("H"), info.get("W")
    gx, gy = info.get("GOAL_X"), info.get("GOAL_Y")
    print(f"H x W = {H} x {W} | Goal = ({gx}, {gy})")
    print(f"Obstacles: {len(info.get('obstacles', []))}")
    if info.get("obstacles", [])[:10]:
        print(f"Sample obstacles (first 10): {info['obstacles'][:10]}")


def print_state_action_overview(dom_info: Dict[str, object], nfl_info: Dict[str, object], tag: str) -> None:
    print("\n" + "-" * 80)
    print(f"[State & Action overview] Map = {tag}")
    pv = dom_info.get("pvariables", {})
    states = pv.get("state", [])
    actions = pv.get("action", [])
    print(f"- State (vector view): dim = {len(states)} | variables = {states or '(none)'}")
    H, W = nfl_info.get("H"), nfl_info.get("W")
    feasible = None
    if isinstance(H, int) and isinstance(W, int):
        feasible = H * W - len(nfl_info.get("obstacles", []))
    print(f"- State (grid view): H={H}, W={W}, obstacles={len(nfl_info.get('obstacles', []))} -> feasible cells = {feasible}")
    print(f"- Actions: count = {len(actions)} | set = {actions or '(none)'}")
    print("- Notes: typical dynamics = move_* changes (agent_x, agent_y); boundary/obstacle -> stay put.")
    print("         instance enforces max-nondef-actions=1 per step.")
    if dom_info.get("has_reward"):
        print("         reward shaping: step penalty; +/- distance delta; collision penalty; goal bonus.")


__all__ = [
    "RE",
    "read_text",
    "strip_comments",
    "parse_pairs",
    "extract_pvariables_between",
    "summarize_domain",
    "summarize_instance",
    "summarize_nonfluents",
    "summarize_file",
    "print_header",
    "pretty_domain",
    "pretty_instance",
    "pretty_nonfluents",
    "print_state_action_overview",
]
