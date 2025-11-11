"""
Experiments for MAPF Step 3A (naive independent A*) and 3B (prioritized planning).

Keeps notebook edits minimal by providing callable helpers that reuse
existing modules, and default to the enhanced 10x15 environment.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt  # noqa: F401
except Exception:
    plt = None

from custom_layouts import make_enhanced_nonfl
from grid_planning import (
    build_grid,
    astar,
    multi_agent_sequential,
    astar_time_aware,
    reserve_path,
    plot_routes_unified,
    animate_routes_unified_gif,
    evaluate_multi_agent,
    auto_pick_starts_goals,
    animate_paths_gif,
    free_cells,
)
from pickplace_utils import astar_with_footprint, plot_explored_heatmap, plot_heuristic_field, plot_open_touched_heatmap


def multi_agent_naive(
    grid: List[List[int]],
    starts: List[Tuple[int, int]],
    goals: List[Tuple[int, int]],
) -> List[Optional[List[Tuple[int, int]]]]:
    """Plan each agent independently with static A* (ignoring others)."""
    plans: List[Optional[List[Tuple[int, int]]]] = []
    for s, g in zip(starts, goals):
        plans.append(astar(grid, s, g))
    return plans


def count_edge_conflicts(paths: List[Optional[List[Tuple[int, int]]]]) -> int:
    """Count edge-swap conflicts across all agent pairs over time.

    Edge conflict (swap) at time t if A at u->v and B at v->u between t and t+1.
    """
    tmax = 0
    for p in paths:
        if p:
            tmax = max(tmax, len(p) - 1)
    conflicts = 0
    for t in range(tmax):
        segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for p in paths:
            if not p or t + 1 >= len(p):
                segments.append(((None, None), (None, None)))
                continue
            segments.append((p[t], p[t + 1]))
        n = len(segments)
        for i in range(n):
            u1, v1 = segments[i]
            if u1[0] is None:
                continue
            for j in range(i + 1, n):
                u2, v2 = segments[j]
                if u2[0] is None:
                    continue
                if u1 == v2 and v1 == u2:
                    conflicts += 1
    return conflicts


def run_3a_3b(
    nonfl: Optional[Dict[str, object]] = None,
    k_agents: int = 3,
    seed: int = 13,
    t_max: int = 256,
    make_gif: bool = False,
    annotate_every: int = 5,
) -> Dict[str, object]:
    """Run both naive (3A) and prioritized (3B) on the same setup and compare.

    Returns a dict with starts/goals, both plan sets, and metrics.
    """
    nonfl = nonfl or make_enhanced_nonfl()
    grid = build_grid(nonfl)

    # Sample tasks from zones: inbound -> packing -> outbound
    tasks = sample_zone_tasks(nonfl, k=k_agents, seed=seed)
    starts = [s for (s, _, _) in tasks]
    goals = [d for (_, _, d) in tasks]

    # 3A: naive independent A*
    naive_segments, plans_naive = plan_pickplace_naive_multi(grid, tasks)
    metrics_naive = evaluate_multi_agent(grid, starts, goals, plans_naive)
    metrics_naive["edge_conflicts"] = count_edge_conflicts(plans_naive)

    # 3B: prioritized sequential with time-aware reservations
    prio_segments, plans_prio = plan_pickplace_prioritized_multi(grid, tasks, t_max=t_max)
    metrics_prio = evaluate_multi_agent(grid, starts, goals, plans_prio)
    metrics_prio["edge_conflicts"] = count_edge_conflicts(plans_prio)

    print("[Setup]")
    print("agents:", k_agents, "seed:", seed)
    print("starts:", starts)
    print("goals:", goals)
    print()
    print("[3A Naive]")
    print("success_rate:", metrics_naive.get("success_rate"))
    print("avg_len:", metrics_naive.get("avg_len"))
    print("vertex_conflicts:", metrics_naive.get("vertex_conflicts"))
    print("edge_conflicts:", metrics_naive.get("edge_conflicts"))
    print()
    print("[3B Prioritized]")
    print("success_rate:", metrics_prio.get("success_rate"))
    print("avg_len:", metrics_prio.get("avg_len"))
    print("vertex_conflicts:", metrics_prio.get("vertex_conflicts"))
    print("edge_conflicts:", metrics_prio.get("edge_conflicts"))

    # Unified visualization with staged logic, offset, zones and one-way arrows
    plot_routes_unified(
        nonfl,
        naive_segments,
        title=f"3A Naive independent A* (k={k_agents})",
        annotate_every=annotate_every,
    )
    plot_routes_unified(
        nonfl,
        prio_segments,
        title=f"3B Prioritized planning (k={k_agents})",
        annotate_every=annotate_every,
    )

    if make_gif:
        animate_routes_unified_gif(
            nonfl,
            naive_segments,
            out_gif="naive_3A.gif",
            fps=1,
            annotate_every=annotate_every,
        )
        animate_routes_unified_gif(
            nonfl,
            prio_segments,
            out_gif="prioritized_3B.gif",
            fps=1,
            annotate_every=annotate_every,
        )

    return {
        "nonfl": nonfl,
        "starts": starts,
        "goals": goals,
        "plans_naive": plans_naive,
        "plans_prioritized": plans_prio,
        "segments_naive": naive_segments,
        "segments_prioritized": prio_segments,
        "metrics_naive": metrics_naive,
        "metrics_prioritized": metrics_prio,
    }


__all__ = [
    "multi_agent_naive",
    "count_edge_conflicts",
    "run_3a_3b",
    "sample_zone_tasks",
    "plan_pickplace_naive_multi",
    "plan_pickplace_prioritized_multi",
    "conflict_timeline",
    "plot_conflict_timeline",
    "plot_wait_gantt",
    "plot_constraint_timeline",
    "list_conflicts",
]


# ---------- New helpers for zone-based tasks and staged planning ----------

def sample_zone_tasks(nonfl: Dict[str, object], k: int = 3, seed: int = 13) -> List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """Sample S (inbound), P (packing), D (outbound) triples per agent from zones.

    - Prefer no replacement by cycling shuffled lists (更少重复)
    - Ensure sampled cells are free in grid (zone cells被障碍膨胀覆盖的情况会被跳过)
    - Try to avoid identical triples across agents
    """
    zones = nonfl.get("zones") or {}
    inbound = list(zones.get("inbound", []))
    packing = list(zones.get("packing", []))
    outbound = list(zones.get("outbound", []))

    from grid_planning import build_grid
    grid = build_grid(nonfl)
    H, W = len(grid), len(grid[0]) if grid else (0, 0)

    def is_free(xy):
        if not grid:
            return True
        x, y = xy
        return 0 <= x < H and 0 <= y < W and grid[x][y] == 0

    if not inbound or not packing or not outbound:
        import random as _r
        cells = [c for c in free_cells(nonfl) if is_free(c)]
        _r.Random(seed).shuffle(cells)
        triples = []
        for i in range(k):
            s = cells[i % len(cells)]
            p = cells[(i + k) % len(cells)]
            d = cells[(i + 2 * k) % len(cells)]
            triples.append((s, p, d))
        return triples

    import random as _r
    rnd = _r.Random(seed)
    rnd.shuffle(inbound)
    rnd.shuffle(packing)
    rnd.shuffle(outbound)

    triples: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = []
    used = set()
    i = 0
    while len(triples) < k and i < k * 5:  # small cap to avoid infinite loops
        s = inbound[len(triples) % max(1, len(inbound))]
        p = packing[len(triples) % max(1, len(packing))]
        d = outbound[len(triples) % max(1, len(outbound))]
        if not (is_free(s) and is_free(p) and is_free(d)):
            # rotate candidates until free
            inbound = inbound[1:] + inbound[:1]
            packing = packing[1:] + packing[:1]
            outbound = outbound[1:] + outbound[:1]
            i += 1
            continue
        triple = (s, p, d)
        if triple in used:
            inbound = inbound[1:] + inbound[:1]
            packing = packing[1:] + packing[:1]
            outbound = outbound[1:] + outbound[:1]
            i += 1
            continue
        used.add(triple)
        triples.append(triple)
        i += 1
    return triples


def plan_pickplace_naive_multi(
    grid: List[List[int]],
    tasks: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]],
) -> Tuple[List[List[Dict[str, object]]], List[Optional[List[Tuple[int, int]]]]]:
    """Plan S->P and P->D independently for each agent (no interaction).

    Returns (agents_segments, merged_paths).
    """
    agents_segments: List[List[Dict[str, object]]] = []
    merged: List[Optional[List[Tuple[int, int]]]] = []
    for aidx, (s, p, d) in enumerate(tasks):
        p1 = astar(grid, s, p)
        p2 = astar(grid, p, d)
        agents_segments.append([
            {"agent": aidx, "task": 0, "phase": "SP", "path": p1},
            {"agent": aidx, "task": 0, "phase": "PD", "path": p2},
        ])
        if p1 and p2:
            merged.append(p1 + p2[1:])
        else:
            merged.append(None)
    return agents_segments, merged


# ---------- Analysis helpers: conflict timeline, wait gantt, A* fields ----------

def conflict_timeline(paths: List[Optional[List[Tuple[int, int]]]]) -> Dict[str, List[int]]:
    tmax = 0
    for p in paths:
        if p:
            tmax = max(tmax, len(p) - 1)
    verts = []
    edges = []
    for t in range(tmax + 1):
        # vertex conflicts
        seen = {}
        vconf = 0
        for p in paths:
            if not p or t >= len(p):
                continue
            cell = p[t]
            seen.setdefault(cell, 0)
            seen[cell] += 1
        for c in seen.values():
            if c > 1:
                vconf += (c - 1)
        verts.append(vconf)
        # edge-swap conflicts at step t (transition t->t+1)
        econ = 0
        segs = []
        for p in paths:
            if not p or t + 1 >= len(p):
                segs.append(None)
            else:
                segs.append((p[t], p[t + 1]))
        n = len(segs)
        for i in range(n):
            if segs[i] is None:
                continue
            u1, v1 = segs[i]
            for j in range(i + 1, n):
                if segs[j] is None:
                    continue
                u2, v2 = segs[j]
                if u1 == v2 and v1 == u2:
                    econ += 1
        edges.append(econ)
    return {"t": list(range(tmax + 1)), "vertex": verts, "edge": edges}


def plot_conflict_timeline(tl: Dict[str, List[int]], title: str = "Conflicts over time") -> None:
    import matplotlib.pyplot as plt
    ts = tl["t"]
    v = tl["vertex"]
    e = tl["edge"]
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(ts, v, label="vertex", linewidth=2.0)
    ax.plot(ts, e, label="edge", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("conflicts")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    plt.show()


def plot_wait_gantt(paths: List[Optional[List[Tuple[int, int]]]], title: str = "Wait vs Move (Gantt)") -> None:
    import matplotlib.pyplot as plt
    def segments_wait_move(p):
        if not p or len(p) < 2:
            return []
        segs = []
        cur_type = None
        start = 0
        for i in range(1, len(p)):
            typ = "wait" if p[i] == p[i-1] else "move"
            if cur_type is None:
                cur_type = typ
                start = i - 1
            elif typ != cur_type:
                segs.append((cur_type, start, i-1))
                cur_type = typ
                start = i - 1
        segs.append((cur_type, start, len(p)-1))
        return segs
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * max(3, len(paths))))
    for idx, p in enumerate(paths):
        offset = 0
        for typ, s, e in segments_wait_move(p):
            width = e - s
            color = "tomato" if typ == "wait" else "deepskyblue"
            ax.barh(idx, width, left=s, height=0.6, color=color, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("agent")
    ax.set_yticks(range(len(paths)))
    ax.set_yticklabels([f"A{i}" for i in range(len(paths))])
    ax.grid(True, axis="x", alpha=0.25)
    plt.show()


def list_conflicts(
    paths: List[Optional[List[Tuple[int, int]]]],
    limit: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Enumerate vertex/edge conflicts for the given set of paths."""
    events: List[Dict[str, object]] = []
    if not paths:
        return events
    max_len = max((len(p) for p in paths if p), default=0)
    if max_len == 0:
        return events

    def cell_at(path: List[Tuple[int, int]], t: int) -> Tuple[int, int]:
        if t < len(path):
            return path[t]
        return path[-1]

    for t in range(max_len):
        seen: Dict[Tuple[int, int], List[int]] = {}
        for idx, path in enumerate(paths):
            if not path:
                continue
            cell = cell_at(path, t)
            if cell is None:
                continue
            seen.setdefault(cell, []).append(idx)
        for cell, agents in seen.items():
            if len(agents) > 1:
                events.append({"type": "vertex", "time": t, "cell": cell, "agents": tuple(agents)})
                if limit and len(events) >= limit:
                    return events

        for i in range(len(paths)):
            pi = paths[i]
            if not pi:
                continue
            u1 = cell_at(pi, t)
            v1 = cell_at(pi, t + 1 if t + 1 < max_len else t)
            if u1 == v1 or u1 is None or v1 is None:
                continue
            for j in range(i + 1, len(paths)):
                pj = paths[j]
                if not pj:
                    continue
                u2 = cell_at(pj, t)
                v2 = cell_at(pj, t + 1 if t + 1 < max_len else t)
                if u2 == v2 or u2 is None or v2 is None:
                    continue
                if u1 == v2 and v1 == u2:
                    events.append({
                        "type": "edge",
                        "time": t,
                        "edge": (u1, v1),
                        "agents": (i, j),
                    })
                    if limit and len(events) >= limit:
                        return events
    return events


def summary_compare(result_obj: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    from grid_planning import path_length
    def waits_in_path(path):
        if not path: return 0
        w = 0
        for (x1,y1),(x2,y2) in zip(path, path[1:]):
            if (x1,y1)==(x2,y2): w += 1
        return w
    out = {}
    for key, name in [("plans_naive","Naive"),("plans_prioritized","Prioritized")]:
        plans = result_obj.get(key) or []
        lens = [path_length(p) or 0 for p in plans]
        soc  = sum(lens)
        mksp = max(lens) if lens else 0
        waits = sum(waits_in_path(p) for p in plans if p)
        edge_c = count_edge_conflicts(plans)
        metrics = result_obj.get("metrics_naive" if name=="Naive" else "metrics_prioritized") or {}
        out[name] = {
            "SoC": soc,
            "Makespan": mksp,
            "AvgLen": (soc/len(lens)) if lens else 0,
            "VertexConflicts": metrics.get("vertex_conflicts", 0),
            "EdgeConflicts": edge_c,
            "TotalWaits": waits,
            "SuccessRate": metrics.get("success_rate", 0.0),
        }
    return out


def analyze_astar_for_agent(nonfl: Dict[str, object], agents_segments: List[List[Dict[str, object]]], agent_idx: int = 0, title_prefix: str = "A") -> None:
    grid = build_grid(nonfl)
    segs = agents_segments[agent_idx]
    sp = next((s for s in segs if s.get("phase") == "SP"), None)
    pd = next((s for s in segs if s.get("phase") == "PD"), None)
    if sp and sp.get("path"):
        s_xy, p_xy = sp["path"][0], sp["path"][-1]
        _, fp = astar_with_footprint(grid, s_xy, p_xy)
        plot_explored_heatmap(nonfl, fp, title=f"{title_prefix}{agent_idx} | SP closed-order")
        plot_open_touched_heatmap(nonfl, fp, title=f"{title_prefix}{agent_idx} | SP open-touched")
        plot_heuristic_field(nonfl, fp, field="h")
    if pd and pd.get("path"):
        p_xy2, d_xy = pd["path"][0], pd["path"][-1]
        _, fp2 = astar_with_footprint(grid, p_xy2, d_xy)
        plot_explored_heatmap(nonfl, fp2, title=f"{title_prefix}{agent_idx} | PD closed-order")
        plot_open_touched_heatmap(nonfl, fp2, title=f"{title_prefix}{agent_idx} | PD open-touched")
        plot_heuristic_field(nonfl, fp2, field="h")


def plot_constraint_timeline(constraints: Dict[int, List[object]], title: str = "CBS constraint timeline", max_time: Optional[int] = None) -> None:
    """Visualize per-agent delay/vertex/edge constraints on a timeline."""
    if plt is None:
        print("matplotlib unavailable; cannot plot constraint timeline.")
        return
    if not constraints:
        print("No constraints to visualize.")
        return
    agents = sorted(constraints.keys())
    fig, ax = plt.subplots(figsize=(8.5, 0.8 * max(2, len(agents))))
    delay_shown = False
    vertex_shown = False
    edge_shown = False
    for idx, agent in enumerate(agents):
        cs = constraints.get(agent, []) or []
        for c in cs:
            kind = getattr(c, "kind", None)
            time = getattr(c, "time", 0)
            if kind == "delay":
                width = max(0, time)
                ax.barh(idx, width if width else 0.4, left=0, height=0.5, color="tab:blue", alpha=0.25, edgecolor="none")
                delay_shown = True
            elif kind == "vertex":
                ax.scatter(time, idx, marker="s", color="tab:red", s=40)
                vertex_shown = True
            elif kind == "edge":
                ax.scatter(time, idx, marker="x", color="black", s=40)
                edge_shown = True
    ax.set_title(title)
    ax.set_xlabel("time step")
    ax.set_ylabel("agent")
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([f"A{a}" for a in agents])
    if max_time is not None:
        ax.set_xlim(-1, max_time + 1)
    ax.grid(True, axis="x", alpha=0.2)
    handles = []
    labels = []
    from matplotlib.patches import Patch
    if delay_shown:
        handles.append(Patch(facecolor="tab:blue", alpha=0.25, label="Delay"))
        labels.append("Delay")
    if vertex_shown:
        handles.append(Patch(facecolor="tab:red", label="Vertex"))
        labels.append("Vertex")
    if edge_shown:
        handles.append(Patch(facecolor="none", edgecolor="black", label="Edge"))
        labels.append("Edge")
    if handles:
        ax.legend(handles, labels, loc="upper right")
    plt.show()



def plan_pickplace_prioritized_multi(
    grid: List[List[int]],
    tasks: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]],
    t_max: int = 512,
) -> Tuple[List[List[Dict[str, object]]], List[Optional[List[Tuple[int, int]]]]]:
    """Sequential reservation planning with time-aware A* per segment.

    For each agent: plan S->P, reserve; then plan P->D starting at the end time, reserve.
    Returns (agents_segments, merged_paths).
    """
    occ_v: Dict[int, set] = {}
    occ_e: Dict[int, set] = {}
    agents_segments: List[List[Dict[str, object]]] = []
    merged: List[Optional[List[Tuple[int, int]]]] = []
    for aidx, (s, p, d) in enumerate(tasks):
        p1 = astar_time_aware(grid, s, p, occ_v, occ_e, t_start=0, t_max=t_max)
        if p1 is None:
            agents_segments.append([
                {"agent": aidx, "task": 0, "phase": "SP", "path": None},
                {"agent": aidx, "task": 0, "phase": "PD", "path": None},
            ])
            merged.append(None)
            continue
        reserve_path(occ_v, occ_e, p1)
        t_start2 = len(p1) - 1
        p2 = astar_time_aware(grid, p, d, occ_v, occ_e, t_start=t_start2, t_max=t_max)
        if p2 is not None:
            reserve_path(occ_v, occ_e, p2)
        agents_segments.append([
            {"agent": aidx, "task": 0, "phase": "SP", "path": p1},
            {"agent": aidx, "task": 0, "phase": "PD", "path": p2},
        ])
        if p1 and p2:
            merged.append(p1 + p2[1:])
        else:
            merged.append(None)
    return agents_segments, merged
