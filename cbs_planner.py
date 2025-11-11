"""
Conflict-Based Search (CBS) planner for the warehouse MAPF tasks.

Designed to reuse the existing grid_planning helpers so that the CBS solver
shares the same low-level A* implementation and visualization pipeline as the
naive/prioritized baselines (Step 3).
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from custom_layouts import make_enhanced_nonfl
from grid_planning import astar_time_aware, build_grid

Coord = Tuple[int, int]
Task = Tuple[Coord, Coord, Coord]  # (start, pick, drop)


@dataclass(frozen=True)
class Constraint:
    """CBS constraint for a single agent."""

    agent: int
    kind: str  # "vertex", "edge", or "delay"
    time: int
    cell: Optional[Coord] = None
    edge: Optional[Tuple[Coord, Coord]] = None

    def __post_init__(self) -> None:
        if self.kind not in {"vertex", "edge", "delay"}:
            raise ValueError(f"Unknown constraint kind: {self.kind}")
        if self.kind == "vertex" and self.cell is None:
            raise ValueError("Vertex constraint requires cell")
        if self.kind == "edge" and self.edge is None:
            raise ValueError("Edge constraint requires edge")
        if self.kind == "delay" and self.time < 0:
            raise ValueError("Delay constraint requires non-negative time")


@dataclass(order=True)
class CBSTreeNode:
    sort_key: Tuple[int, int, int] = field(init=False, repr=False)
    cost: int
    conflicts: int
    node_id: int
    constraints: Dict[int, List[Constraint]]
    paths: List[Optional[List[Coord]]]
    segments: List[List[Dict[str, object]]]
    depth: int = 0

    def __post_init__(self) -> None:
        # ordering tuple: (cost, conflicts, node_id)
        object.__setattr__(self, "sort_key", (self.cost, self.conflicts, self.node_id))


class CBSSearchError(RuntimeError):
    def __init__(self, message: str, stats: Dict[str, object], best_result: Optional[Dict[str, object]] = None):
        super().__init__(message)
        self.stats = stats
        self.best_result = best_result


def _constraint_maps(constraints: Sequence[Constraint]) -> Tuple[Dict[int, set], Dict[int, set]]:
    v_map: Dict[int, set] = {}
    e_map: Dict[int, set] = {}
    for c in constraints:
        if c.kind == "vertex" and c.cell is not None:
            v_map.setdefault(c.time, set()).add(c.cell)
        elif c.kind == "edge" and c.edge is not None:
            e_map.setdefault(c.time, set()).add(c.edge)
    return v_map, e_map


def _delay_from_constraints(constraints: Sequence[Constraint]) -> int:
    delay = 0
    for c in constraints:
        if c.kind == "delay" and c.time > delay:
            delay = c.time
    return delay


def _plan_agent_with_constraints(
    agent_idx: int,
    grid: List[List[int]],
    task: Task,
    agent_constraints: Sequence[Constraint],
    t_max: int,
) -> Tuple[Optional[List[Dict[str, object]]], Optional[List[Coord]]]:
    """Plan S->P->D for one agent under the given constraints."""

    (start_xy, pick_xy, drop_xy) = task
    v_map, e_map = _constraint_maps(agent_constraints)
    delay_steps = _delay_from_constraints(agent_constraints)

    p1_raw = astar_time_aware(
        grid,
        start_xy,
        pick_xy,
        occupied_vertices={},
        occupied_edges={},
        t_start=delay_steps,
        t_max=t_max,
        vertex_constraints=v_map,
        edge_constraints=e_map,
    )
    if p1_raw is None:
        return None, None
    p1_conflict = ([start_xy] * delay_steps) + p1_raw
    p1_plot = ([start_xy] * delay_steps) + p1_raw

    t_start2 = len(p1_conflict) - 1
    p2 = astar_time_aware(
        grid,
        pick_xy,
        drop_xy,
        occupied_vertices={},
        occupied_edges={},
        t_start=t_start2,
        t_max=t_max,
        vertex_constraints=v_map,
        edge_constraints=e_map,
    )
    if p2 is None:
        return None, None

    segments = [
        {"agent": agent_idx, "task": 0, "phase": "SP", "path": p1_plot},
        {"agent": agent_idx, "task": 0, "phase": "PD", "path": p2},
    ]
    p2_conflict = p2
    merged_conflict = p1_conflict + p2_conflict[1:]
    return segments, merged_conflict


def _cell_at(path: List[Coord], t: int) -> Coord:
    if t < len(path):
        return path[t]
    return path[-1]


def _detect_conflict(paths: List[Optional[List[Coord]]]) -> Optional[Dict[str, object]]:
    max_len = max((len(p) for p in paths if p), default=0)
    if max_len == 0:
        return None

    n_agents = len(paths)
    for t in range(max_len):
        # vertex conflicts
        seen: Dict[Coord, List[int]] = {}
        for idx, path in enumerate(paths):
            if not path:
                continue
            cell = _cell_at(path, t)
            if cell is None:
                continue
            seen.setdefault(cell, []).append(idx)
        for cell, agents in seen.items():
            if len(agents) > 1:
                return {"type": "vertex", "time": t, "cell": cell, "agents": tuple(agents[:2])}

        # edge conflicts
        for i in range(n_agents):
            pi = paths[i]
            if not pi:
                continue
            u1 = _cell_at(pi, t)
            v1 = _cell_at(pi, t + 1 if t + 1 < max_len else t)
            if u1 is None or v1 is None or u1 == v1:
                continue
            for j in range(i + 1, n_agents):
                pj = paths[j]
                if not pj:
                    continue
                u2 = _cell_at(pj, t)
                v2 = _cell_at(pj, t + 1 if t + 1 < max_len else t)
                if u2 is None or v2 is None or u2 == v2:
                    continue
                if u1 == v2 and v1 == u2:
                    return {
                        "type": "edge",
                        "time": t,
                        "edge_i": (u1, v1),
                        "edge_j": (u2, v2),
                        "agents": (i, j),
                    }
    return None


def _constraint_from_conflict(agent: int, conflict: Dict[str, object]) -> Constraint:
    if conflict["type"] == "vertex":
        return Constraint(agent=agent, kind="vertex", time=conflict["time"], cell=conflict["cell"])
    edge_key = "edge_i" if conflict["agents"][0] == agent else "edge_j"
    return Constraint(agent=agent, kind="edge", time=conflict["time"], edge=conflict[edge_key])


def _cost_of_paths(paths: List[Optional[List[Coord]]], metric: str = "soc") -> int:
    lens = []
    for p in paths:
        if not p:
            return float("inf")
        lens.append(max(0, len(p) - 1))
    if not lens:
        return float("inf")
    if metric.lower() == "makespan":
        return max(lens)
    return sum(lens)


def run_cbs(
    tasks: List[Task],
    nonfl: Optional[Dict[str, object]] = None,
    grid: Optional[List[List[int]]] = None,
    t_max: int = 256,
    cost_metric: str = "soc",
    max_expansions: int = 20000,
    verbose: bool = True,
    enforce_entry_queue: bool = True,
    entry_queue_spacing: int = 1,
) -> Dict[str, object]:
    """Run Conflict-Based Search and return the solution bundle."""

    if nonfl is None:
        nonfl = make_enhanced_nonfl()
    if grid is None:
        grid = build_grid(nonfl)

    n_agents = len(tasks)
    constraints: Dict[int, List[Constraint]] = {}
    segments: List[List[Dict[str, object]]] = []
    paths: List[Optional[List[Coord]]] = []

    if enforce_entry_queue and entry_queue_spacing > 0:
        start_counts: Dict[Coord, int] = {}
        for agent_idx, task in enumerate(tasks):
            start = task[0]
            delay = start_counts.get(start, 0) * entry_queue_spacing
            start_counts[start] = start_counts.get(start, 0) + 1
            if delay > 0:
                constraints.setdefault(agent_idx, []).append(
                    Constraint(agent=agent_idx, kind="delay", time=delay)
                )

    for agent_idx, task in enumerate(tasks):
        segs, path = _plan_agent_with_constraints(agent_idx, grid, task, constraints.get(agent_idx, []), t_max)
        segments.append(segs or [])
        paths.append(path)
    root_cost = _cost_of_paths(paths, metric=cost_metric)
    if root_cost == float("inf"):
        raise RuntimeError("CBS root node infeasible (check tasks or t_max)")

    node_counter = 0
    open_heap: List[Tuple[Tuple[int, int, int], CBSTreeNode]] = []
    root_conflict = _detect_conflict(paths)
    root = CBSTreeNode(
        cost=root_cost,
        conflicts=0 if root_conflict is None else 1,
        node_id=node_counter,
        constraints={k: list(v) for k, v in constraints.items()},
        paths=paths,
        segments=segments,
        depth=0,
    )
    heapq.heappush(open_heap, (root.sort_key, root))
    best_node = root
    best_cost = root_cost

    stats = {"expanded": 0, "generated": 1, "log": deque(maxlen=50)}

    def log(msg: str) -> None:
        if verbose:
            print(msg)
        stats["log"].append(msg)

    while open_heap:
        _, node = heapq.heappop(open_heap)
        conflict = _detect_conflict(node.paths)
        if conflict is None:
            log(f"[CBS] Solution cost={node.cost}, expanded={stats['expanded']} nodes")
            stats_out = dict(stats)
            stats_out["log_tail"] = list(stats_out.pop("log"))
            return {
                "paths": node.paths,
                "segments": node.segments,
                "constraints": node.constraints,
                "tasks": tasks,
                "nonfl": nonfl,
                "grid": grid,
                "cost": node.cost,
                "metric": cost_metric,
                "stats": stats_out,
            }

        stats["expanded"] += 1
        if verbose and (stats["expanded"] <= 20 or stats["expanded"] % 1000 == 0):
            log(
                f"[CBS] Expand node#{node.node_id} depth={node.depth} cost={node.cost} conflict={conflict}"
            )
        if stats["expanded"] > max_expansions:
            stats_out = dict(stats)
            stats_out["log_tail"] = list(stats_out.pop("log"))
            best_result = {
                "paths": best_node.paths if best_node else None,
                "segments": best_node.segments if best_node else None,
                "constraints": best_node.constraints if best_node else {},
                "tasks": tasks,
                "nonfl": nonfl,
                "grid": grid,
                "cost": _cost_of_paths(best_node.paths, metric=cost_metric) if best_node else float("inf"),
                "metric": cost_metric,
                "stats": stats_out,
                "partial": True,
            } if best_node else None
            raise CBSSearchError(f"CBS exceeded max expansions ({max_expansions})", stats, best_result)

        for agent in conflict["agents"]:
            existing = node.constraints.get(agent, [])
            branch_constraints = [_constraint_from_conflict(agent, conflict)]
            if (
                conflict["type"] == "vertex"
                and conflict.get("cell") == tasks[agent][0]
            ):
                current_delay = _delay_from_constraints(existing)
                delay_target = max(current_delay, conflict["time"] + 1)
                branch_constraints.append(Constraint(agent=agent, kind="delay", time=delay_target))

            for new_constraint in branch_constraints:
                new_constraints = {a: list(lst) for a, lst in node.constraints.items()}
                new_constraints.setdefault(agent, []).append(new_constraint)

                if verbose and (stats["expanded"] <= 20 or stats["expanded"] % 1000 == 0):
                    log(f"  |- branch on agent {agent} with constraint {new_constraint}")
                segs, path = _plan_agent_with_constraints(agent, grid, tasks[agent], new_constraints[agent], t_max)
                if path is None:
                    if verbose and (stats["expanded"] <= 20 or stats["expanded"] % 1000 == 0):
                        log("     infeasible branch (no path)")
                    continue  # infeasible branch

                new_paths = list(node.paths)
                new_segments = [list(seg_list) for seg_list in node.segments]
                new_paths[agent] = path
                new_segments[agent] = segs or []
                new_cost = _cost_of_paths(new_paths, metric=cost_metric)
                if new_cost == float("inf"):
                    continue

                node_counter += 1
                new_conflict = _detect_conflict(new_paths)
                child = CBSTreeNode(
                    cost=new_cost,
                    conflicts=0 if new_conflict is None else 1,
                    node_id=node_counter,
                    constraints=new_constraints,
                    paths=new_paths,
                    segments=new_segments,
                    depth=node.depth + 1,
                )
                heapq.heappush(open_heap, (child.sort_key, child))
                stats["generated"] += 1
                log(
                    f"     -> child node#{child.node_id} cost={child.cost} depth={child.depth} "
                    f"conflict={'none' if new_conflict is None else new_conflict}"
                )
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_node = child

    stats_out = dict(stats)
    stats_out["log_tail"] = list(stats_out.pop("log"))
    best_result = {
        "paths": best_node.paths if best_node else None,
        "segments": best_node.segments if best_node else None,
        "constraints": best_node.constraints if best_node else {},
        "tasks": tasks,
        "nonfl": nonfl,
        "grid": grid,
        "cost": _cost_of_paths(best_node.paths, metric=cost_metric) if best_node else float("inf"),
        "metric": cost_metric,
        "stats": stats_out,
        "partial": True,
    } if best_node else None
    raise CBSSearchError("CBS failed to find a solution", stats_out, best_result)


__all__ = [
    "Constraint",
    "CBSSearchError",
    "run_cbs",
]
