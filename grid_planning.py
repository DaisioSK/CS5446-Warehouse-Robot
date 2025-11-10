"""
Grid construction, single-agent A*, and multi-agent planning helpers.

Spun out of the original CS5446 warehouse notebook to keep notebooks small.
"""

from __future__ import annotations

import heapq
import random
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    from matplotlib.patches import FancyArrowPatch, Patch
except ImportError:  # pragma: no cover - plotting disabled outside notebook env
    pe = plt = PillowWriter = FancyArrowPatch = Patch = None

Action = Tuple[int, int, str]
ACTIONS: List[Action] = [
    (-1, 0, "N"),
    (1, 0, "S"),
    (0, 1, "E"),
    (0, -1, "W"),
]


def _require_matplotlib() -> None:
    if plt is None or pe is None:
        raise ImportError("matplotlib is required for plotting helpers")


def build_grid(nonfl: Dict[str, object]) -> List[List[int]]:
    H, W = nonfl.get("H"), nonfl.get("W")
    grid = [[0] * W for _ in range(H)]
    for (x, y) in nonfl.get("obstacles", []):
        if 0 <= x < H and 0 <= y < W:
            grid[x][y] = 1
    return grid


def plot_grid(
    nonfl: Dict[str, object],
    start_xy: Optional[Tuple[int, int]] = None,
    title: str = "map",
    figsize: Optional[Tuple[float, float]] = None,
    annotate_special: bool = False,
) -> None:
    _require_matplotlib()
    H, W = nonfl.get("H"), nonfl.get("W")
    grid = build_grid(nonfl)
    fig, ax = plt.subplots(figsize=figsize or (4, 4))
    ax.imshow(grid, cmap="cividis")
    ax.set_title(title)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    legend_items: List[Patch] = []
    if start_xy:
        sx, sy = start_xy
        if 0 <= sx < H and 0 <= sy < W:
            ax.text(sy, sx, "S", ha="center", va="center", fontweight="bold")
            legend_items.append(Patch(facecolor="none", edgecolor="none", label="Start"))
    gx, gy = nonfl.get("GOAL_X"), nonfl.get("GOAL_Y")
    if isinstance(gx, int) and isinstance(gy, int):
        ax.text(gy, gx, "G", ha="center", va="center", fontweight="bold")
        legend_items.append(Patch(facecolor="none", edgecolor="none", label="Goal"))

    if annotate_special:
        zones = nonfl.get("zones") or {}
        zone_styles = {
            "inbound": ("I", "tab:blue"),
            "packing": ("P", "darkorange"),
            "outbound": ("O", "tab:red"),
            "charging": ("C", "tab:green"),
        }
        for key, (label, color) in zone_styles.items():
            cells = zones.get(key, [])
            if not cells:
                continue
            legend_items.append(Patch(facecolor=color, alpha=0.4, label=key.capitalize()))
            for (x, y) in cells:
                ax.scatter(y, x, s=90, color=color, alpha=0.4, edgecolors="black", linewidths=0.5)
                ax.text(y, x, label, color="white", ha="center", va="center", fontweight="bold")

        one_way_edges = nonfl.get("one_way_edges", [])
        if one_way_edges:
            legend_items.append(Patch(facecolor="none", edgecolor="none", label="One-way"))
        for (x1, y1), (x2, y2) in one_way_edges or []:
            arrow = FancyArrowPatch(
                (y1, x1),
                (y2, x2),
                arrowstyle="->",
                mutation_scale=12,
                color="white",
                linewidth=1.5,
            )
            ax.add_patch(arrow)

    if legend_items:
        unique = []
        filtered = []
        for item in legend_items:
            label = item.get_label()
            if label in unique:
                continue
            unique.append(label)
            filtered.append(item)
        ax.legend(
            handles=filtered,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            framealpha=0.9,
        )
        fig.subplots_adjust(right=0.8)
    plt.show()


def in_bounds(H: int, W: int, x: int, y: int) -> bool:
    return 0 <= x < H and 0 <= y < W


def neighbors(grid: List[List[int]], x: int, y: int) -> Sequence[Tuple[int, int, str]]:
    H, W = len(grid), len(grid[0])
    for dx, dy, a in ACTIONS:
        nx, ny = x + dx, y + dy
        if in_bounds(H, W, nx, ny) and grid[nx][ny] == 0:
            yield nx, ny, a


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    H, W = len(grid), len(grid[0])
    if not (in_bounds(H, W, start[0], start[1]) and in_bounds(H, W, goal[0], goal[1])):
        return None
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        return None

    openpq: List[Tuple[int, int, Tuple[int, int]]] = []
    heapq.heappush(openpq, (manhattan(start, goal), 0, start))
    came: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    g: Dict[Tuple[int, int], int] = {start: 0}

    while openpq:
        _, _, cur = heapq.heappop(openpq)
        if cur == goal:
            path = []
            node = cur
            while node is not None:
                path.append(node)
                node = came[node]
            return list(reversed(path))
        cx, cy = cur
        for nx, ny, _ in neighbors(grid, cx, cy):
            ng = g[cur] + 1
            if (nx, ny) not in g or ng < g[(nx, ny)]:
                g[(nx, ny)] = ng
                came[(nx, ny)] = cur
                heapq.heappush(openpq, (ng + manhattan((nx, ny), goal), ng, (nx, ny)))
    return None


def path_to_actions(path: List[Tuple[int, int]]) -> List[str]:
    if not path or len(path) < 2:
        return []
    out: List[str] = []
    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        dx, dy = x2 - x1, y2 - y1
        if (dx, dy) == (-1, 0):
            out.append("N")
        elif (dx, dy) == (1, 0):
            out.append("S")
        elif (dx, dy) == (0, 1):
            out.append("E")
        elif (dx, dy) == (0, -1):
            out.append("W")
        else:
            out.append("WAIT")
    return out


def plot_path(nonfl: Dict[str, object], path: Optional[List[Tuple[int, int]]], title: str = "path") -> None:
    _require_matplotlib()
    H, W = nonfl.get("H"), nonfl.get("W")
    grid = build_grid(nonfl)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(grid)
    ax.set_title(title)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    if path:
        for t, (x, y) in enumerate(path):
            ax.text(y, x, str(t), ha="center", va="center")
    gx, gy = nonfl.get("GOAL_X"), nonfl.get("GOAL_Y")
    if isinstance(gx, int) and isinstance(gy, int):
        ax.text(gy, gx, "G", ha="center", va="center", fontweight="bold")
    plt.show()


# ---------- Multi-agent helpers ----------
ALL_MOVES = [(-1, 0, "N"), (1, 0, "S"), (0, 1, "E"), (0, -1, "W"), (0, 0, "WAIT")]


def neighbors4(grid: List[List[int]], x: int, y: int) -> Sequence[Tuple[int, int]]:
    H, W = len(grid), len(grid[0])
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if in_bounds(H, W, nx, ny) and grid[nx][ny] == 0:
            yield nx, ny


def astar_time_aware(
    grid: List[List[int]],
    start_xy: Tuple[int, int],
    goal_xy: Tuple[int, int],
    occupied_vertices: Dict[int, set],
    occupied_edges: Dict[int, set],
    t_start: int = 0,
    t_max: int = 512,
) -> Optional[List[Tuple[int, int]]]:
    H, W = len(grid), len(grid[0])
    sx, sy = start_xy
    gx, gy = goal_xy

    def heuristic(node: Tuple[int, int, int]) -> int:
        x, y, _ = node
        return manhattan((x, y), (gx, gy))

    def is_free_vertex(x: int, y: int, t: int) -> bool:
        if not in_bounds(H, W, x, y) or grid[x][y] == 1:
            return False
        return (t not in occupied_vertices) or ((x, y) not in occupied_vertices[t])

    def is_free_edge(x1: int, y1: int, x2: int, y2: int, t: int) -> bool:
        pair = ((x1, y1), (x2, y2))
        if t in occupied_edges and pair in occupied_edges[t]:
            return False
        opp = ((x2, y2), (x1, y1))
        if t in occupied_edges and opp in occupied_edges[t]:
            return False
        return True

    start = (sx, sy, t_start)
    openpq: List[Tuple[int, int, Tuple[int, int, int]]] = []
    heapq.heappush(openpq, (heuristic(start), 0, start))
    g_cost = {start: 0}
    came = {start: None}

    while openpq:
        _, _, cur = heapq.heappop(openpq)
        x, y, t = cur
        if (x, y) == (gx, gy):
            seq = []
            node = cur
            while node is not None:
                seq.append(node)
                node = came[node]
            seq.reverse()
            return [(xx, yy) for (xx, yy, _) in seq]
        if t >= t_max:
            continue
        for dx, dy, _ in ALL_MOVES:
            nx, ny, nt = x + dx, y + dy, t + 1
            if not is_free_vertex(nx, ny, nt):
                continue
            if not (dx == 0 and dy == 0) and not is_free_edge(x, y, nx, ny, t):
                continue
            ng = g_cost[cur] + 1
            nxt = (nx, ny, nt)
            if nxt not in g_cost or ng < g_cost[nxt]:
                g_cost[nxt] = ng
                came[nxt] = cur
                heapq.heappush(openpq, (ng + heuristic(nxt), ng, nxt))
    return None


def reserve_path(occupied_vertices: Dict[int, set], occupied_edges: Dict[int, set], path_xy: List[Tuple[int, int]]) -> None:
    for t, (x, y) in enumerate(path_xy):
        occupied_vertices.setdefault(t, set()).add((x, y))
        if t + 1 < len(path_xy):
            x2, y2 = path_xy[t + 1]
            occupied_edges.setdefault(t, set()).add(((x, y), (x2, y2)))


def multi_agent_sequential(
    grid: List[List[int]],
    starts: List[Tuple[int, int]],
    goals: List[Tuple[int, int]],
    t_max: int = 512,
) -> List[Optional[List[Tuple[int, int]]]]:
    occupied_vertices: Dict[int, set] = {}
    occupied_edges: Dict[int, set] = {}
    plans: List[Optional[List[Tuple[int, int]]]] = []
    for s, g in zip(starts, goals):
        path = astar_time_aware(grid, s, g, occupied_vertices, occupied_edges, t_start=0, t_max=t_max)
        if path is None:
            plans.append(None)
        else:
            reserve_path(occupied_vertices, occupied_edges, path)
            plans.append(path)
    return plans


def _agent_offset(k: int) -> Tuple[float, float]:
    base_offsets = [(-0.18, -0.18), (0.0, 0.0), (0.18, 0.18), (0.18, -0.18), (-0.18, 0.18)]
    return base_offsets[k % len(base_offsets)]


def plot_multiple_paths_pretty_offset(
    nonfl: Dict[str, object],
    paths: List[Optional[List[Tuple[int, int]]]],
    starts: Optional[List[Tuple[int, int]]] = None,
    goals: Optional[List[Tuple[int, int]]] = None,
    title: str = "Multi-agent plan (offset)",
) -> None:
    _require_matplotlib()
    H, W = nonfl.get("H"), nonfl.get("W")
    grid = build_grid(nonfl)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid)
    ax.set_title(title)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)

    def white_text(xy: Tuple[int, int], text: str, **kw):
        kw.setdefault("fontsize", 10)
        kw.setdefault("color", "white")
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        txt = ax.text(xy[1], xy[0], text, **kw)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])
        return txt

    gx, gy = nonfl.get("GOAL_X"), nonfl.get("GOAL_Y")
    if isinstance(gx, int) and isinstance(gy, int):
        white_text((gx, gy), "G", fontsize=12, fontweight="bold")

    if starts:
        for i, (sx, sy) in enumerate(starts):
            dx, dy = _agent_offset(i)
            white_text((sx + dx, sy + dy), f"S{i}", fontsize=10, fontweight="bold")

    if goals:
        for (gx2, gy2) in goals:
            white_text((gx2, gy2), "G", fontsize=11, fontweight="bold")

    for k, path in enumerate(paths):
        if not path:
            continue
        dx, dy = _agent_offset(k)
        xs = [x + dx for (x, y) in path]
        ys = [y + dy for (x, y) in path]
        ax.plot(ys, xs, linestyle="-", linewidth=2.0, alpha=0.95, label=f"agent {k}")
        ax.scatter(ys, xs, s=22, edgecolors="white", linewidths=0.8, zorder=3)
        for t, (x, y) in enumerate(path):
            white_text((x + dx, y + dy), f"{k}:{t}", fontsize=9)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.show()


def animate_paths_gif(
    nonfl: Dict[str, object],
    paths: List[Optional[List[Tuple[int, int]]]],
    starts: Optional[List[Tuple[int, int]]] = None,
    goals: Optional[List[Tuple[int, int]]] = None,
    out_gif: str = "multi_agent_playback.gif",
    fps: int = 2,
) -> None:
    _require_matplotlib()
    if PillowWriter is None:
        raise ImportError("matplotlib PillowWriter is required for GIF export")
    H, W = nonfl.get("H"), nonfl.get("W")
    grid = build_grid(nonfl)
    T = 0
    for p in paths:
        if p:
            T = max(T, len(p) - 1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    writer = PillowWriter(fps=fps)

    with writer.saving(fig, out_gif, dpi=120):
        for t in range(T + 1):
            ax.clear()
            ax.imshow(grid)
            ax.set_title(f"t = {t}")
            ax.set_xticks(range(W))
            ax.set_yticks(range(H))
            ax.set_xlabel("y")
            ax.set_ylabel("x")
            ax.set_xlim(-0.5, W - 0.5)
            ax.set_ylim(H - 0.5, -0.5)

            gx, gy = nonfl.get("GOAL_X"), nonfl.get("GOAL_Y")
            if isinstance(gx, int) and isinstance(gy, int):
                ax.text(gy, gx, "G", color="white", ha="center", va="center", fontsize=12, path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])

            if starts:
                for i, (sx, sy) in enumerate(starts):
                    dx, dy = _agent_offset(i)
                    ax.text(
                        sy + dy,
                        sx + dx,
                        f"S{i}",
                        color="white",
                        ha="center",
                        va="center",
                        fontsize=10,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
                    )
            if goals:
                for (gx2, gy2) in goals:
                    ax.text(
                        gy2,
                        gx2,
                        "G",
                        color="white",
                        ha="center",
                        va="center",
                        fontsize=11,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
                    )

            for k, path in enumerate(paths):
                if not path:
                    continue
                dx, dy = _agent_offset(k)
                upto = min(t + 1, len(path))
                xs = [x + dx for (x, y) in path[:upto]]
                ys = [y + dy for (x, y) in path[:upto]]
                if len(xs) >= 2:
                    ax.plot(ys, xs, linestyle="-", linewidth=2.0, alpha=0.95, label=f"agent {k}" if t == 0 else None)
                ax.scatter(ys[-1:], xs[-1:], s=28, edgecolors="white", linewidths=0.9, zorder=3)
                ax.text(
                    ys[-1],
                    xs[-1],
                    f"{k}:{upto-1}",
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=10,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
                )
            if t == 0:
                ax.legend(loc="upper right", framealpha=0.9)
            writer.grab_frame()
    print(f"Saved GIF to: {out_gif}")


def free_cells(nonfl: Dict[str, object]) -> List[Tuple[int, int]]:
    grid = build_grid(nonfl)
    H, W = nonfl.get("H"), nonfl.get("W")
    cells = []
    for x in range(H):
        for y in range(W):
            if grid[x][y] == 0:
                cells.append((x, y))
    return cells


def auto_pick_starts_goals(nonfl: Dict[str, object], k: int = 3, seed: int = 42, avoid_goal: bool = True) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    cells = free_cells(nonfl)
    random.Random(seed).shuffle(cells)
    gx, gy = nonfl.get("GOAL_X"), nonfl.get("GOAL_Y")
    if avoid_goal and isinstance(gx, int) and isinstance(gy, int):
        cells = [(x, y) for (x, y) in cells if (x, y) != (gx, gy)]
    starts = cells[:k]
    goals: List[Tuple[int, int]] = []
    for i in range(k):
        if i == 0 and isinstance(gx, int) and isinstance(gy, int):
            goals.append((gx, gy))
        else:
            goals.append(cells[k + i] if k + i < len(cells) else starts[i])
    return starts, goals


def path_length(path: Optional[List[Tuple[int, int]]]) -> Optional[int]:
    if not path:
        return None
    return max(0, len(path) - 1)


def success(path: Optional[List[Tuple[int, int]]], goal: Tuple[int, int]) -> bool:
    if not path:
        return False
    return path[-1] == goal


def count_vertex_conflicts(paths: List[Optional[List[Tuple[int, int]]]]) -> int:
    tmax = 0
    for p in paths:
        if p:
            tmax = max(tmax, len(p) - 1)
    conflicts = 0
    for t in range(tmax + 1):
        seen: Dict[Tuple[int, int], List[int]] = {}
        for i, p in enumerate(paths):
            if not p or t >= len(p):
                continue
            cell = p[t]
            seen.setdefault(cell, []).append(i)
        for agents in seen.values():
            if len(agents) > 1:
                conflicts += len(agents) - 1
    return conflicts


def evaluate_multi_agent(grid: List[List[int]], starts: List[Tuple[int, int]], goals: List[Tuple[int, int]], plans: List[Optional[List[Tuple[int, int]]]]) -> Dict[str, object]:
    lengths = [path_length(p) for p in plans]
    succs = [success(p, g) for p, g in zip(plans, goals)]
    v_conf = count_vertex_conflicts(plans)
    avg_len = None
    if lengths and all(l is not None for l in lengths):
        avg_len = sum(lengths) / len(lengths)
    return {
        "avg_len": avg_len,
        "success_rate": sum(succs) / len(succs) if succs else 0.0,
        "vertex_conflicts": v_conf,
        "lengths": lengths,
        "successes": succs,
    }


def single_agent_demo(nonfl: Dict[str, object], inst: Dict[str, object]) -> None:
    _require_matplotlib()
    grid = build_grid(nonfl)
    start = inst.get("init_agent_xy") or (0, 0)
    goal = (nonfl.get("GOAL_X"), nonfl.get("GOAL_Y"))
    path = astar(grid, start, goal)
    actions = None if path is None else path_to_actions(path)
    print("[Single-agent]")
    print("start:", start, "goal:", goal, "path_len:", None if path is None else len(path) - 1)
    print("actions:", actions)
    plot_path(nonfl, path, title="Single-agent A*")


def multi_agent_demo(nonfl: Dict[str, object], k: int = 3, seed: int = 13, title_suffix: str = "") -> Dict[str, object]:
    _require_matplotlib()
    grid = build_grid(nonfl)
    starts, goals = auto_pick_starts_goals(nonfl, k=k, seed=seed)
    plans = multi_agent_sequential(grid, starts, goals, t_max=256)
    metrics = evaluate_multi_agent(grid, starts, goals, plans)
    print("[Multi-agent]")
    print("starts:", starts)
    print("goals:", goals)
    print("lengths:", metrics["lengths"])
    print("success_rate:", metrics["success_rate"], "vertex_conflicts:", metrics["vertex_conflicts"])
    plot_multiple_paths_pretty_offset(nonfl, plans, starts=starts, goals=goals, title=f"Multi-agent A* (k={k}) {title_suffix}")
    return metrics


__all__ = [
    "ACTIONS",
    "build_grid",
    "plot_grid",
    "in_bounds",
    "neighbors",
    "manhattan",
    "astar",
    "path_to_actions",
    "plot_path",
    "neighbors4",
    "astar_time_aware",
    "reserve_path",
    "multi_agent_sequential",
    "plot_multiple_paths_pretty_offset",
    "animate_paths_gif",
    "free_cells",
    "auto_pick_starts_goals",
    "path_length",
    "success",
    "count_vertex_conflicts",
    "evaluate_multi_agent",
    "single_agent_demo",
    "multi_agent_demo",
]
