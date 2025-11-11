"""
Grid construction, single-agent A*, and multi-agent planning helpers.

Spun out of the original CS5446 warehouse notebook to keep notebooks small.
"""

from __future__ import annotations

import heapq
import random
from typing import Dict, List, Optional, Sequence, Tuple, Set

try:
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
except ImportError:  # pragma: no cover - plotting disabled outside notebook env
    pe = plt = PillowWriter = FancyArrowPatch = Patch = Rectangle = None

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


def build_one_way_forbidden(nonfl: Dict[str, object]) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
    blocked: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    for edge in nonfl.get("one_way_edges", []) or []:
        src, dst = edge
        blocked.add((dst, src))
    return blocked


def _draw_one_way_edges(ax, edges):
    if not edges or FancyArrowPatch is None:
        return
    for (x1, y1), (x2, y2) in edges:
        arrow = FancyArrowPatch(
            (y1, x1),
            (y2, x2),
            arrowstyle="->",
            mutation_scale=13,
            color="white",
            linewidth=1.6,
        )
        ax.add_patch(arrow)


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


def astar(
    grid: List[List[int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    forbidden_edges: Optional[Set[Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
) -> Optional[List[Tuple[int, int]]]:
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
            if forbidden_edges and ((cx, cy), (nx, ny)) in forbidden_edges:
                continue
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


def plot_path(
    nonfl: Dict[str, object],
    path: Optional[List[Tuple[int, int]]],
    title: str = "path",
    color: str = "deepskyblue",
    annotate_every: int = 5,
) -> None:
    _require_matplotlib()
    H, W = nonfl.get("H"), nonfl.get("W")
    grid = build_grid(nonfl)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(grid, cmap="cividis", alpha=0.85)
    ax.set_title(title)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)

    if not path:
        ax.text(0.5, 0.5, "No path", transform=ax.transAxes)
        plt.show()
        return

    xs = [x for x, _ in path]
    ys = [y for _, y in path]
    ax.plot(ys, xs, "-", linewidth=2.8, color=color, alpha=0.95)
    ax.scatter(ys, xs, s=28, c=range(len(path)), cmap="plasma", edgecolors="white", linewidths=0.6)

    for idx, (x, y) in enumerate(path):
        if annotate_every and (idx in (0, len(path) - 1) or idx % annotate_every == 0):
            txt = "S" if idx == 0 else ("G" if idx == len(path) - 1 else str(idx))
            ax.text(
                y,
                x,
                txt,
                color="white",
                ha="center",
                va="center",
                fontsize=9,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
            )

    _draw_one_way_edges(ax, nonfl.get("one_way_edges", []))
    plt.show()


def plot_staged_path(
    nonfl: Dict[str, object],
    segments: List[Dict[str, object]],
    title: str = "staged path",
    annotate_every: int = 5,
) -> None:
    """Plot multiple path segments with per-stage colors and slight offsets."""
    _require_matplotlib()
    if not segments:
        raise ValueError("segments list is empty")

    H, W = nonfl.get("H"), nonfl.get("W")
    grid = build_grid(nonfl)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap="cividis", alpha=0.8)
    ax.set_title(title)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)

    handles = []
    labels = []
    for seg in segments:
        path = seg.get("path")
        if not path or len(path) < 2:
            continue
        color = seg.get("color", "deepskyblue")
        label = seg.get("label")
        dx, dy = seg.get("offset", (0.0, 0.0))
        xs = [x + dx for x, _ in path]
        ys = [y + dy for _, y in path]
        line = ax.plot(ys, xs, "-", linewidth=2.6, color=color, alpha=0.95)[0]
        ax.scatter(ys, xs, s=26, color=color, alpha=0.7, edgecolors="white", linewidths=0.4)
        if label:
            handles.append(line)
            labels.append(label)

        for idx, (x, y) in enumerate(path):
            if not annotate_every:
                continue
            if idx not in (0, len(path) - 1) and idx % annotate_every != 0:
                continue
            txt = f"{label}:{idx}" if label else str(idx)
            ax.text(
                y + dy,
                x + dx,
                txt,
                color="white",
                fontsize=8,
                ha="center",
                va="center",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="black")],
            )

    _draw_one_way_edges(ax, nonfl.get("one_way_edges", []))

    if handles:
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, framealpha=0.9)
        fig.subplots_adjust(right=0.78)

    plt.show()


# A staged-style multi-agent plot, reusing plot_staged_path semantics
def plot_multiagent_staged(
    nonfl: Dict[str, object],
    paths: List[Optional[List[Optional[Tuple[int, int]]]]],
    labels: Optional[List[str]] = None,
    title: str = "Multi-agent plan (staged style)",
    annotate_every: int = 5,
) -> None:
    """Plot multiple agent paths using the same visual style as plot_staged_path.

    Each agent path is mapped to a segment with its own color and slight
    positional offset, and timestamps are sparsely annotated according to
    annotate_every.
    """
    # Palette is short on purpose; it will cycle for many agents
    palette = [
        "deepskyblue",
        "tomato",
        "mediumseagreen",
        "gold",
        "orchid",
        "dodgerblue",
        "darkorange",
    ]

    segs: List[Dict[str, object]] = []
    for k, path in enumerate(paths):
        if not path:
            continue
        spawn = _first_real_index(path)
        clean = [cell for cell in path if cell is not None]
        if not clean or len(clean) < 2:
            continue
        label = labels[k] if labels and k < len(labels) else f"agent{k}"
        color = palette[k % len(palette)]
        off = _agent_offset(k)
        segs.append({
            "label": label,
            "path": clean,
            "color": color,
            "offset": off,
            "t_offset": spawn or 0,
        })
    if not segs:
        _require_matplotlib()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No paths", transform=ax.transAxes, ha="center", va="center")
        plt.show()
        return
    plot_staged_path(nonfl, segs, title=title, annotate_every=annotate_every)

def _draw_zone_patches(ax, nonfl: Dict[str, object], legend_items: List[Patch]) -> None:
    zones = nonfl.get("zones") or {}
    if Rectangle is None:
        return
    # Draw shelves (obstacles) as solid yellow blocks
    for (x, y) in nonfl.get("obstacles", []) or []:
        ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1.0, 1.0, facecolor="#f0d95a", alpha=0.9, edgecolor="black", linewidth=0.4))
    legend_items.append(Patch(facecolor="#f0d95a", edgecolor="black", linewidth=0.5, label="Shelf"))
    styles = {
        "inbound": ("tab:blue", 0.35, "Inbound"),
        "packing": ("darkorange", 0.35, "Packing"),
        "outbound": ("tab:red", 0.35, "Outbound"),
        "charging": ("tab:green", 0.35, "Charging"),
    }
    for key, cells in zones.items():
        if key not in styles:
            continue
        color, alpha, label = styles[key]
        for (x, y) in cells:
            ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1.0, 1.0, facecolor=color, alpha=alpha, edgecolor="black", linewidth=0.4))
        legend_items.append(Patch(facecolor=color, alpha=0.35, edgecolor="black", linewidth=0.5, label=label))


def plot_routes_unified(
    nonfl: Dict[str, object],
    agents_segments: List[List[Dict[str, object]]],
    title: str = "Routes (unified)",
    annotate_every: int = 5,
    figsize: Tuple[float, float] = (9.5, 7.0),
) -> None:
    """Unified visualization inspired by plot_staged_path.

    - Background: grid + zone color patches + one-way arrows
    - Per-agent: slight offset to avoid overlap
    - Per-segment: different linestyles to distinguish phases (e.g., S->P vs P->D)
    - Labels: sparse time indices every k steps, with agent/task prefixes
    - Legend placed outside to avoid occlusion
    """
    _require_matplotlib()
    H, W = nonfl.get("H"), nonfl.get("W")
    grid = build_grid(nonfl)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(grid, cmap="cividis", alpha=0.85)
    ax.set_title(title)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)

    legend_items: List[Patch] = []
    _draw_zone_patches(ax, nonfl, legend_items)
    if nonfl.get("one_way_edges"):
        legend_items.append(Patch(facecolor="none", edgecolor="none", label="One-way"))
    _draw_one_way_edges(ax, nonfl.get("one_way_edges", []))

    palette = [
        "deepskyblue",
        "tomato",
        "mediumseagreen",
        "gold",
        "orchid",
        "dodgerblue",
        "darkorange",
    ]
    ls_for_phase = {"SP": "--", "PD": "-", "GEN": "-"}
    phase_legend_added = set()
    agent_handles = {}

    for aidx, segs in enumerate(agents_segments):
        color = palette[aidx % len(palette)]
        dx, dy = _agent_offset(aidx)
        for seg in segs:
            path = seg.get("path")
            if not path or len(path) < 2:
                continue
            phase = seg.get("phase", "GEN")
            ls = seg.get("linestyle") or ls_for_phase.get(phase, "-")
            task_idx = seg.get("task", 0)
            t_offset = int(seg.get("t_offset", 0) or 0)
            xs = [x + dx for (x, y) in path]
            ys = [y + dy for (x, y) in path]
            line = ax.plot(ys, xs, ls, linewidth=2.6, color=color, alpha=0.96)[0]
            ax.scatter(ys, xs, s=26, color=color, alpha=0.8, edgecolors="white", linewidths=0.5)

            if aidx not in agent_handles:
                agent_handles[aidx] = line

            for t, (x, y) in enumerate(path):
                if not annotate_every:
                    continue
                if t in (0, len(path) - 1) or (t % annotate_every == 0):
                    txt = f"A{aidx}T{task_idx}:{t_offset + t}"
                    ax.text(
                        y + dy,
                        x + dx,
                        txt,
                        color="white",
                        fontsize=8.5,
                        ha="center",
                        va="center",
                        path_effects=[pe.withStroke(linewidth=2.3, foreground="black")],
                    )

            key = (phase, ls)
            if key not in phase_legend_added:
                legend_items.append(Patch(facecolor="none", edgecolor="black", label=("S→P" if phase == "SP" else ("P→D" if phase == "PD" else phase)), linewidth=2.6))
                phase_legend_added.add(key)

    agent_list = sorted(agent_handles.items())
    agent_lines = [h for _, h in agent_list]
    agent_labels = [f"A{idx}" for idx, _ in agent_list]
    if agent_lines:
        lg1 = ax.legend(agent_lines, agent_labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.9, title="Agents")
        ax.add_artist(lg1)

    if legend_items:
        lg2 = ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.02, 0.42), framealpha=0.9, title="Layers")
        ax.add_artist(lg2)

    fig.subplots_adjust(right=0.78)
    plt.show()


def animate_routes_unified_gif(
    nonfl: Dict[str, object],
    agents_segments: List[List[Dict[str, object]]],
    out_gif: str = "routes_unified.gif",
    fps: int = 1,
    annotate_every: int = 5,
    figsize: Tuple[float, float] = (9.5, 7.0),
) -> None:
    _require_matplotlib()
    if PillowWriter is None:
        raise ImportError("matplotlib PillowWriter is required for GIF export")
    H, W = nonfl.get("H"), nonfl.get("W")
    grid = build_grid(nonfl)

    # Compute per-agent concatenated lengths to decide total T
    def concat_len(segs: List[Dict[str, object]]) -> int:
        latest = 0
        fallback = 0
        for seg in segs:
            p = seg.get("path") or []
            if len(p) < 2:
                continue
            dur = len(p) - 1
            start = seg.get("t_offset")
            if start is None:
                start = fallback
            fallback = start + dur
            latest = max(latest, start + dur)
        return latest

    T = 0
    for segs in agents_segments:
        T = max(T, concat_len(segs))

    palette = [
        "deepskyblue",
        "tomato",
        "mediumseagreen",
        "gold",
        "orchid",
        "dodgerblue",
        "darkorange",
    ]
    ls_for_phase = {"SP": "--", "PD": "-", "GEN": "-"}
    head_markers = ["o", "^", "s", "X", "D", "P"]  # circle, triangle, square, cross, diamond, plus-filled

    agent_task_windows: List[Dict[int, Dict[str, int]]] = []
    for segs in agents_segments:
        windows: Dict[int, Dict[str, int]] = {}
        fallback = 0
        for seg in segs:
            path = seg.get("path") or []
            if len(path) < 2:
                continue
            seg_start = seg.get("t_offset")
            if seg_start is None:
                seg_start = fallback
            seg_end = seg_start + len(path) - 1
            fallback = seg_end
            task_idx = seg.get("task", 0)
            info = windows.setdefault(task_idx, {"start": seg_start, "end": seg_end})
            info["start"] = min(info["start"], seg_start)
            info["end"] = max(info["end"], seg_end)
        if windows:
            last_task = max(windows.keys())
            for task_idx, info in windows.items():
                info["display_end"] = T if task_idx == last_task else info["end"]
        agent_task_windows.append(windows)

    fig, ax = plt.subplots(figsize=figsize)
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, out_gif, dpi=120):
        for t in range(T + 1):
            ax.clear()
            ax.imshow(grid, cmap="cividis", alpha=0.85)
            ax.set_title(f"t = {t}")
            ax.set_xticks(range(W))
            ax.set_yticks(range(H))
            ax.set_xlabel("y")
            ax.set_ylabel("x")
            ax.set_xlim(-0.5, W - 0.5)
            ax.set_ylim(H - 0.5, -0.5)

            legend_items: List[Patch] = []
            _draw_zone_patches(ax, nonfl, legend_items)
            if nonfl.get("one_way_edges"):
                legend_items.append(Patch(facecolor="none", edgecolor="none", label="One-way"))
            _draw_one_way_edges(ax, nonfl.get("one_way_edges", []))

            for aidx, segs in enumerate(agents_segments):
                color = palette[aidx % len(palette)]
                dx, dy = _agent_offset(aidx)
                elapsed = 0
                last_end = 0
                windows = agent_task_windows[aidx]
                for seg in segs:
                    p = seg.get("path") or []
                    if len(p) < 2:
                        continue
                    phase = seg.get("phase", "GEN")
                    ls = ls_for_phase.get(phase, "-")
                    Tseg = len(p) - 1
                    seg_start = seg.get("t_offset")
                    if seg_start is None:
                        seg_start = last_end
                    last_end = seg_start + Tseg
                    task_idx = seg.get("task", 0)
                    window = windows.get(task_idx)
                    if not window:
                        continue
                    display_end = window.get("display_end", seg_start + Tseg)
                    if t < seg_start or t > display_end:
                        continue
                    upto = max(0, min(Tseg, t - seg_start))
                    xs = [x + dx for (x, y) in p[: upto + 1]]
                    ys = [y + dy for (x, y) in p[: upto + 1]]
                    ax.plot(ys, xs, ls, linewidth=2.6, color=color, alpha=0.96)

                    is_active = seg_start <= t <= seg_start + Tseg
                    if is_active:
                        head_x, head_y = p[upto]
                        m = head_markers[aidx % len(head_markers)]
                        ax.scatter(
                            [head_y + dy],
                            [head_x + dx],
                            s=70,
                            marker=m,
                            edgecolors="white",
                            linewidths=1.2,
                            c=[color],
                            zorder=4,
                        )
                        if annotate_every:
                            ax.text(
                                head_y + dy,
                                head_x + dx,
                                f"A{aidx}:{seg_start + upto}",
                                color="white",
                                fontsize=8.5,
                                ha="center",
                                va="center",
                                path_effects=[pe.withStroke(linewidth=2.3, foreground="black")],
                            )
                    elapsed = last_end

            if legend_items:
                ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.02, 0.42), framealpha=0.9)
                fig.subplots_adjust(right=0.78)
            writer.grab_frame()
    print(f"Saved GIF to: {out_gif}")

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
    vertex_constraints: Optional[Dict[int, set]] = None,
    edge_constraints: Optional[Dict[int, set]] = None,
    forbidden_edges: Optional[Set[Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
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
        if t in occupied_vertices and (x, y) in occupied_vertices[t]:
            return False
        if vertex_constraints and t in vertex_constraints and (x, y) in vertex_constraints[t]:
            return False
        return True

    def is_free_edge(x1: int, y1: int, x2: int, y2: int, t: int) -> bool:
        pair = ((x1, y1), (x2, y2))
        if t in occupied_edges and pair in occupied_edges[t]:
            return False
        opp = ((x2, y2), (x1, y1))
        if t in occupied_edges and opp in occupied_edges[t]:
            return False
        if edge_constraints and t in edge_constraints and pair in edge_constraints[t]:
            return False
        if forbidden_edges and pair in forbidden_edges:
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
    forbidden_edges: Optional[Set[Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
) -> List[Optional[List[Tuple[int, int]]]]:
    occupied_vertices: Dict[int, set] = {}
    occupied_edges: Dict[int, set] = {}
    plans: List[Optional[List[Tuple[int, int]]]] = []
    for s, g in zip(starts, goals):
        path = astar_time_aware(
            grid,
            s,
            g,
            occupied_vertices,
            occupied_edges,
            t_start=0,
            t_max=t_max,
            forbidden_edges=forbidden_edges,
        )
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
    paths: List[Optional[List[Optional[Tuple[int, int]]]]],
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
        spawn = _first_real_index(path)
        if spawn is None:
            continue
        clean = [cell for cell in path[spawn:] if cell is not None]
        if not clean:
            continue
        dx, dy = _agent_offset(k)
        xs = [x + dx for (x, y) in clean]
        ys = [y + dy for (x, y) in clean]
        ax.plot(ys, xs, linestyle="-", linewidth=2.0, alpha=0.95, label=f"agent {k}")
        ax.scatter(ys, xs, s=22, edgecolors="white", linewidths=0.8, zorder=3)
        for t, (x, y) in enumerate(clean):
            white_text((x + dx, y + dy), f"{k}:{spawn + t}", fontsize=9)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.show()


def animate_paths_gif(
    nonfl: Dict[str, object],
    paths: List[Optional[List[Optional[Tuple[int, int]]]]],
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
    spawn_times: List[Optional[int]] = []
    trimmed: List[List[Tuple[int, int]]] = []
    for p in paths:
        if p:
            T = max(T, len(p) - 1)
            spawn = _first_real_index(p)
            spawn_times.append(spawn)
            if spawn is None:
                trimmed.append([])
            else:
                trimmed.append([cell for cell in p[spawn:] if cell is not None])
        else:
            spawn_times.append(None)
            trimmed.append([])
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

            for k, path in enumerate(trimmed):
                spawn = spawn_times[k]
                if spawn is None or not path:
                    continue
                if t < spawn:
                    continue
                dx, dy = _agent_offset(k)
                rel_t = min(len(path) - 1, t - spawn)
                xs = [x + dx for (x, y) in path[: rel_t + 1]]
                ys = [y + dy for (x, y) in path[: rel_t + 1]]
                if len(xs) >= 2:
                    ax.plot(ys, xs, linestyle="-", linewidth=2.0, alpha=0.95, label=f"agent {k}" if t == 0 else None)
                ax.scatter(ys[-1:], xs[-1:], s=28, edgecolors="white", linewidths=0.9, zorder=3)
                ax.text(
                    ys[-1],
                    xs[-1],
                    f"{k}:{spawn + rel_t}",
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


def _first_real_index(path: List[Optional[Tuple[int, int]]]) -> Optional[int]:
    for idx, cell in enumerate(path):
        if cell is not None:
            return idx
    return None


def _last_real_cell(path: List[Optional[Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
    for cell in reversed(path):
        if cell is not None:
            return cell
    return None


def path_length(path: Optional[List[Optional[Tuple[int, int]]]]) -> Optional[int]:
    if not path:
        return None
    spawn_idx = _first_real_index(path)
    if spawn_idx is None:
        return None
    return max(0, len(path) - 1 - spawn_idx)


def success(path: Optional[List[Optional[Tuple[int, int]]]], goal: Tuple[int, int]) -> bool:
    if not path:
        return False
    last = _last_real_cell(path)
    return last == goal


def _cell_at_time(path: List[Optional[Tuple[int, int]]], t: int) -> Optional[Tuple[int, int]]:
    if not path:
        return None
    if t < len(path) and path[t] is not None:
        return path[t]
    for idx in range(min(t, len(path) - 1), -1, -1):
        if path[idx] is not None:
            return path[idx]
    return None


def count_vertex_conflicts(paths: List[Optional[List[Optional[Tuple[int, int]]]]]) -> int:
    tmax = 0
    for p in paths:
        if p:
            tmax = max(tmax, len(p) - 1)
    conflicts = 0
    for t in range(tmax + 1):
        seen: Dict[Tuple[int, int], List[int]] = {}
        for i, p in enumerate(paths):
            if not p:
                continue
            cell = _cell_at_time(p, t)
            if cell is None:
                continue
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
    "build_one_way_forbidden",
    "plot_grid",
    "in_bounds",
    "neighbors",
    "manhattan",
    "astar",
    "path_to_actions",
    "plot_path",
    "plot_staged_path",
    "neighbors4",
    "astar_time_aware",
    "reserve_path",
    "multi_agent_sequential",
    "plot_multiple_paths_pretty_offset",
    "plot_routes_unified",
    "plot_multiagent_staged",
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
