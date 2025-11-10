"""
Pick-and-place inference utilities plus visualization helpers.
"""

from __future__ import annotations

import heapq
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - fallback for headless tests
    np = None

try:
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
except ImportError:  # pragma: no cover - plotting disabled outside notebook env
    pe = plt = PillowWriter = None

from grid_planning import astar, build_grid, free_cells
from rddl_utils import read_text


# ---------- Parsing helpers ----------
def parse_pick_drop_from_nonfluent_text(nf_text: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Scan non-fluents for pick/drop predicates."""
    picks: List[Tuple[int, int]] = []
    drops: List[Tuple[int, int]] = []
    p_names = ["PICK", "SOURCE", "ITEM_AT", "PICK_LOC", "PICKUP", "PICKUP_LOC"]
    d_names = ["DROP", "TARGET", "DEST", "SINK", "PLACE", "DROP_LOC", "DELIVER_TO"]

    def scan(names: Sequence[str]) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for nm in names:
            for match in re.finditer(rf"\b{nm}\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)\s*=\s*true\s*;", nf_text, re.IGNORECASE):
                out.append((int(match.group(1)), int(match.group(2))))
        uniq: List[Tuple[int, int]] = []
        seen = set()
        for xy in out:
            if xy not in seen:
                seen.add(xy)
                uniq.append(xy)
        return uniq

    picks = scan(p_names)
    drops = scan(d_names)
    return picks, drops


def parse_start_from_instance_text(inst_text: str) -> Optional[Tuple[int, int]]:
    for kx, ky in [("agent_x", "agent_y"), ("robot_x", "robot_y")]:
        mx = re.search(rf"\b{kx}\s*=\s*([0-9]+)\s*;", inst_text)
        my = re.search(rf"\b{ky}\s*=\s*([0-9]+)\s*;", inst_text)
        if mx and my:
            return int(mx.group(1)), int(my.group(1))
    return None


def first_free_cell_from_grid(nonfl: Dict[str, object]) -> Optional[Tuple[int, int]]:
    cells = free_cells(nonfl)
    return cells[0] if cells else None


def validate_or_fallback(nonfl: Dict[str, object], xy: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if xy is None:
        return first_free_cell_from_grid(nonfl)
    H, W = nonfl["H"], nonfl["W"]
    x, y = xy
    if not (isinstance(x, int) and isinstance(y, int) and 0 <= x < H and 0 <= y < W):
        return first_free_cell_from_grid(nonfl)
    grid = build_grid(nonfl)
    if grid[x][y] == 1:
        return first_free_cell_from_grid(nonfl)
    return xy


# ---------- Planning + plotting ----------
def _require_matplotlib() -> None:
    if plt is None or pe is None:
        raise ImportError("matplotlib is required for pick&place visual helpers")


def _require_numpy() -> None:
    if np is None:
        raise ImportError("numpy is required for footprint visualizations")


def plan_pick_and_drop(
    nonfl: Dict[str, object],
    start_xy: Tuple[int, int],
    pick_xy: Tuple[int, int],
    drop_xy: Tuple[int, int],
) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]:
    grid = build_grid(nonfl)
    p1 = astar(grid, start_xy, pick_xy)
    p2 = astar(grid, pick_xy, drop_xy)
    return p1, p2


def _white_text(ax, y, x, s, **kw):
    kw.setdefault("fontsize", 10)
    kw.setdefault("color", "white")
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    txt = ax.text(y, x, s, **kw)
    txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])
    return txt


def plot_pickdrop_offset(
    nonfl: Dict[str, object],
    p1: Optional[List[Tuple[int, int]]],
    p2: Optional[List[Tuple[int, int]]],
    start_xy: Tuple[int, int],
    pick_xy: Tuple[int, int],
    drop_xy: Tuple[int, int],
    title: str = "Pick&Place S→P→D (offset)",
) -> None:
    _require_matplotlib()
    H, W = nonfl["H"], nonfl["W"]
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
    _white_text(ax, start_xy[1], start_xy[0], "S", fontsize=12, fontweight="bold")
    _white_text(ax, pick_xy[1], pick_xy[0], "P", fontsize=12, fontweight="bold")
    _white_text(ax, drop_xy[1], drop_xy[0], "D", fontsize=12, fontweight="bold")

    off_p1 = (-0.18, -0.18)
    off_p2 = (0.18, 0.18)

    if p1 and len(p1) >= 2:
        dx, dy = off_p1
        xs1 = [x + dx for (x, y) in p1]
        ys1 = [y + dy for (x, y) in p1]
        ax.plot(ys1, xs1, "-", linewidth=2.5, alpha=0.95, label="S→P")
        for t, (x, y) in enumerate(p1):
            _white_text(ax, y + dy, x + dx, f"{t}", fontsize=9)
    if p2 and len(p2) >= 2:
        dx, dy = off_p2
        xs2 = [x + dx for (x, y) in p2]
        ys2 = [y + dy for (x, y) in p2]
        ax.plot(ys2, xs2, "-", linewidth=2.5, alpha=0.95, label="P→D")
        base = (len(p1) - 1) if p1 else 0
        for i, (x, y) in enumerate(p2):
            _white_text(ax, y + dy, x + dx, f"{base + i}", fontsize=9)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.show()


def animate_pickdrop_gif_offset(
    nonfl: Dict[str, object],
    p1: Optional[List[Tuple[int, int]]],
    p2: Optional[List[Tuple[int, int]]],
    start_xy: Tuple[int, int],
    pick_xy: Tuple[int, int],
    drop_xy: Tuple[int, int],
    out_gif: str = "pickplace_single_agent_offset.gif",
    fps: int = 2,
) -> None:
    _require_matplotlib()
    if PillowWriter is None:
        raise ImportError("matplotlib PillowWriter is required for GIF export")
    grid = build_grid(nonfl)
    H, W = nonfl["H"], nonfl["W"]
    seq: List[Tuple[str, Tuple[int, int]]] = []
    if p1:
        seq += [("p1", n) for n in p1]
    if p2:
        seq += [("p2", n) for n in (p2[1:] if p1 else p2)]
    T = max(0, len(seq) - 1)
    offsets = {"p1": (-0.18, -0.18), "p2": (0.18, 0.18)}

    fig, ax = plt.subplots(figsize=(6, 6))
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, out_gif, dpi=120):
        for t in range(T + 1):
            ax.clear()
            ax.imshow(grid)
            ax.set_title(f"Pick&Place t={t}")
            ax.set_xticks(range(W))
            ax.set_yticks(range(H))
            ax.set_xlabel("y")
            ax.set_ylabel("x")
            ax.set_xlim(-0.5, W - 0.5)
            ax.set_ylim(H - 0.5, -0.5)
            for label, xy in [("S", start_xy), ("P", pick_xy), ("D", drop_xy)]:
                _white_text(ax, xy[1], xy[0], label, fontsize=12, fontweight="bold")
            xs, ys = [], []
            for phase, (x, y) in seq[: t + 1]:
                dx, dy = offsets[phase]
                xs.append(x + dx)
                ys.append(y + dy)
            if len(xs) >= 2:
                ax.plot(ys, xs, "-", linewidth=2.5, alpha=0.95)
            _white_text(ax, ys[-1], xs[-1], f"{t}", fontsize=10)
            ax.scatter([ys[-1]], [xs[-1]], s=36, edgecolors="white", linewidths=1.0, zorder=3)
            writer.grab_frame()
    print(f"Saved GIF to: {out_gif}")


# ---------- Smooth variant & footprints ----------
def quad_bezier(P0, P1, P2, n: int = 12):
    _require_numpy()
    t = np.linspace(0, 1, n, dtype=float)
    Bx = (1 - t) ** 2 * P0[0] + 2 * (1 - t) * t * P1[0] + t**2 * P2[0]
    By = (1 - t) ** 2 * P0[1] + 2 * (1 - t) * t * P1[1] + t**2 * P2[1]
    return Bx, By


def smooth_polyline_bezier(points: Sequence[Tuple[float, float]], samples_per_seg: int = 12, normal_amp: float = 0.25):
    _require_numpy()
    if points is None or len(points) < 2:
        return None
    xs_s, ys_s = [], []
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x2, y2 = points[i + 1]
        vx, vy = x2 - x0, y2 - y0
        nx, ny = float(vy), float(-vx)
        nlen = math.hypot(nx, ny) or 1.0
        nx, ny = nx / nlen, ny / nlen
        xm, ym = (x0 + x2) / 2.0, (y0 + y2) / 2.0
        P1 = (xm + normal_amp * nx, ym + normal_amp * ny)
        bx, by = quad_bezier((x0, y0), P1, (x2, y2), n=samples_per_seg)
        if i > 0:
            bx, by = bx[1:], by[1:]
        xs_s.extend(bx.tolist())
        ys_s.extend(by.tolist())
    return np.array(xs_s), np.array(ys_s)


def plot_pickdrop_offset_smooth(
    nonfl: Dict[str, object],
    p1: Optional[List[Tuple[int, int]]],
    p2: Optional[List[Tuple[int, int]]],
    start_xy: Tuple[int, int],
    pick_xy: Tuple[int, int],
    drop_xy: Tuple[int, int],
    k_step: int = 5,
    off_p1: Tuple[float, float] = (-0.18, -0.18),
    off_p2: Tuple[float, float] = (0.18, 0.18),
    ls1: str = "--",
    ls2: str = "-",
    use_bezier: bool = True,
    bezier_amp: float = 0.22,
    samples_per_seg: int = 14,
    title: str = "Pick&Place (smooth & offset)",
) -> None:
    _require_matplotlib()
    H, W = nonfl["H"], nonfl["W"]
    grid = build_grid(nonfl)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(grid)
    ax.set_title(title)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    for label, (x, y) in [("S", start_xy), ("P", pick_xy), ("D", drop_xy)]:
        _white_text(ax, y, x, label, fontsize=12, fontweight="bold")

    def draw_phase(path, offset, ls, phase_name, base_t=0):
        if not path or len(path) < 2:
            return base_t
        dx, dy = offset
        pts = [(x + dx, y + dy) for (x, y) in path]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if use_bezier:
            smooth = smooth_polyline_bezier(pts, samples_per_seg=samples_per_seg, normal_amp=bezier_amp)
            if smooth:
                sx, sy = smooth
                ax.plot(sy, sx, ls, linewidth=2.6, alpha=0.95, label=phase_name)
        else:
            ax.plot(ys, xs, ls, linewidth=2.6, alpha=0.95, label=phase_name)
        T = len(path) - 1
        for t, (x, y) in enumerate(path):
            if t in (0, T) or (t % k_step == 0):
                _white_text(ax, y + dy, x + dx, f"{base_t + t}", fontsize=9)
        return base_t + T

    base = 0
    base = draw_phase(p1, off_p1, ls1, "S→P", base_t=base)
    base = draw_phase(p2, off_p2, ls2, "P→D", base_t=base)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.show()


def animate_pickdrop_smooth_gif(
    nonfl: Dict[str, object],
    p1: Optional[List[Tuple[int, int]]],
    p2: Optional[List[Tuple[int, int]]],
    start_xy: Tuple[int, int],
    pick_xy: Tuple[int, int],
    drop_xy: Tuple[int, int],
    out_gif: str = "pickplace_smooth.gif",
    fps: int = 2,
    k_step: int = 5,
    off_p1: Tuple[float, float] = (-0.18, -0.18),
    off_p2: Tuple[float, float] = (0.18, 0.18),
    ls1: str = "--",
    ls2: str = "-",
    use_bezier: bool = True,
    bezier_amp: float = 0.22,
    samples_per_seg: int = 14,
) -> None:
    _require_matplotlib()
    if PillowWriter is None:
        raise ImportError("matplotlib PillowWriter is required for GIF export")
    _require_numpy()
    grid = build_grid(nonfl)
    H, W = nonfl["H"], nonfl["W"]

    seq: List[Tuple[str, Tuple[int, int]]] = []
    if p1:
        seq += [("p1", pt) for pt in p1]
    if p2:
        seq += [("p2", pt) for pt in p2[1:]]
    T_total = max(len(seq) - 1, 0)

    def offset_path(path, off):
        dx, dy = off
        return [(x + dx, y + dy) for (x, y) in path] if path else []

    pts1 = offset_path(p1, off_p1)
    pts2 = offset_path(p2, off_p2)
    if use_bezier:
        curve1 = smooth_polyline_bezier(pts1, samples_per_seg=samples_per_seg, normal_amp=bezier_amp)
        curve2 = smooth_polyline_bezier(pts2, samples_per_seg=samples_per_seg, normal_amp=bezier_amp)
    else:
        curve1 = (np.array([p[0] for p in pts1]), np.array([p[1] for p in pts1])) if pts1 else None
        curve2 = (np.array([p[0] for p in pts2]), np.array([p[1] for p in pts2])) if pts2 else None

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, out_gif, dpi=120):
        for t in range(T_total + 1):
            ax.clear()
            ax.imshow(grid)
            ax.set_title(f"Pick→Deliver   t={t}")
            ax.set_xticks(range(W))
            ax.set_yticks(range(H))
            ax.set_xlabel("y")
            ax.set_ylabel("x")
            ax.set_xlim(-0.5, W - 0.5)
            ax.set_ylim(H - 0.5, -0.5)
            for label, (x, y) in [("S", start_xy), ("P", pick_xy), ("D", drop_xy)]:
                _white_text(ax, y, x, label, fontsize=12, fontweight="bold")

            if curve1 is not None and p1:
                cx, cy = curve1
                T1 = len(p1) - 1
                ratio = min(1.0, t / max(1, T1))
                cut = int(ratio * len(cx))
                ax.plot(cy[:cut], cx[:cut], ls1, color="deepskyblue", linewidth=2.8, alpha=0.95)

            if curve2 is not None and p2 and p1:
                remaining = max(0, t - (len(p1) - 1))
                T2 = len(p2) - 1
                ratio2 = min(1.0, remaining / max(1, T2))
                c2x, c2y = curve2
                cut2 = int(ratio2 * len(c2x))
                ax.plot(c2y[:cut2], c2x[:cut2], ls2, color="tomato", linewidth=2.8, alpha=0.95)

            def need(idx, total):
                return idx in (0, total) or (idx % k_step == 0)

            base = 0
            if p1:
                T1 = len(p1) - 1
                for i, (x, y) in enumerate(p1):
                    if need(i, T1) and i <= t:
                        _white_text(ax, y + off_p1[1], x + off_p1[0], f"{i}", fontsize=9)
                base = T1
            if p2:
                T2 = len(p2) - 1
                for i, (x, y) in enumerate(p2):
                    glob_i = base + i
                    if need(glob_i, T1 + T2) and glob_i <= t:
                        _white_text(ax, y + off_p2[1], x + off_p2[0], f"{glob_i}", fontsize=9)
            writer.grab_frame()
    print(f"Saved GIF to: {out_gif}")


def astar_with_footprint(
    grid: List[List[int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    heuristic: str = "manhattan",
) -> Tuple[Optional[List[Tuple[int, int]]], Dict[str, object]]:
    _require_numpy()
    H, W = len(grid), len(grid[0])
    h_func = (lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]))
    g = np.full((H, W), np.inf, dtype=float)
    h = np.zeros((H, W), dtype=float)
    f = np.full((H, W), np.inf, dtype=float)
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
    closed_order: Dict[Tuple[int, int], int] = {}
    open_heap: List[Tuple[float, int, Tuple[int, int]]] = []
    open_set = set()
    open_touched = set()
    g[start] = 0.0
    h[start] = h_func(start, goal)
    f[start] = h[start]
    came_from[start] = None
    t = 0
    heapq_entry = (f[start], t, start)
    heapq.heappush(open_heap, heapq_entry)
    open_set.add(start)
    open_touched.add(start)

    while open_heap:
        _, _, u = heapq.heappop(open_heap)
        if u not in open_set:
            continue
        open_set.remove(u)
        closed_order[u] = len(closed_order)
        if u == goal:
            path = [u]
            cur = u
            while came_from[cur] is not None:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path, {"closed_order": closed_order, "g": g, "h": h, "f": f, "came_from": came_from, "open_touched": open_touched}
        ux, uy = u
        for vx, vy in [(ux - 1, uy), (ux + 1, uy), (ux, uy - 1), (ux, uy + 1)]:
            if not (0 <= vx < H and 0 <= vy < W) or grid[vx][vy] == 1:
                continue
            alt = g[ux, uy] + 1.0
            if alt < g[vx, vy]:
                g[vx, vy] = alt
                h[vx, vy] = h_func((vx, vy), goal)
                f[vx, vy] = alt + h[vx, vy]
                came_from[(vx, vy)] = u
                t += 1
                heapq.heappush(open_heap, (f[vx, vy], t, (vx, vy)))
                open_set.add((vx, vy))
                open_touched.add((vx, vy))
    return None, {"closed_order": closed_order, "g": g, "h": h, "f": f, "came_from": came_from, "open_touched": open_touched}


def plot_explored_heatmap(nonfl: Dict[str, object], footprint: Dict[str, object], title: str = "A* explored (closed order)") -> None:
    _require_numpy()
    _require_matplotlib()
    H, W = nonfl["H"], nonfl["W"]
    grid = build_grid(nonfl)
    order = np.full((H, W), np.nan, dtype=float)
    for (x, y), idx in footprint["closed_order"].items():
        order[x, y] = idx
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(grid, alpha=0.35)
    im = ax.imshow(order, cmap="magma", alpha=0.9)
    ax.set_title(title)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Closed/set pop order")
    plt.show()


def plot_heuristic_field(nonfl: Dict[str, object], footprint: Dict[str, object], field: str = "h") -> None:
    _require_matplotlib()
    H, W = nonfl["H"], nonfl["W"]
    grid = build_grid(nonfl)
    arr = footprint[field].copy()
    arr[arr == np.inf] = np.nan
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(grid, alpha=0.35)
    im = ax.imshow(arr, cmap="viridis", alpha=0.9)
    ax.set_title(f"{field.upper()} field")
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(field.upper())
    plt.show()


__all__ = [
    "parse_pick_drop_from_nonfluent_text",
    "parse_start_from_instance_text",
    "first_free_cell_from_grid",
    "validate_or_fallback",
    "plan_pick_and_drop",
    "plot_pickdrop_offset",
    "animate_pickdrop_gif_offset",
    "plot_pickdrop_offset_smooth",
    "animate_pickdrop_smooth_gif",
    "astar_with_footprint",
    "plot_explored_heatmap",
    "plot_heuristic_field",
]

# --- Additional visualization: open-set touched heatmap ---
def plot_open_touched_heatmap(nonfl: Dict[str, object], footprint: Dict[str, object], title: str = "A* open touched") -> None:
    _require_numpy()
    _require_matplotlib()
    H, W = nonfl["H"], nonfl["W"]
    grid = build_grid(nonfl)
    arr = np.full((H, W), np.nan, dtype=float)
    for (x, y) in footprint.get("open_touched", set()):
        if 0 <= x < H and 0 <= y < W:
            arr[x, y] = 1.0
    import matplotlib.pyplot as plt  # local import to satisfy headless tests
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(grid, alpha=0.35)
    im = ax.imshow(arr, cmap="YlGnBu", alpha=0.9)
    ax.set_title(title)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.show()

__all__.append("plot_open_touched_heatmap")
