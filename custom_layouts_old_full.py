"""
Custom warehouse layout configurations for the enhanced MAPF experiments.

Creates a richer 10×15 grid with shelves, narrow corridors, semantic zones,
and optional one-way hints, without touching the original RDDL files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Coord = Tuple[int, int]


@dataclass(frozen=True)
class LayoutConfig:
    size: Tuple[int, int]
    ascii_rows: Sequence[str]
    legend: Dict[str, str]


ENHANCED_WAREHOUSE = LayoutConfig(
    size=(10, 15),
    ascii_rows=(
        ".II............",
        ".....>>.#......",
        "..#####.###^#..",
        "....#...#.#^...",
        ".PP.#.....#....",
        "...v...........",
        "..#v###.#####..",
        "....#...#.#....",
        ".....<<.#....C.",
        "............OO.",
    ),
    legend={
        ".": "free",
        "#": "shelf",
        "I": "inbound",
        "O": "outbound",
        "P": "packing",
        "C": "charging",
        ">": "one_way_east",
        "<": "one_way_west",
        "^": "one_way_north",
        "v": "one_way_south",
    },
)


def _validate_ascii(config: LayoutConfig) -> None:
    H, W = config.size
    if len(config.ascii_rows) != H:
        raise ValueError(f"Expected {H} rows, got {len(config.ascii_rows)}")
    for idx, row in enumerate(config.ascii_rows):
        if len(row) != W:
            raise ValueError(f"Row {idx} expected width {W}, got {len(row)}")


def _parse_layout(config: LayoutConfig) -> Dict[str, object]:
    H, W = config.size
    obstacles: Set[Coord] = set()
    zones: Dict[str, List[Coord]] = {"inbound": [], "outbound": [], "packing": [], "charging": []}
    one_way_edges: Set[Tuple[Coord, Coord]] = set()

    char_to_direction = {
        ">": (0, 1),
        "<": (0, -1),
        "^": (-1, 0),
        "v": (1, 0),
    }

    for x, row in enumerate(config.ascii_rows):
        for y, ch in enumerate(row):
            if ch == "#":
                obstacles.add((x, y))
            elif ch == "I":
                zones["inbound"].append((x, y))
            elif ch == "O":
                zones["outbound"].append((x, y))
            elif ch == "P":
                zones["packing"].append((x, y))
            elif ch == "C":
                zones["charging"].append((x, y))
            elif ch in char_to_direction:
                dx, dy = char_to_direction[ch]
                tgt = (x + dx, y + dy)
                if 0 <= tgt[0] < H and 0 <= tgt[1] < W:
                    one_way_edges.add(((x, y), tgt))
    return {
        "H": H,
        "W": W,
        "obstacles": sorted(obstacles),
        "zones": zones,
        "one_way_edges": sorted(one_way_edges),
        "legend": config.legend,
    }


def inflate_obstacles(obstacles: Iterable[Coord], H: int, W: int, radius: int) -> List[Coord]:
    if radius <= 0:
        return sorted(set(obstacles))
    inflated: Set[Coord] = set()
    for x, y in obstacles:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    inflated.add((nx, ny))
    return sorted(inflated)


def make_enhanced_nonfl(config: LayoutConfig = ENHANCED_WAREHOUSE, inflation_radius: int = 0) -> Dict[str, object]:
    """
    Build a non-fluent-like dictionary compatible with grid_planning helpers.
    """
    _validate_ascii(config)
    parsed = _parse_layout(config)
    H, W = parsed["H"], parsed["W"]
    obstacles = inflate_obstacles(parsed["obstacles"], H, W, inflation_radius)
    nonfl = {
        "H": H,
        "W": W,
        "GOAL_X": parsed["zones"]["outbound"][0][0] if parsed["zones"]["outbound"] else None,
        "GOAL_Y": parsed["zones"]["outbound"][0][1] if parsed["zones"]["outbound"] else None,
        "obstacles": obstacles,
        "zones": parsed["zones"],
        "one_way_edges": parsed["one_way_edges"],
        "legend": parsed["legend"],
    }
    return nonfl


def ascii_preview(config: LayoutConfig = ENHANCED_WAREHOUSE) -> str:
    _validate_ascii(config)
    lines = ["Legend: " + ", ".join(f"{k}={v}" for k, v in config.legend.items())]
    for i, row in enumerate(config.ascii_rows):
        lines.append(f"{i:02d} { ' '.join(row) }")
    return "\n".join(lines)


__all__ = [
    "LayoutConfig",
    "ENHANCED_WAREHOUSE",
    "make_enhanced_nonfl",
    "ascii_preview",
]
