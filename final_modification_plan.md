# Warehouse MAPF — Final Modification Plan

This document summarizes the final agreed **4‑step modification plan** for Member‑2 deliverables in the Warehouse Multi‑Agent Planning project.

The goal is to fully demonstrate:
1) A more realistic warehouse navigation environment
2) Three progressively stronger MAPF planning baselines:
   - Baseline: Single‑agent sequential task planning
   - Upgrade: Multi‑agent prioritized A*
   - Final: Multi‑agent CBS

No detailed code or CLI specifications are included here. Instead, this document focuses on conceptual structure, rationale, and implementation intent (for both humans and AI tools).

---

## ✅ Step 1 — Environment Upgrade (10×15 → Enhanced Warehouse)

### ✔ Purpose
The original 10×15 grid has too few obstacles, causing A* to behave trivially (straight‑line paths, low interaction).
We introduce moderate complexity to ensure:
- Actual route deviation
- Bottlenecks → natural conflict points
- Slight domain richness for trajectory visualization

### ✔ Modifications (No need to modify RDDL files)
All modifications are done at the **notebook/grid level** for minimal effort and maximal control.

Recommended enhancements:
- **Shelves (row/column obstacles)**
  e.g. row 3 and 8, or column 5 and 10, forming cross‑shaped racks.
- **Narrow corridors (1‑cell width)**
  Force route competition → conflict opportunities.
- **Optional directional edges (one‑way aisles)**
  Certain edges disallow reverse traversal; encourages detours.
- **Optional geometry constraints: inflated agent radius**
  Narrow aisles become “exclusive access”, encouraging staggered use.

### ✔ Why
- Creates meaningful A* search & footprint heatmaps
- Induces natural agent conflicts → highlights MAPF value
- Maintain simplicity to ensure reproducibility

---

## ✅ Step 2 — Baseline: Single Agent, Sequential Pick‑→‑Deliver Tasks

### ✔ Purpose
Provide a clear reference baseline for path quality, footprint visualization, and sequential task execution.
This is the foundation against which future multi‑agent methods are compared.

### ✔ Structure
1) Read upgraded grid
2) Define a list of independent tasks: (Start → Pick → Drop)
3) For each sub‑task:
   - Run A* independently
   - Record path & search footprint
4) Stitch these segments together into a sequential route

### ✔ Visualization
- Grid with racks/chokepoints
- Path overlays:
  - Start→Pick: dashed (`--`)
  - Pick→Drop: solid (`-`)
  - Optional Bezier smoothing + small offsets
- Sparse timestamps (every k steps) to reduce clutter
- Heatmaps:
  - Closed‑set exploration
  - Optional heuristic field (h or f)

### ✔ Why
- Establishes how the agent performs with no interaction
- Provides baseline cost (SoC, makespan)
- Creates consistent visual language for later comparisons

---

## ✅ Step 3 — Upgrade: Multi‑Agent Planning

### ✔ Purpose
Demonstrate emerging inter‑agent conflicts and the benefit of conflict‑aware planning.

Two variants will be shown:

---

### 3A) Multi‑Agent — Naïve (Ignore Each Other)

**Description**
Each agent solves its task using A* as if other agents do not exist.

**Expected outcome**
- Clean individual trajectories
- **Collisions detected afterward** (vertex / edge conflicts)
- Heuristic fields/footprints show overlapping exploration zones

**Why**
- Highlights that naïve independent planning fails under resource contention

---

### 3B) Multi‑Agent — Prioritized Planning (Sequential Reservation)

**Description**
Agents are ordered (e.g. A→B→C).
Each agent plans using time‑expanded A*, but respects earlier agents’ reserved occupancy (vertex & edge).

**Expected outcome**
- No runtime conflicts
- Paths may be less optimal (detours, delays)

**Why**
- Very strong practical baseline
- Low‑level planner already implemented → minimal new code
- Establishes performance gap before introducing CBS

---

## ✅ Step 4 — Final: Conflict‑Based Search (CBS)

### ✔ Purpose
Demonstrate optimal or near‑optimal MAPF by resolving conflicts using a top‑down constraint tree.

### ✔ Structure (Concept only — no code)
1) Root node: no constraints → plan individual paths
2) If a conflict exists:
   - Extract earliest conflict (time, agents, type)
   - Create two children nodes:
     - Child A: Add constraint on agent‑i
     - Child B: Add constraint on agent‑j
   - Only replan affected agent in each branch
3) Each node tracks:
   - Constraint sets
   - Paths
   - Cost (SoC/makespan)
4) Best‑first search on constraint tree until:
   - No conflict → solution
   - Queue empty → infeasible

### ✔ Why
- Shows fully general MAPF reasoning hierarchy
- Demonstrates how constraints guide re‑planning
- Clear progression from Prioritized Planning

### ✔ Visualization
- Final path overlays
- Optional: small CT visualization (text tree)
- Compare CBS vs Prioritized:
  - Cost SoC
  - Makespan
  - #Constraints added
  - #Low‑level replans

---

# Deliverables Summary

| Phase | Method | Agents | Conflict Handling | Expected Benefit |
|---:|---|---:|---:|---|
| 1 | Single‑agent A* | 1 | N/A | Baseline |
| 2A | Naive Multi | 2–3 | ❌ | Show failures |
| 2B | Prioritized | 2–3 | ✅ via reservation | Simple + robust |
| 3 | CBS | 2–3 | ✅ optimal-level | MAPF complete |

---

# Final Notes

- The three‑stage pipeline (single → prioritized → CBS) provides a compelling narrative for MAPF algorithm evolution.
- Environment upgrades introduce just enough structural difficulty without increasing implementation overhead.
- All enhancements are additive; previous steps remain reusable.
