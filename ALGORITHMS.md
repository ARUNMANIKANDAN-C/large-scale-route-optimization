# Optimization Algorithms

This document explains the mathematical models and heuristics used to solve the cargo loading and route planning problem in this application.

## Overview: The Two-Stage Approach with Feedback

The problem is a variant of the **Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)**, combined with 3D-ish bin packing (weight and area) and hazard compatibility constraints. Because solving this monolithically is computationally intractable for thousands of items, we use a two-stage approach with a feedback loop:

1. **Stage 1 (Loading / Packing)**: Group items into valid truckloads.
2. **Stage 2 (Routing)**: Sequence the stops for each loaded truck to minimize distance and meet time windows.
3. **Feedback**: If Stage 2 fails to route a truck, that specific grouping of items is forbidden, and Stage 1 is re-solved.

---

## Stage 1: Cargo Loading

We provide three different methods for Stage 1, allowing a tradeoff between speed and optimality.

### 1. Cluster Master (Overlapping Candidates + Set Partitioning)
*This is the default and most powerful method.*

Instead of blindly packing items, this method generates hundreds of thousands of *candidate* truckloads, and then uses an exact Mixed Integer Program (MIP) to pick the best non-overlapping subset.

- **Candidate Generation (`candidate_clusters.py`)**:
  - **Local packing**: Groups items going to the same destination using a Best-Fit Decreasing algorithm.
  - **Deadline variants**: Sorts items by deadline before packing, creating alternative groups for time-sensitive cargo.
  - **Geographic adjacencies**: Merges part-loaded candidates from nearby cities.
  - *Result*: A massive pool of valid, capacity-respecting "candidate clusters".

- **Set-Partitioning Master Problem (`master.py`)**:
  - We define a binary variable $y_{p,t} \in \{0,1\}$ for each candidate $p$ assigned to a truck type $t$.
  - **Objective**: Minimize the total estimated route cost + fixed truck costs.
  - **Constraints**:
    - *Coverage*: Every item must be in exactly one selected candidate.
    - *Fleet Limits*: The number of selected candidates of type $t$ cannot exceed the available trucks $K_t$.
  - Solved exactly using the **CBC** solver via `pulp`.

### 2. Heuristic (Best-Fit Packing + Merging)
*Extremely fast, used for massive datasets (thousands of items) where the MIP would time out.*

- **Phase A**: Sort destinations by distance from the depot. Pack items for each destination using Best-Fit Decreasing (handling weight, area, and hazard compatibility).
- **Phase B**: Iteratively merge part-loaded trucks if their destinations are geographically close, they share compatible hazard classes, and the merged route remains deadline-feasible.
- **Phase C**: Greedily assign the cheapest capable truck type from the available fleet to each load.

### 3. Exact MIP (Integer Program)
*Solves the full Stage 1 monolithically. Only feasible for very small instances (<25 items).*

- A massive integer program with binary variables $x_{i,v} \in \{0,1\}$ (item $i$ on truck $v$) and $z_v \in \{0,1\}$ (truck $v$ used).
- Enforces weight, area, hazard, and time-window constraints directly in the solver. 

---

## Stage 2: Route Planning

Once a truck is assigned a set of items, it must visit a set of destinations $C$. Stage 2 sequences these stops.

### 1. Heuristic Routing
*Used for trucks with many stops.*

- **Initialization**: 
  - Generates a **Nearest Neighbour** route (always visit the closest unvisited city).
  - Generates an **Earliest Deadline First (EDF)** route.
  - Picks the shorter of the two that is deadline-feasible.
- **Local Search (2-opt)**: 
  - Iteratively swaps pairs of edges to uncross paths and shorten the route, ensuring no time windows are violated during the swap.
- **Repair**: 
  - If no feasible route is found, a local search attempts to minimize *tardiness* (lateness) by shifting stops.

### 2. Exact MIP Routing
*Used for trucks with a small number of stops (default $\le 9$).*

- Uses an MTZ-style (Miller-Tucker-Zemlin) formulation for the Traveling Salesperson Problem with Time Windows (TSPTW).
- Binary variables $y_{a,b}$ indicate travel from stop $a$ to stop $b$.
- Continuous variables $T_c$ track the arrival time at stop $c$.
- **Objective**: Minimize total travel distance.
- **Constraints**:
  - Enter and leave each stop exactly once.
  - Time propagation: $T_b \ge T_a + \text{service\_time} + \text{travel\_time}(a,b) - \text{BigM}(1 - y_{a,b})$.
  - Time windows: $T_c \le \text{Deadline}_c$.

---

## The Feedback Loop

A major innovation in this pipeline is the **integrated feedback loop** (`pipeline.py`). 

Because Stage 1 uses estimated distances to group items, it might propose a cluster that is actually impossible to route in Stage 2 (due to tight deadlines). 
- When Stage 2 reports a failure, the pipeline extracts the exact items in that failed truck.
- That specific combination of items is added as a **cut** (forbidden signature).
- Stage 1 is re-solved from scratch. The Cluster Master will simply pick a different combination of overlapping candidates to cover those items.
- This iterates until all trucks are successfully routed.
