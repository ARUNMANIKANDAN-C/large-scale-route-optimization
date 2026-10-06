# Cargo Loading and Route Planning — Operations Research

An interactive optimization application for complex logistics planning. This project uses an **iterative candidate-pattern master problem** to efficiently handle cargo loading and truck routing as a unified challenge, complete with a rich Streamlit UI for visual analysis.

## Features

- **End-to-End Optimization Pipeline**: 
  - **Stage 1 (Loading)**: Overlapping-candidate clustering via a set-partitioning master problem (CBC exact solver) or scalable heuristics.
  - **Stage 2 (Routing)**: Time-window aware route optimization (nearest-neighbour + 2-opt, or exact MIP).
- **Interactive UI**:
  - Detailed pre-check validation ensuring data feasibility (capacity, deadlines).
  - Configurable truck fleet and hazard compatibility constraints.
  - Granular breakdown of optimization results (costs, utilization, on-time percentage).
- **Network & Route Visualization**: 
  - Rich interactive Plotly network graphs.
  - Stress-minimization algorithm (MDS) to estimate 2D map layouts from distance matrices.
  - Detailed truck inspector (showing utilization bars, loaded items, stop arrival times, and routing paths).
- **Scenario Analysis**: Run sensitivity analysis on max stops, availability gaps, and fixed costs natively in the app.

---

## Architecture Overview

1. **Pre-check Validation (`precheck.py`)**
   - Validates items against fleet capacity, distance reachability, and time deadlines before optimization begins. Calculates theoretical lower bounds for the required fleet.
2. **Candidate Generation (`candidate_clusters.py`)**
   - Builds overlapping loading clusters (destination-local packing, deadline-aware alternatives, geographic adjacencies).
3. **Cluster Master (`master.py`)**
   - Solves a binary **set-partitioning** model with CBC (using `pulp`).
   - Each item is assigned exactly one selected candidate, constrained by fleet limits.
4. **Routing (`routing.py`)**
   - Evaluates tight deadline and travel-time constraints.
5. **Feedback Loop (`pipeline.py`)**
   - If a selected candidate fails Stage 2 routing, its pattern is forbidden and Stage 1 re-solves, allowing alternative clusters to win.
6. **Visualization (`viz.py`)**
   - Renders the interactive map layout, edges, directional arrows, and tooltips using Plotly.

---

## Installation & Setup

This project uses `uv` for lightning-fast dependency management and virtual environments.

1. **Install dependencies** via `uv`:
   ```bash
   uv sync
   ```
   *(Alternatively, if you don't use uv: `pip install -r requirements.txt`)*

2. **Run the Streamlit app**:
   ```bash
   uv run streamlit run app.py
   ```

3. **Open the browser**: 
   The app will automatically open at `http://localhost:8501`. 

---

## Using the Application

1. **Upload Data** (Sidebar): Upload your `order_large.csv` and `distance.csv` files, or simply leave the default "Use bundled sample files" checked.
2. **Configure Fleet**: Adjust the truck types, capacities, costs, and availability constraints.
3. **Set Constraints**: Tweak max stops per truck, unloading hours, and hazard compatibility.
4. **Pre-check Tab**: Verify that the selected subset of orders is theoretically feasible.
5. **Optimize Tab**: Choose your pipeline mode (Full, Loading only, or Routing only) and click **Run optimization**.
6. **Network & Routes Tab**: Once optimized, visually explore the resulting network map and inspect individual truck loads and routes in the interactive sidebar.
7. **Results & Compare**: Review KPI metrics or compare the exact model vs. the heuristic model on small subsets.

## Important Optimality Statement

The cluster master is **exact within the generated candidate pool**. It is not automatically globally optimal over the original continuous problem. Global optimality would require a full integrated formulation or exact column-generation scheme. The advantage here is that Stage 1 has multiple routing-aware alternatives for each item, and Stage 2 feeds failures back into the master iteratively.
