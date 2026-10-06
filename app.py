"""Streamlit UI - Cargo loading and route planning (Operations Research project).

Run:  streamlit run app.py
"""
import copy
import os
import pathlib
import time

import pandas as pd
import plotly.express as px
import streamlit as st

import data
import pipeline
import stage1
import viz
from common import Params, TruckType, default_fleet, default_incompatible
from precheck import precheck


HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "sample_data")
ORDER_LARGE = os.path.join(HERE, "order_large(1).csv")
DISTANCE_FILE = os.path.join(HERE, "distance(1).csv")

st.set_page_config(page_title="Cargo Loading & Route Planning", layout="wide")
st.title("Cargo Loading and Route Planning")
st.caption("Operations Research project - overlapping candidate-cluster master (set partitioning) + Stage 2 TSP with time windows.")

# =====================================================================================
# SIDEBAR : inputs
# =====================================================================================
with st.sidebar:
    st.header("1. Data")
    up_orders = st.file_uploader("Order file(s) (order_small.csv / order_large.csv)", type="csv",
                                 accept_multiple_files=True)
    up_dist = st.file_uploader("Distance file (distance.csv)", type="csv")
    use_sample = st.checkbox("Use the bundled sample files when nothing is uploaded", value=True)
    merge_orders = st.radio("If several order files are uploaded", ["Merge them", "Use only the first"],
                            horizontal=True)

    st.header("2. Fleet")
    fleet_df = pd.DataFrame([{"Truck type": t.name, "Area (m2)": t.area, "Weight cap (kg)": t.weight_cap,
                              "Cost per km": t.cost_km, "Speed (km/h)": t.speed,
                              "No. of trucks K_t (0 = none, blank = unlimited)": t.count,
                              "Fixed cost / truck": t.fixed_cost} for t in default_fleet()])
    fleet_df = st.data_editor(fleet_df, num_rows="dynamic", width="stretch", key="fleet")

    st.header("3. Constraint parameters")
    N = st.number_input("N - max stops per truck", 1, 20, 3)
    M = st.number_input("M - unloading hours per stop", 0.0, 24.0, 1.0, 0.5)
    f_cost = st.number_input("f - fixed cost per stop", 0.0, 100000.0, 500.0, 50.0)
    delta = st.number_input("Delta - max availability gap in one truck (h)", 0.0, 500.0, 4.0, 0.5)
    ret = st.checkbox("Include return to depot", value=False,
                      help="Off = one-way routes ending at the last stop (as in the Kaggle sample output).")

    st.markdown("**Hazard compatibility** (tick = may share a truck)")
    classes = ["type_1", "type_2", "non_danger"]
    inc0 = default_incompatible()
    comp = pd.DataFrame([[a == b or frozenset((a, b)) not in inc0 for b in classes] for a in classes],
                        index=classes, columns=classes)
    comp = st.data_editor(comp, key="compat")

# ---------------- build objects from the sidebar ---------------------------------------
fleet = []
for _, r in fleet_df.dropna(subset=["Truck type"]).iterrows():
    cnt = r["No. of trucks K_t (0 = none, blank = unlimited)"]
    fleet.append(TruckType(str(r["Truck type"]), float(r["Area (m2)"]), float(r["Weight cap (kg)"]),
                           float(r["Cost per km"]), float(r["Speed (km/h)"]),
                           None if pd.isna(cnt) else int(cnt),
                           0.0 if pd.isna(r["Fixed cost / truck"]) else float(r["Fixed cost / truck"])))
incompatible = set()
for a in classes:
    for b in classes:
        if a != b and not (bool(comp.loc[a, b]) and bool(comp.loc[b, a])):
            incompatible.add(frozenset((a, b)))
params = Params(max_stops=int(N), unload_h=float(M), stop_cost=float(f_cost), delta_h=float(delta),
                return_to_depot=bool(ret), incompatible=incompatible)


# ---------------- load data ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load(order_blobs, dist_blob, merge):
    import io
    srcs = [io.BytesIO(b) for b in order_blobs]
    if not merge:
        srcs = srcs[:1]
    orders = data.read_orders(srcs)
    dist = data.read_csv_any(io.BytesIO(dist_blob))
    return orders, dist


order_blobs, dist_blob = None, None
if up_orders:
    order_blobs = tuple(f.getvalue() for f in up_orders)
elif use_sample:
    order_path = os.path.join(SAMPLE, "order_large.csv") if os.path.exists(os.path.join(SAMPLE, "order_large.csv")) else ORDER_LARGE
    order_blobs = (pathlib.Path(order_path).read_bytes(),)
if up_dist:
    dist_blob = up_dist.getvalue()
elif use_sample:
    dist_path = os.path.join(SAMPLE, "distance.csv") if os.path.exists(os.path.join(SAMPLE, "distance.csv")) else DISTANCE_FILE
    dist_blob = pathlib.Path(dist_path).read_bytes()

if not order_blobs or not dist_blob:
    st.info("Upload the order file(s) and the distance file in the sidebar to begin.")
    st.stop()
try:
    orders_raw, dist_raw = _load(order_blobs, dist_blob, merge_orders == "Merge them")
    P_all = data.Problem(orders_raw, dist_raw)
except Exception as e:  # noqa
    st.error(f"Could not read the files: {e}")
    st.stop()
for msg in data.check_coverage(P_all):
    st.warning(msg)

tab_data, tab_pre, tab_opt, tab_res, tab_viz, tab_cmp, tab_sens = st.tabs(
    ["Data and statistics", "Feasibility pre-check", "Optimize", "Results",
     "Network & Routes", "Compare methods", "Sensitivity"])

# =====================================================================================
# TAB 1 : data + statistics (on demand)
# =====================================================================================
with tab_data:
    st.subheader("Orders")
    c1, c2, c3 = st.columns(3)
    c1.metric("Items", f"{len(P_all.items):,}")
    c2.metric("Orders", f"{P_all.items['Order_ID'].nunique():,}")
    c3.metric("Destinations", P_all.items["d"].nunique())
    st.caption(f"Depot (source of every order): {P_all.depot}.  Units converted: Area cm2 -> m2, Weight (0.1 g) -> kg, "
               f"Distance m -> km.")
    st.dataframe(P_all.items[data.ORDER_COLS].head(200), width="stretch", height=220)
    if st.button("Generate statistics"):
        s = data.statistics(P_all)
        a, b, c, d_ = st.columns(4)
        a.metric("Total cargo (t)", f"{s['total_weight_t']:,.0f}")
        b.metric("Total area (m2)", f"{s['total_area_m2']:,.0f}")
        c.metric("Heaviest item (kg)", f"{s['weight_kg']['max']:,.0f}")
        d_.metric("Depot -> farthest city (km)", f"{s['by_dest']['distance_km'].max():,.0f}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Cargo by hazard class**")
            st.dataframe(s["by_hazard"])
            fig = px.pie(s["by_hazard"].reset_index(), names="h", values="items", title="Items by hazard class")
            st.plotly_chart(fig, width="stretch")
        with col2:
            st.markdown("**Delivery window length (days)**")
            wd = s["window_days"].reset_index()
            wd.columns = ["days", "items"]
            st.plotly_chart(px.bar(wd, x="days", y="items"), width="stretch")
        st.markdown("**Items, weight and distance per destination**")
        bd = s["by_dest"].reset_index().rename(columns={"d": "Destination"})
        st.dataframe(bd, width="stretch", height=300)
        st.plotly_chart(px.scatter(bd, x="distance_km", y="weight_t", size="items", hover_name="Destination",
                                   title="Destination weight vs distance from depot"), width="stretch")

# =====================================================================================
# scope selection (shared by the tabs below)
# =====================================================================================
with tab_pre:
    st.subheader("Choose the part of the data to plan")
    cc1, cc2 = st.columns([2, 1])
    dests = cc1.multiselect("Destinations (empty = all)", sorted(P_all.items["d"].unique()))
    max_items = cc2.number_input("Max items (0 = all)", 0, len(P_all.items), 0, step=10)
P_scope = P_all.subset(destinations=dests or None, max_items=max_items or None)

with tab_pre:
    st.write(f"**{len(P_scope.items):,} items** selected.")
    pre = precheck(P_scope, fleet, params)
    for m_ in pre["messages"]:
        (st.success if pre["ok"] else st.error)(m_)
    if "bound_table" in pre:
        st.markdown("**Lower bound on the number of trucks** (weight and area, per group that may share a truck; "
                    "assumes the largest allowed truck type, so the real need can be higher)")
        st.dataframe(pre["bound_table"], width="stretch")
        st.metric("Minimum trucks needed (at least)", pre["n_min"])
    if len(pre["bad_items"]):
        st.markdown("**Items that can never be delivered**")
        st.dataframe(pre["bad_items"], width="stretch")

# =====================================================================================
# TAB 3 : optimize
# =====================================================================================
with tab_opt:
    st.subheader("Run an optimization")
    mode_label = st.radio(
        "What to optimize",
        ["Full pipeline (loading, then routing)", "Loading only (Stage 1)", "Routing only (Stage 2 on a baseline loading)"],
        horizontal=False)
    mode = {"Full pipeline (loading, then routing)": "full",
            "Loading only (Stage 1)": "loading",
            "Routing only (Stage 2 on a baseline loading)": "routing"}[mode_label]
    mc1, mc2 = st.columns(2)
    s1m = mc1.selectbox("Stage 1 method", [
        "Cluster master (overlapping candidates + CBC)",
        "Heuristic (best-fit packing + merging)",
        "Exact (integer program, CBC)"])
    s2m = mc2.selectbox("Stage 2 method", ["Heuristic (nearest neighbour + 2-opt)", "Exact (MIP, CBC)"])
    s1m = "cluster" if s1m.startswith("Cluster") else ("exact" if s1m.startswith("Exact") else "heuristic")
    s2m = "exact" if s2m.startswith("Exact") else "heuristic"
    tl = st.slider("Time limit for exact Stage 1 (seconds)", 5, 300, 60)
    if s1m == "exact" and mode != "routing":
        if len(P_scope.items) > stage1.MAX_EXACT_ITEMS:
            st.error(f"The exact integer program accepts at most {stage1.MAX_EXACT_ITEMS} items (you selected "
                     f"{len(P_scope.items)}). Select a few destinations or set 'Max items'; otherwise the heuristic "
                     f"loading will be used.")
        elif len(P_scope.items) > 25:
            st.warning("The exact model is only guaranteed to finish for roughly 10-25 items; for more it returns "
                       "the best plan found within the time limit.")
    if st.button("Run optimization", type="primary"):
        if len(P_scope.items) == 0:
            st.error("No items selected.")
        else:
            with st.spinner("Optimizing..."):
                res = pipeline.run(P_scope, fleet, params, mode=mode, stage1_method=s1m, stage2_method=s2m,
                                   time_limit=tl)
            st.session_state["res"] = res
            st.session_state["res_ctx"] = (P_scope, copy.deepcopy(fleet), copy.deepcopy(params))
            st.success(f"Done in {res['seconds']:.1f} s - open the Results tab.")
            for msg in res["log"]:
                st.write("-", msg)

# =====================================================================================
# TAB 4 : results
# =====================================================================================
with tab_res:
    if "res" not in st.session_state:
        st.info("Run an optimization first.")
    else:
        res = st.session_state["res"]
        Pr, fl, pr = st.session_state["res_ctx"]
        m = res["metrics"]
        st.caption(f"Mode: {res['mode']} - Stage 1: {res['stage1_method']} ({res['info1']['status']}) - "
                   f"Stage 2: {res['stage2_method']} - {res['seconds']:.1f} s")
        k = st.columns(5)
        k[0].metric("Trucks used", m["trucks_used"])
        k[1].metric("Total cost" if "total_cost" in m else "Stage 1 cost estimate",
                    f"{m.get('total_cost', m['stage1_estimate']):,.0f}")
        k[2].metric("Distance (km)", f"{m['distance_km']:,.0f}" if "distance_km" in m else "-")
        k[3].metric("Weight utilisation", f"{m.get('avg_weight_util_pct', 0):.1f}%")
        k[4].metric("On-time", f"{m['on_time_pct']:.1f}%" if "on_time_pct" in m else "-")
        st.write("Trucks by type:", m["by_type"], "  |  stops:", m["stops_total"], "  |  items loaded:",
                 f"{m['items_loaded']:,}")
        if res["unassigned"]:
            st.error(f"{len(res['unassigned'])} item(s) could not be loaded.")
            ua = pd.DataFrame([(Pr.items.at[i, "Order_ID"], Pr.items.at[i, "Item_ID"], Pr.items.at[i, "d"], why)
                               for i, why in res["unassigned"]], columns=["Order_ID", "Item_ID", "Destination", "Reason"])
            st.dataframe(ua.groupby("Reason").size().rename("items"))
            with st.expander("List of unassigned items"):
                st.dataframe(ua, width="stretch")
        viol = pipeline.validate(Pr, res, fl, pr)
        if viol:
            st.error(f"Constraint check FAILED ({len(viol)}):")
            st.write(viol[:20])
        else:
            st.success("Constraint check passed: every item loaded once; capacity, hazard, stops, availability gap, "
                       "deadline and fleet limits all hold.")
        tt = pipeline.truck_table(Pr, res, fl)
        a, b = st.columns(2)
        with a:
            if len(tt):
                st.plotly_chart(px.histogram(tt, x="Weight_util_%", color="Truck_Type", nbins=20,
                                             title="Weight utilisation per truck"), width="stretch")
        with b:
            if len(tt):
                st.plotly_chart(px.histogram(tt, x="Stops", color="Truck_Type", title="Stops per truck"),
                                width="stretch")
        st.markdown("**Truck plan**")
        st.dataframe(tt, width="stretch", height=300)
        it = pipeline.item_table(Pr, res)
        st.markdown("**Item assignment (Kaggle output format)**")
        st.dataframe(it.head(500), width="stretch", height=250)
        d1, d2 = st.columns(2)
        d1.download_button("Download truck plan (CSV)", tt.to_csv(index=False).encode(), "truck_plan.csv", "text/csv")
        d2.download_button("Download item assignment (CSV)", it.to_csv(index=False).encode(), "item_assignment.csv",
                           "text/csv")

# =====================================================================================
# TAB 5 : network & route visualisation
# =====================================================================================
with tab_viz:

    # ============================================================
    # CARGO LOADING & ROUTE PLANNING
    # PROFESSIONAL ROUTE ANALYTICS DASHBOARD
    # ============================================================

    # ------------------------------------------------------------
    # Custom styling
    # ------------------------------------------------------------
    st.markdown("""
    <style>

    .viz-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .viz-subtitle {
        color: #8b949e;
        font-size: 0.85rem;
        margin-bottom: 1.2rem;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 650;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .status-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        padding: 12px 15px;
        background: rgba(128,128,128,0.035);
    }

    .route-pill {
        display: inline-block;
        padding: 4px 9px;
        margin: 2px;
        border-radius: 12px;
        background: rgba(50, 150, 250, 0.12);
        border: 1px solid rgba(50, 150, 250, 0.25);
        font-size: 0.75rem;
    }

    </style>
    """, unsafe_allow_html=True)

    # ============================================================
    # HEADER
    # ============================================================

    st.markdown(
        '<div class="viz-title">🚚 Network & Route Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="viz-subtitle">'
        'Interactive analysis of optimized fleet routes, vehicle utilization, '
        'delivery schedules and cargo allocation.'
        '</div>',
        unsafe_allow_html=True
    )

    # ============================================================
    # CHECK OPTIMIZATION RESULT
    # ============================================================

    if "res" not in st.session_state:

        st.info(
            "🚀 **No optimization result available.**\n\n"
            "Run the optimization from the **Optimize** tab first."
        )

        st.stop()

    try:

        res_v = st.session_state["res"]

        Pv, flv, prv = st.session_state["res_ctx"]

        trucks_v = res_v.get("trucks", [])

    except Exception as e:

        st.error(
            f"Unable to load optimization results: {e}"
        )

        st.stop()

    if not trucks_v:

        st.warning(
            "⚠️ The optimization completed, but no trucks were assigned. "
            "Try changing fleet capacity, demand, or optimization parameters."
        )

        st.stop()

    # ============================================================
    # BASIC DATA PREPARATION
    # ============================================================

    ftv = {
        t.name: t
        for t in flv
    }

    truck_ids = [
        str(t.get("id", f"Truck-{i+1}"))
        for i, t in enumerate(trucks_v)
    ]

    # ------------------------------------------------------------
    # Layout cache
    # ------------------------------------------------------------

    try:

        dest_key = tuple(
            sorted(
                str(x)
                for x in Pv.items["d"]
                .dropna()
                .unique()
            )
        )

        if st.session_state.get("_viz_key") != dest_key:

            with st.spinner("Preparing network layout..."):

                st.session_state["_viz_pos"] = (
                    viz.compute_layout(Pv)
                )

            st.session_state["_viz_key"] = dest_key

        positions = st.session_state["_viz_pos"]

    except Exception as e:

        st.error(
            f"Unable to compute network layout: {e}"
        )

        st.stop()

    # ============================================================
    # CALCULATE FLEET METRICS
    # ============================================================

    total_trucks = len(trucks_v)

    total_items = sum(
        len(t.get("items", []))
        for t in trucks_v
    )

    total_stops = sum(
        len(t.get("stops", []))
        for t in trucks_v
    )

    total_distance = sum(
        float(t.get("route_km", 0) or 0)
        for t in trucks_v
    )

    total_cost = sum(
        float(t.get("route_cost", 0) or 0)
        for t in trucks_v
    )

    valid_route_status = [
        t for t in trucks_v
        if t.get("route_ok") is not None
    ]

    successful_routes = sum(
        1
        for t in valid_route_status
        if bool(t.get("route_ok"))
    )

    route_success_pct = (
        successful_routes /
        len(valid_route_status) * 100
        if valid_route_status
        else 0
    )

    avg_distance = (
        total_distance / total_trucks
        if total_trucks
        else 0
    )

    avg_items = (
        total_items / total_trucks
        if total_trucks
        else 0
    )

    # ============================================================
    # CAPACITY UTILIZATION
    # ============================================================

    weight_utilizations = []
    area_utilizations = []

    for tr in trucks_v:

        ttype = ftv.get(
            tr.get("type")
        )

        if not ttype:
            continue

        weight_cap = getattr(
            ttype,
            "weight_cap",
            0
        )

        area_cap = getattr(
            ttype,
            "area",
            0
        )

        if weight_cap:

            weight_utilizations.append(
                min(
                    float(tr.get("w", 0) or 0) /
                    float(weight_cap),
                    1
                )
            )

        if area_cap:

            area_utilizations.append(
                min(
                    float(tr.get("a", 0) or 0) /
                    float(area_cap),
                    1
                )
            )

    avg_weight_util = (
        sum(weight_utilizations) /
        len(weight_utilizations) *
        100
        if weight_utilizations
        else 0
    )

    avg_area_util = (
        sum(area_utilizations) /
        len(area_utilizations) *
        100
        if area_utilizations
        else 0
    )

    # ============================================================
    # KPI HEADER
    # ============================================================

    st.markdown(
        '<div class="section-title">📊 Fleet Overview</div>',
        unsafe_allow_html=True
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    k1.metric(
        "🚛 Fleet",
        f"{total_trucks:,}"
    )

    k2.metric(
        "📦 Items",
        f"{total_items:,}"
    )

    k3.metric(
        "📍 Stops",
        f"{total_stops:,}"
    )

    k4.metric(
        "🛣 Distance",
        f"{total_distance:,.0f} km"
    )

    k5.metric(
        "💰 Cost",
        f"{total_cost:,.0f}"
    )

    k6.metric(
        "⏱ Route Success",
        f"{route_success_pct:.1f}%"
    )

    # ============================================================
    # SECONDARY METRICS
    # ============================================================

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Avg Distance / Truck",
        f"{avg_distance:,.1f} km"
    )

    s2.metric(
        "Avg Items / Truck",
        f"{avg_items:,.1f}"
    )

    s3.metric(
        "Avg Weight Utilization",
        f"{avg_weight_util:.1f}%"
    )

    s4.metric(
        "Avg Area Utilization",
        f"{avg_area_util:.1f}%"
    )

    st.divider()

    # ============================================================
    # MAIN DASHBOARD TABS
    # ============================================================

    overview_tab, route_tab, cargo_tab = st.tabs(
        [
            "🌐 Fleet Network",
            "🚛 Route Analysis",
            "📦 Cargo Analysis"
        ]
    )

    # ============================================================
    # TAB 1 — FLEET NETWORK
    # ============================================================

    with overview_tab:

        control_left, control_mid, control_right = st.columns(
            [1.4, 1.2, 1.2]
        )

        # --------------------------------------------------------
        # Route display mode
        # --------------------------------------------------------

        with control_left:

            overview_mode = st.radio(
                "Network view",
                [
                    "Top routes",
                    "All routes"
                ],
                horizontal=True,
                key="network_view_mode"
            )

        # --------------------------------------------------------
        # Number of routes
        # --------------------------------------------------------

        with control_mid:

            if overview_mode == "Top routes":

                max_routes = st.slider(
                    "Routes displayed",
                    min_value=10,
                    max_value=min(
                        total_trucks,
                        150
                    ),
                    value=min(
                        total_trucks,
                        40
                    ),
                    step=10,
                    key="network_max_routes"
                )

            else:

                max_routes = total_trucks

                st.caption(
                    f"Displaying all {total_trucks:,} routes"
                )

        # --------------------------------------------------------
        # Sorting
        # --------------------------------------------------------

        with control_right:

            route_sort = st.selectbox(
                "Route ranking",
                [
                    "Longest routes",
                    "Highest cost",
                    "Most items",
                    "Most stops"
                ],
                key="network_route_sort"
            )

        # --------------------------------------------------------
        # Sort trucks
        # --------------------------------------------------------

        if route_sort == "Longest routes":

            sorted_trucks = sorted(
                trucks_v,
                key=lambda x: x.get(
                    "route_km",
                    0
                ),
                reverse=True
            )

        elif route_sort == "Highest cost":

            sorted_trucks = sorted(
                trucks_v,
                key=lambda x: x.get(
                    "route_cost",
                    0
                ),
                reverse=True
            )

        elif route_sort == "Most items":

            sorted_trucks = sorted(
                trucks_v,
                key=lambda x: len(
                    x.get("items", [])
                ),
                reverse=True
            )

        else:

            sorted_trucks = sorted(
                trucks_v,
                key=lambda x: len(
                    x.get("stops", [])
                ),
                reverse=True
            )

        visible_trucks = sorted_trucks[:max_routes]

        # --------------------------------------------------------
        # Network summary
        # --------------------------------------------------------

        st.caption(
            f"Showing **{len(visible_trucks):,}** of "
            f"**{total_trucks:,}** optimized routes."
        )

        # --------------------------------------------------------
        # Network map
        # --------------------------------------------------------

        try:

            fig_net = viz.build_figure(
                Pv,
                trucks_v,
                selected_id=None,
                positions=positions,
                max_routes=max_routes
            )

            st.plotly_chart(
                fig_net,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "displayModeBar": True
                },
                key="fleet_network_plot"
            )

        except Exception as e:

            st.error(
                f"Unable to render network map: {e}"
            )

        # --------------------------------------------------------
        # Route ranking table
        # --------------------------------------------------------

        st.markdown(
            '<div class="section-title">🏆 Route Ranking</div>',
            unsafe_allow_html=True
        )

        ranking_rows = []

        for rank, tr in enumerate(
            visible_trucks,
            start=1
        ):

            ranking_rows.append(
                {
                    "Rank": rank,
                    "Truck": tr.get(
                        "id",
                        f"Truck-{rank}"
                    ),
                    "Vehicle": tr.get(
                        "type",
                        "Unknown"
                    ),
                    "Distance (km)": round(
                        tr.get(
                            "route_km",
                            0
                        ),
                        1
                    ),
                    "Cost": round(
                        tr.get(
                            "route_cost",
                            0
                        ),
                        2
                    ),
                    "Items": len(
                        tr.get(
                            "items",
                            []
                        )
                    ),
                    "Stops": len(
                        tr.get(
                            "stops",
                            []
                        )
                    ),
                    "Status": (
                        "OK"
                        if tr.get(
                            "route_ok"
                        ) is True
                        else
                        "Check"
                        if tr.get(
                            "route_ok"
                        ) is False
                        else
                        "N/A"
                    )
                }
            )

        if ranking_rows:

            ranking_df = pd.DataFrame(
                ranking_rows
            )

            st.dataframe(
                ranking_df,
                width="stretch",
                height=350,
                hide_index=True
            )

    # ============================================================
    # TAB 2 — ROUTE ANALYSIS
    # ============================================================

    with route_tab:

        # --------------------------------------------------------
        # Truck selection
        # --------------------------------------------------------

        route_select_col, route_info_col = st.columns(
            [1.2, 2.8]
        )

        with route_select_col:

            selected_truck_id = st.selectbox(
                "🚛 Select truck",
                truck_ids,
                key="route_analysis_truck"
            )

        selected_truck = next(
            (
                t
                for t in trucks_v
                if str(t.get("id")) ==
                str(selected_truck_id)
            ),
            None
        )

        if selected_truck is None:

            st.error(
                "Unable to find selected truck."
            )

        else:

            truck_type = selected_truck.get(
                "type",
                "Unknown"
            )

            truck_class = ftv.get(
                truck_type
            )

            # ----------------------------------------------------
            # Truck header
            # ----------------------------------------------------

            with route_info_col:

                st.markdown(
                    f"### 🚛 {selected_truck_id}"
                )

                st.caption(
                    f"Vehicle type: **{truck_type}**"
                )

            # ----------------------------------------------------
            # Truck metrics
            # ----------------------------------------------------

            r1, r2, r3, r4, r5 = st.columns(5)

            r1.metric(
                "Distance",
                f"{selected_truck.get('route_km', 0):,.1f} km"
            )

            r2.metric(
                "Cost",
                f"{selected_truck.get('route_cost', 0):,.0f}"
            )

            r3.metric(
                "Items",
                f"{len(selected_truck.get('items', [])):,}"
            )

            r4.metric(
                "Stops",
                f"{len(selected_truck.get('stops', [])):,}"
            )

            route_status = selected_truck.get(
                "route_ok"
            )

            r5.metric(
                "Status",
                "✅ OK"
                if route_status is True
                else
                "❌ Check"
                if route_status is False
                else
                "—"
            )

            st.divider()

            # ----------------------------------------------------
            # Selected route map
            # ----------------------------------------------------

            map_col, detail_col = st.columns(
                [2.4, 1]
            )

            with map_col:

                st.markdown(
                    "#### 🗺 Selected Route"
                )

                try:

                    selected_fig = viz.build_figure(
                        Pv,
                        trucks_v,
                        selected_id=selected_truck_id,
                        positions=positions,
                        max_routes=1
                    )

                    st.plotly_chart(
                        selected_fig,
                        use_container_width=True,
                        config={
                            "displaylogo": False,
                            "scrollZoom": True,
                            "displayModeBar": True
                        },
                        key="selected_route_plot"
                    )

                except Exception as e:

                    st.error(
                        f"Unable to render selected route: {e}"
                    )

            # ----------------------------------------------------
            # Route details
            # ----------------------------------------------------

            with detail_col:

                st.markdown(
                    "#### 📍 Route"
                )

                route_nodes = (
                    [Pv.depot] +
                    selected_truck.get(
                        "route",
                        selected_truck.get(
                            "stops",
                            []
                        )
                    )
                )

                if route_nodes:

                    for i, node in enumerate(
                        route_nodes
                    ):

                        if i == 0:

                            st.markdown(
                                f"🏠 **{node}**"
                            )

                        else:

                            previous = route_nodes[
                                i - 1
                            ]

                            distance = (
                                Pv.D
                                .get(previous, {})
                                .get(node, 0)
                            )

                            st.markdown(
                                f"↓ **{node}**  \n"
                                f"<small>{distance:.1f} km</small>",
                                unsafe_allow_html=True
                            )

                st.markdown("---")

                # ------------------------------------------------
                # Capacity
                # ------------------------------------------------

                st.markdown(
                    "#### 📊 Capacity"
                )

                if truck_class:

                    weight_cap = getattr(
                        truck_class,
                        "weight_cap",
                        0
                    )

                    area_cap = getattr(
                        truck_class,
                        "area",
                        0
                    )

                    current_weight = (
                        selected_truck.get(
                            "w",
                            0
                        )
                    )

                    current_area = (
                        selected_truck.get(
                            "a",
                            0
                        )
                    )

                    weight_ratio = (
                        min(
                            current_weight /
                            weight_cap,
                            1
                        )
                        if weight_cap
                        else 0
                    )

                    area_ratio = (
                        min(
                            current_area /
                            area_cap,
                            1
                        )
                        if area_cap
                        else 0
                    )

                    st.write(
                        f"Weight — "
                        f"{current_weight:,.0f} / "
                        f"{weight_cap:,.0f} kg"
                    )

                    st.progress(
                        weight_ratio
                    )

                    st.write(
                        f"Area — "
                        f"{current_area:.1f} / "
                        f"{area_cap:.1f} m²"
                    )

                    st.progress(
                        area_ratio
                    )

                # ------------------------------------------------
                # Hazard
                # ------------------------------------------------

                hazards = selected_truck.get(
                    "classes",
                    []
                )

                st.markdown(
                    "#### ⚠️ Hazard Classes"
                )

                if hazards:

                    st.write(
                        ", ".join(
                            sorted(
                                str(x)
                                for x in hazards
                            )
                        )
                    )

                else:

                    st.caption(
                        "No hazard classes"
                    )

            # ----------------------------------------------------
            # Schedule
            # ----------------------------------------------------

            st.markdown(
                "#### ⏱ Delivery Schedule"
            )

            arrivals = selected_truck.get(
                "arrivals",
                {}
            )

            schedule_rows = []

            for sequence, node in enumerate(
                route_nodes,
                start=1
            ):

                arrival = arrivals.get(
                    node
                )

                schedule_rows.append(
                    {
                        "Sequence": sequence,
                        "Location": node,
                        "Arrival": (
                            Pv.to_time(arrival)
                            if arrival is not None
                            else "—"
                        )
                    }
                )

            if schedule_rows:

                st.dataframe(
                    pd.DataFrame(
                        schedule_rows
                    ),
                    width="stretch",
                    hide_index=True
                )

            # ----------------------------------------------------
            # Cargo for selected truck
            # ----------------------------------------------------

            st.markdown(
                "#### 📦 Loaded Cargo"
            )

            selected_items = selected_truck.get(
                "items",
                []
            )

            if selected_items:

                try:

                    cargo_columns = [
                        "Order_ID",
                        "Item_ID",
                        "d",
                        "w",
                        "a",
                        "h",
                        "Available_Time",
                        "Deadline"
                    ]

                    available_columns = [
                        c
                        for c in cargo_columns
                        if c in Pv.items.columns
                    ]

                    route_cargo = (
                        Pv.items
                        .iloc[selected_items]
                        [
                            available_columns
                        ]
                        .copy()
                    )

                    route_cargo.rename(
                        columns={
                            "d": "Destination",
                            "w": "Weight (kg)",
                            "a": "Area (m²)",
                            "h": "Hazard"
                        },
                        inplace=True
                    )

                    st.dataframe(
                        route_cargo.reset_index(
                            drop=True
                        ),
                        width="stretch",
                        height=350,
                        hide_index=True
                    )

                except Exception as e:

                    st.error(
                        f"Unable to load cargo table: {e}"
                    )

            else:

                st.info(
                    "No cargo assigned to this truck."
                )

    # ============================================================
    # TAB 3 — CARGO ANALYSIS
    # ============================================================

    with cargo_tab:

        st.markdown(
            '<div class="section-title">📦 Cargo Distribution</div>',
            unsafe_allow_html=True
        )

        # --------------------------------------------------------
        # Build cargo dataframe
        # --------------------------------------------------------

        all_items_data = []

        for tr in trucks_v:

            truck_id = tr.get(
                "id",
                "Unknown"
            )

            for item_index in tr.get(
                "items",
                []
            ):

                try:

                    row = Pv.items.iloc[
                        item_index
                    ].copy()

                    row["Truck_ID"] = truck_id

                    all_items_data.append(
                        row
                    )

                except Exception:

                    continue

        if not all_items_data:

            st.info(
                "No cargo data available."
            )

        else:

            cargo_df = pd.DataFrame(
                all_items_data
            )

            # ----------------------------------------------------
            # Cargo KPIs
            # ----------------------------------------------------

            total_weight = (
                pd.to_numeric(
                    cargo_df.get(
                        "w",
                        pd.Series(dtype=float)
                    ),
                    errors="coerce"
                )
                .fillna(0)
                .sum()
            )

            total_area = (
                pd.to_numeric(
                    cargo_df.get(
                        "a",
                        pd.Series(dtype=float)
                    ),
                    errors="coerce"
                )
                .fillna(0)
                .sum()
            )

            unique_orders = (
                cargo_df["Order_ID"]
                .nunique()
                if "Order_ID"
                in cargo_df.columns
                else 0
            )

            unique_destinations = (
                cargo_df["d"]
                .nunique()
                if "d"
                in cargo_df.columns
                else 0
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "📦 Items",
                f"{len(cargo_df):,}"
            )

            c2.metric(
                "⚖️ Total Weight",
                f"{total_weight:,.1f} kg"
            )

            c3.metric(
                "📐 Total Area",
                f"{total_area:,.2f} m²"
            )

            c4.metric(
                "📍 Destinations",
                f"{unique_destinations:,}"
            )

            st.divider()

            # ----------------------------------------------------
            # Cargo filters
            # ----------------------------------------------------

            filter1, filter2 = st.columns(2)

            with filter1:

                if "d" in cargo_df.columns:

                    destinations = sorted(
                        cargo_df["d"]
                        .dropna()
                        .astype(str)
                        .unique()
                    )

                    selected_destination = st.selectbox(
                        "Filter by destination",
                        ["All"] + destinations,
                        key="cargo_destination_filter"
                    )

                else:

                    selected_destination = "All"

            with filter2:

                if "Truck_ID" in cargo_df.columns:

                    cargo_truck = st.selectbox(
                        "Filter by truck",
                        ["All"] +
                        [
                            str(x)
                            for x in sorted(
                                cargo_df[
                                    "Truck_ID"
                                ]
                                .dropna()
                                .unique()
                            )
                        ],
                        key="cargo_truck_filter"
                    )

                else:

                    cargo_truck = "All"

            filtered_cargo = cargo_df.copy()

            if (
                selected_destination != "All"
                and "d" in filtered_cargo.columns
            ):

                filtered_cargo = (
                    filtered_cargo[
                        filtered_cargo["d"]
                        .astype(str)
                        ==
                        str(selected_destination)
                    ]
                )

            if (
                cargo_truck != "All"
                and "Truck_ID"
                in filtered_cargo.columns
            ):

                filtered_cargo = (
                    filtered_cargo[
                        filtered_cargo[
                            "Truck_ID"
                        ].astype(str)
                        ==
                        str(cargo_truck)
                    ]
                )

            # ----------------------------------------------------
            # Display cargo
            # ----------------------------------------------------

            display_columns = [
                "Truck_ID",
                "Order_ID",
                "Item_ID",
                "d",
                "w",
                "a",
                "h",
                "Available_Time",
                "Deadline"
            ]

            display_columns = [
                c
                for c in display_columns
                if c in filtered_cargo.columns
            ]

            display_cargo = (
                filtered_cargo[
                    display_columns
                ]
                .copy()
            )

            display_cargo.rename(
                columns={
                    "d": "Destination",
                    "w": "Weight (kg)",
                    "a": "Area (m²)",
                    "h": "Hazard"
                },
                inplace=True
            )

            st.caption(
                f"Showing **{len(display_cargo):,}** "
                f"cargo items."
            )

            st.dataframe(
                display_cargo.reset_index(
                    drop=True
                ),
                width="stretch",
                height=500,
                hide_index=True
            )

            # ----------------------------------------------------
            # Download
            # ----------------------------------------------------

            csv_data = display_cargo.to_csv(
                index=False
            )

            st.download_button(
                "⬇️ Download Cargo CSV",
                data=csv_data,
                file_name="optimized_cargo.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ============================================================
    # FOOTER
    # ============================================================

    st.divider()

    st.caption(
        f"Optimization result: {total_trucks:,} trucks · "
        f"{total_items:,} items · "
        f"{total_stops:,} stops · "
        f"{total_distance:,.0f} km total route distance"
    )
# =====================================================================================
# TAB 6 : compare heuristic vs exact on the same subset
# =====================================================================================
with tab_cmp:
    st.subheader("Heuristic vs exact on the same subset")
    st.caption("The exact model can only be solved for small subsets, so use a few destinations or 'Max items' "
               "(about 10-25 items).")
    tl2 = st.slider("Time limit for the exact model (s)", 5, 300, 60, key="tl2")
    if st.button("Run comparison"):
        if len(P_scope.items) > stage1.MAX_EXACT_ITEMS:
            st.warning(f"More than {stage1.MAX_EXACT_ITEMS} items: the exact model is skipped and the heuristic "
                       f"plan is shown for both rows. Select fewer destinations / 'Max items'.")
        rows = []
        with st.spinner("Solving both..."):
            for name, a1, a2 in [
                ("Heuristic", "heuristic", "heuristic"),
                ("Cluster master", "cluster", "exact"),
                ("Exact (MIP)", "exact", "exact")]:
                r = pipeline.run(P_scope, fleet, params, "full", a1, a2, time_limit=tl2)
                mm = r["metrics"]
                rows.append({"Method": name, "Status": r["info1"]["status"], "Trucks": mm["trucks_used"],
                             "Stage 1 cost": round(mm["stage1_estimate"]),
                             "Total cost": round(mm.get("total_cost", float("nan"))),
                             "Distance km": round(mm.get("distance_km", float("nan"))),
                             "Unassigned": mm["items_unassigned"], "Seconds": round(r["seconds"], 1),
                             "Violations": len(pipeline.validate(P_scope, r, fleet, params))})
        cmp_df = pd.DataFrame(rows)
        st.dataframe(cmp_df, width="stretch")
        st.plotly_chart(px.bar(cmp_df, x="Method", y="Total cost", text="Total cost"), width="stretch")
        st.caption("Cluster master is exact only over the generated candidate pool. Routing feedback removes failed patterns and re-solves the master; this is stronger than independent Stage 1/Stage 2 optimization but is still not a proof of global optimality over all possible item groupings.")

# =====================================================================================
# TAB 6 : sensitivity (scenario analysis with the heuristic)
# =====================================================================================
with tab_sens:
    st.subheader("Scenario (sensitivity) analysis")
    st.caption("Re-solve with the heuristic while one parameter changes. Uses the destinations / item limit chosen "
               "in the pre-check tab.")
    param = st.selectbox("Parameter to vary", ["N (max stops)", "Delta (availability gap, h)", "f (cost per stop)",
                                               "Trucks of the largest type K_t"])
    vals = st.text_input("Values (comma separated)", "1,2,3,4,5" if param.startswith("N") else
                         "1,2,4,8,24" if param.startswith("Delta") else
                         "0,250,500,1000,2000" if param.startswith("f") else "100,200,300,400,600")
    if st.button("Run scenarios"):
        out = []
        for v in [float(x) for x in vals.split(",") if x.strip()]:
            p2, f2 = copy.deepcopy(params), copy.deepcopy(fleet)
            if param.startswith("N"):
                p2.max_stops = int(v)
            elif param.startswith("Delta"):
                p2.delta_h = v
            elif param.startswith("f"):
                p2.stop_cost = v
            else:
                big = max(f2, key=lambda t: t.weight_cap)
                big.count = int(v)
            r = pipeline.run(P_scope, f2, p2, "full", "heuristic", "heuristic")
            mm = r["metrics"]
            out.append({param: v, "Trucks": mm["trucks_used"], "Total cost": round(mm.get("total_cost", 0)),
                        "Distance km": round(mm.get("distance_km", 0)), "Unassigned items": mm["items_unassigned"],
                        "Weight util %": round(mm.get("avg_weight_util_pct", 0), 1)})
        sdf = pd.DataFrame(out)
        st.dataframe(sdf, width="stretch")
        st.plotly_chart(px.line(sdf, x=param, y="Total cost", markers=True), width="stretch")
        st.plotly_chart(px.line(sdf, x=param, y="Trucks", markers=True), width="stretch")
