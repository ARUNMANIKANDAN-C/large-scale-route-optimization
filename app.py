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
    st.subheader("Network & Route Visualization")
    if "res" not in st.session_state:
        st.info("Run an optimization first (in the **Optimize** tab) to see route visualizations.")
    else:
        res_v = st.session_state["res"]
        Pv, flv, prv = st.session_state["res_ctx"]
        trucks_v = res_v["trucks"]
        ftv = {t.name: t for t in flv}

        if not trucks_v:
            st.warning("The optimization produced no trucks. Try different parameters.")
        else:
            # compute layout once per destination-set
            dest_key = tuple(sorted(Pv.items["d"].unique()))
            if st.session_state.get("_viz_key") != dest_key:
                with st.spinner("Computing network layout..."):
                    st.session_state["_viz_pos"] = viz.compute_layout(Pv)
                st.session_state["_viz_key"] = dest_key
            positions = st.session_state["_viz_pos"]

            left_col, right_col = st.columns([1, 3])

            with left_col:
                st.markdown("### 🚛 Truck Inspector")
                truck_ids = [tr["id"] for tr in trucks_v]
                sel_truck = st.selectbox("Select a truck", ["All trucks"] + truck_ids,
                                         key="viz_truck_select")

                if sel_truck != "All trucks":
                    tr = next(t for t in trucks_v if t["id"] == sel_truck)
                    ttype = ftv[tr["type"]]

                    # ── header info ──
                    st.markdown(f"**Type:** `{tr['type']}`")
                    route_nodes = [Pv.depot] + tr.get("route", tr["stops"])
                    # route with distances
                    parts = [f"**{route_nodes[0]}**"]
                    for ri in range(1, len(route_nodes)):
                        seg_km = Pv.D.get(route_nodes[ri - 1], {}).get(route_nodes[ri], 0)
                        parts.append(f" —({seg_km:.0f} km)→ **{route_nodes[ri]}**")
                    st.markdown("".join(parts))

                    # ── metrics row ──
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Distance", f"{tr.get('route_km', 0):.0f} km")
                    mc2.metric("Cost", f"{tr.get('route_cost', 0):,.0f}")
                    mc3, mc4 = st.columns(2)
                    mc3.metric("Items", len(tr["items"]))
                    mc4.metric("Stops", len(tr["stops"]))

                    # ── utilisation bars ──
                    w_pct = min(tr["w"] / ttype.weight_cap, 1.0) if ttype.weight_cap else 0
                    a_pct = min(tr["a"] / ttype.area, 1.0) if ttype.area else 0
                    st.markdown(f"**Weight:** {tr['w']:.0f} / {ttype.weight_cap:.0f} kg "
                                f"({w_pct * 100:.0f}%)")
                    st.progress(w_pct)
                    st.markdown(f"**Area:** {tr['a']:.1f} / {ttype.area:.1f} m² "
                                f"({a_pct * 100:.0f}%)")
                    st.progress(a_pct)

                    st.markdown(f"**Hazard:** {', '.join(sorted(tr['classes']))}")
                    on_time = tr.get("route_ok", None)
                    if on_time is not None:
                        st.markdown(f"**On-time:** {'✅ Yes' if on_time else '❌ No'}")
                    st.markdown(f"**Depart:** {Pv.to_time(tr['S'])}")

                    # ── stop arrivals ──
                    if "arrivals" in tr and tr["arrivals"]:
                        st.markdown("**Arrivals:**")
                        for stop in tr.get("route", tr["stops"]):
                            arr_h = tr["arrivals"].get(stop)
                            if arr_h is not None:
                                st.write(f"📍 {stop}: {Pv.to_time(arr_h)}")

                    # ── items table ──
                    st.markdown("---")
                    st.markdown(f"**📦 Items ({len(tr['items'])})**")
                    items_df = Pv.items.iloc[tr["items"]][
                        ["Order_ID", "Item_ID", "d", "w", "a", "h",
                         "Available_Time", "Deadline"]
                    ].rename(columns={"d": "Dest", "w": "Wt (kg)",
                                      "a": "Area (m²)", "h": "Hazard"})
                    st.dataframe(items_df.reset_index(drop=True),
                                 width="stretch", height=300)
                else:
                    # ── summary view ──
                    max_vis = st.slider("Max routes to show", 5, min(len(trucks_v), 100),
                                        min(len(trucks_v), 40), key="max_routes")
                    st.caption(f"Showing {min(len(trucks_v), max_vis)} of "
                               f"{len(trucks_v)} truck routes.")
                    st.markdown("Select a truck above to inspect its cargo in detail.")
                    st.markdown("---")
                    st.markdown("**Fleet summary**")
                    type_counts = {}
                    for tr in trucks_v:
                        type_counts[tr["type"]] = type_counts.get(tr["type"], 0) + 1
                    for tname, cnt in sorted(type_counts.items()):
                        st.write(f"`{tname}` — {cnt} truck(s)")
                    st.metric("Total distance",
                              f"{sum(t.get('route_km', 0) for t in trucks_v):,.0f} km")
                    st.metric("Total items loaded",
                              f"{sum(len(t['items']) for t in trucks_v):,}")
                    
                    # ── global items list ──
                    st.markdown("---")
                    st.markdown("**📦 All Loaded Items**")
                    all_items_data = []
                    for tr in trucks_v:
                        for i in tr["items"]:
                            row = Pv.items.iloc[i].copy()
                            row["Truck_ID"] = tr["id"]
                            all_items_data.append(row)
                    if all_items_data:
                        df_all = pd.DataFrame(all_items_data)[
                            ["Truck_ID", "Order_ID", "Item_ID", "d", "w", "a"]
                        ].rename(columns={"d": "Dest", "w": "Wt", "a": "Area"})
                        st.dataframe(df_all, width="stretch", height=300)

            with right_col:
                sel_id = None if sel_truck == "All trucks" else sel_truck
                max_rt = st.session_state.get("max_routes", 40) if not sel_id else 1
                fig_net = viz.build_figure(Pv, trucks_v, selected_id=sel_id,
                                           positions=positions, max_routes=max_rt)
                st.plotly_chart(fig_net, use_container_width=True)

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
