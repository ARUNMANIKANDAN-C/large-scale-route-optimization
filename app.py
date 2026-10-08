"""Streamlit UI - Cargo loading and route planning (Operations Research project).

Run:  streamlit run app.py
"""
import copy
import os
import pathlib
import time
import math
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import data
import pipeline
import stage1
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

# =====================================================================================
# scope summary, planning window and feasibility checks
# =====================================================================================
with tab_pre:
    st.write(f"**{len(P_scope.items):,} items** selected.")
    pre = precheck(P_scope, fleet, params)
    w = pre.get("window") or {}

    # ---------------------------------------------------------------- planning window
    if w:
        st.markdown("#### Planning window")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Planning starts", w["start"],
                  help="The moment the first selected item becomes available at the depot. Nothing can leave earlier.")
        c2.metric("Everything delivered by", w["end"],
                  help="The latest deadline among the selected items = end of the planning horizon.")
        c3.metric("Horizon", f"{w['horizon_days']:.1f} days")
        c4.metric("Depot (all orders start here)", w["depot"])

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Last item available", w["last_available"])
        d2.metric("Earliest deadline", w["earliest_deadline"])
        lo, med, hi = w["window_days"]
        d3.metric("Delivery window / item", f"{med:.1f} d (median)", help=f"Shortest {lo:.1f} d, longest {hi:.1f} d.")
        d4.metric("Items with < 24 h slack", f"{w['tight_items']:,}",
                  help="Slack = deadline - availability - fastest possible travel - unloading. "
                       "Items with little slack force early, dedicated trips.")

        with st.expander("What do these terms mean?"):
            st.markdown(
                "- **Available time** – when an item is ready to leave the depot.\n"
                "- **Deadline** – the latest time the item may arrive at its destination. A truck that arrives "
                "after it breaks the plan (the pre-check is stricter and also adds the unloading time).\n"
                "- **Truck departure** – a truck leaves when the *last* item loaded on it becomes available, "
                "so mixing early and late items makes the early ones wait.\n"
                "- **Delta (availability gap)** – the maximum difference in available times between items "
                "sharing one truck.\n"
                "- **Slack** – how much extra waiting or detour an item can absorb before its deadline is missed."
            )

        # ------------------------------------------------------------ other important factors
        st.markdown("#### Factors that shape the plan")
        n_trucks = sum(t.count for t in fleet if t.count is not None)
        unlimited = any(t.count is None for t in fleet)
        factors = pd.DataFrame([
            ("Max stops per truck (N)", f"{params.max_stops}", "More stops = fewer trucks but longer routes and later arrivals"),
            ("Unloading time per stop (M)", f"{params.unload_h:g} h", "Added at every stop; delays all following stops"),
            ("Cost per stop (f)", f"{params.stop_cost:,.0f}", "Penalises many small deliveries"),
            ("Availability gap (Delta)", f"{params.delta_h:g} h", "Limits how different the ready times in one truck may be"),
            ("Return to depot", "Yes" if params.return_to_depot else "No", "Adds the way back to distance and cost"),
            ("Fleet size", "unlimited" if unlimited else f"{n_trucks:,} trucks", "Limited fleets can leave items unassigned"),
            ("Hazard classes", ", ".join(f"{k} ({v:,})" for k, v in w["hazard_counts"].items()),
             "Incompatible classes can never share a truck"),
            ("Destinations", f"{w['n_destinations']:,}", f"Farthest is {w['farthest_km']:,.0f} km from the depot"),
            ("Heaviest item", f"{w['heaviest_kg']:,.0f} kg", "Must fit the largest allowed truck"),
            ("Tightest slack", f"{w['slack_min_h']:.1f} h", f"Median slack {w['slack_median_h']:.1f} h"),
        ], columns=["Factor", "Value", "Why it matters"])
        st.dataframe(factors, hide_index=True, width="stretch")

    # ---------------------------------------------------------------- feasibility checks
    st.markdown("#### Feasibility checks")
    for m_ in pre["messages"]:
        (st.success if pre["ok"] else st.error)(m_)
    if "bound_table" in pre and len(pre["bound_table"]):
        st.markdown("**Lower bound on the number of trucks** (weight and area, per group that may share a truck; "
                    "assumes the largest allowed truck type, so the real need can be higher)")
        st.dataframe(pre["bound_table"], width="stretch", hide_index=True)
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
        "Heuristic (best-fit packing + merging)",
        "Exact (integer program, CBC)"])
    s2m = mc2.selectbox("Stage 2 method", ["Heuristic (nearest neighbour + 2-opt)", "Exact (MIP, CBC)"])
    s1m = "exact" if s1m.startswith("Exact") else "heuristic"
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

                
PALETTE = ["#4C9BE8", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
           "#76B7B2", "#EDC948", "#FF9DA7", "#9C755F", "#86BCB6"]
RENAME = {"d": "Destination", "w": "Weight (kg)", "a": "Area (m²)", "h": "Hazard"}
CARGO_COLS = ["Truck_ID", "Order_ID", "Item_ID", "d", "w", "a", "h", "Available_Time", "Deadline"]


# ----------------------------------------------------------------- helpers
def path_of(P, t):
    stops = [n for n in (t.get("route") or t.get("stops") or []) if n != P.depot]
    return [P.depot] + stops + [P.depot]


def radial_layout(P, trucks):
    """Depot at centre; each destination sits on a ring whose radius = distance
    from depot. Nodes of the same truck share an angular sector."""
    order, seen = [], set()
    for t in trucks:
        for n in path_of(P, t)[1:-1]:
            if n not in seen:
                seen.add(n)
                order.append(n)
    dist = {n: float(P.D.get(P.depot, {}).get(n, 0) or 0) for n in order}
    rmax = max(dist.values(), default=1) or 1
    pos = {P.depot: (0.0, 0.0)}
    for i, n in enumerate(order):
        ang = 2 * math.pi * i / max(len(order), 1)
        r = 0.15 + 0.85 * dist[n] / rmax
        pos[n] = (r * math.cos(ang), r * math.sin(ang))
    return pos, dist, rmax


def network_fig(P, trucks, pos, dist, rmax, selected=None):
    fig = go.Figure()

    # distance rings (circles)
    for f in (0.25, 0.5, 0.75, 1.0):
        r = 0.15 + 0.85 * f
        fig.add_shape(type="circle", x0=-r, y0=-r, x1=r, y1=r,
                      line=dict(color="rgba(128,128,128,.35)", dash="dot", width=1))
        fig.add_annotation(x=0, y=r, text=f"{f * rmax:,.0f} km", showarrow=False,
                           font=dict(size=9, color="gray"), yshift=8)

    # routes
    for i, t in enumerate(trucks):
        tid = str(t.get("id", i))
        on = selected is None or tid == selected
        p = [n for n in path_of(P, t) if n in pos]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in p], y=[pos[n][1] for n in p], mode="lines",
            line=dict(color=PALETTE[i % len(PALETTE)], width=3 if selected == tid else 1.4,
                      shape="spline", smoothing=0.6),
            opacity=0.95 if on else 0.08, name=tid, showlegend=selected is not None and on,
            hovertemplate=f"<b>{tid}</b><br>{t.get('route_km', 0):,.0f} km · "
                          f"{len(t.get('items', []))} items<extra></extra>"))

    # destination nodes
    cnt = {}
    for t in trucks:
        for n in t.get("stops", []):
            cnt[n] = cnt.get(n, 0) + len(t.get("items", [])) / max(len(t.get("stops", [])), 1)
    nodes = [n for n in pos if n != P.depot]
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers",
        marker=dict(size=[8 + 4 * math.sqrt(cnt.get(n, 1)) for n in nodes],
                    color=[dist[n] for n in nodes], colorscale="Viridis",
                    line=dict(width=1, color="white"), colorbar=dict(title="km", thickness=10)),
        text=[f"<b>{n}</b><br>{dist[n]:,.0f} km from depot" for n in nodes],
        hovertemplate="%{text}<extra></extra>", showlegend=False))

    # depot
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", text=["DEPOT"],
                             textposition="bottom center", showlegend=False,
                             marker=dict(size=26, color="#E15759", symbol="star",
                                         line=dict(width=2, color="white"))))
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False),
                      plot_bgcolor="rgba(0,0,0,0)", hovermode="closest")
    return fig


def route_fig(P, t, pos, dist, rmax, step=None):
    """Single-truck view: zoomed to the path, only its legs/stops are highlighted."""
    p = [n for n in path_of(P, t) if n in pos]
    step = len(p) - 1 if step is None else step
    arr = t.get("arrivals", {})
    fig = go.Figure()
    for f in (0.25, 0.5, 0.75, 1.0):
        r = 0.15 + 0.85 * f
        fig.add_shape(type="circle", x0=-r, y0=-r, x1=r, y1=r,
                      line=dict(color="rgba(128,128,128,.2)", dash="dot", width=1))
    rest = [n for n in pos if n not in p]                       # context only, barely visible
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in rest], y=[pos[n][1] for n in rest], mode="markers",
                             marker=dict(size=5, color="rgba(128,128,128,.25)"), hoverinfo="skip",
                             showlegend=False))
    for k in range(len(p) - 1):                                 # legs as arrows
        (x0, y0), (x1, y1) = pos[p[k]], pos[p[k + 1]]
        done = k < step
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.3, standoff=12,
                           arrowwidth=3 if done else 1.2,
                           arrowcolor="#4C9BE8" if done else "rgba(128,128,128,.45)")
    stops = list(range(1, len(p) - 1))
    fig.add_trace(go.Scatter(
        x=[pos[p[k]][0] for k in stops], y=[pos[p[k]][1] for k in stops], mode="markers+text",
        text=[str(k) for k in stops], textfont=dict(color="white", size=12), showlegend=False,
        marker=dict(size=30, line=dict(width=2, color="white"),
                    color=["#4C9BE8" if k <= step else "#9AA0A6" for k in stops]),
        customdata=[[p[k], str(P.to_time(arr[p[k]])) if arr.get(p[k]) is not None else "-",
                     P.D.get(p[k - 1], {}).get(p[k], 0)] for k in stops],
        hovertemplate="<b>#%{text} %{customdata[0]}</b><br>Arrival %{customdata[1]}"
                      "<br>Leg %{customdata[2]:.1f} km<extra></extra>"))
    cx, cy = pos[p[min(step, len(p) - 1)]]                      # current position ring
    fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(size=44, color="rgba(0,0,0,0)", line=dict(width=3, color="#F28E2B"))))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", text=["DEPOT"], textposition="bottom center",
                             showlegend=False, marker=dict(size=26, color="#E15759", symbol="star",
                                                           line=dict(width=2, color="white"))))
    xs, ys = [pos[n][0] for n in p], [pos[n][1] for n in p]
    pad = 0.2 + 0.2 * max(max(xs) - min(xs), max(ys) - min(ys))
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False, range=[min(xs) - pad, max(xs) + pad], scaleanchor="y"),
                      yaxis=dict(visible=False, range=[min(ys) - pad, max(ys) + pad]))
    return fig


def _step(d, ids):
    cur = ids.index(st.session_state.get("rt_sel", ids[0]))
    st.session_state["rt_sel"] = ids[(cur + d) % len(ids)]


def cargo_frame(P, trucks):
    parts = []
    for t in trucks:
        if t.get("items"):
            df = P.items.iloc[t["items"]].copy()
            df["Truck_ID"] = t.get("id")
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts)
    return df[[c for c in CARGO_COLS if c in df.columns]].rename(columns=RENAME).reset_index(drop=True)


def status(t):
    return {True: "✅ OK", False: "❌ Check"}.get(t.get("route_ok"), "—")


# ------------------------------------------------------------------ tab
def render_viz():
    st.subheader("Network & Route Analytics")

    if "res" not in st.session_state:
        return st.info("Run the optimization from the **Optimize** tab first.")
    res = st.session_state["res"]
    P, fleet, _ = st.session_state["res_ctx"]
    trucks = res.get("trucks", [])
    if not trucks:
        return st.warning("No trucks were assigned. Adjust fleet, demand or parameters.")

    ftype = {f.name: f for f in fleet}
    for i, t in enumerate(trucks):
        t.setdefault("id", f"Truck-{i + 1}")

    # ---- KPIs
    n = len(trucks)
    km = sum(float(t.get("route_km") or 0) for t in trucks)
    cost = sum(float(t.get("route_cost") or 0) for t in trucks)
    items = sum(len(t.get("items", [])) for t in trucks)
    stops = sum(len(t.get("stops", [])) for t in trucks)
    chk = [t["route_ok"] for t in trucks if t.get("route_ok") is not None]

    def util(key, cap_attr):
        v = [min(float(t.get(key) or 0) / c, 1) for t in trucks
             if (c := getattr(ftype.get(t.get("type")), cap_attr, 0))]
        return 100 * sum(v) / len(v) if v else 0

    kpis = [("Fleet", f"{n:,}"), ("Items", f"{items:,}"), ("Stops", f"{stops:,}"),
            ("Distance", f"{km:,.0f} km"), ("Cost", f"{cost:,.0f}"),
            ("Route OK", f"{100 * sum(chk) / len(chk):.0f}%" if chk else "—"),
            ("Avg km/truck", f"{km / n:,.1f}"), ("Avg items/truck", f"{items / n:,.1f}"),
            ("Weight util.", f"{util('w', 'weight_cap'):.1f}%"), ("Area util.", f"{util('a', 'area'):.1f}%")]
    for row in (kpis[:5], kpis[5:]):
        for col, (k, v) in zip(st.columns(5), row):
            col.metric(k, v)
    st.divider()

    pos, dist, rmax = radial_layout(P, trucks)
    t_net, t_route, t_cargo = st.tabs(["Fleet Network", "Route Analysis", "Cargo"])

    # ---- Network
    with t_net:
        c1, c2 = st.columns(2)
        sort = c1.selectbox("Rank routes by", ["Longest", "Highest cost", "Most items", "Most stops"])
        key = {"Longest": lambda t: t.get("route_km", 0), "Highest cost": lambda t: t.get("route_cost", 0),
               "Most items": lambda t: len(t.get("items", [])), "Most stops": lambda t: len(t.get("stops", []))}[sort]
        ranked = sorted(trucks, key=key, reverse=True)
        top = c2.slider("Routes shown", 1, min(n, 150), min(n, 40)) if n > 1 else 1
        vis = ranked[:top]
        st.plotly_chart(network_fig(P, vis, pos, dist, rmax), width="stretch",
                        config={"displaylogo": False, "scrollZoom": True})
        st.dataframe(pd.DataFrame([{
            "Rank": i, "Truck": t["id"], "Vehicle": t.get("type", "?"),
            "km": round(t.get("route_km", 0), 1), "Cost": round(t.get("route_cost", 0), 2),
            "Items": len(t.get("items", [])), "Stops": len(t.get("stops", [])), "Status": status(t)}
            for i, t in enumerate(vis, 1)]), hide_index=True, height=300, width="stretch")

    # ---- Route analysis
    with t_route:
        ids = [str(t["id"]) for t in trucks]
        c1, c2, c3 = st.columns([1, 1, 6])
        c1.button("◀ Prev", on_click=_step, args=(-1, ids), width="stretch")
        c2.button("Next ▶", on_click=_step, args=(1, ids), width="stretch")
        tid = c3.selectbox("Truck", ids, key="rt_sel", label_visibility="collapsed")
        t = trucks[ids.index(tid)]
        route = path_of(P, t)
        legs = len(route) - 1

        for c, (k, v) in zip(st.columns(5), [("Distance", f"{t.get('route_km', 0):,.1f} km"),
                                             ("Cost", f"{t.get('route_cost', 0):,.0f}"),
                                             ("Items", len(t.get("items", []))),
                                             ("Stops", len(t.get("stops", []))), ("Status", status(t))]):
            c.metric(k, v)

        step = st.slider("Follow the route (legs driven)", 0, legs, legs, key=f"rt_step_{tid}") if legs > 1 else legs
        left, right = st.columns([2.2, 1])
        left.plotly_chart(route_fig(P, t, pos, dist, rmax, step), width="stretch",
                          config={"displaylogo": False, "scrollZoom": True})
        with right:
            ft = ftype.get(t.get("type"))
            st.caption(f"Vehicle: **{t.get('type', '?')}**")
            for label, key_, attr, unit in (("Weight", "w", "weight_cap", "kg"), ("Area", "a", "area", "m²")):
                cap = getattr(ft, attr, 0) if ft else 0
                val = float(t.get(key_) or 0)
                st.caption(f"{label}: {val:,.1f} / {cap:,.1f} {unit}")
                st.progress(min(val / cap, 1.0) if cap else 0.0)
            hz = t.get("classes") or []
            st.caption("Hazard: " + (", ".join(sorted(map(str, hz))) or "none"))
            arr = t.get("arrivals", {})
            st.dataframe(pd.DataFrame([{
                "#": k, "Stop": nd,
                "km": round(P.D.get(route[k - 1], {}).get(nd, 0), 1) if k else 0,
                "Arrival": str(P.to_time(arr[nd])) if arr.get(nd) is not None else "-"}
                for k, nd in enumerate(route)]), hide_index=True, width="stretch", height=300)

        cargo = cargo_frame(P, [t])
        st.markdown("**Loaded cargo**")
        if len(cargo):
            st.dataframe(cargo, hide_index=True, width="stretch")
        else:
            st.info("No cargo.")

    # ---- Cargo
    with t_cargo:
        cargo = cargo_frame(P, trucks)
        if cargo.empty:
            return st.info("No cargo data available.")
        c = st.columns(4)
        c[0].metric("Items", f"{len(cargo):,}")
        c[1].metric("Weight", f"{pd.to_numeric(cargo.get('Weight (kg)'), errors='coerce').sum():,.1f} kg")
        c[2].metric("Area", f"{pd.to_numeric(cargo.get('Area (m²)'), errors='coerce').sum():,.2f} m²")
        c[3].metric("Destinations", f"{cargo['Destination'].nunique():,}")

        f1, f2 = st.columns(2)
        dest = f1.selectbox("Destination", ["All"] + sorted(cargo["Destination"].astype(str).unique()))
        trk = f2.selectbox("Truck", ["All"] + sorted(cargo["Truck_ID"].astype(str).unique()))
        if dest != "All":
            cargo = cargo[cargo["Destination"].astype(str) == dest]
        if trk != "All":
            cargo = cargo[cargo["Truck_ID"].astype(str) == trk]
        st.caption(f"Showing **{len(cargo):,}** items")
        st.dataframe(cargo, hide_index=True, height=450, width="stretch")
        st.download_button("⬇Download CSV", cargo.to_csv(index=False), "optimized_cargo.csv",
                           "text/csv", width="stretch")

    st.divider()
    st.caption(f"{n:,} trucks · {items:,} items · {stops:,} stops · {km:,.0f} km total")


# usage:  with tab_viz: render_viz()
with tab_viz: render_viz()

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
# TAB 6 : compare heuristic vs exact on the same subset
# =====================================================================================
with tab_cmp:
    st.subheader("Heuristic vs exact on the same subset")
    st.caption("The exact model can only be solved for small subsets, so use a few destinations or 'Max items' "
               "(about 10-25 items).")
    tl2 = st.slider("Time limit for the exact model (s)", 5, 300, 60, key="cmp_time_limit")

    if st.button("Run comparison", key="cmp_run"):
        if len(P_scope.items) > stage1.MAX_EXACT_ITEMS:
            st.warning(f"More than {stage1.MAX_EXACT_ITEMS} items: the exact model is skipped and the heuristic "
                       f"plan is shown for both rows. Select fewer destinations / 'Max items'.")
        rows = []
        with st.spinner("Solving both..."):
            for name, a1, a2 in [("Heuristic", "heuristic", "heuristic"),
                                 ("Exact (MIP)", "exact", "exact")]:
                r = pipeline.run(P_scope, fleet, params, "full", a1, a2, time_limit=tl2)
                mm = r["metrics"]
                rows.append({
                    "Method": name,
                    "Status": r["info1"]["status"],
                    "Trucks": mm["trucks_used"],
                    "Stage 1 cost": round(mm["stage1_estimate"]),
                    "Total cost": round(mm.get("total_cost", float("nan"))),
                    "Distance km": round(mm.get("distance_km", float("nan"))),
                    "Unassigned": mm["items_unassigned"],
                    "Seconds": round(r["seconds"], 1),
                    "Violations": len(pipeline.validate(P_scope, r, fleet, params)),
                })
        cmp_df = pd.DataFrame(rows)
        st.dataframe(cmp_df, width="stretch", hide_index=True)
        st.plotly_chart(px.bar(cmp_df, x="Method", y="Total cost", text="Total cost"),
                        width="stretch", key="cmp_chart")


# =====================================================================================
# TAB 7 : sensitivity (scenario analysis with the heuristic)
# =====================================================================================
SENS_PARAMS = {
    "N (max stops)":               ("max_stops",  "1,2,3,4,5"),
    "Delta (availability gap, h)": ("delta_h",    "1,2,4,8,24"),
    "f (cost per stop)":           ("stop_cost",  "0,250,500,1000,2000"),
    "Trucks of the largest type K_t": (None,      "100,200,300,400,600"),
}

with tab_sens:
    st.subheader("Scenario (sensitivity) analysis")
    st.caption("Re-solve while one parameter changes, using the heuristic or the exact model. Uses the destinations / item limit "
               "chosen in the pre-check tab.")

    param = st.selectbox("Parameter to vary", list(SENS_PARAMS), key="sens_param")
    attr, default_vals = SENS_PARAMS[param]
    # key includes the parameter so the default text resets when the parameter changes
    vals = st.text_input("Values (comma separated)", default_vals, key=f"sens_vals_{attr or 'trucks'}")

    m1, m2 = st.columns([1.4, 1])
    method = m1.radio("Solver", ["Heuristic", "Exact (MIP)"], horizontal=True, key="sens_method",
                      help="Exact is only practical for small subsets (about 10-25 items).")
    algo = "exact" if method.startswith("Exact") else "heuristic"
    tl_s = m2.slider("Time limit per scenario (s)", 5, 300, 60, key="sens_time_limit",
                     disabled=algo == "heuristic")
    if algo == "exact" and len(P_scope.items) > stage1.MAX_EXACT_ITEMS:
        st.warning(f"More than {stage1.MAX_EXACT_ITEMS} items: the exact model is skipped and the heuristic "
                   f"plan is used instead. Select fewer destinations / 'Max items'.")

    if st.button("Run scenarios", key="sens_run"):
        try:
            values = [float(x) for x in vals.split(",") if x.strip()]
        except ValueError:
            st.error("Values must be numbers separated by commas, e.g. 1,2,3.")
            values = []

        out = []
        with st.spinner(f"Re-solving scenarios ({method})..."):
            for v in values:
                p2, f2 = copy.deepcopy(params), copy.deepcopy(fleet)
                if attr:
                    setattr(p2, attr, int(v) if attr == "max_stops" else v)
                else:
                    max(f2, key=lambda t: t.weight_cap).count = int(v)

                kw = {"time_limit": tl_s} if algo == "exact" else {}
                r = pipeline.run(P_scope, f2, p2, "full", algo, algo, **kw)
                mm = r["metrics"]
                out.append({
                    param: v,
                    "Solver": method,
                    "Status": r["info1"]["status"],
                    "Trucks": mm["trucks_used"],
                    "Total cost": round(mm.get("total_cost", 0)),
                    "Distance km": round(mm.get("distance_km", 0)),
                    "Unassigned items": mm["items_unassigned"],
                    "Weight util %": round(mm.get("avg_weight_util_pct", 0), 1),
                    "Seconds": round(r["seconds"], 1),
                })

        if out:
            sdf = pd.DataFrame(out)
            st.dataframe(sdf, width="stretch", hide_index=True)
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.line(sdf, x=param, y="Total cost", markers=True),
                            width="stretch", key="sens_cost_chart")
            c2.plotly_chart(px.line(sdf, x=param, y="Trucks", markers=True),
                            width="stretch", key="sens_trucks_chart")