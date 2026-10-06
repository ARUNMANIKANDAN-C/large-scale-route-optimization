"""Network graph and route visualisation helpers for the Streamlit app."""

import math
import numpy as np
import plotly.graph_objects as go


# ── Vibrant colours that pop on dark backgrounds ─────────────────────────────
PALETTE = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#F8C471", "#82E0AA", "#AED6F1", "#F1948A",
    "#BB8FCE", "#85C1E9", "#D2B4DE", "#F9E79F", "#A3E4D7",
    "#F5B7B1", "#D5F5E3", "#FADBD8", "#E8DAEF", "#D4EFDF",
]


# ══════════════════════════════════════════════════════════════════════════════
# Layout
# ══════════════════════════════════════════════════════════════════════════════
def compute_layout(P, iterations=300, seed=42):
    """Position nodes in 2-D so Euclidean distances approximate the real km
    distances (stress-minimisation).  The depot is pinned at the origin.

    Returns ``{node_name: (x, y)}``.
    """
    depot = P.depot
    dests = sorted(set(P.items["d"].unique()) - {depot})
    nodes = [depot] + dests
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: (0.0, 0.0)}

    # target distance matrix (symmetric, normalised)
    Dt = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = P.D.get(nodes[i], {}).get(nodes[j],
                    P.D.get(nodes[j], {}).get(nodes[i], None))
            Dt[i, j] = Dt[j, i] = d if d is not None else 0.0
    scale = Dt.max() or 1.0
    Dt /= scale

    # initial positions: polar around depot
    pos = np.zeros((n, 2))
    for i in range(1, n):
        angle = 2 * math.pi * (i - 1) / max(n - 1, 1)
        r = max(Dt[0, i], 0.05)
        pos[i] = [r * math.cos(angle), r * math.sin(angle)]

    # simple stress-majorisation
    lr = 0.04
    for _ in range(iterations):
        for i in range(1, n):
            grad = np.zeros(2)
            for j in range(n):
                if i == j or Dt[i, j] == 0:
                    continue
                diff = pos[i] - pos[j]
                d_act = np.linalg.norm(diff) + 1e-12
                grad += 2.0 * (d_act - Dt[i, j]) / d_act * diff
            pos[i] -= lr * grad / n

    return {nodes[k]: (float(pos[k, 0]), float(pos[k, 1])) for k in range(n)}


# ══════════════════════════════════════════════════════════════════════════════
# Network figure
# ══════════════════════════════════════════════════════════════════════════════
def build_figure(P, trucks, selected_id=None, positions=None, max_routes=40):
    """Build a Plotly Figure showing the network graph with truck routes.

    Parameters
    ----------
    P : data.Problem
    trucks : list[dict]
    selected_id : str | None – a single truck ID, or None for "all"
    positions : dict | None – pre-computed layout
    max_routes : int – cap on visible routes in "all" mode
    """
    if positions is None:
        positions = compute_layout(P)

    depot = P.depot
    D = P.D
    destinations = sorted(set(P.items["d"].unique()) - {depot})
    item_counts = P.items.groupby("d").size().to_dict()

    fig = go.Figure()

    # ── 1. background edges (depot → each destination) ───────────────────
    for d in destinations:
        if d not in positions or depot not in positions:
            continue
        x0, y0 = positions[depot]
        x1, y1 = positions[d]
        dist_km = D.get(depot, {}).get(d, 0)
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=0.6, color="rgba(80,100,130,0.18)"),
            hovertext=[f"{depot} → {d}: {dist_km:.0f} km", None, None],
            hoverinfo="text", showlegend=False,
        ))

    # ── 2. truck routes ──────────────────────────────────────────────────
    if selected_id:
        show = [t for t in trucks if t["id"] == selected_id]
    else:
        show = trucks[:max_routes]

    active_stops = set()
    for tr in show:
        active_stops.update(tr.get("route", tr["stops"]))

    for idx, tr in enumerate(show):
        color = PALETTE[idx % len(PALETTE)]
        route_nodes = [depot] + tr.get("route", tr["stops"])
        rx = [positions.get(nd, (0, 0))[0] for nd in route_nodes]
        ry = [positions.get(nd, (0, 0))[1] for nd in route_nodes]

        # hover for each point on the route
        hover = []
        for i, nd in enumerate(route_nodes):
            txt = f"<b>{tr['id']}</b> ({tr['type']})<br>{nd}"
            if i < len(route_nodes) - 1:
                nxt = route_nodes[i + 1]
                seg = D.get(nd, {}).get(nxt, 0)
                txt += f" → {nxt}  ({seg:.0f} km)"
            hover.append(txt)

        fig.add_trace(go.Scatter(
            x=rx, y=ry, mode="lines+markers",
            line=dict(width=3.5 if selected_id else 2, color=color,
                      dash="solid"),
            marker=dict(size=6, color=color),
            name=f"{tr['id']} ({tr['type']})",
            hovertext=hover, hoverinfo="text",
            legendgroup=tr["id"],
        ))

        # direction arrows at midpoint of each segment
        for i in range(len(route_nodes) - 1):
            p0 = positions.get(route_nodes[i], (0, 0))
            p1 = positions.get(route_nodes[i + 1], (0, 0))
            ax = p0[0] * 0.55 + p1[0] * 0.45
            ay = p0[1] * 0.55 + p1[1] * 0.45
            hx = p0[0] * 0.32 + p1[0] * 0.68
            hy = p0[1] * 0.32 + p1[1] * 0.68
            fig.add_annotation(
                x=hx, y=hy, ax=ax, ay=ay,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.6,
                arrowwidth=2, arrowcolor=color, opacity=0.85,
            )

    # ── 3. destination nodes ─────────────────────────────────────────────
    dx = [positions[d][0] for d in destinations if d in positions]
    dy = [positions[d][1] for d in destinations if d in positions]
    d_labels = [d for d in destinations if d in positions]
    d_sizes = [max(16, min(50, 8 + item_counts.get(d, 0) * 1.5)) for d in d_labels]
    # dim nodes not on the active route(s)
    d_border = [
        "#4ECDC4" if (not selected_id or d in active_stops) else "rgba(78,205,196,0.25)"
        for d in d_labels
    ]
    d_hover = [
        f"<b>{d}</b><br>"
        f"Items: {item_counts.get(d, 0)}<br>"
        f"Distance from depot: {D.get(depot, {}).get(d, 0):.0f} km"
        for d in d_labels
    ]
    fig.add_trace(go.Scatter(
        x=dx, y=dy, mode="markers+text",
        marker=dict(size=d_sizes, color="#0f1129",
                    line=dict(width=2.5, color=d_border)),
        text=d_labels, textposition="top center",
        textfont=dict(size=9, color="rgba(175,185,200,0.85)"),
        hovertext=d_hover, hoverinfo="text",
        name="Destinations", showlegend=True,
    ))

    # ── 4. depot node ────────────────────────────────────────────────────
    if depot in positions:
        fig.add_trace(go.Scatter(
            x=[positions[depot][0]], y=[positions[depot][1]],
            mode="markers+text",
            marker=dict(size=26, color="#FF6B6B", symbol="star-diamond",
                        line=dict(width=2, color="white")),
            text=["DEPOT"], textposition="bottom center",
            textfont=dict(size=11, color="#FF6B6B", family="Arial Black"),
            hovertext=f"<b>DEPOT</b>: {depot}", hoverinfo="text",
            name="Depot",
        ))

    # ── layout ───────────────────────────────────────────────────────────
    title_txt = (f"Route: {selected_id}" if selected_id
                 else f"Network  ·  {len(show)} of {len(trucks)} routes shown")
    fig.update_layout(
        title=dict(text=title_txt,
                   font=dict(color="#d0d0d0", size=14)),
        template="plotly_dark",
        paper_bgcolor="rgba(10,10,20,1)",
        plot_bgcolor="rgba(14,14,28,1)",
        height=720,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(
            font=dict(size=10, color="#a0a0a0"),
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(80,80,80,0.3)", borderwidth=1,
        ),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False,
                   scaleanchor="x"),
        hoverlabel=dict(bgcolor="rgba(25,25,45,0.95)",
                        font_size=12, font_color="#e0e0e0"),
    )

    return fig
