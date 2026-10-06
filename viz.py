"""Network graph and route visualisation helpers."""

import math
import numpy as np
import plotly.graph_objects as go


PALETTE = [
    "#2563EB", "#DC2626", "#059669", "#7C3AED",
    "#EA580C", "#0891B2", "#DB2777", "#65A30D",
    "#4F46E5", "#CA8A04", "#9333EA", "#0F766E",
]


def compute_layout(P, iterations=300, seed=42):
    depot = P.depot
    dests = sorted(set(P.items["d"].unique()) - {depot})
    nodes = [depot] + dests
    n = len(nodes)

    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: (0.0, 0.0)}

    Dt = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = P.D.get(nodes[i], {}).get(
                nodes[j],
                P.D.get(nodes[j], {}).get(nodes[i], None)
            )
            Dt[i, j] = Dt[j, i] = d if d is not None else 0.0

    Dt /= Dt.max() or 1.0

    pos = np.zeros((n, 2))

    for i in range(1, n):
        angle = 2 * math.pi * (i - 1) / max(n - 1, 1)
        r = max(Dt[0, i], 0.05)
        pos[i] = [r * math.cos(angle), r * math.sin(angle)]

    lr = 0.04

    for _ in range(iterations):
        for i in range(1, n):
            grad = np.zeros(2)

            for j in range(n):
                if i == j or Dt[i, j] == 0:
                    continue

                diff = pos[i] - pos[j]
                d_act = np.linalg.norm(diff) + 1e-12
                grad += 2 * (d_act - Dt[i, j]) / d_act * diff

            pos[i] -= lr * grad / n

    return {
        nodes[k]: (float(pos[k, 0]), float(pos[k, 1]))
        for k in range(n)
    }


def build_figure(P, trucks, selected_id=None, positions=None, max_routes=40):

    if positions is None:
        positions = compute_layout(P)

    depot = P.depot
    D = P.D
    destinations = sorted(set(P.items["d"].unique()) - {depot})
    item_counts = P.items.groupby("d").size().to_dict()

    fig = go.Figure()

    # Background network
    for d in destinations:
        if d not in positions or depot not in positions:
            continue

        x0, y0 = positions[depot]
        x1, y1 = positions[d]
        dist_km = D.get(depot, {}).get(d, 0)

        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=0.8, color="rgba(100,116,139,0.18)"),
            hovertext=f"{depot} → {d}: {dist_km:.0f} km",
            hoverinfo="text",
            showlegend=False
        ))

    # Routes
    show = (
        [t for t in trucks if t["id"] == selected_id]
        if selected_id else trucks[:max_routes]
    )

    active_stops = set()

    for tr in show:
        active_stops.update(tr.get("route", tr["stops"]))

    for idx, tr in enumerate(show):

        color = PALETTE[idx % len(PALETTE)]
        route_nodes = [depot] + tr.get("route", tr["stops"])

        rx = [positions.get(n, (0, 0))[0] for n in route_nodes]
        ry = [positions.get(n, (0, 0))[1] for n in route_nodes]

        hover = []

        for i, nd in enumerate(route_nodes):
            txt = f"<b>{tr['id']}</b> ({tr['type']})<br>{nd}"

            if i < len(route_nodes) - 1:
                nxt = route_nodes[i + 1]
                seg = D.get(nd, {}).get(nxt, 0)
                txt += f" → {nxt} ({seg:.0f} km)"

            hover.append(txt)

        fig.add_trace(go.Scatter(
            x=rx,
            y=ry,
            mode="lines+markers",
            line=dict(
                width=4 if selected_id else 2.5,
                color=color
            ),
            marker=dict(
                size=7,
                color=color,
                line=dict(width=1.5, color="white")
            ),
            name=f"{tr['id']} ({tr['type']})",
            hovertext=hover,
            hoverinfo="text",
            legendgroup=tr["id"]
        ))

        # Direction arrows
        for i in range(len(route_nodes) - 1):

            p0 = positions.get(route_nodes[i], (0, 0))
            p1 = positions.get(route_nodes[i + 1], (0, 0))

            hx = p0[0] * 0.32 + p1[0] * 0.68
            hy = p0[1] * 0.32 + p1[1] * 0.68

            ax = p0[0] * 0.55 + p1[0] * 0.45
            ay = p0[1] * 0.55 + p1[1] * 0.45

            fig.add_annotation(
                x=hx, y=hy, ax=ax, ay=ay,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.4,
                arrowwidth=1.8,
                arrowcolor=color
            )

    # Destination nodes
    d_labels = [d for d in destinations if d in positions]

    dx = [positions[d][0] for d in d_labels]
    dy = [positions[d][1] for d in d_labels]

    d_sizes = [
        max(14, min(48, 8 + item_counts.get(d, 0) * 1.5))
        for d in d_labels
    ]

    d_border = [
        "#2563EB"
        if not selected_id or d in active_stops
        else "rgba(148,163,184,0.35)"
        for d in d_labels
    ]

    fig.add_trace(go.Scatter(
        x=dx,
        y=dy,
        mode="markers+text",
        marker=dict(
            size=d_sizes,
            color="#EFF6FF",
            line=dict(width=2.5, color=d_border)
        ),
        text=d_labels,
        textposition="top center",
        textfont=dict(size=9, color="#334155"),
        hovertext=[
            f"<b>{d}</b><br>"
            f"Items: {item_counts.get(d, 0)}<br>"
            f"Distance: {D.get(depot, {}).get(d, 0):.0f} km"
            for d in d_labels
        ],
        hoverinfo="text",
        name="Destinations"
    ))

    # Depot
    if depot in positions:
        fig.add_trace(go.Scatter(
            x=[positions[depot][0]],
            y=[positions[depot][1]],
            mode="markers+text",
            marker=dict(
                size=28,
                color="#DC2626",
                symbol="star-diamond",
                line=dict(width=2, color="white")
            ),
            text=["DEPOT"],
            textposition="bottom center",
            textfont=dict(
                size=11,
                color="#B91C1C",
                family="Arial Black"
            ),
            hovertext=f"<b>DEPOT</b>: {depot}",
            hoverinfo="text",
            name="Depot"
        ))

    # White theme
    title_txt = (
        f"Route: {selected_id}"
        if selected_id
        else f"Network · {len(show)} of {len(trucks)} routes shown"
    )

    fig.update_layout(
        title=dict(
            text=title_txt,
            font=dict(color="#0F172A", size=14)
        ),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=720,
        margin=dict(l=10, r=10, t=50, b=10),

        legend=dict(
            font=dict(size=10, color="#334155"),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#E2E8F0",
            borderwidth=1
        ),

        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False
        ),

        yaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
            scaleanchor="x"
        ),

        hoverlabel=dict(
            bgcolor="white",
            font=dict(size=12, color="#0F172A")
        )
    )

    return fig