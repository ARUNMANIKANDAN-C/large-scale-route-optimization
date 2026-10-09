"""Feasibility pre-check: item validation, deadline verification, fleet capacity bounds,
and a summary of the planning window (start date, deadlines, slack)."""
import math
import pandas as pd
from common import available_types


# Converts elapsed hours from time zero into a formatted calendar string or fallback hour label.
def _when(P, hours):
    """Hours since time zero -> readable date/time (falls back to 't = x h')."""
    try:
        return str(P.to_time(float(hours)))
    except Exception:
        return f"t = {float(hours):.1f} h"


# Calculates the overall planning horizon, arrival deadlines, and deadline slack for all items.
def planning_window(P, params, max_speed):
    """When does planning start, when must everything be delivered, and how tight are the deadlines?

    Time zero of the data is converted with ``P.to_time``. Planning starts when the first item
    becomes available; every truck then leaves when the LAST item on it is available.
    """
    df = P.items
    if df.empty:
        return {}

    e, l = df["e"].astype(float), df["l"].astype(float)
    start_h, end_h = float(e.min()), float(l.max())

    # slack = how much waiting/delay an item can still absorb on the fastest possible trip
    dist = df["d"].map(lambda d: P.D.get(P.depot, {}).get(d))
    travel = pd.to_numeric(dist, errors="coerce") / max_speed if max_speed > 0 else float("nan")
    slack = (l - e - travel - params.unload_h).dropna()
    window_d = (l - e) / 24.0

    return {
        "depot": P.depot,
        "start": _when(P, start_h),
        "end": _when(P, end_h),
        "horizon_days": (end_h - start_h) / 24.0,
        "last_available": _when(P, e.max()),
        "earliest_deadline": _when(P, l.min()),
        "latest_deadline": _when(P, l.max()),
        "window_days": (float(window_d.min()), float(window_d.median()), float(window_d.max())),
        "slack_min_h": float(slack.min()) if len(slack) else float("nan"),
        "slack_median_h": float(slack.median()) if len(slack) else float("nan"),
        "tight_items": int((slack < 24).sum()),
        "n_items": len(df),
        "n_destinations": int(df["d"].nunique()),
        "farthest_km": float(pd.to_numeric(dist, errors="coerce").max()),
        "heaviest_kg": float(df["w"].max()),
        "hazard_counts": df["h"].value_counts().to_dict(),
    }


# Validates item delivery feasibility against fleet capacities and computes theoretical lower bounds on required trucks.
def precheck(P, fleet, params):
    """Perform pre-optimization feasibility checks and calculate theoretical bounds.

    Returns:
        dict with keys:
            - 'ok': bool
            - 'messages': list of str
            - 'bound_table': pd.DataFrame
            - 'n_min': int
            - 'bad_items': pd.DataFrame
            - 'window': dict  (planning start, deadlines, slack; see planning_window)
    """
    avail = available_types(fleet)
    messages = []
    bad_reasons = {}

    if not avail:
        return {
            "ok": False,
            "messages": ["No truck types available in the fleet configuration."],
            "bound_table": pd.DataFrame(),
            "n_min": 0,
            "bad_items": pd.DataFrame(),
            "window": {},
        }

    max_weight_cap = max(t.weight_cap for t in avail)
    max_area_cap = max(t.area for t in avail)
    max_speed = max(t.speed for t in avail)

    df = P.items
    depot = P.depot

    # Check each item
    for idx, row in df.iterrows():
        reasons = []

        # 1. Weight capacity
        if row["w"] > max_weight_cap:
            reasons.append(f"Weight {row['w']:.1f} kg exceeds max truck capacity {max_weight_cap:.1f} kg")

        # 2. Area capacity
        if row["a"] > max_area_cap:
            reasons.append(f"Area {row['a']:.2f} m2 exceeds max truck area {max_area_cap:.2f} m2")

        # 3. Distance & Reachability
        dest = row["d"]
        dist_km = P.D.get(depot, {}).get(dest, None)
        if dist_km is None:
            reasons.append(f"No distance path from depot '{depot}' to destination '{dest}'")
        else:
            # 4. Earliest possible delivery vs deadline
            min_travel_h = dist_km / max_speed if max_speed > 0 else float("inf")
            min_arrival_h = row["e"] + min_travel_h + params.unload_h
            if min_arrival_h > row["l"]:
                reasons.append(
                    f"Impossible deadline: available at {row['e']:.1f}h + travel {min_travel_h:.1f}h + unload {params.unload_h:.1f}h = {min_arrival_h:.1f}h > deadline {row['l']:.1f}h"
                )

        if reasons:
            bad_reasons[idx] = "; ".join(reasons)

    bad_items = df.loc[list(bad_reasons.keys())].copy() if bad_reasons else pd.DataFrame(columns=list(df.columns) + ["Reason"])
    if not bad_items.empty:
        bad_items["Reason"] = [bad_reasons[i] for i in bad_items.index]

    # Lower bound on number of trucks
    hazard_classes = sorted(df["h"].unique())
    groups = []
    visited = set()
    for h in hazard_classes:
        if h in visited:
            continue
        comp_group = {h}
        for other in hazard_classes:
            if other != h and params.compatible(h, other):
                comp_group.add(other)

        all_comp = True
        for a in comp_group:
            for b in comp_group:
                if not params.compatible(a, b):
                    all_comp = False
                    break
        if all_comp:
            visited.update(comp_group)
            groups.append(sorted(comp_group))
        else:
            visited.add(h)
            groups.append([h])

    bound_rows = []
    total_min_trucks = 0
    for grp in groups:
        grp_df = df[df["h"].isin(grp)]
        if grp_df.empty:
            continue
        tot_w = grp_df["w"].sum()
        tot_a = grp_df["a"].sum()
        cnt_items = len(grp_df)

        min_t_wt = math.ceil(tot_w / max_weight_cap) if max_weight_cap > 0 else 0
        min_t_ar = math.ceil(tot_a / max_area_cap) if max_area_cap > 0 else 0
        min_t = max(min_t_wt, min_t_ar)
        total_min_trucks += min_t

        bound_rows.append({
            "Hazard group": " + ".join(grp),
            "Items": cnt_items,
            "Total weight (kg)": round(tot_w, 1),
            "Total area (m2)": round(tot_a, 2),
            "Min trucks (weight)": min_t_wt,
            "Min trucks (area)": min_t_ar,
            "Min trucks": min_t,
        })

    bound_table = pd.DataFrame(bound_rows)

    total_fleet_count = sum(t.count for t in avail if t.count is not None)
    has_unlimited = any(t.count is None for t in avail)

    ok = len(bad_items) == 0

    if not ok:
        messages.append(f"Feasibility pre-check found {len(bad_items)} item(s) that cannot be delivered.")
    else:
        messages.append("All items passed individual feasibility checks (capacity & deadlines).")

    if not has_unlimited and total_fleet_count < total_min_trucks:
        ok = False
        messages.append(
            f"Fleet size is insufficient: {total_fleet_count} trucks available vs minimum bound of {total_min_trucks} trucks required."
        )
    else:
        messages.append(f"Theoretical minimum fleet required: at least {total_min_trucks} truck(s).")

    return {
        "ok": ok,
        "messages": messages,
        "bound_table": bound_table,
        "n_min": total_min_trucks,
        "bad_items": bad_items,
        "window": planning_window(P, params, max_speed),
    }