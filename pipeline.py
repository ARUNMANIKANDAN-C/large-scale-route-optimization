"""Pipeline: Stage 1 (loading) -> Stage 2 (routing) -> result table, with feedback loop,
repair and an INDEPENDENT constraint validator."""
import copy
import time
import pandas as pd

from common import available_types
import stage1
import routing
from precheck import precheck


def _fleet_map(fleet):
    return {t.name: t for t in fleet}


def _stage2(P, trucks, fleet, params, method, time_limit):
    ft = _fleet_map(fleet)
    for tr in trucks:
        routing.route_truck(P, tr, ft[tr["type"]], params, method=method, time_limit=time_limit)
    return [tr for tr in trucks if not tr.get("route_ok", True)]


def _repair_split(P, bad_trucks, fleet, params, remaining):
    """Split trucks whose stops cannot all meet their deadlines into single-stop trucks."""
    p1 = copy.copy(params)
    p1.allow_multi_stop = False
    p1.max_stops = 1
    sub_items = [i for tr in bad_trucks for i in tr["items"]]
    new_fleet = copy.deepcopy(fleet)
    for t in new_fleet:
        if t.count is not None:
            t.count = remaining.get(t.name, 0)
    sub = P.subset()
    sub.items = P.items.iloc[sub_items].reset_index(drop=True)
    tr, un, _ = stage1.heuristic_loading(sub, new_fleet, p1)
    # map positions back to P
    for t in tr:
        t["items"] = [sub_items[i] for i in t["items"]]
    return tr, [(sub_items[i], r) for i, r in un]


def _load(P, fleet, params, method, time_limit, forbidden_pairs, log):
    """Stage 1: exact MIP (falls back to the heuristic if it cannot produce a plan) or heuristic."""
    if method == "exact":
        trucks, unassigned, info = stage1.mip_loading(
            P, fleet, params, time_limit=time_limit, forbidden_pairs=forbidden_pairs)
        if info["status"] in ("infeasible", "no solution in time limit", "too large"):
            why = info.get("message", info["status"])
            log.append(f"Exact loading not used ({why}) -> heuristic loading used instead.")
            trucks, unassigned, info = stage1.heuristic_loading(P, fleet, params)
            info["status"] = f"heuristic fallback ({why})"
        return trucks, unassigned, info
    return stage1.heuristic_loading(P, fleet, params)


def run(P, fleet, params, mode="full", stage1_method="heuristic", stage2_method="heuristic",
        time_limit=60, max_feedback=3, route_time_limit=15):
    """Run loading + routing.

    stage1_method : "heuristic" or "exact".
    mode          : "full" (loading + routing), "loading" (Stage 1 only) or
                    "routing" (Stage 2 on a heuristic baseline loading).

    With the exact Stage 1, trucks that fail routing contribute city-pair cuts and the
    loading is re-solved (up to ``max_feedback`` times). Anything still unroutable is
    repaired afterwards by splitting it into single-stop trucks.
    """
    t0 = time.time()
    log = []
    pre = precheck(P, fleet, params)
    log += pre["messages"]
    forbidden_pairs = set()
    trucks = unassigned = info1 = None

    if mode == "routing":
        stage1_method = "heuristic"

    for it in range(max_feedback + 1):
        trucks, unassigned, info1 = _load(P, fleet, params, stage1_method, time_limit, forbidden_pairs, log)

        if mode == "loading":
            break

        bad = _stage2(P, trucks, fleet, params, stage2_method, route_time_limit)
        if not bad:
            break
        # Only the exact model can learn from a routing failure, and only if it was really used.
        if stage1_method != "exact" or info1["status"].startswith("heuristic fallback"):
            break

        before = len(forbidden_pairs)
        for tr in bad:
            for a in range(len(tr["stops"])):
                for b in range(a + 1, len(tr["stops"])):
                    forbidden_pairs.add((tr["stops"][a], tr["stops"][b]))
        log.append(f"Feedback {it + 1}: {len(bad)} truck(s) failed routing; "
                   f"{len(forbidden_pairs)} city-pair cut(s) in total.")
        if len(forbidden_pairs) == before:      # nothing new to learn -> re-solving would repeat itself
            break

    if trucks is None:
        trucks, unassigned, info1 = [], [], {"status": "no solution"}

    # Repair only after the feedback loop is exhausted.
    if mode != "loading":
        bad = [tr for tr in trucks if not tr.get("route_ok", True)]
        if bad:
            used = {}
            for tr in trucks:
                if tr not in bad:
                    used[tr["type"]] = used.get(tr["type"], 0) + 1
            remaining = {t.name: (10 ** 9 if t.count is None else max(t.count - used.get(t.name, 0), 0))
                         for t in available_types(fleet)}
            keep = [tr for tr in trucks if tr not in bad]
            new, un = _repair_split(P, bad, fleet, params, remaining)
            _stage2(P, new, fleet, params, stage2_method, route_time_limit)
            trucks = keep + new
            unassigned += un
            log.append(f"Repair: split {len(bad)} unroutable truck(s) into {len(new)} single-stop truck(s).")
            for k, tr in enumerate(trucks):
                tr["id"] = f"T{k + 1:04d}"

    z1 = stage1.stage1_objective(P, trucks, fleet, params)
    res = dict(trucks=trucks, unassigned=unassigned or [], info1=info1, stage1_objective=z1,
               pre=pre, log=log, mode=mode, stage1_method=stage1_method, stage2_method=stage2_method)
    res["metrics"] = metrics(P, res, fleet, params)
    res["seconds"] = time.time() - t0
    if res["unassigned"]:
        res["log"].append(f"{len(res['unassigned'])} item(s) could not be loaded (see 'unassigned').")
    return res


# --------------------------------------------------------------------------------------
def metrics(P, res, fleet, params):
    ft = _fleet_map(fleet)
    trucks = res["trucks"]
    m = {}
    m["trucks_used"] = len(trucks)
    m["by_type"] = {t.name: sum(1 for tr in trucks if tr["type"] == t.name) for t in fleet}
    m["items_loaded"] = sum(len(tr["items"]) for tr in trucks)
    m["items_unassigned"] = len(res["unassigned"])
    m["stops_total"] = sum(len(tr["stops"]) for tr in trucks)
    m["stop_cost"] = params.stop_cost * m["stops_total"]
    m["fixed_cost"] = sum(ft[tr["type"]].fixed_cost for tr in trucks)
    m["stage1_estimate"] = stage1.stage1_objective(P, trucks, fleet, params)
    routed = [tr for tr in trucks if "route_km" in tr]
    if routed and len(routed) == len(trucks):
        m["distance_km"] = sum(tr["route_km"] for tr in routed)
        m["distance_cost"] = sum(tr["route_cost"] for tr in routed)
        m["total_cost"] = m["distance_cost"] + m["stop_cost"] + m["fixed_cost"]
        m["on_time_pct"] = 100.0 * sum(len(tr["items"]) for tr in routed if tr["route_ok"]) / max(m["items_loaded"], 1)
    if trucks:
        m["avg_weight_util_pct"] = 100 * sum(tr["w"] for tr in trucks) / sum(ft[tr["type"]].weight_cap for tr in trucks)
        m["avg_area_util_pct"] = 100 * sum(tr["a"] for tr in trucks) / sum(ft[tr["type"]].area for tr in trucks)
    return m


def truck_table(P, res, fleet):
    ft = _fleet_map(fleet)
    rows = []
    for tr in res["trucks"]:
        t = ft[tr["type"]]
        rows.append({
            "Truck_ID": tr["id"], "Truck_Type": tr["type"],
            "Route": "->".join([P.depot] + tr.get("route", tr["stops"])),
            "Stops": len(tr["stops"]), "Items": len(tr["items"]),
            "Weight_kg": round(tr["w"], 1), "Weight_util_%": round(100 * tr["w"] / t.weight_cap, 1),
            "Area_m2": round(tr["a"], 2), "Area_util_%": round(100 * tr["a"] / t.area, 1),
            "Hazard_classes": "+".join(sorted(tr["classes"])),
            "Start_Time": P.to_time(tr["S"]), "Distance_km": round(tr.get("route_km", float("nan")), 1),
            "Cost": round(tr.get("route_cost", float("nan")), 1), "On_time": tr.get("route_ok", None),
        })
    return pd.DataFrame(rows)


def item_table(P, res):
    """Kaggle-style output: one row per item."""
    df = P.items
    rows = []
    for tr in res["trucks"]:
        orders = set(df.at[i, "Order_ID"] for i in tr["items"])
        route = "->".join([P.depot] + tr.get("route", tr["stops"]))
        for i in tr["items"]:
            r = df.iloc[i]
            arr = tr.get("arrivals", {}).get(r["d"])
            rows.append({
                "Truck_ID": tr["id"], "Truck_Route": route, "Order_ID": r["Order_ID"],
                "Material_ID": r["Material_ID"], "Item_ID": r["Item_ID"], "Danger_Type": r["h"],
                "Source": r["Source"], "Destination": r["d"], "Start_Time": P.to_time(tr["S"]),
                "Arrival_Time": P.to_time(arr) if arr is not None else pd.NaT,
                "Deadline": r["Deadline"], "Shared_Truck": "Y" if len(orders) > 1 else "N",
                "Truck_Type": tr["type"],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
def validate(P, res, fleet, params, tol=1e-6):
    """Independent re-check of every constraint on the FINAL plan. Returns list of violations."""
    ft = _fleet_map(fleet)
    A = stage1._arr(P)
    bad = []
    seen = {}
    for tr in res["trucks"]:
        t = ft[tr["type"]]
        for i in tr["items"]:
            seen[i] = seen.get(i, 0) + 1
        w = sum(A["w"][i] for i in tr["items"])
        a = sum(A["a"][i] for i in tr["items"])
        if w > t.weight_cap + tol:
            bad.append(f"{tr['id']}: weight {w:.0f} > {t.weight_cap}")
        if a > t.area + tol:
            bad.append(f"{tr['id']}: area {a:.2f} > {t.area}")
        cls = sorted(set(A["h"][i] for i in tr["items"]))
        for x in range(len(cls)):
            for y in range(x + 1, len(cls)):
                if not params.compatible(cls[x], cls[y]):
                    bad.append(f"{tr['id']}: hazard classes {cls[x]} and {cls[y]} share a truck")
        stops = set(A["d"][i] for i in tr["items"])
        if len(stops) > params.max_stops:
            bad.append(f"{tr['id']}: {len(stops)} stops > N={params.max_stops}")
        es = [A["e"][i] for i in tr["items"]]
        if max(es) - min(es) > params.delta_h + tol:
            bad.append(f"{tr['id']}: availability gap {max(es) - min(es):.2f} h > {params.delta_h}")
        if "route" in tr:
            if sorted(tr["route"]) != sorted(stops):
                bad.append(f"{tr['id']}: route does not visit exactly its stops")
            # recompute arrivals from scratch
            tt, prev = max(es), P.depot
            for c in tr["route"]:
                tt += P.D[prev][c] / t.speed
                for i in tr["items"]:
                    if A["d"][i] == c and tt > A["l"][i] + 1e-6:
                        bad.append(f"{tr['id']}: item {i} arrives {tt:.1f} h after deadline {A['l'][i]:.1f} h")
                        break
                tt += params.unload_h
                prev = c
    for i, k in seen.items():
        if k > 1:
            bad.append(f"item {i} loaded {k} times")
    n_loaded = len(seen)
    n_un = len(res["unassigned"])
    if n_loaded + n_un != len(P.items):
        bad.append(f"items loaded ({n_loaded}) + unassigned ({n_un}) != total ({len(P.items)})")
    for t in fleet:
        if t.count is not None:
            used = sum(1 for tr in res["trucks"] if tr["type"] == t.name)
            if used > t.count:
                bad.append(f"truck type {t.name}: {used} used > K_t={t.count}")
    return bad