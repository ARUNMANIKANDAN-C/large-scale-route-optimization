"""Set-partitioning master problem over overlapping candidate clusters.

This is the key new Stage 1.  The candidate generator proposes alternative loading
patterns; this MIP chooses exactly one pattern containing every item.  Each pattern is
then assigned a truck type.  The result is globally optimal *within the generated
candidate pool* when CBC proves optimality.
"""
import math
import time
import pulp
from candidate_clusters import generate_candidates
from stage1 import make_truck


def solve_master(P, fleet, params, candidates=None, time_limit=60, max_clusters_per_item=4):
    t0 = time.time()
    types = [t for t in fleet if t.count is None or t.count > 0]
    if candidates is None:
        candidates = generate_candidates(P, fleet, params, max_clusters_per_item=max_clusters_per_item)
    n = len(P.items)
    if not candidates:
        return [], [(i, "no candidate cluster") for i in range(n)], {
            "method":"cluster-master", "status":"no candidates", "seconds":time.time()-t0,
            "candidate_count":0}

    A = P.items
    compatible = []
    for p, c in enumerate(candidates):
        for t in types:
            if c.weight <= t.weight_cap + 1e-9 and c.area <= t.area + 1e-9:
                # A candidate with a failed route is still selectable, but receives a
                # strong penalty so the feedback loop can replace it if possible.
                route_penalty = 1e5 if not c.route_feasible else 0.0
                cost = t.cost_km * c.estimated_km + t.fixed_cost + params.stop_cost * len(c.stops) + route_penalty
                compatible.append((p, t.name, cost))
    if not compatible:
        return [], [(i, "no truck type can carry any candidate") for i in range(n)], {
            "method":"cluster-master", "status":"infeasible", "seconds":time.time()-t0,
            "candidate_count":len(candidates)}

    prob = pulp.LpProblem("cluster_master", pulp.LpMinimize)
    y = {(p, tn): pulp.LpVariable(f"y_{p}_{j}", cat="Binary") for j,(p,tn,_) in enumerate(compatible)}
    # Re-key by pattern/type for easier constraints.
    cost = {(p,tn): c for p,tn,c in compatible}
    prob += pulp.lpSum(cost[k] * v for k,v in y.items())

    by_item = {i: [] for i in range(n)}
    by_type = {t.name: [] for t in types}
    for key, var in y.items():
        p, tn = key
        for i in candidates[p].items:
            by_item[i].append(var)
        by_type[tn].append(var)

    # Every item is in exactly one selected candidate.
    for i in range(n):
        if not by_item[i]:
            return [], [(i, "item has no generated candidate")], {
                "method":"cluster-master", "status":"incomplete candidate pool", "seconds":time.time()-t0,
                "candidate_count":len(candidates)}
        prob += pulp.lpSum(by_item[i]) == 1

    # A candidate can be assigned to at most one truck type.
    for p in range(len(candidates)):
        vars_p = [v for (pp, _),v in y.items() if pp == p]
        prob += pulp.lpSum(vars_p) <= 1

    # Fleet availability.
    for t in types:
        if t.count is not None:
            prob += pulp.lpSum(by_type[t.name]) <= t.count

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=4))
    secs = time.time() - t0
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Integer Feasible"):
        return [], [(i, "cluster master could not find a complete selection") for i in range(n)], {
            "method":"cluster-master", "status":status.lower(), "seconds":secs,
            "candidate_count":len(candidates)}

    trucks = []
    for (p, tn), var in y.items():
        if var.value() and var.value() > 0.5:
            c = candidates[p]
            tr = make_truck(P, c.items, tn)
            tr["candidate_id"] = c.id
            tr["candidate_route"] = c.route_order
            tr["candidate_route_feasible"] = c.route_feasible
            tr["candidate_estimated_km"] = c.estimated_km
            trucks.append(tr)
    for k,tr in enumerate(trucks):
        tr["id"] = f"T{k+1:04d}"
    proven = status == "Optimal"
    info = {"method":"cluster-master", "status":"optimal in candidate pool" if proven else "best found within time limit",
            "seconds":secs, "candidate_count":len(candidates), "objective":pulp.value(prob.objective),
            "candidate_optimal":proven}
    return trucks, [], info
