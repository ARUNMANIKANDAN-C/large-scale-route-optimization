"""STAGE 2 - route planning for ONE loaded truck.

Given the stops C_v, start time S_v and stop deadlines L_vc (from Stage 1) find the visiting
order that minimises distance (cost = c_t * distance) while arriving at every stop before its
deadline.   Routes are one-way (end at last stop) unless params.return_to_depot is True.

Two methods:
  * heuristic : nearest-neighbour / earliest-deadline start + 2-opt improvement
  * exact     : the MIP of the report (MTZ-style time constraints), solved with PuLP/CBC
"""
import itertools
import pulp


# ----------------------------------------------------------------------------------
def eval_order(order, start, L, D, depot, speed, M, ret):
    """Evaluate a visiting order. Returns (distance_km, feasible, arrivals dict)."""
    t, prev, dist, feas = start, depot, 0.0, True
    arr = {}
    for c in order:
        d = D[prev][c]
        dist += d
        t += d / speed
        arr[c] = t
        if t > L[c] + 1e-7:
            feas = False
        t += M
        prev = c
    if ret and order:
        dist += D[prev][depot]
    return dist, feas, arr


def best_route_enum(stops, start, L, D, depot, speed, M, ret):
    """Exact by enumeration (use only for a handful of stops)."""
    best = None
    for perm in itertools.permutations(stops):
        dist, feas, _ = eval_order(perm, start, L, D, depot, speed, M, ret)
        if feas and (best is None or dist < best[0] - 1e-9):
            best = (dist, list(perm))
    return best  # None if no feasible order


def heuristic_route(stops, start, L, D, depot, speed, M, ret):
    """Nearest neighbour + earliest-deadline-first starts, then 2-opt. Returns (dist, order) or None."""
    stops = list(stops)
    if len(stops) <= 1:
        d, f, _ = eval_order(stops, start, L, D, depot, speed, M, ret)
        return (d, stops) if f else None

    def ev(o):
        d, f, _ = eval_order(o, start, L, D, depot, speed, M, ret)
        return d, f

    # nearest neighbour
    rem, cur, nn = set(stops), depot, []
    while rem:
        nxt = min(rem, key=lambda c: D[cur][c])
        nn.append(nxt)
        rem.remove(nxt)
        cur = nxt
    edf = sorted(stops, key=lambda c: L[c])
    cands = [(ev(o), o) for o in (nn, edf)]
    feas = [(d, o) for (d, f), o in cands if f]
    if not feas:
        fixed = _repair(edf, start, L, D, depot, speed, M, ret)
        if fixed is None:
            return None
        feas = [(ev(fixed)[0], fixed)]
    bd, best = min(feas, key=lambda x: x[0])
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                d, f = ev(cand)
                if f and d < bd - 1e-9:
                    best, bd, improved = cand, d, True
    return bd, best


def _tardiness(order, start, L, D, depot, speed, M):
    t, prev, tot = start, depot, 0.0
    for c in order:
        t += D[prev][c] / speed
        tot += max(0.0, t - L[c])
        t += M
        prev = c
    return tot


def _repair(order, start, L, D, depot, speed, M, ret):
    """Local search (move one stop to another position) that reduces total lateness to 0."""
    order = list(order)
    cur = _tardiness(order, start, L, D, depot, speed, M)
    while cur > 1e-9:
        best, bo = cur, None
        for i in range(len(order)):
            for j in range(len(order)):
                if i == j:
                    continue
                cand = order[:i] + order[i + 1:]
                cand.insert(j, order[i])
                tv = _tardiness(cand, start, L, D, depot, speed, M)
                if tv < best - 1e-9:
                    best, bo = tv, cand
        if bo is None:
            break
        order, cur = bo, best
    if cur > 1e-9 and len(order) <= 7:          # tiny cases: enumerate
        r = best_route_enum(order, start, L, D, depot, speed, M, ret)
        return r[1] if r else None
    return order if cur <= 1e-9 else None


def mip_route(stops, start, L, D, depot, speed, M, ret, time_limit=20):
    """Exact MIP (equations of Stage 2).  Returns (dist, order) or None if infeasible."""
    stops = list(stops)
    if len(stops) == 1:
        return heuristic_route(stops, start, L, D, depot, speed, M, ret)
    nodes = [depot] + stops
    prob = pulp.LpProblem("route", pulp.LpMinimize)
    y = {(a, b): pulp.LpVariable(f"y_{i}_{j}", cat="Binary")
         for i, a in enumerate(nodes) for j, b in enumerate(nodes)
         if a != b and (b != depot or ret)}
    maxt = max((D.get(a, {}).get(b, 0) for a in nodes for b in nodes if a != b), default=0) / speed
    theta = max(L.values()) + M + maxt - start + 1.0
    T = {c: pulp.LpVariable(f"T_{k}", lowBound=start, upBound=L[c]) for k, c in enumerate(stops)}
    prob += pulp.lpSum(D[a][b] * v for (a, b), v in y.items())
    prob += pulp.lpSum(y[(depot, c)] for c in stops) == 1                       # (1) leave depot once
    for c in stops:
        prob += pulp.lpSum(y[(a, c)] for a in nodes if a != c) == 1               # (2) enter each stop once
        outs = [y[(c, b)] for b in nodes if b != c and (b != depot or ret)]
        if ret:
            prob += pulp.lpSum(outs) == 1                                         # (3) round trip
        else:
            prob += pulp.lpSum(outs) <= 1                                         # (3) last stop not left
    if ret:
        prob += pulp.lpSum(y[(c, depot)] for c in stops) == 1
    for (a, b), v in y.items():                                                   # (4) time propagation
        if b == depot:
            continue
        ta = start if a == depot else T[a]
        mm = 0.0 if a == depot else M
        prob += T[b] >= ta + mm + D[a][b] / speed - theta * (1 - v)
    # (5) deadlines are the upper bounds of T
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    if pulp.LpStatus[prob.status] != "Optimal" or prob.sol_status not in (1, 2):
        return None
    nxt = {a: b for (a, b), v in y.items() if v.value() and v.value() > 0.5 and a != depot}
    first = [c for c in stops if y[(depot, c)].value() > 0.5][0]
    order, cur = [first], first
    while cur in nxt and nxt[cur] != depot and len(order) < len(stops):
        cur = nxt[cur]
        order.append(cur)
    if len(order) != len(stops):
        return None
    d, f, _ = eval_order(order, start, L, D, depot, speed, M, ret)
    return (d, order) if f else None


# ----------------------------------------------------------------------------------
def route_truck(P, truck, ttype, params, method="heuristic", mip_limit=9, time_limit=20):
    """Fill truck['route'], ['route_km'], ['arrivals'], ['route_ok'], ['route_cost'], ['route_method']."""
    D, depot = P.D, P.depot
    stops, start, L = truck["stops"], truck["S"], truck["L"]
    used = method
    if method == "exact" and 1 < len(stops) <= mip_limit:
        r = mip_route(stops, start, L, D, depot, ttype.speed, params.unload_h, params.return_to_depot, time_limit)
    else:
        used = "heuristic"
        r = heuristic_route(stops, start, L, D, depot, ttype.speed, params.unload_h, params.return_to_depot)
    if r is None:
        # no deadline-feasible order: report the distance-best order as infeasible
        order = sorted(stops, key=lambda c: L[c])
        dist, _, arr = eval_order(order, start, L, D, depot, ttype.speed, params.unload_h, params.return_to_depot)
        truck.update(route=order, route_km=dist, arrivals=arr, route_ok=False)
    else:
        dist, order = r
        _, _, arr = eval_order(order, start, L, D, depot, ttype.speed, params.unload_h, params.return_to_depot)
        truck.update(route=order, route_km=dist, arrivals=arr, route_ok=True)
    truck["route_cost"] = ttype.cost_km * truck["route_km"]
    truck["route_method"] = used
    return truck
