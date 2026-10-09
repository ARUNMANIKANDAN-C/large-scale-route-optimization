"""STAGE 1 - cargo loading and truck selection.

Two methods:
  * heuristic_loading : best-fit-decreasing bin packing per destination, merging of nearby
                        part-loaded trucks (savings), then fleet-aware truck-type assignment.
                        Scales to all 4,635 items.
  * mip_loading       : the exact integer program of the report (PuLP / CBC). Use on a subset.

Both return  (trucks, unassigned, info)  where each truck is a dict:
   id, type, items (row positions in P.items), stops, L {stop: tightest deadline}, S (start time h),
   emin, emax, w, a, classes, R (farthest stop distance from depot)
"""
import collections
import heapq
import math
import time
import pulp

from common import available_types
import routing


# --------------------------------------------------------------------------------
# Caches and returns item attributes as contiguous python lists for fast indexing.
def _arr(P):
    if not hasattr(P, "_arr"):
        df = P.items
        P._arr = dict(w=df["w"].tolist(), a=df["a"].tolist(), e=df["e"].tolist(), l=df["l"].tolist(),
                      d=df["d"].tolist(), h=df["h"].tolist())
    return P._arr


# Creates a truck dictionary summarizing cargo weights, area, hazard classes, and earliest deadlines.
def make_truck(P, idxs, type_name=None):
    A = _arr(P)
    idxs = list(idxs)
    stops = list(dict.fromkeys(A["d"][i] for i in idxs))
    L = {}
    for i in idxs:
        c = A["d"][i]
        L[c] = min(L.get(c, 1e18), A["l"][i])
    return dict(
        id=None, type=type_name, items=idxs, stops=stops, L=L,
        S=max(A["e"][i] for i in idxs), emin=min(A["e"][i] for i in idxs), emax=max(A["e"][i] for i in idxs),
        w=sum(A["w"][i] for i in idxs), a=sum(A["a"][i] for i in idxs),
        classes=set(A["h"][i] for i in idxs),
    )


# Computes a theoretical lower-bound distance in km for a truck serving a given set of stops.
def route_lb(P, stops, ret):
    """Lower bound R_v on the route length of a truck that visits `stops` (km).
       single stop : D(0,c)            (x2 for a round trip)
       pair (c,c') : min(D0c+Dcc', D0c'+Dc'c)   (round trip: shortest closed tour through both)
       R_v = the largest of these bounds (exact for <= 2 stops)."""
    D, d0 = P.D, P.depot
    lb = max(D[d0][c] for c in stops) * (2 if ret else 1)
    for i in range(len(stops)):
        for j in range(i + 1, len(stops)):
            a, b = stops[i], stops[j]
            if ret:
                val = min(D[d0][a] + D[a][b] + D[b][d0], D[d0][b] + D[b][a] + D[a][d0])
            else:
                val = min(D[d0][a] + D[a][b], D[d0][b] + D[b][a])
            lb = max(lb, val)
    return lb


# Evaluates the Stage 1 objective cost combining estimated distance, stop charges, and truck fixed costs.
def stage1_objective(P, trucks, fleet, params):
    """Z1 = sum_v c_t R_v + f * stops + fixed truck cost,  R_v = route_lb(...)"""
    ft = {t.name: t for t in fleet}
    z = 0.0
    for tr in trucks:
        t = ft[tr["type"]]
        z += t.cost_km * route_lb(P, tr["stops"], params.return_to_depot)
        z += params.stop_cost * len(tr["stops"]) + t.fixed_cost
    return z


# ================================================================================
# HEURISTIC
# ================================================================================
# Assigns items to trucks using best-fit-decreasing packing, savings-based merges, and fleet-aware sizing.
def heuristic_loading(P, fleet, params):
    t_start = time.time()
    types = available_types(fleet)
    A = _arr(P)
    D, depot = P.D, P.depot
    vmin = min(t.speed for t in types)
    N = params.max_stops if params.allow_multi_stop else 1
    delta = params.delta_h
    unassigned = []

    # Checks if the given weight and area can fit into at least one available truck type.
    def fits_some(w, a):
        return any(t.weight_cap + 1e-9 >= w and t.area + 1e-9 >= a for t in types)

    # Checks if all hazard classes between two cargo collections are mutually compatible.
    def hz_ok(c1, c2):
        return all(params.compatible(x, y) for x in c1 for y in c2)


    # ---------------- 1. best-fit-decreasing packing per destination ------------
    by_dest = {}
    for i, c in enumerate(A["d"]):
        by_dest.setdefault(c, []).append(i)
    bins = []
    for dest in sorted(by_dest, key=lambda c: -D[depot][c]):
        dist0 = D[depot][dest]
        dbins = []
        for i in sorted(by_dest[dest], key=lambda i: (-A["w"][i], -A["a"][i])):
            if not fits_some(A["w"][i], A["a"][i]):
                unassigned.append((i, "too heavy/large for every allowed truck type"))
                continue
            if A["e"][i] + dist0 / vmin > A["l"][i] + 1e-9:
                unassigned.append((i, "deadline impossible"))
                continue
            best, best_w = None, -1
            for b in dbins:
                if not hz_ok(b["classes"], {A["h"][i]}):
                    continue
                if not fits_some(b["w"] + A["w"][i], b["a"] + A["a"][i]):
                    continue
                emax, emin = max(b["emax"], A["e"][i]), min(b["emin"], A["e"][i])
                if emax - emin > delta + 1e-9:
                    continue
                if emax + dist0 / vmin > min(b["L"][dest], A["l"][i]) + 1e-9:
                    continue
                if b["w"] > best_w:            # best fit = fullest feasible bin
                    best, best_w = b, b["w"]
            if best is None:
                dbins.append(make_truck(P, [i]))
            else:
                nb = make_truck(P, best["items"] + [i])
                best.update(nb)
        bins.extend(dbins)

    # ---------------- 2. merge part-loaded trucks of nearby cities ---------------
    def routing_best(b):
        L, S, stops = b["L"], b["S"], b["stops"]
        if len(stops) == 1:
            c = stops[0]
            km = D[depot][c] * (2 if params.return_to_depot else 1)
            if S + D[depot][c] / vmin > L[c] + 1e-9:
                return None
            return km, [c]
        if len(stops) <= 6:
            r = routing.best_route_enum(stops, S, L, D, depot, vmin, params.unload_h, params.return_to_depot)
        else:
            r = routing.heuristic_route(stops, S, L, D, depot, vmin, params.unload_h, params.return_to_depot)
        return r

    def unit_cost(b, km):
        best = None
        for t in types:
            if t.weight_cap + 1e-9 >= b["w"] and t.area + 1e-9 >= b["a"]:
                c = t.cost_km * km + t.fixed_cost
                if best is None or c < best:
                    best = c
        return best

    for b in bins:
        r = routing_best(b)
        if r is None:
            b["km"], b["order"] = float("inf"), b["stops"]
        else:
            b["km"], b["order"] = r
        b["cost"] = unit_cost(b, b["km"]) + params.stop_cost * len(b["stops"])

    if N >= 2 and len(bins) > 1:
        alive = {k: b for k, b in enumerate(bins)}
        nxt_id = len(bins)
        heap = []

        def merged(b1, b2):
            stops = list(dict.fromkeys(b1["stops"] + b2["stops"]))
            if len(stops) > N:
                return None
            if b1["w"] + b2["w"] > max(t.weight_cap for t in types) + 1e-9:
                return None
            if not fits_some(b1["w"] + b2["w"], b1["a"] + b2["a"]):
                return None
            if not hz_ok(b1["classes"], b2["classes"]):
                return None
            if max(b1["emax"], b2["emax"]) - min(b1["emin"], b2["emin"]) > delta + 1e-9:
                return None
            m = make_truck(P, b1["items"] + b2["items"])
            r = routing_best(m)
            if r is None:
                return None
            m["km"], m["order"] = r
            m["cost"] = unit_cost(m, m["km"]) + params.stop_cost * len(m["stops"])
            return m

        def push_pairs(k1, only_with=None):
            b1 = alive[k1]
            for k2 in (only_with if only_with is not None else list(alive)):
                if k2 == k1 or k2 not in alive:
                    continue
                if only_with is None and k2 < k1:
                    continue
                b2 = alive[k2]
                m = merged(b1, b2)
                if m is None:
                    continue
                sav = b1["cost"] + b2["cost"] - m["cost"]
                if sav > 1e-6:
                    heapq.heappush(heap, (-sav, k1, k2, m["items"].__len__(), id(m), m))

        keys = sorted(alive)
        for k in keys:
            push_pairs(k)
        while heap:
            nsav, k1, k2, _, _, m = heapq.heappop(heap)
            if k1 not in alive or k2 not in alive:
                continue
            del alive[k1]
            del alive[k2]
            alive[nxt_id] = m
            push_pairs(nxt_id, only_with=list(alive))
            nxt_id += 1
        bins = list(alive.values())

    # ---------------- 3. assign truck types respecting K_t ----------------------
    avail = {t.name: (math.inf if t.count is None else t.count) for t in types}
    tmap = {t.name: t for t in types}
    trucks = []
    queue = collections.deque(sorted(bins, key=lambda b: -b["km"]))
    while queue:
        b = queue.popleft()
        cand = [t for t in types if avail[t.name] > 0 and t.weight_cap + 1e-9 >= b["w"] and t.area + 1e-9 >= b["a"]]
        if cand:
            t = min(cand, key=lambda t: (t.cost_km * b["km"] + t.fixed_cost, t.weight_cap))
            avail[t.name] -= 1
            b["type"] = t.name
            trucks.append(b)
            continue
        # no truck left that can carry this load -> split the load into two
        if len(b["items"]) == 1:
            unassigned.append((b["items"][0], "no truck of a suitable type left (fleet exhausted)"))
            continue
        items = sorted(b["items"], key=lambda i: -A["w"][i])
        h1, h2 = items[0::2], items[1::2]
        for h in (h1, h2):
            nb = make_truck(P, h)
            r = routing_best(nb)
            if r is None:
                nb["km"], nb["order"] = float("inf"), nb["stops"]
            else:
                nb["km"], nb["order"] = r
            queue.appendleft(nb)

    for k, tr in enumerate(trucks):
        tr["id"] = f"T{k + 1:04d}"
    info = dict(method="heuristic", seconds=time.time() - t_start, status="done")
    return trucks, unassigned, info


# ================================================================================
# EXACT MIP
# ================================================================================
MAX_EXACT_ITEMS = 60


# Formulates and solves an exact mixed-integer programming (MIP) model to optimally load items into trucks.
def mip_loading(P, fleet, params, time_limit=60, forbidden_pairs=(), margin=2, max_items=MAX_EXACT_ITEMS):
    t_start = time.time()
    types = available_types(fleet)
    A = _arr(P)
    D, depot = P.D, P.depot
    n = len(A["w"])
    unassigned = []
    if n > max_items:
        return [], [], dict(method="exact", seconds=0.0, status="too large",
                            message=f"{n} items is more than the {max_items} the exact model accepts")

    # heuristic plan = warm start (guarantees the MIP starts from a feasible incumbent)
    h_trucks, h_un, _ = heuristic_loading(P, fleet, params)
    h_count = {}
    for tr in h_trucks:
        h_count[tr["type"]] = h_count.get(tr["type"], 0) + 1

    # candidate trucks per type (upper bound on how many could be needed)
    sw, sa = sum(A["w"]), sum(A["a"])
    V = []
    for t in types:
        need = max(math.ceil(sw / t.weight_cap), math.ceil(sa / t.area)) + margin
        need = max(min(need, n), h_count.get(t.name, 0))
        cnt = need if t.count is None else min(t.count, need)
        V += [(t, k) for k in range(max(cnt, 0))]
    if not V:
        return [], [(i, "no trucks") for i in range(n)], dict(method="exact", status="no trucks", seconds=0)

    # allowed (item, truck) pairs: item fits the truck type and can reach its city in time
    allowed = {}
    ok_item = [False] * n
    for i in range(n):
        for v, (t, k) in enumerate(V):
            if A["w"][i] <= t.weight_cap + 1e-9 and A["a"][i] <= t.area + 1e-9 and \
                    A["e"][i] + D[depot][A["d"][i]] / t.speed <= A["l"][i] + 1e-9:
                allowed[(i, v)] = True
                ok_item[i] = True
        if not ok_item[i]:
            unassigned.append((i, "cannot be carried / delivered by any allowed truck"))
    items = [i for i in range(n) if ok_item[i]]
    if not items:
        return [], unassigned, dict(method="exact", status="nothing to load", seconds=0)

    cities = sorted(set(A["d"][i] for i in items))
    classes = sorted(set(A["h"][i] for i in items))
    maxl = max(A["l"][i] for i in items)
    theta = maxl + max(D[depot][c] for c in cities) / min(t.speed for t in types) + 1.0
    ret = 2 if params.return_to_depot else 1

    prob = pulp.LpProblem("stage1_loading", pulp.LpMinimize)
    x = {(i, v): pulp.LpVariable(f"x_{i}_{v}", cat="Binary") for (i, v) in allowed}
    z = {v: pulp.LpVariable(f"z_{v}", cat="Binary") for v in range(len(V))}
    q = {(v, k): pulp.LpVariable(f"q_{v}_{ci}", cat="Binary") for v in range(len(V))
         for ci, k in enumerate(classes)}
    u = {(v, c): pulp.LpVariable(f"u_{v}_{ci}", cat="Binary") for v in range(len(V)) for ci, c in enumerate(cities)}
    R = {v: pulp.LpVariable(f"R_{v}", lowBound=0) for v in range(len(V))}
    emax_all = max(A["e"][i] for i in items)
    S = {v: pulp.LpVariable(f"S_{v}", lowBound=0, upBound=emax_all) for v in range(len(V))}
    Emax = {v: pulp.LpVariable(f"Emax_{v}", lowBound=0, upBound=emax_all) for v in range(len(V))}
    Emin = {v: pulp.LpVariable(f"Emin_{v}", lowBound=0, upBound=emax_all) for v in range(len(V))}

    # objective  Z1
    prob += pulp.lpSum(V[v][0].cost_km * R[v] + V[v][0].fixed_cost * z[v] for v in range(len(V))) + \
        params.stop_cost * pulp.lpSum(u.values())

    by_v = {}
    for (i, v) in allowed:
        by_v.setdefault(v, []).append(i)
    for i in items:
        prob += pulp.lpSum(x[(i, v)] for v in range(len(V)) if (i, v) in x) == 1                # (1)
    for v, (t, k) in enumerate(V):
        its = by_v.get(v, [])
        prob += pulp.lpSum(A["w"][i] * x[(i, v)] for i in its) <= t.weight_cap * z[v]             # (2)
        prob += pulp.lpSum(A["a"][i] * x[(i, v)] for i in its) <= t.area * z[v]                   # (3)
        for i in its:
            prob += x[(i, v)] <= q[(v, A["h"][i])]                                                # (4a/4b) class present
            prob += x[(i, v)] <= z[v]                                                             # (5)
            prob += x[(i, v)] <= u[(v, A["d"][i])]                                                # (7)
            prob += R[v] >= ret * D[depot][A["d"][i]] * x[(i, v)]                                 # (9) farthest stop
            prob += S[v] >= A["e"][i] * x[(i, v)]                                                 # (10)
            prob += Emax[v] >= A["e"][i] * x[(i, v)]                                              # (11a)
            prob += Emin[v] <= A["e"][i] + (emax_all - A["e"][i]) * (1 - x[(i, v)])               # (11b) tight big-M
            th_i = max(0.0, emax_all + D[depot][A["d"][i]] / t.speed - A["l"][i])
            prob += S[v] + D[depot][A["d"][i]] / t.speed <= A["l"][i] + th_i * (1 - x[(i, v)])    # (12) tight big-M
        prob += Emax[v] - Emin[v] <= params.delta_h                                               # (11c)
        prob += pulp.lpSum(u[(v, c)] for c in cities) <= params.max_stops * z[v]                  # (8)
        for a_c in range(len(cities)):                                                            # (9') pair detour
            for b_c in range(a_c + 1, len(cities)):
                ca, cb = cities[a_c], cities[b_c]
                rho = route_lb(P, [ca, cb], params.return_to_depot)
                prob += R[v] >= rho * (u[(v, ca)] + u[(v, cb)] - 1)
        for a_i, ka in enumerate(classes):                                                        # hazard rule
            for kb in classes[a_i + 1:]:
                if not params.compatible(ka, kb):
                    prob += q[(v, ka)] + q[(v, kb)] <= 1
        for (c1, c2) in forbidden_pairs:                                                          # feedback cuts
            if (v, c1) in u and (v, c2) in u:
                prob += u[(v, c1)] + u[(v, c2)] <= 1
    # (6) K_t is enforced by the number of candidate trucks of each type.
    # valid cuts: total capacity of used trucks must cover the cargo
    prob += pulp.lpSum(V[v][0].weight_cap * z[v] for v in range(len(V))) >= sum(A["w"][i] for i in items)
    prob += pulp.lpSum(V[v][0].area * z[v] for v in range(len(V))) >= sum(A["a"][i] for i in items)
    # symmetry breaking: trucks of one type are interchangeable -> truck k may carry item i only if
    # truck k-1 of the same type already carries a lower-numbered item
    for ti, t in enumerate(types):
        vs = [v for v, (tt, k) in enumerate(V) if tt is t]
        for a_v, b_v in zip(vs, vs[1:]):
            prob += z[a_v] >= z[b_v]
            for i in by_v.get(b_v, []):
                prob += x[(i, b_v)] <= pulp.lpSum(x[(j, a_v)] for j in by_v.get(a_v, []) if j < i)

    # ---- warm start from the heuristic solution ----
    warm = not h_un and not forbidden_pairs
    if warm:
        slots, used_slot = {}, {}
        for v, (t, k) in enumerate(V):
            slots.setdefault(t.name, []).append(v)
        assign = {}
        for tr in h_trucks:
            lst = slots.get(tr["type"], [])
            pos = used_slot.get(tr["type"], 0)
            if pos >= len(lst):
                warm = False
                break
            used_slot[tr["type"]] = pos + 1
            assign[lst[pos]] = tr
        if warm:
            for var in list(x.values()) + list(z.values()) + list(q.values()) + list(u.values()) + \
                    list(R.values()) + list(S.values()) + list(Emax.values()) + list(Emin.values()):
                var.setInitialValue(0)
            for v, tr in assign.items():
                for i in tr["items"]:
                    if (i, v) in x:
                        x[(i, v)].setInitialValue(1)
                    else:
                        warm = False
                z[v].setInitialValue(1)
                for k in tr["classes"]:
                    q[(v, k)].setInitialValue(1)
                for c in tr["stops"]:
                    u[(v, c)].setInitialValue(1)
                R[v].setInitialValue(route_lb(P, tr["stops"], params.return_to_depot))
                S[v].setInitialValue(tr["S"])
                Emax[v].setInitialValue(tr["emax"])
                Emin[v].setInitialValue(tr["emin"])
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=4, warmStart=warm))
    secs = time.time() - t_start
    st = pulp.LpStatus[prob.status]
    has_sol = prob.sol_status in (1, 2) and all(v.value() is not None for v in list(z.values())[:1])
    if st == "Infeasible" or not has_sol:
        return [], unassigned + [(i, "MIP found no solution") for i in items], \
            dict(method="exact", status="infeasible" if st == "Infeasible" else "no solution in time limit",
                 seconds=secs)
    trucks = []
    for v, (t, k) in enumerate(V):
        its = [i for i in by_v.get(v, []) if x[(i, v)].value() and x[(i, v)].value() > 0.5]
        if its:
            tr = make_truck(P, its, t.name)
            trucks.append(tr)
    for k, tr in enumerate(trucks):
        tr["id"] = f"T{k + 1:04d}"
    proven = (prob.sol_status == 1) and secs < time_limit * 0.97
    info = dict(method="exact", seconds=secs, objective=pulp.value(prob.objective),
                status="optimal" if proven else "best found within time limit")
    return trucks, unassigned, info
