"""Scalable overlapping candidate-pattern generation.

Each item can occur in at most K candidate clusters.  The generator is deliberately
cheap: it builds destination-local packing patterns, deadline variants and geographically
adjacent merged patterns.  The master optimizer, not this module, decides which pattern
survives.
"""
from dataclasses import dataclass
import routing


@dataclass
class CandidateCluster:
    id: str
    items: list
    stops: list
    estimated_km: float
    route_order: list
    route_feasible: bool
    weight: float
    area: float
    classes: set
    emin: float
    emax: float


def _arr(P):
    if not hasattr(P, "_arr"):
        df = P.items
        P._arr = dict(w=df.w.tolist(), a=df.a.tolist(), e=df.e.tolist(), l=df.l.tolist(),
                      d=df.d.tolist(), h=df.h.tolist())
    return P._arr


def _compatible(classes, params):
    c = list(classes)
    return all(params.compatible(c[i], c[j]) for i in range(len(c)) for j in range(i + 1, len(c)))


def _route(P, items, params, speed):
    A = _arr(P)
    stops = list(dict.fromkeys(A["d"][i] for i in items))
    L = {c: min(A["l"][i] for i in items if A["d"][i] == c) for c in stops}
    start = max(A["e"][i] for i in items)
    if len(stops) <= 7:
        return routing.best_route_enum(stops, start, L, P.D, P.depot, speed,
                                       params.unload_h, params.return_to_depot)
    return routing.heuristic_route(stops, start, L, P.D, P.depot, speed,
                                   params.unload_h, params.return_to_depot)


def make_candidate(P, items, cid, params, speed, allow_infeasible=True):
    A = _arr(P)
    items = list(dict.fromkeys(items))
    if not items:
        return None
    stops = list(dict.fromkeys(A["d"][i] for i in items))
    if len(stops) > params.max_stops:
        return None
    w = sum(A["w"][i] for i in items)
    a = sum(A["a"][i] for i in items)
    classes = set(A["h"][i] for i in items)
    if not _compatible(classes, params):
        return None
    emin = min(A["e"][i] for i in items)
    emax = max(A["e"][i] for i in items)
    if emax - emin > params.delta_h + 1e-9:
        return None
    r = _route(P, items, params, speed)
    if r is None:
        if not allow_infeasible:
            return None
        L = {c: min(A["l"][i] for i in items if A["d"][i] == c) for c in stops}
        start = emax
        order = sorted(stops, key=lambda c: L[c])
        km, _, _ = routing.eval_order(order, start, L, P.D, P.depot, speed,
                                      params.unload_h, params.return_to_depot)
        feasible = False
    else:
        km, order = r
        feasible = True
    return CandidateCluster(cid, items, stops, km, list(order), feasible,
                            w, a, classes, emin, emax)


def _pack_group(indices, A, max_w, max_a):
    """Best-fit-ish packing of one destination into capacity-safe item groups."""
    bins = []
    for i in sorted(indices, key=lambda x: (-A["w"][x], -A["a"][x])):
        placed = None
        best_slack = None
        for b in bins:
            nw = b["w"] + A["w"][i]
            na = b["a"] + A["a"][i]
            if nw <= max_w + 1e-9 and na <= max_a + 1e-9:
                slack = (max_w - nw) / max_w + (max_a - na) / max_a
                if best_slack is None or slack < best_slack:
                    best_slack, placed = slack, b
        if placed is None:
            placed = {"items": [], "w": 0.0, "a": 0.0}
            bins.append(placed)
        placed["items"].append(i)
        placed["w"] += A["w"][i]
        placed["a"] += A["a"][i]
    return [b["items"] for b in bins]


def generate_candidates(P, fleet, params, max_clusters_per_item=4, max_candidates=12000):
    A = _arr(P)
    n = len(P.items)
    types = [t for t in fleet if t.count is None or t.count > 0]
    if n == 0 or not types:
        return []
    max_w = max(t.weight_cap for t in types)
    max_a = max(t.area for t in types)
    speed = min(t.speed for t in types)
    by_dest = {}
    for i, d in enumerate(A["d"]):
        by_dest.setdefault(d, []).append(i)
    dests = list(by_dest)
    depot = P.depot
    nearest = {}
    for d in dests:
        nearest[d] = sorted((x for x in dests if x != d),
                            key=lambda x: P.D.get(d, {}).get(x, P.D.get(x, {}).get(d, 1e9)))

    raw = []
    seen = set()

    def add(items):
        key = tuple(sorted(set(items)))
        if not key or key in seen:
            return
        c = make_candidate(P, key, f"C{len(raw)+1:06d}", params, speed)
        if c is not None:
            seen.add(key)
            raw.append(c)

    # One local packing family per destination.
    local_bins = {}
    for d in dests:
        local_bins[d] = _pack_group(by_dest[d], A, max_w, max_a)
        for b in local_bins[d]:
            add(b)

    # Deadline variants: same destination, but packing in deadline order. This gives
    # the master an alternative when the weight-first pattern misses a tight deadline.
    for d in dests:
        idx = sorted(by_dest[d], key=lambda i: (A["l"][i], -A["w"][i]))
        for b in _pack_group(idx, A, max_w, max_a):
            add(b)

    # Geographic alternatives: merge each local bin with the nearest destination's
    # compatible local bin.  Do not mutate the local bins; these are overlapping options.
    for d in dests:
        for nd in nearest[d][:max(0, params.max_stops - 1)]:
            if nd not in local_bins:
                continue
            for b1 in local_bins[d][:max_clusters_per_item]:
                # Pick the nearest/heaviest compatible bin from the neighboring city.
                for b2 in local_bins[nd][:max_clusters_per_item]:
                    trial = b1 + b2
                    c = make_candidate(P, trial, "tmp", params, speed, allow_infeasible=False)
                    if c is not None:
                        add(trial)
                        break

    # Keep at most K incident patterns per item.  Prefer route-feasible and short patterns.
    raw.sort(key=lambda c: (not c.route_feasible, c.estimated_km, len(c.items)))
    count = [0] * n
    selected = []
    for c in raw:
        if all(count[i] < max_clusters_per_item for i in c.items):
            selected.append(c)
            for i in c.items:
                count[i] += 1
        if len(selected) >= max_candidates:
            break

    # Guarantee coverage. A singleton is always the cleanest fallback candidate.
    for i in range(n):
        if count[i] == 0:
            c = make_candidate(P, [i], f"C{len(selected)+1:06d}", params, speed, allow_infeasible=False)
            if c is not None:
                selected.append(c)
                count[i] = 1

    return selected
