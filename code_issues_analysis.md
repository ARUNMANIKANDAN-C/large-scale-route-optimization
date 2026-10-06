# Code Issues Analysis

## Issues Already Fixed ✅

| # | File | Issue | Status |
|---|------|-------|--------|
| 1 | [`app.py`](file:///d:/7th%20project/operational%20rescarch/code/app.py#L17) / [`pipeline.py`](file:///d:/7th%20project/operational%20rescarch/code/pipeline.py#L12) | `precheck` was an external PyPI package whose API didn't match usage (`TypeError: 'module' object is not callable`) | ✅ Replaced with local [`precheck.py`](file:///d:/7th%20project/operational%20rescarch/code/precheck.py) |
| 2 | [`pyproject.toml`](file:///d:/7th%20project/operational%20rescarch/code/pyproject.toml#L12) | `pulp>=4.0.0` — PuLP 4.x removed the `cat=` keyword from `LpVariable`, breaking all MIP formulations | ✅ Pinned to `pulp>=2.8,<4` |

---

## Remaining Issues

### 🔴 Critical (will crash or produce wrong results)

#### 3. [`stage1.py:156`](file:///d:/7th%20project/operational%20rescarch/code/stage1.py#L154-L156) — Unpacking `None` crashes heuristic
When `routing_best(b)` returns `None` (no feasible route), the next line does:
```python
b["km"], b["order"] = r   # r is None → TypeError: cannot unpack non-iterable NoneType
```
This will crash for any truck whose stops can't be visited in deadline order.

**Fix:** Guard the unpacking:
```python
if r is None:
    b["km"], b["order"] = float("inf"), b["stops"]
else:
    b["km"], b["order"] = r
```

#### 4. [`stage1.py:235`](file:///d:/7th%20project/operational%20rescarch/code/stage1.py#L234-L236) — Same `None` crash in repair-split fallback
Identical bug during fleet-exhaustion splitting:
```python
nb["km"], nb["order"] = r   # r could be None
```

#### 5. [`master.py:42`](file:///d:/7th%20project/operational%20rescarch/code/master.py#L42) — Still uses `cat="Binary"` (PuLP 3.x uses `cat="Binary"` but PuLP 4.x does not)
With `pulp>=2.8,<4` pinned this now works, but the code is fragile. If PuLP is ever upgraded, every `cat=` call in [`master.py`](file:///d:/7th%20project/operational%20rescarch/code/master.py), [`stage1.py`](file:///d:/7th%20project/operational%20rescarch/code/stage1.py), and [`routing.py`](file:///d:/7th%20project/operational%20rescarch/code/routing.py) will break.

> [!NOTE]
> This is now guarded by the version pin, but worth documenting.

#### 6. [`stage1.py:84`](file:///d:/7th%20project/operational%20rescarch/code/stage1.py#L84) — Slowest speed used for deadline check instead of truck-specific speed
```python
vmin = min(t.speed for t in types)
```
The heuristic uses the **minimum** speed across all truck types for the deadline check, which is correct for a conservative feasibility test, but then the same `vmin` is passed into `routing_best` for actual route evaluation. This means a truck of a faster type gets its route quality evaluated at the speed of the slowest type, leading to **suboptimal routing decisions** during merging.

---

### 🟡 Medium (logic / correctness concerns)

#### 7. [`data.py:71-72`](file:///d:/7th%20project/operational%20rescarch/code/data.py#L71-L72) — `KeyError` if distance is missing
```python
def dist(self, a, b):
    return self.D[a][b]
```
No fallback. If a city pair is missing from the distance matrix, this crashes with `KeyError` rather than returning a sensible error or `inf`.

#### 8. [`pipeline.py:65`](file:///d:/7th%20project/operational%20rescarch/code/pipeline.py#L65) — Precheck result is never used to halt
```python
pre = precheck(P, fleet, params)
log += pre["messages"]
```
Even if `pre["ok"]` is `False`, the pipeline continues to optimise. The precheck result is only informational — the pipeline never stops or adjusts when items are infeasible.

#### 9. [`candidate_clusters.py:27-32`](file:///d:/7th%20project/operational%20rescarch/code/candidate_clusters.py#L27-L32) — `_arr` cache collision with `stage1._arr`
Both `candidate_clusters.py` and `stage1.py` define their own `_arr(P)` that cache on `P._arr`. If both modules are called on the same `Problem` object, the first one to run wins. They produce the same structure, so it works *by accident*, but it's fragile — any divergence in the dict keys would silently corrupt the cache.

#### 10. [`app.py:96-97`](file:///d:/7th%20project/operational%20rescarch/code/app.py#L96-L97) — File handles are never closed
```python
order_blobs = (open(order_path, "rb").read(),)
```
File handles are opened but never closed. Should use `with` or at least call `.close()`.

#### 11. [`app.py:188`](file:///d:/7th%20project/operational%20rescarch/code/app.py#L188) — Fragile mode mapping
```python
mode = {"Full": "full", "Load": "loading", "Rout": "routing"}[mode_label[:4]]
```
Maps by the first 4 characters of the radio label. If the label text ever changes, this silently breaks with a `KeyError`.

#### 12. [`routing.py:127`](file:///d:/7th%20project/operational%20rescarch/code/routing.py#L127) — `maxt` can trigger `KeyError`
```python
maxt = max(D[a][b] for a in nodes for b in nodes) / speed
```
If the distance matrix doesn't contain all node-to-node pairs, this crashes. Should at least skip `a == b`.

#### 13. [`pipeline.py:243`](file:///d:/7th%20project/operational%20rescarch/code/pipeline.py#L243) — `sorted()` on route may differ from `stops`
```python
if sorted(tr["route"]) != sorted(stops):
```
Comparing sorted lists of strings works, but `stops` is a set, so `sorted(stops)` is fine. However, this validation silently passes if a route visits a superset of stops (can't happen in practice, but the validator doesn't catch it separately).

---

### 🟢 Minor (style / robustness / performance)

#### 14. [`stage1.py:218`](file:///d:/7th%20project/operational%20rescarch/code/stage1.py#L218) — `queue.pop(0)` is O(n)
```python
b = queue.pop(0)
```
Using `list.pop(0)` in a loop is O(n²). Use `collections.deque` for O(1) popleft.

#### 15. [`common.py:6-7`](file:///d:/7th%20project/operational%20rescarch/code/common.py#L6-L7) — `Optional` / `Set` / `FrozenSet` imports are unnecessary on Python 3.14
With `requires-python = ">=3.14"`, you can use `int | None`, `set[frozenset[str]]` etc. directly. Not a bug, just legacy typing.

#### 16. [`app.py:216`](file:///d:/7th%20project/operational%20rescarch/code/app.py#L216) — Variable `l` shadows built-in
```python
for l in res["log"]:
```
`l` shadows the built-in `list`. Use `msg` or `line` instead.

#### 17. [`pyproject.toml:6`](file:///d:/7th%20project/operational%20rescarch/code/pyproject.toml#L6) — `requires-python = ">=3.14"` is very restrictive
Python 3.14 is bleeding edge. Unless you specifically need 3.14 features, `>=3.10` or `>=3.12` would be more practical.

#### 18. No `__pycache__` in `.gitignore` check
There's a `__pycache__` directory but it's unclear if `.gitignore` covers it properly (no commits yet to verify).

---

## Summary

| Severity | Count |
|----------|-------|
| ✅ Already fixed | 2 |
| 🔴 Critical | 4 (issues 3-6) |
| 🟡 Medium | 7 (issues 7-13) |
| 🟢 Minor | 5 (issues 14-18) |

> [!IMPORTANT]
> **Issues #3 and #4** (the `None` unpacking crashes in `stage1.py`) are the most likely to hit you next — they'll crash whenever a truck has stops whose deadlines can't all be met. These should be fixed first.
