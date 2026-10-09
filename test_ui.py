"""Smoke test for app.py using Streamlit's AppTest (run: python test_app.py)."""
import time
from streamlit.testing.v1 import AppTest

STAGE1_HEUR = "Heuristic (best-fit packing + merging)"
STAGE2_HEUR = "Heuristic (nearest neighbour + 2-opt)"
STAGE1_EXACT = "Exact (integer program, CBC)"
STAGE2_EXACT = "Exact (MIP, CBC)"


# Finds and returns the first button in the Streamlit AppTest matching the given label.
def button(label):
    return [b for b in at.button if b.label == label][0]


# Finds a selectbox in the Streamlit AppTest by label and selects the given option.
def pick(label, option):
    for sb in at.selectbox:
        if sb.label == label:
            sb.select(option)


# Executes a test action callback, measures execution time, and logs exceptions or errors.
def step(name, fn):
    t = time.time()
    fn()
    print(f"{name}: {time.time() - t:.1f}s "
          f"exc={[e.value for e in at.exception]} err={[e.value for e in at.error]}", flush=True)


at = AppTest.from_file("app.py", default_timeout=300)
step("load", lambda: at.run())
step("stats", lambda: button("Generate statistics").click().run())

# 1) full pipeline, heuristic for both stages (all items)
pick("Stage 1 method", STAGE1_HEUR)
pick("Stage 2 method", STAGE2_HEUR)
at.run()
step("optimize full/heuristic", lambda: button("Run optimization").click().run())

# 2) restrict scope to 12 items for the exact model and the comparison tab
[n for n in at.number_input if n.label.startswith("Max items")][0].set_value(12)
at.run()
step("compare small (heuristic vs exact)", lambda: button("Run comparison").click().run())
print("dataframes after compare:", [d.value.to_dict("records") for d in at.dataframe][-1:])

# 3) exact optimisation on the small scope
pick("Stage 1 method", STAGE1_EXACT)
pick("Stage 2 method", STAGE2_EXACT)
at.run()
step("optimize exact", lambda: button("Run optimization").click().run())
print("success:", [s.value for s in at.success][-2:])

# 4) sensitivity scenarios (solver defaults to the heuristic)
step("scenarios (heuristic)", lambda: button("Run scenarios").click().run())
