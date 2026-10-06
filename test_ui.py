from streamlit.testing.v1 import AppTest
import time
def step(name, fn):
    t=time.time(); fn(); print(f"{name}: {time.time()-t:.1f}s exc={[e.value for e in at.exception]} err={[e.value for e in at.error]}", flush=True)
at = AppTest.from_file("app.py", default_timeout=300)
step("load", lambda: at.run())
step("stats", lambda: [b for b in at.button if b.label=="Generate statistics"][0].click().run())
step("optimize full/heuristic", lambda: [b for b in at.button if b.label=="Run optimization"][0].click().run())
# cluster-master smoke test
for sb in at.selectbox:
    if sb.label=="Stage 1 method": sb.select("Cluster master (overlapping candidates + CBC)")
at.run()
step("optimize cluster", lambda: [b for b in at.button if b.label=="Run optimization"][0].click().run())
# restrict scope to 12 items for exact + compare
[n for n in at.number_input if n.label.startswith("Max items")][0].set_value(12)
at.run()
step("compare small", lambda: [b for b in at.button if b.label=="Run comparison"][0].click().run())
print("dataframes after compare:", [d.value.to_dict('records') for d in at.dataframe][-1:] )
# exact optimize
for sb in at.selectbox:
    if sb.label=="Stage 1 method": sb.select("Exact (integer program, CBC)")
    if sb.label=="Stage 2 method": sb.select("Exact (MIP, CBC)")
at.run()
step("optimize exact", lambda: [b for b in at.button if b.label=="Run optimization"][0].click().run())
print("success:", [s.value for s in at.success][-2:])
step("scenarios", lambda: [b for b in at.button if b.label=="Run scenarios"][0].click().run())
# infeasible fleet
