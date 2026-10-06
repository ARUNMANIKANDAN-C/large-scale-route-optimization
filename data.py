"""Loading, unit conversion and descriptive statistics."""
import io
import numpy as np
import pandas as pd

ORDER_COLS = ["Order_ID", "Material_ID", "Item_ID", "Source", "Destination",
              "Available_Time", "Deadline", "Danger_Type", "Area", "Weight"]

# File units -> SI units used by the models.
#   Area   : cm^2  -> m^2  (divide by 10,000)
#   Weight : 0.1 g -> kg   (divide by 10,000)   [max item = 3,130 kg, which fits the 10 t truck]
AREA_DIV = 1e4
WEIGHT_DIV = 1e4


def read_csv_any(src):
    """src can be a path, a file-like object or an uploaded Streamlit file."""
    if hasattr(src, "read"):
        return pd.read_csv(src)
    return pd.read_csv(src)


def read_orders(sources):
    """Read one or several order files and concatenate them (duplicate Item_IDs dropped)."""
    frames = []
    for s in sources:
        df = read_csv_any(s)
        missing = [c for c in ORDER_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Order file is missing columns: {missing}")
        frames.append(df[ORDER_COLS])
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="Item_ID", keep="first").reset_index(drop=True)
    return df


class Problem:
    """Orders + distance matrix in model units."""

    def __init__(self, orders: pd.DataFrame, dist: pd.DataFrame, t0=None):
        df = orders.copy()
        df["Available_Time"] = pd.to_datetime(df["Available_Time"])
        df["Deadline"] = pd.to_datetime(df["Deadline"])
        self.t0 = pd.Timestamp(t0) if t0 is not None else df["Available_Time"].min()
        df["w"] = df["Weight"] / WEIGHT_DIV
        df["a"] = df["Area"] / AREA_DIV
        df["e"] = (df["Available_Time"] - self.t0).dt.total_seconds() / 3600.0
        df["l"] = (df["Deadline"] - self.t0).dt.total_seconds() / 3600.0
        df["d"] = df["Destination"]
        df["h"] = df["Danger_Type"]
        self.items = df.reset_index(drop=True)
        self.depot = self.items["Source"].mode().iloc[0]
        self.D = build_distance(dist)

    # ------------------------------------------------------------------
    def subset(self, destinations=None, max_items=None, order_ids=None):
        df = self.items
        mask = pd.Series(True, index=df.index)
        if destinations:
            mask &= df["d"].isin(destinations)
        if order_ids:
            mask &= df["Order_ID"].isin(order_ids)
        df = df[mask]
        if max_items:
            df = df.head(int(max_items))
        new = Problem.__new__(Problem)
        new.t0, new.depot, new.D = self.t0, self.depot, self.D
        new.items = df.reset_index(drop=True)
        return new

    def dist(self, a, b):
        try:
            return self.D[a][b]
        except KeyError:
            raise ValueError(f"No distance from '{a}' to '{b}' in the distance matrix")

    def to_time(self, hours):
        return (self.t0 + pd.to_timedelta(round(hours * 3600), unit="s"))


def build_distance(df: pd.DataFrame):
    """Dict-of-dicts distance in km. Missing (a,b) falls back to (b,a)."""
    if not {"Source", "Destination"}.issubset(df.columns):
        raise ValueError("Distance file needs Source, Destination and a distance column")
    col = [c for c in df.columns if c.lower().startswith("distance")]
    if not col:
        raise ValueError("Distance file needs a 'Distance(M)' column")
    km = df[col[0]].astype(float) / 1000.0
    D = {}
    for s, d, v in zip(df["Source"], df["Destination"], km):
        D.setdefault(s, {})[d] = v
        D.setdefault(d, {})
    for a in list(D):
        D[a][a] = 0.0
        for b in list(D):
            if b not in D[a]:
                if a in D[b]:
                    D[a][b] = D[b][a]
    return D


def check_coverage(P: Problem):
    """Return a list of problems with the data (cities with no distance, etc.)."""
    msgs = []
    cities = set(P.items["d"].unique())
    miss = [c for c in cities if c not in P.D.get(P.depot, {})]
    if miss:
        msgs.append(f"No distance from depot {P.depot} to: {sorted(miss)}")
    return msgs


# ----------------------------------------------------------------------
def statistics(P: Problem):
    df = P.items
    out = {}
    out["n_items"] = len(df)
    out["n_orders"] = df["Order_ID"].nunique()
    out["n_dest"] = df["d"].nunique()
    out["depot"] = P.depot
    out["total_weight_t"] = df["w"].sum() / 1000.0
    out["total_area_m2"] = df["a"].sum()
    out["weight_kg"] = df["w"].describe()[["min", "mean", "max"]].to_dict()
    out["area_m2"] = df["a"].describe()[["min", "mean", "max"]].to_dict()
    out["by_hazard"] = (df.groupby("h").agg(items=("w", "size"), weight_t=("w", lambda s: s.sum() / 1000),
                                            area_m2=("a", "sum")).round(2))
    by_dest = df.groupby("d").agg(items=("w", "size"), orders=("Order_ID", "nunique"),
                                  weight_t=("w", lambda s: s.sum() / 1000), area_m2=("a", "sum"))
    by_dest["distance_km"] = [P.D[P.depot].get(c, np.nan) for c in by_dest.index]
    out["by_dest"] = by_dest.sort_values("items", ascending=False).round(2)
    win = (df["l"] - df["e"]) / 24.0
    out["window_days"] = win.round(1).value_counts().sort_index()
    out["avail_range"] = (df["Available_Time"].min(), df["Available_Time"].max())
    out["deadline_max"] = df["Deadline"].max()
    return out
