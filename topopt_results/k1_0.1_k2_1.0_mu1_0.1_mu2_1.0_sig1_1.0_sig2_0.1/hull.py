import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

# -----------------------------
# PARAMETERS
# -----------------------------
NEAR_HULL_TOL = 1e-3  # distance threshold
csv_files = [
    ("final_obs", Path("final_observables.csv")),
    ("final_obs2", Path("final_observables2.csv")),
]
required = {"morphology", "conductivity", "shear_modulus"}

# -----------------------------
# LOAD + TAG DATA
# -----------------------------
dfs = []
for source_name, path in csv_files:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found.")
    d = pd.read_csv(path)
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    d = d.copy()
    d["source_csv"] = source_name
    dfs.append(d)

df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)

# -----------------------------
# CONVEX HULL (Andrew)
# -----------------------------
def convex_hull_indices(points):
    pts = sorted(points, key=lambda t: (t[0], t[1]))

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower, upper = [], []

    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return [p[2] for p in hull]

points = [(r.conductivity, r.shear_modulus, i) for i, r in df.iterrows()]
hull_idx = convex_hull_indices(points)

hull_df = df.loc[hull_idx].reset_index(drop=True)

# Hull polygon (closed)
H = hull_df[["conductivity", "shear_modulus"]].to_numpy()
H = np.vstack([H, H[0]])

# -----------------------------
# DISTANCE POINT → SEGMENT
# -----------------------------
def point_segment_distance(p, a, b):
    ap = p - a
    ab = b - a
    t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def distance_to_hull(point, hull_poly):
    return min(
        point_segment_distance(point, hull_poly[i], hull_poly[i+1])
        for i in range(len(hull_poly)-1)
    )

# -----------------------------
# FIND NEAR-HULL POINTS
# -----------------------------
near_mask = []
for _, r in df.iterrows():
    p = np.array([r.conductivity, r.shear_modulus])
    d = distance_to_hull(p, H)
    near_mask.append(d <= NEAR_HULL_TOL)

df["near_hull"] = near_mask
df["on_hull"] = df.index.isin(hull_idx)

label_df = df[df["on_hull"] | df["near_hull"]].copy()

# -----------------------------
# PLOT
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 7))

marker_by_source = {"final_obs": "o", "final_obs2": "s"}

for src, sub in df.groupby("source_csv"):
    ax.scatter(
        sub["conductivity"],
        sub["shear_modulus"],
        marker=marker_by_source.get(src, "o"),
        alpha=0.6,
        edgecolors="none",
        label=src,
    )

# Hull outline
ax.plot(H[:, 0], H[:, 1], lw=1.5)

# -----------------------------
# LABEL HULL + NEAR-HULL
# -----------------------------
texts = []
for _, r in label_df.iterrows():
    tag = "H" if r.on_hull else "NH"
    label = f"{r.morphology}[{r.source_csv},{tag}]"
    texts.append(
        ax.text(
            r.conductivity,
            r.shear_modulus,
            label,
            fontsize=4,   # very small
            alpha=0.9,
        )
    )

adjust_text(
    texts,
    ax=ax,
    x=label_df["conductivity"].to_numpy(),
    y=label_df["shear_modulus"].to_numpy(),
    expand_text=(1.2, 1.2),
    expand_points=(1.2, 1.2),
    force_text=(0.3, 0.3),
    force_points=(0.2, 0.2),
    lim=3000,
    arrowprops=dict(arrowstyle="-", lw=0.4, alpha=0.8),
)

ax.set_xlabel("conductivity")
ax.set_ylabel("shear_modulus")
ax.set_title("Convex Hull + Near-Hull Points")
ax.legend()
plt.tight_layout()
plt.savefig("convex_hull_with_near_points.png", dpi=300)
plt.show()
