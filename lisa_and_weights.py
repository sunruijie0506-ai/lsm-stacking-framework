# src/lisa_and_weights.py
# -*- coding: utf-8 -*-
"""
3.3 Spatial Autocorrelation (Global Moran's I + LISA) + kNN weights

This script supports TWO reproducibility modes:

MODE A (full geospatial, local use only):
- Input: 5 m probability raster GeoTIFF
- Build kNN weights on raster cell centroids (k=10)
- Compute Global Moran's I and Local Moran (LISA) at pixel level
- Export: global_moran.json, lisa_raster.tif (encoded HH/HL/LH/LL/N)

MODE B (de-identified / no-geometry release):
- Input: slope-unit probability table (CSV) + anonymized weights_adjacency.csv
- Compute Global Moran's I and LISA at slope-unit scale (no coordinates needed)
- Export: global_moran.json, lisa_units.csv (unit_id, lisa_label, p_value, quadrant, etc.)

Why MODE B?
- If coordinates/geometry cannot be shared due to regulations, you can still provide:
  (i) per-unit probabilities and (ii) anonymized adjacency/weight table.
  This allows reviewers to reproduce Moran/LISA statistics without any geographic data.

Dependencies:
- numpy, pandas
- libpysal, esda
Optional for MODE A:
- rasterio, sklearn (NearestNeighbors), scipy (sparse)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -----------------------
# CONFIG (edit paths)
# -----------------------
CFG = {
    # Choose "A" (raster pixel LISA) or "B" (unit table + adjacency weights)
    "mode": "B",

    # ---- MODE A inputs ----
    "prob_raster": r"F:\your_project\susceptibility_prob_5m.tif",

    # ---- MODE B inputs ----
    # Slope-unit table must contain:
    #   unit_id, p_mean (or your probability column)
    "unit_table_csv": r"F:\your_project\data\processed\slopeunit_attributes_no_geo.csv",
    "unit_id_col": "unit_id",
    "prob_col": "p_mean",

    # Anonymized weights table (no coordinates):
    # CSV with columns: i, j, w  (unit_id i connected to unit_id j with weight w)
    "weights_adjacency_csv": r"F:\your_project\data\processed\weights_adjacency.csv",
    "w_i_col": "i",
    "w_j_col": "j",
    "w_w_col": "w",

    # Spatial parameters
    "k_neighbors": 10,         # for MODE A kNN
    "alpha": 0.05,             # LISA significance threshold
    "permutations": 999,       # for p-values

    # Output
    "out_dir": r"F:\your_project\outputs_lisa",
}


# -----------------------
# LISA category encoding
# -----------------------
# Quadrant convention from PySAL Moran_Local (ml.q):
#   1: HH, 2: LH, 3: LL, 4: HL
LISA_CODE = {"N": 0, "HH": 1, "HL": 2, "LH": 3, "LL": 4}
CODE_LISA = {v: k for k, v in LISA_CODE.items()}


def lisa_labels_from_moran_local(ml, alpha=0.05):
    """
    Convert esda.Moran_Local output to LISA labels:
      - If p_sim < alpha: assign HH/HL/LH/LL by quadrant
      - Else: N (non-significant)
    """
    sig = ml.p_sim < alpha
    labels = np.full(ml.Is.shape[0], "N", dtype=object)
    q = ml.q
    labels[sig & (q == 1)] = "HH"
    labels[sig & (q == 2)] = "LH"
    labels[sig & (q == 3)] = "LL"
    labels[sig & (q == 4)] = "HL"
    return labels


# -----------------------
# MODE B: from adjacency weights (no geometry)
# -----------------------
def build_W_from_adjacency(edges_df, node_ids, i_col="i", j_col="j", w_col="w"):
    """
    Build a PySAL W object from an edge list.

    edges_df: DataFrame with columns (i, j, w)
    node_ids: iterable of all node ids (unit ids). Ensures isolated nodes handled.

    Returns:
      W: libpysal.weights.W
    """
    from libpysal.weights import W

    # neighbors dict: {i: [j1, j2, ...]}
    neighbors = {int(n): [] for n in node_ids}
    weights = {int(n): [] for n in node_ids}

    for r in edges_df.itertuples(index=False):
        i = int(getattr(r, i_col))
        j = int(getattr(r, j_col))
        w = float(getattr(r, w_col))
        if i not in neighbors:
            neighbors[i] = []
            weights[i] = []
        neighbors[i].append(j)
        weights[i].append(w)

    W_obj = W(neighbors, weights)
    # Row-standardize (common in Moran/LISA)
    W_obj.transform = "R"
    return W_obj


def run_mode_B():
    """
    Unit-scale Moran/LISA using anonymized adjacency/weight table.
    """
    os.makedirs(CFG["out_dir"], exist_ok=True)

    # 1) Load unit probability table
    df = pd.read_csv(CFG["unit_table_csv"])
    if CFG["unit_id_col"] not in df.columns:
        raise ValueError(f"Missing unit_id_col={CFG['unit_id_col']} in unit table.")
    if CFG["prob_col"] not in df.columns:
        raise ValueError(f"Missing prob_col={CFG['prob_col']} in unit table.")

    # Ensure deterministic ordering
    df = df[[CFG["unit_id_col"], CFG["prob_col"]]].copy()
    df = df.dropna(subset=[CFG["unit_id_col"], CFG["prob_col"]])
    df[CFG["unit_id_col"]] = df[CFG["unit_id_col"]].astype(int)
    df = df.sort_values(CFG["unit_id_col"]).reset_index(drop=True)

    unit_ids = df[CFG["unit_id_col"]].values
    y = df[CFG["prob_col"]].astype(float).values

    # 2) Load adjacency weights
    edges = pd.read_csv(CFG["weights_adjacency_csv"])
    for c in (CFG["w_i_col"], CFG["w_j_col"], CFG["w_w_col"]):
        if c not in edges.columns:
            raise ValueError(f"Missing column '{c}' in weights_adjacency_csv.")

    W = build_W_from_adjacency(
        edges_df=edges,
        node_ids=unit_ids,
        i_col=CFG["w_i_col"],
        j_col=CFG["w_j_col"],
        w_col=CFG["w_w_col"]
    )

    # 3) Global Moran's I
    from esda.moran import Moran, Moran_Local

    moran = Moran(y, W, permutations=CFG["permutations"])
    global_out = {
        "mode": "B_unit_adjacency",
        "n_units": int(len(y)),
        "moran_I": float(moran.I),
        "expected_I": float(moran.EI),
        "z_norm": float(moran.z_norm),
        "p_norm": float(moran.p_norm),
        "z_sim": float(moran.z_sim),
        "p_sim": float(moran.p_sim),
        "permutations": int(CFG["permutations"]),
        "alpha": float(CFG["alpha"]),
    }
    with open(os.path.join(CFG["out_dir"], "global_moran.json"), "w", encoding="utf-8") as f:
        json.dump(global_out, f, ensure_ascii=False, indent=2)

    # 4) Local Moran (LISA)
    ml = Moran_Local(y, W, permutations=CFG["permutations"])
    labels = lisa_labels_from_moran_local(ml, alpha=CFG["alpha"])

    out = pd.DataFrame({
        "unit_id": unit_ids,
        "p": y,
        "local_I": ml.Is.astype(float),
        "q": ml.q.astype(int),               # quadrant code
        "p_sim": ml.p_sim.astype(float),     # permutation p-value
        "lisa_label": labels,
    })
    out.to_csv(os.path.join(CFG["out_dir"], "lisa_units.csv"), index=False, encoding="utf-8-sig")

    print("=== MODE B done ===")
    print("Saved: global_moran.json, lisa_units.csv")


# -----------------------
# MODE A: raster pixel LISA (local use)
# -----------------------
def run_mode_A():
    """
    Pixel-scale Moran/LISA on raster grid using kNN weights built from cell centroids.

    WARNING:
    - For large 5 m rasters, this can be computationally heavy.
    - Use for internal runs; do NOT publish raw rasters if restricted.
    """
    os.makedirs(CFG["out_dir"], exist_ok=True)

    import rasterio
    from sklearn.neighbors import NearestNeighbors
    from scipy import sparse
    from libpysal.weights import WSP
    from esda.moran import Moran, Moran_Local

    # 1) Read raster and valid pixels
    with rasterio.open(CFG["prob_raster"]) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is None:
            valid = np.isfinite(arr)
        else:
            valid = np.isfinite(arr) & (arr != nodata)

        rows, cols = np.where(valid)
        y = arr[rows, cols].astype(float)

        # Build centroids from affine transform
        t = src.transform
        a, b, c, d, e, f = t.a, t.b, t.c, t.d, t.e, t.f
        x = a * cols + b * rows + c + a * 0.5 + b * 0.5
        ycoord = d * cols + e * rows + f + d * 0.5 + e * 0.5
        coords = np.column_stack([x, ycoord])

        raster_meta = src.meta.copy()

    # 2) Build kNN weights (k=10) as sparse matrix and row-standardize
    n = coords.shape[0]
    k = min(CFG["k_neighbors"] + 1, n)
    nn = NearestNeighbors(n_neighbors=k).fit(coords)
    _, idx = nn.kneighbors(coords)
    neigh = idx[:, 1:]  # drop self

    row_idx = np.repeat(np.arange(n), neigh.shape[1])
    col_idx = neigh.reshape(-1)
    data = np.ones_like(col_idx, dtype=float)

    W_csr = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
    row_sums = np.array(W_csr.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    W_rs = sparse.diags(1.0 / row_sums) @ W_csr
    wsp = WSP(W_rs)
    W = wsp.to_W()
    W.transform = "R"

    # 3) Global Moran's I
    moran = Moran(y, W, permutations=CFG["permutations"])
    global_out = {
        "mode": "A_raster_knn",
        "k": int(CFG["k_neighbors"]),
        "n_cells": int(len(y)),
        "moran_I": float(moran.I),
        "expected_I": float(moran.EI),
        "z_sim": float(moran.z_sim),
        "p_sim": float(moran.p_sim),
        "permutations": int(CFG["permutations"]),
        "alpha": float(CFG["alpha"]),
    }
    with open(os.path.join(CFG["out_dir"], "global_moran.json"), "w", encoding="utf-8") as f:
        json.dump(global_out, f, ensure_ascii=False, indent=2)

    # 4) Local Moran (LISA)
    ml = Moran_Local(y, W, permutations=CFG["permutations"])
    labels = lisa_labels_from_moran_local(ml, alpha=CFG["alpha"])

    # 5) Export LISA raster (uint8 codes)
    lisa_code_full = np.zeros_like(arr, dtype=np.uint8)
    lisa_code_full[rows, cols] = np.array([LISA_CODE[x] for x in labels], dtype=np.uint8)

    lisa_path = os.path.join(CFG["out_dir"], "lisa_raster.tif")
    raster_meta.update({"count": 1, "dtype": "uint8", "nodata": 0})
    with rasterio.open(lisa_path, "w", **raster_meta) as dst:
        dst.write(lisa_code_full, 1)

    print("=== MODE A done ===")
    print("Saved: global_moran.json, lisa_raster.tif")


def main():
    mode = str(CFG["mode"]).strip().upper()
    if mode == "A":
        run_mode_A()
    elif mode == "B":
        run_mode_B()
    else:
        raise ValueError("CFG['mode'] must be 'A' or 'B'.")


if __name__ == "__main__":
    main()