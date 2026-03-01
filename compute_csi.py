# -*- coding: utf-8 -*-
"""
3.3 Spatial Autocorrelation and CSI (kNN weights, k=10)
- Input: 5 m susceptibility probability raster (GeoTIFF) from stacking ensemble
         slope-unit polygons (SHP/GPKG) with a unique unit id field
- Output:
    1) global_moran.json
    2) lisa_raster.tif (HH/HL/LH/LL/N encoded)
    3) slopeunit_csi.csv (p_mean, dom_lisa, Z, X, CSI, preliminary_class)

Dependencies:
  numpy, pandas, geopandas, rasterio, shapely
  libpysal, esda
  scipy (for sparse)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import xy as tf_xy

from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from libpysal.weights import WSP
from esda.moran import Moran
from esda.moran import Moran_Local

warnings.filterwarnings("ignore")


# -----------------------------
# CONFIG (edit paths/fields)
# -----------------------------
CFG = {
    # 5 m susceptibility surface (0-1), GeoTIFF
    "prob_raster": r"F:\your_project\susceptibility_prob_5m.tif",

    # slope units polygons (must align CRS with raster)
    "slope_units": r"F:\your_project\slope_units.gpkg",
    "unit_id_field": "unit_id",  # unique id per slope unit

    # spatial weights
    "k_neighbors": 10,

    # LISA significance
    "lisa_alpha": 0.05,
    "permutations": 999,

    # performance / safety
    "MAX_CELLS": None,   # e.g., 200000 for quick test; set None for full run

    # probability thresholds used in Table 3
    "p_low": 0.3,
    "p_high": 0.7,

    # output folder
    "out_dir": r"F:\your_project\outputs_spatial_csi",
}


# -----------------------------
# Helpers: probability level, Table 3 mapping
# -----------------------------
def prob_level(p, p_low=0.3, p_high=0.7):
    # Low / Moderate / High
    if p < p_low:
        return "L"
    if p < p_high:
        return "M"
    return "H"


# Table 3 mapping (Probability level x Dominant LISA)
# Columns: HH, HL, LH, N, LL
# Rows: High(H), Moderate(M), Low(L)
TABLE3 = {
    ("H", "HH"): "Very high risk",
    ("H", "HL"): "Very high risk",
    ("H", "LH"): "High risk",
    ("H", "N"):  "High risk",
    ("H", "LL"): "Moderate risk",

    ("M", "HH"): "High risk",
    ("M", "HL"): "High risk",
    ("M", "LH"): "Uncertain",
    ("M", "N"):  "Uncertain",
    ("M", "LL"): "Moderate risk",

    ("L", "HH"): "High risk",
    ("L", "HL"): "Uncertain",
    ("L", "LH"): "Low risk",
    ("L", "N"):  "Low risk",
    ("L", "LL"): "Low risk",
}


def preliminary_class(p_mean, dom_lisa, p_low=0.3, p_high=0.7):
    pl = prob_level(p_mean, p_low, p_high)
    # dom_lisa must be one of HH, HL, LH, LL, N
    return TABLE3.get((pl, dom_lisa), "Uncertain")


# -----------------------------
# Raster cell extraction
# -----------------------------
def raster_valid_cells(src):
    """Return row/col indices of valid pixels (not nodata)."""
    arr = src.read(1)
    nodata = src.nodata
    if nodata is None:
        mask = np.isfinite(arr)
    else:
        mask = np.isfinite(arr) & (arr != nodata)
    rows, cols = np.where(mask)
    vals = arr[rows, cols].astype(float)
    return rows, cols, vals, arr, mask


def cell_centroids(transform, rows, cols):
    """Compute centroid coords for given row/col arrays."""
    # rasterio.transform.xy supports vectorized access if we loop; do it in numpy-ish way:
    # Use affine: x = a*col + b*row + c; y = d*col + e*row + f
    # For north-up rasters, b=d=0 typically.
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    x = a * cols + b * rows + c + a * 0.5 + b * 0.5
    y = d * cols + e * rows + f + d * 0.5 + e * 0.5
    return np.column_stack([x, y])


# -----------------------------
# Build kNN sparse weights
# -----------------------------
def knn_weights_sparse(coords, k=10):
    """
    Build row-standardized kNN weights as a sparse matrix.
    coords: (n,2)
    returns: WSP (PySAL sparse weights wrapper)
    """
    n = coords.shape[0]
    # find k+1 because first neighbor is itself
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="auto").fit(coords)
    dist, idx = nn.kneighbors(coords)

    # exclude self (first column)
    neigh = idx[:, 1:]
    # build CSR
    row_idx = np.repeat(np.arange(n), neigh.shape[1])
    col_idx = neigh.reshape(-1)
    data = np.ones_like(col_idx, dtype=float)

    W = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
    # row-standardize (sum to 1)
    row_sums = np.array(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    inv = 1.0 / row_sums
    Dinv = sparse.diags(inv)
    W_rs = Dinv @ W
    return WSP(W_rs)


# -----------------------------
# LISA category encoding
# -----------------------------
# Use integer codes for raster export
# 0: N (not significant), 1: HH, 2: HL, 3: LH, 4: LL
LISA_CODE = {"N": 0, "HH": 1, "HL": 2, "LH": 3, "LL": 4}
CODE_LISA = {v: k for k, v in LISA_CODE.items()}


def lisa_labels_from_moran_local(ml: Moran_Local, alpha=0.05):
    """
    Convert Moran_Local results to labels HH/HL/LH/LL/N using quadrant + significance.
    PySAL q convention:
      q=1: HH
      q=2: LH
      q=3: LL
      q=4: HL
    Significance:
      p_sim < alpha => significant
    """
    sig = ml.p_sim < alpha
    labels = np.full(ml.Is.shape[0], "N", dtype=object)
    q = ml.q

    labels[sig & (q == 1)] = "HH"
    labels[sig & (q == 2)] = "LH"
    labels[sig & (q == 3)] = "LL"
    labels[sig & (q == 4)] = "HL"
    return labels


# -----------------------------
# Aggregate raster metrics to slope units (fast, no rasterstats)
# -----------------------------
def aggregate_to_units(unit_ids_raster, prob_vals_full, lisa_code_full, valid_mask, unit_nodata=0):
    """
    unit_ids_raster: 2D int array, same shape as raster, unit id per pixel (0 for background)
    prob_vals_full: 2D float array, probability raster full
    lisa_code_full: 2D int array, lisa code per pixel full (0..4)
    valid_mask: 2D bool, valid pixels of probability raster
    returns: DataFrame with per-unit metrics
    """
    # consider only pixels that are valid and within a unit (unit_id != 0)
    unit_id = unit_ids_raster[valid_mask]
    prob_v = prob_vals_full[valid_mask]
    lisa_c = lisa_code_full[valid_mask]

    in_unit = unit_id != unit_nodata
    unit_id = unit_id[in_unit].astype(np.int64)
    prob_v = prob_v[in_unit].astype(float)
    lisa_c = lisa_c[in_unit].astype(np.int64)

    # map unit ids to 0..m-1 for bincount operations
    uniq_units, inv = np.unique(unit_id, return_inverse=True)
    m = uniq_units.shape[0]

    # total pixels per unit
    n_total = np.bincount(inv, minlength=m).astype(float)

    # mean probability per unit
    sum_prob = np.bincount(inv, weights=prob_v, minlength=m)
    p_mean = sum_prob / np.maximum(n_total, 1.0)

    # significant pixels: lisa code != 0
    sig = (lisa_c != 0)
    inv_sig = inv[sig]
    lisa_sig = lisa_c[sig]

    n_sig = np.bincount(inv_sig, minlength=m).astype(float)

    # counts of each cluster type among significant pixels, but we need Z relative to total pixels
    # We'll compute counts per cluster among significant pixels:
    cnt_hh = np.bincount(inv_sig[lisa_sig == 1], minlength=m).astype(float)
    cnt_hl = np.bincount(inv_sig[lisa_sig == 2], minlength=m).astype(float)
    cnt_lh = np.bincount(inv_sig[lisa_sig == 3], minlength=m).astype(float)
    cnt_ll = np.bincount(inv_sig[lisa_sig == 4], minlength=m).astype(float)

    # dominant LISA type based on highest count among significant pixels
    stack = np.vstack([cnt_hh, cnt_hl, cnt_lh, cnt_ll])  # shape (4, m)
    dom_idx = np.argmax(stack, axis=0)  # 0..3
    dom_label = np.array(["HH", "HL", "LH", "LL"], dtype=object)[dom_idx]

    # if no significant pixels, dominant should be N
    dom_label[n_sig == 0] = "N"

    # Z_i: proportion of pixels belonging to dominant LISA cluster type within unit i
    # Here: dominant significant cluster count / total pixels
    dom_cnt = stack[dom_idx, np.arange(m)]
    Z = dom_cnt / np.maximum(n_total, 1.0)

    # X_i: proportion of statistically significant pixels within unit i
    X = n_sig / np.maximum(n_total, 1.0)

    CSI = Z * X

    out = pd.DataFrame({
        "unit_id": uniq_units,
        "n_total_pix": n_total.astype(int),
        "n_sig_pix": n_sig.astype(int),
        "p_mean": p_mean,
        "dom_lisa": dom_label,
        "Z_dom_prop": Z,
        "X_sig_prop": X,
        "CSI": CSI,
    })

    return out


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    os.makedirs(CFG["out_dir"], exist_ok=True)

    # 1) Read probability raster
    with rasterio.open(CFG["prob_raster"]) as src:
        transform = src.transform
        crs = src.crs
        rows, cols, vals, prob_full, valid_mask = raster_valid_cells(src)

        # optional subsampling for fast test
        if CFG["MAX_CELLS"] is not None and len(vals) > CFG["MAX_CELLS"]:
            rng = np.random.default_rng(CFG["random_state"])
            idx = rng.choice(len(vals), size=CFG["MAX_CELLS"], replace=False)
            rows_s, cols_s, vals_s = rows[idx], cols[idx], vals[idx]
            sample_mode = True
        else:
            rows_s, cols_s, vals_s = rows, cols, vals
            sample_mode = False

        coords_s = cell_centroids(transform, rows_s, cols_s)

    # 2) Build kNN weights (k=10) in sparse form, row-standardized
    wsp = knn_weights_sparse(coords_s, k=CFG["k_neighbors"])
    w = wsp.to_W()  # PySAL W object

    # 3) Global Moran's I
    moran = Moran(vals_s, w, permutations=CFG["permutations"])
    global_out = {
        "k": CFG["k_neighbors"],
        "n_cells_used": int(len(vals_s)),
        "sample_mode": bool(sample_mode),
        "moran_I": float(moran.I),
        "expected_I": float(moran.EI),
        "z_norm": float(moran.z_norm),
        "p_norm": float(moran.p_norm),
        "z_sim": float(moran.z_sim),
        "p_sim": float(moran.p_sim),
        "permutations": int(CFG["permutations"]),
    }
    with open(os.path.join(CFG["out_dir"], "global_moran.json"), "w", encoding="utf-8") as f:
        json.dump(global_out, f, ensure_ascii=False, indent=2)

    # 4) Local Moran (LISA)
    ml = Moran_Local(vals_s, w, permutations=CFG["permutations"])
    lisa_labels_s = lisa_labels_from_moran_local(ml, alpha=CFG["lisa_alpha"])

    # 5) Build full-size LISA raster (if sample_mode=True, we can only output sampled LISA;
    #    for publication-quality maps, you should run full mode)
    if sample_mode:
        print("[WARN] MAX_CELLS is enabled -> LISA raster will contain labels only for sampled pixels.")
        lisa_code_full = np.zeros_like(prob_full, dtype=np.uint8)
        # fill only sampled positions (rows_s, cols_s)
        lisa_code_full[rows_s, cols_s] = np.array([LISA_CODE[x] for x in lisa_labels_s], dtype=np.uint8)
    else:
        lisa_code_full = np.zeros_like(prob_full, dtype=np.uint8)
        lisa_code_full[rows, cols] = np.array([LISA_CODE[x] for x in lisa_labels_s], dtype=np.uint8)

    # 6) Save LISA raster
    lisa_path = os.path.join(CFG["out_dir"], "lisa_raster.tif")
    with rasterio.open(CFG["prob_raster"]) as src:
        meta = src.meta.copy()
        meta.update({"count": 1, "dtype": "uint8", "nodata": 0})
        with rasterio.open(lisa_path, "w", **meta) as dst:
            dst.write(lisa_code_full.astype(np.uint8), 1)

    # 7) Read slope units, rasterize unit ids to raster grid
    gdf = gpd.read_file(CFG["slope_units"])
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    if CFG["unit_id_field"] not in gdf.columns:
        raise ValueError(f"Missing unit_id_field={CFG['unit_id_field']} in slope units.")

    shapes = [(geom, int(uid)) for geom, uid in zip(gdf.geometry, gdf[CFG["unit_id_field"]])]
    unit_id_raster = rasterize(
        shapes=shapes,
        out_shape=prob_full.shape,
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False
    )

    # 8) Aggregate to slope units: p_mean + dominant LISA + Z/X/CSI
    df_units = aggregate_to_units(
        unit_ids_raster=unit_id_raster,
        prob_vals_full=prob_full,
        lisa_code_full=lisa_code_full,
        valid_mask=valid_mask,
        unit_nodata=0
    )

    # 9) Preliminary susceptibility class (Table 3)
    df_units["prob_level"] = df_units["p_mean"].apply(lambda x: prob_level(x, CFG["p_low"], CFG["p_high"]))
    df_units["prelim_class"] = [
        preliminary_class(p, d, CFG["p_low"], CFG["p_high"])
        for p, d in zip(df_units["p_mean"].values, df_units["dom_lisa"].values)
    ]

    # 10) Save slope-unit table
    out_csv = os.path.join(CFG["out_dir"], "slopeunit_csi.csv")
    df_units.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("=== Done ===")
    print("Global Moran ->", os.path.join(CFG["out_dir"], "global_moran.json"))
    print("LISA raster  ->", lisa_path)
    print("Slope-unit CSI table ->", out_csv)


if __name__ == "__main__":
    main()