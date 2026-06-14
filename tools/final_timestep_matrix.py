"""
Build a 9-case x 6-column matrix of 2D top-down maps showing PRESSURE and SWAT
at the FINAL timestep, comparing Eclipse truth vs the X-MGN proxy.

Columns (left to right):
  1. PRESSURE truth (bar)
  2. PRESSURE pred  (bar)
  3. PRESSURE diff  (pred - truth, bar)
  4. SWAT     truth (fraction)
  5. SWAT     pred  (fraction)
  6. SWAT     diff  (pred - truth)

Rows: 9 cases from representative_cases.json, in order BEST -> MEDIAN -> WORST.

Method: read each case's truth UNRST and pred UNRST. Take the LAST PRESSURE/SWAT
keyword block per file (final report). Map active-cell values back into the
(nx, ny, nz) grid using ACTNUM. Average over Z (depth) to get a top-down
(nx, ny) plan view. Plot all panels with consistent colormaps per column.
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from resdata.resfile import ResdataFile
from resdata.resfile import ResdataFile  # type: ignore

CASES_JSON = "/home/raseena/ML-Reservoir-Simulation-Proxy-Exercise/xmgn/outputs/XMGN_Norne/accuracy/representative_cases.json"
TRUTH_BASE = "/home/raseena/data/NORNE_FLAT"
PRED_BASE = "/mnt/e/NVIDIA/reservoir_simulation/eval_results/predictions_unrst"
OUT_PNG = "/mnt/e/NVIDIA/reservoir_simulation/eval_results/final_timestep_matrix.png"


def load_grid_dims(egrid_path):
    """Read nx, ny, nz from an EGRID via the FILEHEAD keyword."""
    f = ResdataFile(egrid_path)
    nx, ny, nz = None, None, None
    for kw in f:
        if kw.getName().strip() == "GRIDHEAD":
            arr = kw.numpy_view()
            # GRIDHEAD layout: [type, nx, ny, nz, ...] (1-indexed dims)
            nx, ny, nz = int(arr[1]), int(arr[2]), int(arr[3])
            break
    return nx, ny, nz


def load_actnum(egrid_path, total_cells):
    """Read ACTNUM (active-cell flag) from EGRID. Returns bool array of length nx*ny*nz."""
    f = ResdataFile(egrid_path)
    for kw in f:
        if kw.getName().strip() == "ACTNUM":
            arr = np.asarray(kw.numpy_view(), dtype=np.int32)
            if arr.size == total_cells:
                return arr.astype(bool)
    return None


def load_actnum_from_init(init_path, total_cells):
    """Some EGRIDs omit ACTNUM — derive it from INIT's PORV > 0 instead."""
    f = ResdataFile(init_path)
    for kw in f:
        if kw.getName().strip() == "PORV":
            arr = np.asarray(kw.numpy_view(), dtype=np.float64)
            if arr.size == total_cells:
                return (arr > 0)
    return None


def load_final_kw(unrst_path, kw_name):
    """Return the LAST occurrence of kw_name in the UNRST (final report's value)."""
    f = ResdataFile(unrst_path)
    last = None
    for kw in f:
        if kw.getName().strip() == kw_name:
            last = kw.numpy_view().copy()
    return last


def load_final_pressure_and_swat(unrst_path):
    """Open the UNRST exactly ONCE and return (final_PRESSURE, final_SWAT).

    Earlier version called load_final_kw twice per file, parsing 470 MB through
    9p four times per case (truth+pred × PRESSURE+SWAT) → ~10 min per case.
    This single-pass version cuts that in half.
    """
    print(f"  scanning {os.path.basename(unrst_path)}...", flush=True)
    f = ResdataFile(unrst_path)
    last_p = None
    last_s = None
    for kw in f:
        n = kw.getName().strip()
        if n == "PRESSURE":
            last_p = kw.numpy_view().copy()
        elif n == "SWAT":
            last_s = kw.numpy_view().copy()
    return last_p, last_s


def active_to_grid(active_values, actnum, nx, ny, nz):
    """Map active-cell vector into a 3D grid array (Fortran-order), with NaN
    for inactive cells. Returns (nx, ny, nz) array.
    """
    full = np.full(nx * ny * nz, np.nan, dtype=np.float64)
    full[actnum] = active_values
    return full.reshape((nx, ny, nz), order="F")


def topdown_mean(grid3d):
    """Average over Z (k axis), ignoring NaN (inactive cells)."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(grid3d, axis=2)


def main():
    with open(CASES_JSON) as f:
        cases_spec = json.load(f)
    cases = cases_spec["cases"]
    annotations = cases_spec.get("annotations", {})

    # Sort: BEST -> MEDIAN -> WORST in the order they were picked
    label_order = {"BEST": 0, "MEDIAN": 1, "WORST": 2}
    cases.sort(key=lambda c: (label_order.get(annotations.get(c, ""), 9), c))

    # Pre-load grid dims (same for all Norne cases)
    egrid0 = os.path.join(TRUTH_BASE, cases[0], f"{cases[0]}.EGRID")
    nx, ny, nz = load_grid_dims(egrid0)
    print(f"Norne grid: nx={nx}, ny={ny}, nz={nz}, total={nx*ny*nz}")

    # Collect data per case
    rows = []
    for case in cases:
        truth_unrst = os.path.join(TRUTH_BASE, case, f"{case}.UNRST")
        pred_unrst = os.path.join(PRED_BASE, f"{case}.UNRST")
        init_path = os.path.join(TRUTH_BASE, case, f"{case}.INIT")
        egrid_path = os.path.join(TRUTH_BASE, case, f"{case}.EGRID")

        # actnum from EGRID; if missing, from PORV in INIT
        actnum = load_actnum(egrid_path, nx * ny * nz)
        if actnum is None:
            actnum = load_actnum_from_init(init_path, nx * ny * nz)
        if actnum is None:
            print(f"!! Could not get ACTNUM for {case}")
            continue

        truth_p, truth_s = load_final_pressure_and_swat(truth_unrst)
        pred_p, pred_s = load_final_pressure_and_swat(pred_unrst)

        # Build (nx, ny) plan views
        rows.append(
            {
                "case": case,
                "label": annotations.get(case, ""),
                "P_truth": topdown_mean(active_to_grid(truth_p, actnum, nx, ny, nz)),
                "P_pred": topdown_mean(active_to_grid(pred_p, actnum, nx, ny, nz)),
                "S_truth": topdown_mean(active_to_grid(truth_s, actnum, nx, ny, nz)),
                "S_pred": topdown_mean(active_to_grid(pred_s, actnum, nx, ny, nz)),
            }
        )
        print(f"  loaded {case} ({annotations.get(case, '')})")

    # Global color ranges so panels are comparable
    all_p_truth = np.concatenate([r["P_truth"].ravel() for r in rows])
    all_p_pred = np.concatenate([r["P_pred"].ravel() for r in rows])
    p_lo = np.nanpercentile(np.concatenate([all_p_truth, all_p_pred]), 1)
    p_hi = np.nanpercentile(np.concatenate([all_p_truth, all_p_pred]), 99)

    all_p_diff = np.concatenate(
        [(r["P_pred"] - r["P_truth"]).ravel() for r in rows]
    )
    p_diff_abs = max(
        abs(np.nanpercentile(all_p_diff, 1)),
        abs(np.nanpercentile(all_p_diff, 99)),
    )

    all_s_diff = np.concatenate(
        [(r["S_pred"] - r["S_truth"]).ravel() for r in rows]
    )
    s_diff_abs = max(
        abs(np.nanpercentile(all_s_diff, 1)),
        abs(np.nanpercentile(all_s_diff, 99)),
    )

    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, 6, figsize=(18, 2.6 * n_rows), squeeze=False
    )
    col_titles = [
        "PRESSURE truth (bar)",
        "PRESSURE pred (bar)",
        "PRESSURE diff (bar)",
        "SWAT truth",
        "SWAT pred",
        "SWAT diff",
    ]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=11)

    for i, r in enumerate(rows):
        # PRESSURE
        for j, (key, vmin, vmax, cmap, norm) in enumerate(
            [
                ("P_truth", p_lo, p_hi, "viridis", None),
                ("P_pred", p_lo, p_hi, "viridis", None),
                ("diff_P", None, None, "RdBu_r",
                 TwoSlopeNorm(vmin=-p_diff_abs, vcenter=0, vmax=p_diff_abs)),
                ("S_truth", 0, 1, "Blues", None),
                ("S_pred", 0, 1, "Blues", None),
                ("diff_S", None, None, "RdBu_r",
                 TwoSlopeNorm(vmin=-s_diff_abs, vcenter=0, vmax=s_diff_abs)),
            ]
        ):
            ax = axes[i, j]
            if key == "diff_P":
                img = r["P_pred"] - r["P_truth"]
                im = ax.imshow(img.T, origin="lower", cmap=cmap, norm=norm,
                               aspect="auto")
            elif key == "diff_S":
                img = r["S_pred"] - r["S_truth"]
                im = ax.imshow(img.T, origin="lower", cmap=cmap, norm=norm,
                               aspect="auto")
            else:
                img = r[key]
                im = ax.imshow(img.T, origin="lower", cmap=cmap,
                               vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(
                    f"{r['case']}\n[{r['label']}]", fontsize=10, rotation=0,
                    labelpad=40, va="center", ha="right"
                )
            # one shared colorbar per column at row 0
            if i == n_rows - 1:
                cbar = fig.colorbar(im, ax=axes[:, j], shrink=0.6,
                                    pad=0.02, location="bottom")

    fig.suptitle(
        "Final timestep: Eclipse truth vs X-MGN proxy — 9 representative cases\n"
        "(top-down view, mean over Z; 46x112 active-cell grid)",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.05, hspace=0.05, wspace=0.05)
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\nWrote {OUT_PNG}")


if __name__ == "__main__":
    main()
