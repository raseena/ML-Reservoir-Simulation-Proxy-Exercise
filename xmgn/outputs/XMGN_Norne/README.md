# XMGN_Norne — Submission Outputs

This directory contains all deliverables for the technical assessment submission.

## Directory Structure

| Path | Description |
|------|-------------|
| `best_checkpoints/MeshGraphNet.0.85.mdlus` | Best trained model checkpoint (epoch 85, val loss 0.005454) |
| `best_checkpoints/checkpoint.0.85.pt` | Corresponding optimizer state |
| `inference/NORNE_002.hdf5` | Inference predictions & targets — test case (worst) |
| `inference/NORNE_008.hdf5` | Inference predictions & targets — test case |
| `inference/NORNE_016.hdf5` | Inference predictions & targets — test case |
| `inference/NORNE_018.hdf5` | Inference predictions & targets — test case (best) |
| `inference/NORNE_041.hdf5` | Inference predictions & targets — test case |
| `inference/NORNE_048.hdf5` | Inference predictions & targets — test case |
| `viz1_histogram_NORNE_002.png` | TRUE/PRED/DIFF value distribution (Figure 1) |
| `viz2_spatial_NORNE_002.png` | TRUE/PRED/DIFF spatial top-down map (Figure 2) |
| `loss_curves.png` | Training & validation loss curves |
| `methodology_figure.png` | Pipeline + X-MGN architecture diagram |
| `accuracy/per_case_metrics.csv` | RMSE/MAE per test case |
| `accuracy/error_vs_timestep.csv` | Error stratified by rollout timestep |
| `accuracy/summary_matrix.csv` / `.txt` | Aggregate metrics (mean, P10/P50/P90, worst) |
| `accuracy/representative_cases.json` | Best/median/worst case selection |
| `accuracy/heatmap_PRESSURE.png` / `heatmap_SWAT.png` | Per-case error heatmaps |

## Key Results

| Variable | RMSE | MAE |
|----------|------|-----|
| PRESSURE | 9.59 bar | 4.74 bar |
| SWAT | 0.01237 | 0.00331 |

## Related Files (Repo Root / Other Locations)

| Path | Description |
|------|-------------|

| `Assessment_Report_Raseena_Haris.pdf` | Full writeup (PDF) |
| `xmgn/conf/config_raseena.yaml` | Final training configuration |
| `xmgn/src/inference.py` | Inference script (with bug fixes, see writeup Section 5) |
