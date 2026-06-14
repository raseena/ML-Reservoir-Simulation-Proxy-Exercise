# Neural Reservoir Surrogate — Technical Assessment Report
**Candidate**: Dr. Raseena Haris  
**Institution**: Qatar University (RH17989@qu.edu.qa)  
**Repository**: https://github.com/raseena/ML-Reservoir-Simulation-Proxy-Exercise  
**Submission Date**: June 13, 2026  

---

## 1. Environment Setup

### 1.1 Hardware & Software

| Component | Details |
|-----------|---------|
| Machine | Dell Precision 5860 Tower |
| OS | Ubuntu 24.04 LTS (WSL2 on Windows 11 Enterprise) |
| GPU | NVIDIA RTX A4000 (16 GB VRAM) |
| RAM | 128 GB |
| Storage | 1.86 TB |
| CUDA | 12.1 |
| Driver | 550.144.06 |
| Python | 3.10.20 (conda environment: xmgn) |
| PyTorch | 2.4.0+cu121 |
| PhysicsNeMo | 1.3.0 |

### 1.2 Environment Installation

PhysicsNeMo requires Python 3.10 and a CUDA-capable NVIDIA GPU. Setup followed SETUP.md using conda (Option A — no sudo required):

```bash
conda create -n xmgn python=3.10 -c conda-forge --override-channels -y
conda activate xmgn
pip install -r requirements.txt
```

Verification:
### 1.3 Issues Encountered & Solutions

**Issue 1 — Python Version Conflict**  
The system had Python 3.13 but PhysicsNeMo requires 3.10 (torch 2.4.0 has no wheels for 3.12+). Solution: Created a dedicated conda environment with Python 3.10.

**Issue 2 — MLflow Deprecated File Store**  
MLflow ≥3.4.0 deprecated the filesystem tracking store, raising `MlflowException`. Solution: Set `MLFLOW_ALLOW_FILE_STORE=true` environment variable before training and inference.

**Issue 3 — Dataset Flat Layout Required**  
The NORNE dataset contains `INCLUDE/SUMMARY/summary.data` files that the glob pattern `**/*.DATA` picks up as fake cases, causing `FileNotFoundError`. Solution: Created a flat layout per TROUBLESHOOTING.md, copying only case-level files (`*.DATA`, `*.EGRID`, `*.INIT`, `*.UNRST`, `*.UNSMRY`, `*.SMSPEC`) to `~/data/NORNE_FLAT/`.

**Issue 4 — NORNE_008 Corrupted Timestep**  
Timestep 17 of NORNE_008 was unreadable (`unpack requires a buffer of 4 bytes`). The preprocessor automatically skipped it and continued — 59 usable cases out of 60.

**Issue 5 — Inference Shape Mismatch Bug**  
Inference failed with a shape mismatch error. Full details in Section 5 (Bug Fix).

---

## 2. Dataset & Preprocessing

### 2.1 Dataset Description

- **Source**: 60 Norne LHS cases (NORNE_001 to NORNE_060)
- **Grid**: 46×112×22 structured grid, ~44,431 active cells
- **Each case**: ~62 timesteps spanning ~9 years of simulated production
- **Variables predicted**: PRESSURE (bar), SWAT (water saturation 0–1)
- **Fault connections**: ~47 Non-Neighbor Connections (NNCs) encoding geological faults
- **Uncertainty axis**: Fault transmissibility multipliers varied via Latin Hypercube Sampling

### 2.2 Data Split

Exactly as mandated by TASK.md:

```yaml
train_ratio: 0.8    # 48 cases
val_ratio:   0.1    # 6 cases
test_ratio:  0.1    # 6 cases
random_seed: 42
```

### 2.3 Preprocessing Results

| Metric | Value |
|--------|-------|
| Total graphs created | 3,720 |
| Average graphs per case | 62 |
| Train partition files | 2,976 |
| Validation partition files | 372 |
| Test partition files | 372 |
| Node features | 13 |
| Edge features | 1 |
| Target features | 2 (PRESSURE, SWAT) |
| Partitions per graph | 3 |
| Halo size | 5 |
| Preprocessing time | ~25 minutes |

### 2.4 Config Changes from Default

| Parameter | Default | Changed To | Reason |
|-----------|---------|------------|--------|
| `sim_dir` | `../dataset/norne/NORNE_LHS.sim` | `/home/raseena/data/NORNE_FLAT` | Flat layout required to avoid INCLUDE/ glob issue |
| `random_seed` | not set | `42` | Mandated by TASK.md for reproducibility |
| `num_epochs` | `1000` | `200` | Finish within assessment deadline |
| `resume` | `false` | `true` | Continue from existing checkpoint |

---

## 3. Model Architecture

### 3.1 X-MeshGraphNet (X-MGN)

X-MGN is a graph neural network designed for physics simulation on unstructured meshes. Each reservoir grid cell is a **graph node**; fluid connections between cells are **graph edges**. The model predicts the next timestep's state autoregressively from the current state.

**Why GNN over CNN or FNO?**  
The Norne reservoir has ~47 geological faults encoded as Non-Neighbor Connections (NNCs) — edges connecting non-adjacent grid cells that break regular grid structure. CNNs and Fourier Neural Operators (FNOs) assume structured, regular grids and cannot represent NNCs. X-MGN handles NNCs naturally as additional graph edges, making it inherently suited for faulted reservoirs (see Section 9 for FNO vs X-MGN discussion).

### 3.2 Model Configuration

| Parameter | Value |
|-----------|-------|
| Message passing layers | 5 |
| Hidden dimension | 128 |
| Activation function | SiLU |
| Loss — PRESSURE | L2 (MSE) |
| Loss — SWAT | L1 (MAE) |
| Loss weights | [1.0, 1.0] |

### 3.3 Node Features (13 total)

- **Static**: PERMX (log-scaled), PORV, X, Y, Z coordinates
- **Dynamic**: PRESSURE, SWAT, WCID (current + 2 previous timesteps)

### 3.4 Edge Features (1 total)

Combined transmissibilities: TRANX, TRANY, TRANZ, TRANNNC (log-scaled)

### 3.5 Global Features

- Time step size (delta_t)
- Normalized simulation time (0–1)

---

## 4. Training

### 4.1 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Final learning rate | 0.000001 |
| LR schedule | Cosine annealing |
| Weight decay | 0.001 |
| Early stopping patience | 20 validation checks |
| Early stopping min_delta | 1e-6 |
| Validation frequency | Every 5 epochs |
| Training samples | 2,976 |
| Validation samples | 372 |

### 4.2 Training Progress

Training ran in two phases due to resumption:

- **Run 1**: Started June 10, 2026 at 17:37 — epochs 1–75
- **Run 2**: Resumed June 11, 2026 at 14:37 from epoch 76 — epochs 76–105

| Epoch | Train Loss | Best Val Loss | Note |
|-------|-----------|--------------|------|
| 46 | 0.01000 | 0.009561 | First significant improvement |
| 61 | 0.00900 | 0.008243 | Continued improvement |
| 71 | 0.00792 | 0.007714 | Improving |
| 76 | 0.00768 | 0.005674 | Large improvement |
| **85** | **0.00700** | **0.005454** | **Best checkpoint saved** |
| 105 | 0.00651 | 0.005454 | Early stopping triggered |

**Total training time**: ~29 hours on NVIDIA RTX A4000  
**Time per epoch**: ~15 minutes  
**Best checkpoint**: `MeshGraphNet.0.85.mdlus` (epoch 85)  
**Best validation loss**: 0.005454 (43% improvement from initial 0.009561)

### 4.3 Loss Curve Summary

Training loss decreased steadily from ~0.012 to ~0.006 over 105 epochs. Validation loss showed two phases of improvement (epochs 46–76) followed by plateau, triggering early stopping at epoch 105. The gap between train and val loss remained small, indicating no significant overfitting.

---

## 5. Bug Fix — Inference Shape Mismatch

### 5.1 Symptom
### 5.2 Root Cause Analysis

**Bug 1 — Wrong output array size in `_reorder_to_natural()`**

The function `_reorder_to_natural()` reorders partition-order predictions to EGRID natural order. The original code:

```python
# BUGGY:
out = np.empty_like(arr)   # creates (14941,2) — same size as input
out[perm] = arr            # perm has 44431 indices -> CRASH
```

`np.empty_like(arr)` creates an array the same shape as `arr` (14941 inner nodes of one partition). But `perm` has 44431 entries (one per active grid cell), so the assignment fails.

**Fix:**

```python
# FIXED:
n_active = len(perm)       # = 44431 (full grid)
if arr.ndim > 1:
    out = np.zeros((n_active, arr.shape[1]), dtype=arr.dtype)
else:
    out = np.zeros(n_active, dtype=arr.dtype)
out[perm] = arr            # now works correctly
```

**Bug 2 — `vec_list` contains individual partition arrays, not concatenated**

After fixing Bug 1, a second error revealed that `vec_list` (the list of prediction arrays per timestep) contained **one array per partition** (14941 each), not the concatenated full-grid array (44431). The fix concatenates before reordering:

```python
# FIXED:
concatenated = np.concatenate(vec_list, axis=0)  # 44431 cells
case_results[case_name]["predictions"][ts] = [
    self._reorder_to_natural(concatenated, perm)
]
```

### 5.3 Why This Matters

Without this fix, predictions cannot be mapped back to physical grid locations. All 6 test cases would fail inference, making visualization and per-cell analysis impossible. The fix ensures spatial correctness of all HDF5 outputs.

---

## 6. Test Set Results

### 6.1 Test Cases (seed=42)

| Case | Category | PRESSURE RMSE | SWAT RMSE |
|------|----------|--------------|-----------|
| NORNE_018 | Best | 9.14 bar | 0.01229 |
| NORNE_008 | Good | 9.32 bar | 0.01230 |
| NORNE_048 | Good | 9.35 bar | 0.01233 |
| NORNE_041 | Median | 9.66 bar | 0.01237 |
| NORNE_016 | Poor | 9.76 bar | 0.01246 |
| NORNE_002 | Worst | 10.32 bar | 0.01247 |

### 6.2 Aggregate Metrics

| Variable | Metric | Mean | P50 | P10 | P90 | Worst |
|----------|--------|------|-----|-----|-----|-------|
| PRESSURE (bar) | RMSE | 9.591 | 9.504 | 9.229 | 10.04 | 10.32 |
| PRESSURE (bar) | MAE | 4.736 | 4.748 | 4.662 | 4.797 | 4.803 |
| SWAT (fraction) | RMSE | 0.01237 | 0.01235 | 0.01230 | 0.01246 | 0.01247 |
| SWAT (fraction) | MAE | 0.003311 | 0.003294 | 0.003272 | 0.003367 | 0.003369 |

### 6.3 Discussion

PRESSURE predictions are within ~4.7 bar MAE for a reservoir with pressure range of 87–549 bar — a relative error of approximately 1.4%. SWAT predictions are very accurate at MAE=0.003 on a 0–1 scale, indicating the model correctly tracks water saturation evolution across all test cases.

Performance is remarkably consistent across cases (PRESSURE RMSE range: 9.14–10.32 bar), suggesting good generalization across different fault multiplier configurations despite only 48 training cases.

### 6.4 Autoregressive Error Drift

| Timestep | PRESSURE RMSE (bar) | SWAT RMSE |
|----------|--------------------:|----------:|
| 3 (early) | 0.90 | 0.00118 |
| 8 | 7.04 | 0.01087 |
| 13 | 12.45 | 0.03529 |
| 23 | 3.32 | 0.00440 |
| 33 | 7.66 | 0.01484 |
| 43 | 5.74 | 0.00438 |
| 53 | 12.97 | 0.01104 |
| 63 (final) | 12.80 | 0.00650 |

Early timestep predictions are highly accurate (RMSE=0.90 bar at t=3) but errors accumulate autoregressively. The non-monotonic pattern (errors decrease around t=23 then increase again) suggests the model captures medium-term dynamics but struggles at very late timesteps when error compounds significantly.

---

## 7. Visualization

Two visualizations were generated for NORNE_002 (worst case) at timestep 23:

**Visualization 1 — Value Distribution (Histogram)**  
Shows the distribution of TRUE vs PRED values across all 44,431 active cells. The TRUE and PRED distributions overlap closely for both PRESSURE and SWAT, with the DIFF histogram centered near zero, confirming no systematic bias.

**Visualization 2 — Spatial Map (Top-Down View)**  
Shows 2D spatial maps of PRESSURE and SWAT averaged over the Z (depth) dimension. The spatial patterns of high/low pressure regions are well captured. The DIFF map shows where errors concentrate — primarily at reservoir boundaries and near fault regions.

Key observations:
- Pressure spatial gradients are correctly predicted
- Water saturation fronts are tracked but with spatial smearing near NNC fault connections
- Error hotspots concentrate at grid boundaries, not in the reservoir interior

---

## 8. Failure Analysis

### 8.1 Worst Case: NORNE_002

- PRESSURE RMSE: **10.32 bar** (highest among 6 test cases)
- SWAT RMSE: **0.01247** (highest among 6 test cases)
- Early timestep error (t=3): 0.90 bar — accurate
- Late timestep error (t=53): 12.97 bar — significant drift

### 8.2 Where It Fails

The spatial visualization shows error hotspots at:
- Reservoir boundaries (edges of the active grid)
- Near fault regions (NNC connections)
- Later timesteps (t > 30) where autoregressive error has accumulated

### 8.3 Hypotheses

**1. Extreme fault multiplier values**  
NORNE_002 likely has fault multiplier values at the extremes of the Latin Hypercube design. With only 48 training cases, the model may not have seen similar fault configurations, reducing accuracy.

**2. Autoregressive error accumulation**  
The model predicts t+1 from t, then feeds its own prediction as input for t+2. Small initial errors compound over 62 timesteps. NORNE_002 shows the steepest late-timestep error growth, suggesting its early predictions are slightly off in ways that cascade.

**3. No explicit fault multiplier features**  
The model has access to transmissibility values (TRAN) as edge features but not the underlying fault multiplier values (MULTFLT). The model must infer fault behavior indirectly, which may be insufficient for extreme multiplier configurations.

**4. Limited model capacity**  
Hidden dimension of 128 with 5 message passing layers may be insufficient to fully capture the complex multi-scale interactions at fault boundaries in the Norne grid.

### 8.4 Key Insight

The primary failure mode is **autoregressive error accumulation** rather than fundamental model failure. The model is accurate at early timesteps (RMSE=0.90 bar at t=3) but errors grow substantially by t=53 (RMSE=12.97 bar). This is a known limitation of autoregressive surrogates and could be mitigated with scheduled sampling or teacher forcing during training.

---

## 9. Bonus: FNO vs X-MGN for Faulted Reservoirs

Fourier Neural Operators (FNOs) operate in the frequency domain by applying global convolutions via FFT. They assume **regular structured grids** where the convolution theorem applies. This makes them highly efficient for structured problems but fundamentally limited for faulted reservoirs like Norne.

**Why FNO would fail on Norne:**

1. **NNCs break grid regularity**: Norne's ~47 faults create Non-Neighbor Connections — edges between non-adjacent cells. These connections break the regular grid topology that FFT-based convolutions require.

2. **Cannot represent non-adjacent flow**: FNO applies convolutions on a regular mesh and cannot model fluid flow between non-adjacent cells (NNCs), which is precisely how fault-controlled flow works.

3. **Frequency domain is inadequate**: The discontinuous pressure jumps across sealed faults require high-frequency components that the FFT truncation in FNO would smooth away.

**Where FNO would beat X-MGN:**

FNO would outperform X-MGN on unfaulted reservoirs with perfectly regular grids where:
- No NNCs exist
- Grid is structurally regular
- Frequency-domain efficiency matters (FNO is faster per forward pass)
- The dataset is large enough that global convolution generalizes well

**Conclusion**: For faulted reservoirs like Norne, X-MGN's graph structure is not just preferable — it is fundamentally necessary to correctly represent fault-controlled fluid flow.

---

## 10. What I Would Do With More Time

**1. Longer training with lower learning rate**  
Resume from epoch 85 checkpoint with lr=1e-4 for 200 more epochs. The loss was still decreasing when early stopping triggered — more training with a finer learning rate may extract additional accuracy.

**2. Larger model capacity**  
Increase hidden dimension from 128 to 256 and message passing layers from 5 to 8. The current model may lack capacity to capture complex multi-scale fault interactions.

**3. Explicit fault multiplier features**  
Add fault multiplier values (MULTFLT) as explicit node/edge features. Currently the model must infer fault behavior from transmissibility values alone. Direct access to multiplier values would reduce the learning burden significantly.

**4. Scheduled sampling to reduce error drift**  
Implement scheduled sampling during training — gradually replace true inputs with model predictions. This trains the model to handle its own errors and reduces autoregressive drift, directly addressing the primary failure mode observed.

**5. Ensemble predictions**  
Train 3–5 models with different random seeds and average predictions. Ensemble averaging typically reduces variance and improves accuracy by 10–20% for surrogate models.

**6. Loss weighting by physical units**  
The current equal weighting [1.0, 1.0] treats pressure (100s of bar) and saturation (0–1) on equal footing after normalization. Tuning these weights based on physical importance and downstream use case could improve relevant metrics.

**7. Stratified evaluation**  
Analyze error patterns stratified by fault multiplier values, reservoir layer (k-index), and distance from wells. This would identify which geological scenarios the model handles poorly and guide targeted data collection.

---

## Appendix: Submission Contents

| File | Description |
|------|-------------|
| `xmgn/outputs/XMGN_Norne/best_checkpoints/MeshGraphNet.0.85.mdlus` | Best model checkpoint (epoch 85) |
| `xmgn/outputs/XMGN_Norne/best_checkpoints/checkpoint.0.85.pt` | Best optimizer checkpoint |
| `xmgn/conf/config_raseena.yaml` | Final training configuration |
| `xmgn/outputs/XMGN_Norne/inference/NORNE_002.hdf5` | Test case inference output |
| `xmgn/outputs/XMGN_Norne/inference/NORNE_008.hdf5` | Test case inference output |
| `xmgn/outputs/XMGN_Norne/inference/NORNE_016.hdf5` | Test case inference output |
| `xmgn/outputs/XMGN_Norne/inference/NORNE_018.hdf5` | Test case inference output |
| `xmgn/outputs/XMGN_Norne/inference/NORNE_041.hdf5` | Test case inference output |
| `xmgn/outputs/XMGN_Norne/inference/NORNE_048.hdf5` | Test case inference output |
| `xmgn/outputs/XMGN_Norne/viz1_histogram_NORNE_002.png` | TRUE/PRED/DIFF histogram visualization |
| `xmgn/outputs/XMGN_Norne/viz2_spatial_NORNE_002.png` | TRUE/PRED/DIFF spatial map visualization |
| `xmgn/outputs/XMGN_Norne/loss_curves.png` | Training loss curves |
| `xmgn/outputs/XMGN_Norne/accuracy/summary_matrix.txt` | Full accuracy matrix |
| `xmgn/outputs/XMGN_Norne/accuracy/per_case_metrics.csv` | Per-case metrics CSV |
| `assessment_writeup.md` | This writeup |
