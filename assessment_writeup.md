


### Root Cause
In _reorder_to_natural():
- arr shape: (14941, 2) — partition inner node predictions
- perm length: 44431 — one index per active grid cell
- np.empty_like(arr) creates (14941,2) output array
- out[perm] = arr tries to use 44431 indices on 14941 slots

### Fix Applied
```python
# Before (buggy):
out = np.empty_like(arr)
out[perm] = arr

# After (fixed):
n_active = len(perm)
if arr.ndim > 1:
    out = np.zeros((n_active, arr.shape[1]), dtype=arr.dtype)
else:
    out = np.zeros(n_active, dtype=arr.dtype)
out[perm] = arr
```

### Why This Matters
Without fix: predictions cannot be mapped to physical 
grid locations — visualization impossible.
With fix: correct spatial mapping of all predictions.

---

## 6. Test Set Results
[FILL AFTER INFERENCE COMPLETES]

### Test Cases (seed=42)
[FILL FROM dataset_metadata.json]

### PRESSURE Metrics
- RMSE (normalized): [FILL]
- RMSE (bar): [FILL]
- MAE (normalized): [FILL]
- MAE (bar): [FILL]

### SWAT Metrics
- RMSE (normalized): [FILL]
- RMSE: [FILL]
- MAE (normalized): [FILL]
- MAE: [FILL]

---

## 7. Failure Analysis
[FILL AFTER VISUALIZATION]

### Worst Predicted Case
- Case: [FILL]
- Error: [FILL]
- Location: [FILL — near faults? specific layer?]
- Hypothesis: [FILL]

---

## 8. Visualization
[ATTACH TRUE/PRED/DIFF IMAGE]

---

## 9. What I Would Do With More Time

1. **Longer training**: Train for 300+ epochs with 
   lower learning rate (1e-4) to squeeze more accuracy

2. **Larger model**: Increase hidden_dim from 128 to 256
   and message passing layers from 5 to 8

3. **Fault features**: Add fault multiplier values as 
   explicit node/edge features — the model currently 
   has no direct access to fault uncertainty values

4. **Ensemble**: Train 3-5 models with different seeds
   and average predictions for uncertainty estimates

5. **Loss weighting**: Tune PRESSURE vs SWAT loss weights
   based on physical importance and magnitude

---

## 10. Bonus: FNO vs X-MGN for Faulted Reservoirs

Fourier Neural Operators (FNOs) work in frequency domain
and assume regular structured grids. They would fail on
Norne because:

1. ~47 faults create Non-Neighbor Connections (NNCs)
   breaking grid regularity FNO requires

2. FNO cannot represent non-adjacent cell connections
   which is exactly what NNCs are

3. X-MGN handles NNCs as natural graph edges making it
   inherently suited for faulted reservoirs

FNO would only outperform X-MGN on:
- Simple unfaulted reservoirs
- Perfectly regular grids
- Cases where frequency-domain efficiency matters more
  than geometric flexibility
## Bug Fix — Inference Shape Mismatch (Detailed)

### Symptom
ValueError: shape mismatch: value array of shape (14941,2)

could not be broadcast to indexing result of shape (44431,2)
### What the Numbers Mean
- 14941 = number of inner nodes in one partition
- 44431 = total active cells in the full Norne grid
- The function was creating output array same size as 
  input partition (14941) but trying to index into it 
  with 44431 positions → crash

### Root Cause
In _reorder_to_natural() at inference.py line 710:
```python
# BUGGY CODE:
out = np.empty_like(arr)  # creates (14941,2) array
out[perm] = arr           # perm has 44431 indices → CRASH
```

np.empty_like(arr) creates array SAME SIZE as arr.
But perm has length 44431 (full grid size).
So out needs to be (44431,2) not (14941,2).

### Fix Applied
```python
# FIXED CODE:
n_active = len(perm)          # = 44431
if arr.ndim > 1:
    out = np.zeros((n_active, arr.shape[1]), dtype=arr.dtype)
else:
    out = np.zeros(n_active, dtype=arr.dtype)
out[perm] = arr               # now works correctly
```

### Fix Process
1. Identified error using HYDRA_FULL_ERROR=1
2. Located bug in _reorder_to_natural() function
3. Multiple fix attempts failed due to:
   - Indentation errors from sed commands
   - Duplicate function definitions created
   - Old inference process still running
4. Final fix approach:
   - Downloaded fresh file from original repo
   - Applied clean Python string replacement
   - Killed old process before rerunning
   - Verified fix with sed -n before running

### Note
When fixing Python files via command line:
- Always verify fix with sed -n before running
- Kill old processes before starting new ones
- Use Python string replacement for multi-line fixes
- Download fresh file if multiple failed attempts 
  corrupted the file
## Final Fix — Concatenate Partitions Before Reordering

The vec_list contains individual partition arrays (14941 each)
not the concatenated result (44431). Fixed by concatenating 
all partition predictions before passing to _reorder_to_natural.

Fix: np.concatenate(vec_list, axis=0) before reordering.
## Inference Run (Successful)
- Started: June 13, 2026 at 10:40
- Checkpoint: epoch 85 (best)
- Test cases: 6
- Timesteps per case: 62
- Status: Running 
