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

## Inference Run (Successful)
- Started: June 13, 2026 at 10:40
- Checkpoint: epoch 85 (best)
- Test cases: 6
- Timesteps per case: 62
- Status: Running 
