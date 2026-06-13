# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Inference script for XMeshGraphNet on reservoir simulation data.
Loads the best checkpoint and performs autoregressive inference on test samples.
Generates GRDECL files with predictions for post-processing.
"""

import os
import sys
import json
import glob
from datetime import datetime, timezone

# Add repository root to Python path for sim_utils import
current_dir = os.path.dirname(os.path.abspath(__file__))  # This is src/
repo_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels from src/
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn as nn
import numpy as np
import h5py
import hydra
from omegaconf import DictConfig

# Patched for nvidia-physicsnemo >= 1.3.0: modules relocated.
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.launch.utils.checkpoint import load_checkpoint
from data.dataloader import GraphDataset, load_stats, find_pt_files
from sim_utils import EclReader, Grid
from utils import get_dataset_paths, fix_layernorm_compatibility

# Fix LayerNorm compatibility issue
fix_layernorm_compatibility()


def InitializeLoggers(cfg: DictConfig):
    """Initialize distributed manager and loggers for inference."""
    DistributedManager.initialize()
    dist = DistributedManager()
    logger = PythonLogger(name="xmgn_inference")

    logger.info("XMeshGraphNet - Autoregressive Inference for Reservoir Simulation")

    return dist, RankZeroLoggingWrapper(logger, dist)


class InferenceRunner:
    """Inference runner for XMeshGraphNet."""

    def __init__(self, cfg: DictConfig, dist, logger):
        """Initialize the inference runner."""
        self.cfg = cfg
        self.dist = dist
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up paths with job name
        paths = get_dataset_paths(cfg)
        self.dataset_dir = paths["dataset_dir"]
        self.stats_file = paths["stats_file"]
        self.test_partitions_path = paths["test_partitions_path"]

        # Set up inference output directory
        self.inference_output_dir = "inference"
        self.inference_metadata_file = os.path.join(
            self.inference_output_dir, "inference_metadata.json"
        )

        # Load global statistics
        self.stats = load_stats(self.stats_file)

        # Get model dimensions
        input_dim_nodes = len(self.stats["node_features"]["mean"])
        input_dim_edges = len(self.stats["edge_features"]["mean"])
        output_dim = len(cfg.dataset.graph.target_vars.node_features)

        # Initialize model
        self.model = MeshGraphNet(
            input_dim_nodes=input_dim_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=output_dim,
            processor_size=cfg.model.num_message_passing_layers,
            aggregation="sum",
            hidden_dim_node_encoder=cfg.model.hidden_dim,
            hidden_dim_edge_encoder=cfg.model.hidden_dim,
            hidden_dim_node_decoder=cfg.model.hidden_dim,
            mlp_activation_fn=cfg.model.activation,
            do_concat_trick=cfg.performance.use_concat_trick,
        ).to(self.device)

        # Load best checkpoint using PhysicsNeMo's load_checkpoint (same as training)
        # Set up checkpoint arguments (same as training)
        base_output_dir = os.getcwd()
        best_checkpoint_dir = os.path.join(base_output_dir, "best_checkpoints")

        # Set up checkpoint arguments (following training pattern - same as bst_ckpt_args)
        ckpt_args = {
            "path": best_checkpoint_dir,
            "models": self.model,
        }

        # Check for explicit checkpoint paths in config
        explicit_checkpoint = getattr(cfg.inference, "checkpoint_path", None)
        explicit_model = getattr(cfg.inference, "model_path", None)

        if explicit_checkpoint or explicit_model:
            # Use explicit checkpoint/model paths
            if explicit_checkpoint:
                self.logger.info(f"Using explicit checkpoint: {explicit_checkpoint}")
                checkpoint = torch.load(explicit_checkpoint, map_location=self.device)

                # Load model state
                if "models" in checkpoint:
                    self.model.load_state_dict(checkpoint["models"])
                else:
                    self.model.load_state_dict(checkpoint)

                # Extract epoch from filename if possible
                filename = os.path.basename(explicit_checkpoint)
                try:
                    parts = filename.split(".")
                    if len(parts) >= 3:
                        loaded_epoch = int(parts[2])
                    else:
                        loaded_epoch = 0
                except Exception:
                    loaded_epoch = 0

                self.logger.info(
                    f"Loaded explicit checkpoint from epoch {loaded_epoch}"
                )

            elif explicit_model:
                self.logger.info(f"Using explicit model: {explicit_model}")
                # For .mdlus files, we need to use PhysicsNeMo's load_checkpoint
                model_ckpt_args = {
                    "path": os.path.dirname(explicit_model),
                    "models": self.model,
                }
                loaded_epoch = load_checkpoint(**model_ckpt_args, device=self.device)
                self.logger.info(f"Loaded explicit model from epoch {loaded_epoch}")
        else:
            # Use automatic best checkpoint selection
            self.logger.info("Using automatic best checkpoint selection")

            # Check for multiple checkpoint files and log them
            if os.path.exists(best_checkpoint_dir):
                checkpoint_files = [
                    f for f in os.listdir(best_checkpoint_dir) if f.endswith(".pt")
                ]
                if len(checkpoint_files) > 1:
                    self.logger.info(
                        f"Found {len(checkpoint_files)} checkpoint files in best_checkpoints:"
                    )
                    for file in sorted(checkpoint_files):
                        self.logger.info(f"   - {file}")
                    self.logger.info(
                        "PhysicsNeMo will automatically select the best performing checkpoint"
                    )

            # Load checkpoint using PhysicsNeMo's system
            loaded_epoch = load_checkpoint(**ckpt_args, device=self.device)
            self.logger.info(f"Loaded BEST checkpoint from epoch {loaded_epoch}")

        self.model.eval()
        self.logger.info(f"Checkpoint directory: {best_checkpoint_dir}")

        # Create test dataset (following training pattern)
        # Find partition files
        file_paths = find_pt_files(self.test_partitions_path)

        # Load per-feature statistics
        node_mean = torch.tensor(self.stats["node_features"]["mean"])
        node_std = torch.tensor(self.stats["node_features"]["std"])
        edge_mean = torch.tensor(self.stats["edge_features"]["mean"])
        edge_std = torch.tensor(self.stats["edge_features"]["std"])

        # Load target feature statistics (if available)
        target_mean = None
        target_std = None
        if "target_features" in self.stats:
            target_mean = torch.tensor(self.stats["target_features"]["mean"])
            target_std = torch.tensor(self.stats["target_features"]["std"])

        # Create dataset
        self.test_dataset = GraphDataset(
            file_paths,
            node_mean,
            node_std,
            edge_mean,
            edge_std,
            target_mean,
            target_std,
        )

        self.logger.info(f"Test dataset loaded with {len(self.test_dataset)} samples")

    def denormalize_predictions(self, pred):
        """Denormalize predictions using global statistics."""
        target_mean = torch.tensor(
            self.stats["target_features"]["mean"], device=self.device
        )
        target_std = torch.tensor(
            self.stats["target_features"]["std"], device=self.device
        )
        return pred * target_std + target_mean

    def denormalize_targets(self, target):
        """Denormalize targets using global statistics."""
        target_mean = torch.tensor(
            self.stats["target_features"]["mean"], device=self.device
        )
        target_std = torch.tensor(
            self.stats["target_features"]["std"], device=self.device
        )
        return target * target_std + target_mean

    def _get_target_feature_indices(self):
        """
        Get the indices in node features that correspond to target variables.
        These are the features we need to replace with predictions during autoregressive inference.
        """
        # Get target variable names
        target_vars = self.cfg.dataset.graph.target_vars.node_features

        # Get dynamic variable names from config
        dynamic_vars = self.cfg.dataset.graph.node_features.dynamic.variables

        # Find indices of target variables in dynamic variables
        target_indices = []
        for target_var in target_vars:
            if target_var in dynamic_vars:
                idx = dynamic_vars.index(target_var)
                target_indices.append(idx)

        return target_indices

    def _update_node_features_with_predictions(
        self, partitions_list, predictions_normalized
    ):
        """
        Update node features in partitions with predictions from previous timestep.
        Replace only the features that correspond to target variables.

        Parameters
            partitions_list: List of graph partitions
            predictions_normalized: Normalized predictions from previous timestep (list of arrays per partition)

        Returns
            Updated partitions_list with predictions in node features
        """
        target_indices = self._get_target_feature_indices()

        # Get the number of dynamic variables to know the offset in node features
        num_static_features = len(self.cfg.dataset.graph.node_features.static)
        num_dynamic_features = len(
            self.cfg.dataset.graph.node_features.dynamic.variables
        )
        prev_timesteps = self.cfg.dataset.graph.node_features.dynamic.prev_timesteps

        # Dynamic features start after static features
        # For prev_timesteps=0: dynamic features are at indices [num_static: num_static+num_dynamic]
        # For prev_timesteps>0: current timestep is at the end of dynamic features

        if prev_timesteps == 0:
            # Current timestep dynamic features start at num_static_features
            dynamic_offset = num_static_features
        else:
            # Current timestep is at the last block of dynamic features
            dynamic_offset = num_static_features + prev_timesteps * num_dynamic_features

        # Update each partition
        updated_partitions = []
        for partition, pred_array in zip(partitions_list, predictions_normalized):
            # Clone the partition to avoid modifying the original
            # PyTorch Geometric Data objects need special handling
            if hasattr(partition, "clone"):
                partition = partition.clone()

            # Clone the node features tensor
            partition.x = partition.x.clone()

            # Convert prediction array to tensor if needed
            if isinstance(pred_array, np.ndarray):
                pred_tensor = torch.tensor(
                    pred_array, dtype=torch.float32, device=partition.x.device
                )
            else:
                pred_tensor = (
                    pred_array.clone() if hasattr(pred_array, "clone") else pred_array
                )

            # Replace target features in node features with predictions
            # Note: predictions are only for inner nodes (excluding halo nodes)
            for i, target_idx in enumerate(target_indices):
                feature_idx = dynamic_offset + target_idx
                if hasattr(partition, "inner_node"):
                    # Update only inner nodes (predictions don't include halo nodes)
                    partition.x[partition.inner_node, feature_idx] = pred_tensor[:, i]
                else:
                    # No halo nodes, update all nodes
                    partition.x[:, feature_idx] = pred_tensor[:, i]

            updated_partitions.append(partition)

        return updated_partitions

    def evaluate_sample(
        self,
        partitions_list,
        use_predictions_as_input=False,
        prev_predictions_normalized=None,
    ):
        """
        Evaluate a single sample (list of partitions).

        Parameters
            partitions_list: List of graph partitions for this timestep
            use_predictions_as_input: If True, replace target features with predictions from previous timestep
            prev_predictions_normalized: Normalized predictions from previous timestep (for autoregressive inference)

        Returns
            avg_loss, avg_denorm_loss, predictions, targets, predictions_normalized
        """
        total_loss = 0.0
        total_denorm_loss = 0.0
        num_partitions = 0

        predictions = []
        targets = []
        predictions_normalized = []  # Store normalized predictions for next timestep

        with torch.no_grad():
            # If using autoregressive mode, update node features with previous predictions
            if use_predictions_as_input and prev_predictions_normalized is not None:
                partitions_list = self._update_node_features_with_predictions(
                    partitions_list, prev_predictions_normalized
                )

            for partition in partitions_list:
                partition = partition.to(self.device)

                # Ensure data is in float32
                if hasattr(partition, "x") and partition.x is not None:
                    partition.x = partition.x.float()
                if hasattr(partition, "edge_attr") and partition.edge_attr is not None:
                    partition.edge_attr = partition.edge_attr.float()
                if hasattr(partition, "y") and partition.y is not None:
                    partition.y = partition.y.float()

                # Forward pass
                pred = self.model(partition.x, partition.edge_attr, partition)

                # Get inner nodes if available
                if hasattr(partition, "inner_node"):
                    pred_inner = pred[partition.inner_node]
                    target_inner = partition.y[partition.inner_node]
                else:
                    pred_inner = pred
                    target_inner = partition.y

                # Calculate losses
                loss = torch.nn.functional.mse_loss(pred_inner, target_inner)

                # Denormalize for physical units
                pred_denorm = self.denormalize_predictions(pred_inner)
                target_denorm = self.denormalize_targets(target_inner)
                denorm_loss = torch.nn.functional.mse_loss(pred_denorm, target_denorm)

                total_loss += loss.item()
                total_denorm_loss += denorm_loss.item()
                num_partitions += 1

                # Store predictions and targets
                predictions.append(pred_denorm.cpu().numpy())
                targets.append(target_denorm.cpu().numpy())

                # Store normalized predictions for next timestep's input
                predictions_normalized.append(pred_inner.cpu().numpy())

        avg_loss = total_loss / num_partitions if num_partitions > 0 else 0.0
        avg_denorm_loss = (
            total_denorm_loss / num_partitions if num_partitions > 0 else 0.0
        )

        return avg_loss, avg_denorm_loss, predictions, targets, predictions_normalized

    def _extract_case_and_timestep(self, filename):
        """Extract case name and time step from filename.

        PATCHED #5 (3-part case-name parsing):
        Handles two filename layouts:
          * ``CASE_2D_1_000.pt``        → 4+ underscore parts  (xaeronet-style)
          * ``NORNE_001_012.pt``         → 3 underscore parts   (Norne-style)

        The original code only handled the 4+ part case and fell through to a
        broken fallback for 3-part names, treating every (case, timestep) as
        a unique single-timestep "case" — which disabled autoregressive rollout
        entirely. The 3-part branch fixes that for the Norne pipeline.
        """

        if filename.startswith("partitions_"):
            # Remove 'partitions_' prefix
            filename = filename[11:]  # Remove 'partitions_'

        # Remove .pt extension
        filename = filename.replace(".pt", "")

        # Split by underscore and extract case name and time step
        parts = filename.split("_")
        if len(parts) >= 4:
            # Format: CASE_2D_1_000
            case_name = "_".join(parts[:-1])  # CASE_2D_1
            timestep = parts[-1]  # 000
        elif len(parts) == 3:
            # Format: NORNE_001_012  (Norne reservoir dataset)
            case_name = "_".join(parts[:-1])  # NORNE_001
            timestep = parts[-1]  # 012
        else:
            # Fallback (unknown layout — treat as single-timestep case)
            case_name = filename
            timestep = "000"

        return case_name, timestep

    def run_inference(self):
        """Run autoregressive inference on test dataset."""
        self.logger.info("=" * 70)
        self.logger.info("STARTING INFERENCE")
        self.logger.info("=" * 70)

        # Get prev_timesteps config for determining initial conditions
        prev_timesteps = self.cfg.dataset.graph.node_features.dynamic.prev_timesteps
        num_initial_true_timesteps = (
            prev_timesteps + 1
        )  # Initial + prev_timesteps as true inputs

        self.logger.info(
            f"Initial timesteps with true features: {num_initial_true_timesteps}"
        )
        self.logger.info(
            f"Subsequent timesteps: predictions feed into next timestep (autoregressive)"
        )

        # First, organize all samples by case and timestep
        case_timestep_data = {}
        for idx in range(len(self.test_dataset)):
            file_path = self.test_dataset.file_paths[idx]
            filename = os.path.basename(file_path)
            case_name, timestep = self._extract_case_and_timestep(filename)

            if case_name not in case_timestep_data:
                case_timestep_data[case_name] = {}

            case_timestep_data[case_name][timestep] = idx

        # Now process each case autoregressively
        # PATCHED: per-case streaming save to bound memory. The original code
        # accumulated all 432 cases × 62 timesteps × 44k cells of predictions
        # in RAM until the loop ended, then wrote HDF5s in one batch. On the
        # 432-case Norne eval that's ~18 GB peak — past WSL2's default 16 GB
        # memory cap → swap thrash → hang. Now we save each case's HDF5 as
        # soon as its rollout completes and drop the in-memory data, plus
        # accumulate aggregate metrics as running sums (mathematically equiv).
        total_loss = 0.0
        total_denorm_loss = 0.0
        num_samples = 0
        case_results = {}  # holds only the CURRENT case's data

        # Running sums for aggregate MAE/MSE (cell-weighted, exact)
        # Per-variable: list of (abs_err_sum, sq_err_sum, count) tuples
        n_vars = len(self.cfg.dataset.graph.target_vars.node_features)
        abs_err_sum_per_var = np.zeros(n_vars, dtype=np.float64)
        sq_err_sum_per_var = np.zeros(n_vars, dtype=np.float64)
        cells_per_var = np.zeros(n_vars, dtype=np.int64)
        completed_case_names = []  # for the final log line

        all_cases = sorted(case_timestep_data.keys())
        total_cases = len(all_cases)
        self.logger.info(f"Processing {total_cases} cases...")

        for case_idx, case_name in enumerate(all_cases, 1):
            case_results[case_name] = {
                "predictions": {},
                "targets": {},
                "losses": [],
                "denorm_losses": [],
            }

            # Get sorted timesteps for this case
            timesteps = sorted(case_timestep_data[case_name].keys())

            self.logger.info(
                f"[{case_idx}/{total_cases}] Processing case: {case_name} ({len(timesteps)} timesteps)"
            )

            # Track predictions from previous timestep (normalized)
            prev_predictions_normalized = None

            # Process each timestep in order
            for timestep_idx, timestep in enumerate(timesteps):
                idx = case_timestep_data[case_name][timestep]
                partitions_list, label = self.test_dataset[idx]

                # Determine if we should use predictions as input
                # Use true features for first num_initial_true_timesteps, then use predictions
                use_predictions_as_input = timestep_idx >= num_initial_true_timesteps

                # Evaluate this timestep
                loss, denorm_loss, predictions, targets, predictions_normalized = (
                    self.evaluate_sample(
                        partitions_list,
                        use_predictions_as_input=use_predictions_as_input,
                        prev_predictions_normalized=prev_predictions_normalized,
                    )
                )

                total_loss += loss
                total_denorm_loss += denorm_loss
                num_samples += 1

                # Store results
                case_results[case_name]["predictions"][timestep] = predictions
                case_results[case_name]["targets"][timestep] = targets
                case_results[case_name]["losses"].append(loss)
                case_results[case_name]["denorm_losses"].append(denorm_loss)

                # Update running aggregate metrics per variable
                # (cell-weighted, denormalized — matches build_accuracy_matrix.py)
                if predictions and targets:
                    pred_arr = np.concatenate(predictions, axis=0)
                    tgt_arr = np.concatenate(targets, axis=0)
                    diff = pred_arr.astype(np.float64) - tgt_arr.astype(np.float64)
                    for v in range(min(n_vars, diff.shape[1])):
                        abs_err_sum_per_var[v] += float(np.sum(np.abs(diff[:, v])))
                        sq_err_sum_per_var[v] += float(np.sum(diff[:, v] ** 2))
                        cells_per_var[v] += int(diff.shape[0])

                # Store predictions for next timestep
                prev_predictions_normalized = predictions_normalized

            # PATCHED: reorder predictions/targets from METIS-partition order
            # to EGRID-natural (i,j,k active-cell) order before saving HDF5.
            # The inference loop produces values in the order it processed
            # partitions; without this reorder, every per-cell value lands at
            # the wrong physical location downstream (e.g. in ResInsight),
            # even though aggregate metrics are still correct. See PATCHES.md
            # patch #7 for the diagnostic story.
            try:
                perm = self._build_natural_order_permutation(case_name)
            except Exception as e:
                self.logger.warning(
                    f"Could not build natural-order permutation for "
                    f"{case_name} ({e}); HDF5 will be in partition order."
                )
                perm = None
            if perm is not None:
                for ts, vec_list in case_results[case_name]["predictions"].items():
                    case_results[case_name]["predictions"][ts] = [
                        self._reorder_to_natural(arr, perm) for arr in vec_list
                    ]
                for ts, vec_list in case_results[case_name]["targets"].items():
                    case_results[case_name]["targets"][ts] = [
                        self._reorder_to_natural(arr, perm) for arr in vec_list
                    ]

            # Save THIS case's HDF5 immediately, then free its in-memory data
            self._save_case_results_hdf5(
                {case_name: case_results[case_name]},
                ordering=("natural" if perm is not None else "partition"),
            )
            del case_results[case_name]
            completed_case_names.append(case_name)

        # Calculate final metrics
        avg_loss = total_loss / num_samples
        avg_denorm_loss = total_denorm_loss / num_samples

        # Aggregate MAE/RMSE from running sums (exact, never materialized full arrays)
        total_abs = float(abs_err_sum_per_var.sum())
        total_sq = float(sq_err_sum_per_var.sum())
        total_cells = int(cells_per_var.sum())
        mae = total_abs / total_cells if total_cells else 0.0
        mse = total_sq / total_cells if total_cells else 0.0
        rmse = float(np.sqrt(mse))

        # all_predictions/all_targets are intentionally None — the original code
        # returned them concatenated for downstream consumers, but the only such
        # consumer (the HDF5 file) is now written per-case in the loop above.
        all_predictions = None
        all_targets = None

        # Write master inference metadata (drives the GRDECL post-processor).
        # The per-case _save_case_results_hdf5 no longer writes this — it would
        # have overwritten the file 432 times, leaving only the last case.
        target_names_str = [
            str(name) for name in self.cfg.dataset.graph.target_vars.node_features
        ]
        all_hdf5_files = [f"{cn}.hdf5" for cn in completed_case_names]
        master_metadata = {
            "hdf5_files": all_hdf5_files,
            "num_cases": len(completed_case_names),
            "target_variables": target_names_str,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        os.makedirs(self.inference_output_dir, exist_ok=True)
        with open(self.inference_metadata_file, "w") as f:
            json.dump(master_metadata, f, indent=2)
        self.logger.info(
            f"Master metadata: {self.inference_metadata_file} "
            f"({len(all_hdf5_files)} HDF5 files listed)"
        )

        # Log final results
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("AUTOREGRESSIVE INFERENCE RESULTS")
        self.logger.info("=" * 70)
        self.logger.info(f"Test samples processed: {num_samples}")
        self.logger.info(f"Simulation cases: {len(completed_case_names)}")
        self.logger.info(f"Average normalized MSE: {avg_loss:.6e}")
        self.logger.info(f"Average denormalized MSE: {avg_denorm_loss:.6e}")
        self.logger.info(f"Overall MAE: {mae:.6e}")
        self.logger.info(f"Overall RMSE: {rmse:.6e}")
        self.logger.info("")
        self.logger.info("Per-Variable Metrics:")
        self.logger.info("-" * 70)

        # Per-variable metrics from running sums
        target_names = self.cfg.dataset.graph.target_vars.node_features
        for i, var_name in enumerate(target_names):
            var_count = cells_per_var[i] if i < n_vars else 0
            var_mae = abs_err_sum_per_var[i] / var_count if var_count else 0.0
            var_rmse = float(np.sqrt(sq_err_sum_per_var[i] / var_count)) if var_count else 0.0
            self.logger.info(
                f"  {var_name:>12s}  |  MAE: {var_mae:>12.6e}  |  RMSE: {var_rmse:>12.6e}"
            )

        self.logger.info("=" * 70)

        return {
            "avg_loss": avg_loss,
            "avg_denorm_loss": avg_denorm_loss,
            "mae": mae,
            "rmse": rmse,
            "predictions": all_predictions,
            "targets": all_targets,
            "num_samples": num_samples,
            "case_results": case_results,
        }

    def _build_natural_order_permutation(self, case_name):
        """Return a numpy array `perm` of length n_active where
        `pred_natural[perm] = pred_partition_order` maps the inference loop's
        partition-order concatenation into EGRID natural (i,j,k) active-cell
        order. Recovered from any partition .pt file for this case — every
        timestep's partitioning is identical, so we just load one.
        """
        import glob as _glob
        # Try the organized layout first (partitions/test/...) then top-level
        partitions_dir = self.cfg_partitions_dir if hasattr(self, "cfg_partitions_dir") else None
        if partitions_dir is None:
            # Derive from dataset metadata
            dataset_dir = os.path.dirname(self.stats_file) if hasattr(self, "stats_file") else None
            if dataset_dir:
                partitions_dir = os.path.join(dataset_dir, "partitions")
        if not partitions_dir or not os.path.isdir(partitions_dir):
            raise FileNotFoundError("partitions directory not found for case_name")

        candidates = (
            _glob.glob(os.path.join(partitions_dir, "test", f"partitions_{case_name}_*.pt"))
            + _glob.glob(os.path.join(partitions_dir, "val", f"partitions_{case_name}_*.pt"))
            + _glob.glob(os.path.join(partitions_dir, "train", f"partitions_{case_name}_*.pt"))
            + _glob.glob(os.path.join(partitions_dir, f"partitions_{case_name}_*.pt"))
        )
        if not candidates:
            raise FileNotFoundError(f"no partition .pt for {case_name}")
        pt_path = sorted(candidates)[0]
        partitions_list = torch.load(pt_path, weights_only=False, map_location="cpu")
        global_indices = []
        for part in partitions_list:
            inner_global = part.part_node[part.inner_node].cpu().numpy()
            global_indices.append(inner_global)
        return np.concatenate(global_indices).astype(np.int64)

    @staticmethod
    def _reorder_to_natural(arr, perm):
        """Reorder a partition-order array into natural order: natural[perm] = arr."""
        arr = np.asarray(arr)
        n_active = len(perm)
        if arr.ndim > 1:
            out = np.zeros((n_active, arr.shape[1]), dtype=arr.dtype)
        else:
            out = np.zeros(n_active, dtype=arr.dtype)
        out[perm] = arr
        return out

    def _save_case_results_hdf5(self, case_results, ordering="partition"):
        """Save inference results per simulation case as HDF5 files.

        `ordering`: either "natural" (EGRID i,j,k active-cell order — the
        convention UNRST and downstream tools expect) or "partition" (the
        order the inference loop produced; legacy / unpatched output). The
        chosen value is written as an HDF5 root attribute so consumers can
        detect stale-format files.
        """
        os.makedirs(self.inference_output_dir, exist_ok=True)

        self.logger.info("")
        self.logger.info("Saving inference results to HDF5 files...")

        target_names = self.cfg.dataset.graph.target_vars.node_features

        for case_name, case_data in case_results.items():
            hdf5_file = os.path.join(self.inference_output_dir, f"{case_name}.hdf5")

            with h5py.File(hdf5_file, "w") as f:
                # Create groups for predictions and targets
                pred_group = f.create_group("predictions")
                target_group = f.create_group("targets")

                # Save metadata
                f.attrs["case_name"] = case_name
                f.attrs["num_timesteps"] = len(case_data["predictions"])
                f.attrs["target_variables"] = [
                    str(name) for name in target_names
                ]  # Convert to list of strings
                # PATCHED: cell-ordering attribute. "natural" means
                # values are in EGRID active-cell (i,j,k) order — the
                # UNRST convention. "partition" means METIS-partition
                # order — the legacy NVIDIA output. See PATCHES.md #7.
                f.attrs["ordering"] = ordering
                f.attrs["avg_loss"] = np.mean(case_data["losses"])
                f.attrs["avg_denorm_loss"] = np.mean(case_data["denorm_losses"])

                # Organize data by variable (PRESSURE, SWAT) with lists of vectors per timestep
                for i, var_name in enumerate(target_names):
                    var_name_clean = var_name.upper()  # Use capital case

                    # Collect all timestep data for this variable
                    pred_vectors = []
                    target_vectors = []
                    timestep_numbers = []  # Track actual timestep numbers

                    for timestep in sorted(case_data["predictions"].keys()):
                        predictions = case_data["predictions"][timestep]
                        targets = case_data["targets"][timestep]

                        if predictions:
                            pred_array = np.concatenate(predictions, axis=0)
                            target_array = np.concatenate(targets, axis=0)

                            # Extract this variable's data (column i)
                            pred_vectors.append(pred_array[:, i])
                            target_vectors.append(target_array[:, i])
                            # Store actual timestep number (predictions are FOR next timestep)
                            timestep_numbers.append(int(timestep))

                    # Save as variable groups with lists of vectors
                    if pred_vectors:
                        # Create variable groups
                        var_pred_group = pred_group.create_group(var_name_clean)
                        var_target_group = target_group.create_group(var_name_clean)

                        # Save each timestep as a separate dataset within the variable group
                        for input_timestep, pred_vec, target_vec in zip(
                            timestep_numbers, pred_vectors, target_vectors
                        ):
                            # Predictions are FOR the next timestep after the input
                            predicted_timestep = input_timestep + 1
                            var_pred_group.create_dataset(
                                f"timestep_{predicted_timestep:04d}", data=pred_vec
                            )
                            var_target_group.create_dataset(
                                f"timestep_{predicted_timestep:04d}", data=target_vec
                            )

                        # Save metadata for this variable
                        var_pred_group.attrs["num_timesteps"] = len(pred_vectors)
                        var_pred_group.attrs["num_nodes"] = (
                            len(pred_vectors[0]) if pred_vectors else 0
                        )
                        var_target_group.attrs["num_timesteps"] = len(target_vectors)
                        var_target_group.attrs["num_nodes"] = (
                            len(target_vectors[0]) if target_vectors else 0
                        )

        # NOTE: metadata-file writing intentionally removed from this method.
        # The original implementation wrote inference_metadata.json on every
        # call to _save_case_results_hdf5 — fine when the original code passed
        # the FULL case_results dict once at the end, but broken under the
        # per-case streaming pattern we now use (each call would overwrite
        # the JSON with just one case → post-processor would only see 1 file).
        # Metadata is now written once at the end of run_inference() from the
        # accumulated `completed_case_names` list.
        self.logger.info(
            f"Saved {len(case_results)} HDF5 file(s) to: {self.inference_output_dir}"
        )

    def _load_partition_and_halo_info(self, case_name):
        """
        Load partition assignments and halo information for a given case.
        First tries to read from partition .pt files (preferred), then falls back to JSON.

        Parameters
            case_name: Name of the simulation case

        Returns
            tuple: (partition_assignment, halo_info)
                - partition_assignment: numpy array with partition IDs for active cells (1-indexed), or None
                - halo_info: numpy array indicating which partition includes this cell as halo (0=none, partition_id if halo), or None
        """
        # First, try to find partition .pt file (from first timestep)
        partitions_dir = os.path.join(self.dataset_dir, "partitions")

        # Check in test/train/val subdirectories
        for split in ["test", "train", "val"]:
            split_dir = os.path.join(partitions_dir, split)

            # Find the first partition file for this case (any timestep)
            partition_pt_file = None
            if os.path.exists(split_dir):
                pattern = os.path.join(split_dir, f"partitions_{case_name}_*.pt")
                matching_files = sorted(glob.glob(pattern))

                if matching_files:
                    partition_pt_file = matching_files[
                        0
                    ]  # Use first available timestep

            if partition_pt_file and os.path.exists(partition_pt_file):
                try:
                    # Load partition data from .pt file
                    partitions = torch.load(
                        partition_pt_file, map_location="cpu", weights_only=False
                    )

                    num_partitions = len(partitions)

                    # Build partition assignment and halo info
                    partition_assignments_dict = {}
                    halo_info_dict = {}  # Which partition includes this cell as halo

                    for part_idx, partition in enumerate(partitions):
                        if hasattr(partition, "part_node") and hasattr(
                            partition, "inner_node"
                        ):
                            # Get inner nodes (these belong to this partition)
                            inner_global_indices = (
                                partition.part_node[partition.inner_node].cpu().numpy()
                            )
                            for global_idx in inner_global_indices:
                                partition_assignments_dict[global_idx] = (
                                    part_idx + 1
                                )  # 1-indexed

                            # Get halo nodes (all nodes NOT in inner_node)
                            all_local_indices = torch.arange(partition.num_nodes)
                            halo_mask = torch.ones(
                                partition.num_nodes, dtype=torch.bool
                            )
                            halo_mask[partition.inner_node] = False
                            halo_local_indices = all_local_indices[halo_mask]
                            halo_global_indices = (
                                partition.part_node[halo_local_indices].cpu().numpy()
                            )

                            # Mark these cells as being halo in this partition
                            for global_idx in halo_global_indices:
                                halo_info_dict[global_idx] = (
                                    part_idx + 1
                                )  # Which partition includes this as halo

                    # Sort by node index and create assignment lists
                    # Include both inner nodes AND halo nodes
                    all_node_indices = set(partition_assignments_dict.keys()) | set(
                        halo_info_dict.keys()
                    )
                    sorted_indices = sorted(all_node_indices)

                    partition_assignment = np.array(
                        [
                            partition_assignments_dict.get(idx, 0)
                            for idx in sorted_indices
                        ],
                        dtype=int,
                    )

                    # Create halo info array (0 = not halo, partition_id = included as halo in that partition)
                    halo_info = np.array(
                        [halo_info_dict.get(idx, 0) for idx in sorted_indices],
                        dtype=int,
                    )

                    num_halo_cells = np.count_nonzero(halo_info)

                    # Debug: check how many cells are halo-only vs inner+halo
                    num_inner_cells = np.count_nonzero(partition_assignment)
                    num_halo_only = np.sum(
                        (halo_info > 0) & (partition_assignment == 0)
                    )
                    num_inner_and_halo = np.sum(
                        (halo_info > 0) & (partition_assignment > 0)
                    )

                    self.logger.info(
                        f"Loaded partition assignments from {split}/{os.path.basename(partition_pt_file)}: "
                        f"{num_partitions} partitions, {len(partition_assignment)} active cells"
                    )
                    self.logger.info(
                        f"   Inner cells: {num_inner_cells}, Halo-only: {num_halo_only}, Inner+Halo: {num_inner_and_halo}"
                    )

                    return partition_assignment, halo_info

                except Exception as e:
                    self.logger.warning(
                        f"Failed to load partitions from {partition_pt_file}: {e}"
                    )
                    continue

        # Fall back to JSON file if .pt file not found
        partition_json_file = os.path.join(
            self.dataset_dir, f"{case_name}_partitions.json"
        )

        if os.path.exists(partition_json_file):
            try:
                with open(partition_json_file, "r") as f:
                    partition_data = json.load(f)

                partition_assignment = np.array(
                    partition_data["partition_assignment"], dtype=int
                )

                self.logger.info(
                    f"Loaded partition assignments from JSON: "
                    f"{partition_data['num_partitions']} partitions, "
                    f"{partition_data['num_nodes']} active cells "
                    f"(halo info not available from JSON)"
                )

                # JSON doesn't have halo info, return None for halo
                return partition_assignment, None

            except Exception as e:
                self.logger.warning(
                    f"Failed to load partition assignments from JSON: {e}"
                )

        # Neither .pt nor JSON found
        self.logger.warning(
            f"No partition data found for {case_name}. PARTITION block will be skipped."
        )
        return None, None

    def _extract_coordinates_from_grid(self, sample_idx):
        """Extract coordinates from grid files using the general Grid approach."""
        # Load dataset metadata from preprocessing
        dataset_metadata_file = os.path.join(self.dataset_dir, "dataset_metadata.json")
        if not os.path.exists(dataset_metadata_file):
            raise FileNotFoundError(
                f"Dataset metadata not found at {dataset_metadata_file}. Please run preprocessing first."
            )

        with open(dataset_metadata_file, "r") as f:
            dataset_metadata = json.load(f)

        # Get the case name from the HDF5 metadata
        if not os.path.exists(self.inference_metadata_file):
            raise FileNotFoundError(
                f"No inference metadata found at {self.inference_metadata_file}"
            )

        with open(self.inference_metadata_file, "r") as f:
            inference_metadata = json.load(f)

        hdf5_files = inference_metadata.get("hdf5_files", [])
        if sample_idx >= len(hdf5_files):
            raise IndexError(
                f"Sample index {sample_idx} exceeds available cases ({len(hdf5_files)})"
            )

        # Extract case name from HDF5 filename (remove .hdf5 extension)
        case_name = hdf5_files[sample_idx].replace(".hdf5", "")

        # Get the original sim_dir from dataset metadata
        original_sim_dir = dataset_metadata.get("sim_dir")
        if not original_sim_dir:
            raise KeyError("sim_dir not found in dataset metadata")

        # Construct the path to the simulator data directory using the original path
        data_file = os.path.join(original_sim_dir, f"{case_name}.DATA")

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Simulator data file not found: {data_file}")

        # Create reader and read grid data
        reader = EclReader(data_file)

        # Read grid data (COORD, ZCORN for coordinates)
        egrid_keys = ["COORD", "ZCORN", "FILEHEAD", "NNC1", "NNC2"]
        egrid_data = reader.read_egrid(egrid_keys)

        # Read init data for grid dimensions and porosity
        init_keys = ["INTEHEAD", "PORV"]
        init_data = reader.read_init(init_keys)

        # Create grid object to get coordinates (same as in reservoir_graph_builder.py)
        grid = Grid(init_data, egrid_data)
        X, Y, Z = grid.X, grid.Y, grid.Z

        # Get grid dimensions from the grid object
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        nact = grid.nact  # number of active cells

        self.logger.info(f"Extracted coordinates from grid for {case_name}:")
        self.logger.info(
            f"   Grid dimensions: {nx} × {ny} × {nz} = {nx * ny * nz} total cells"
        )
        self.logger.info(
            f"   Active cells: {nact} ({nact / (nx * ny * nz) * 100:.1f}% of total)"
        )
        self.logger.info(f"   Coordinates: {len(X)} active nodes")

        return X, Y, Z, (nx, ny, nz), nact, grid

    def run_post(self):
        """
        Generate Eclipse-style GRDECL ASCII files from HDF5 inference results.
        Each HDF5 file is converted to a GRDECL file with format:
        KEY_<timestep>
        <values>
        /
        """
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("POST-PROCESSING: GENERATING GRDECL FILES")
        self.logger.info("=" * 70)

        # Load metadata to get HDF5 files
        if not os.path.exists(self.inference_metadata_file):
            self.logger.warning(
                f"No inference metadata found at {self.inference_metadata_file}"
            )
            return

        with open(self.inference_metadata_file, "r") as f:
            metadata = json.load(f)

        hdf5_files = metadata.get("hdf5_files", [])
        if not hdf5_files:
            self.logger.warning("No HDF5 files found in metadata")
            return

        # Output directory (directly under inference/)
        grdecl_output_dir = self.inference_output_dir
        os.makedirs(grdecl_output_dir, exist_ok=True)

        self.logger.info(f"Output directory: {grdecl_output_dir}")
        self.logger.info(f"Processing {len(hdf5_files)} HDF5 file(s)...")

        # Process each HDF5 file
        for sample_idx, hdf5_filename in enumerate(hdf5_files, 1):
            hdf5_file = os.path.join(self.inference_output_dir, hdf5_filename)
            case_name = os.path.basename(hdf5_filename).replace(".hdf5", "")

            self.logger.info(
                f"[{sample_idx}/{len(hdf5_files)}] Generating GRDECL for {case_name}..."
            )

            # Output GRDECL filename
            grdecl_filename = f"{case_name}.GRDECL"
            grdecl_filepath = os.path.join(grdecl_output_dir, grdecl_filename)

            try:
                # Get grid information and actnum for this case
                # Note: sample_idx is 1-based for display, convert to 0-based for indexing
                X, Y, Z, grid_dims, nact, grid = self._extract_coordinates_from_grid(
                    sample_idx - 1
                )
                nx, ny, nz = grid_dims
                total_cells = nx * ny * nz
                actnum = grid.actnum_bool

                # Load partition assignments and halo info for this case
                partition_data_active, halo_data_active = (
                    self._load_partition_and_halo_info(case_name)
                )

                with h5py.File(hdf5_file, "r") as f:
                    with open(grdecl_filepath, "w") as grdecl_file:
                        # Write combined PARTITION block (if available)
                        # Positive values = partition ID (inner nodes that are NOT halo anywhere)
                        # Negative values = -partition_id for boundary nodes (cells that serve as halo)
                        if partition_data_active is not None:
                            # Start with partition assignments for active cells
                            combined_data_active = partition_data_active.copy()

                            # Mark ALL halo cells with negative values (even if they're also inner somewhere)
                            if halo_data_active is not None:
                                # All cells where halo_data_active > 0 get marked as halo (negative)
                                halo_mask = halo_data_active > 0
                                num_halo = np.sum(halo_mask)
                                num_halo_only = np.sum(
                                    (halo_data_active > 0)
                                    & (partition_data_active == 0)
                                )
                                num_boundary = np.sum(
                                    (halo_data_active > 0) & (partition_data_active > 0)
                                )

                                self.logger.info(
                                    f"   Total halo cells: {num_halo} (Halo-only: {num_halo_only}, Boundary: {num_boundary})"
                                )

                                # Mark halo cells with negative of their halo partition ID
                                # Use halo_data_active (not partition_data_active) to avoid -0 for halo-only cells
                                combined_data_active[halo_mask] = -halo_data_active[
                                    halo_mask
                                ]

                            # Initialize full array with zeros (for inactive cells)
                            partition_data_full = np.zeros((total_cells,), dtype=int)
                            # Populate only active cells with combined partition/halo info
                            partition_data_full[actnum] = combined_data_active

                            # Write PARTITION block
                            grdecl_file.write("PARTITION\n")
                            grdecl_file.write(
                                "-- Positive values: partition ID (inner nodes, not serving as halo)\n"
                            )
                            grdecl_file.write(
                                "-- Negative values: -partition_id for boundary/halo nodes (e.g., -2 = owned by partition 2, serves as halo)\n"
                            )
                            grdecl_file.write("-- Zero: inactive cells\n")
                            for i, value in enumerate(partition_data_full):
                                grdecl_file.write(f"{value} ")
                                if (i + 1) % 10 == 0:  # 10 values per line for integers
                                    grdecl_file.write("\n")

                            # Ensure newline before '/'
                            if len(partition_data_full) % 10 != 0:
                                grdecl_file.write("\n")

                            # Terminator
                            grdecl_file.write("/\n\n")

                        # Get target variables
                        target_variables = f.attrs.get("target_variables", [])

                        # Process each target variable
                        for var_name in target_variables:
                            var_name_clean = var_name.upper()

                            if var_name_clean not in f["predictions"]:
                                self.logger.warning(
                                    f"Variable {var_name_clean} not found in {hdf5_filename}"
                                )
                                continue

                            # Get all timesteps for this variable
                            timesteps = sorted(f["predictions"][var_name_clean].keys())

                            for timestep_key in timesteps:
                                # Extract timestep number from key (e.g., "timestep_001" -> 1)
                                timestep_num = int(timestep_key.split("_")[-1])

                                # Read prediction data (only active cells)
                                pred_data_active = f["predictions"][var_name_clean][
                                    timestep_key
                                ][:]

                                # Initialize full array with zeros for all cells
                                pred_data_full = np.zeros((total_cells,))

                                # Populate only active cells with predicted values
                                pred_data_full[actnum] = pred_data_active

                                # Write in Eclipse GRDECL format
                                # KEY_<timestep> with 4-digit formatting (e.g., 0001, 0010, 0120)
                                grdecl_file.write(
                                    f"{var_name_clean}_{timestep_num:04d}\n"
                                )

                                # Values (write 5 values per line for readability)
                                for i, value in enumerate(pred_data_full):
                                    grdecl_file.write(f"{value:.6e} ")
                                    if (i + 1) % 5 == 0:
                                        grdecl_file.write("\n")

                                # Ensure newline before '/'
                                if len(pred_data_full) % 5 != 0:
                                    grdecl_file.write("\n")

                                # Terminator
                                grdecl_file.write("/\n")

            except Exception as e:
                self.logger.error(f"Failed to process {hdf5_filename}: {e}")
                continue

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(f"GRDECL files saved to: {grdecl_output_dir}")
        self.logger.info("=" * 70)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main inference entry point.
    Performs autoregressive inference and generates GRDECL files.
    """

    dist, logger = InitializeLoggers(cfg)

    runner = InferenceRunner(cfg, dist, logger)

    runner.run_inference()

    runner.run_post()

    logger.success("Inference and post-processing completed successfully!")


if __name__ == "__main__":
    main()
