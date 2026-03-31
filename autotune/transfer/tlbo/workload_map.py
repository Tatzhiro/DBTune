import os
import sys
import pdb
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from autotune.transfer.tlbo.base import BaseTLSurrogate
from autotune.utils.parser import get_action_data_json
from autotune.utils.binner import Bin
from autotune.knobs import knobDF2action
from autotune.gp import gp_predict
from autotune.utils.history_container import HistoryContainer
from openbox.utils.config_space import Configuration
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.config_space import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter


class WorkloadMapping(BaseTLSurrogate):
    def __init__(self, config_space, source_hpo_data, seed,
                 surrogate_type='rf', num_src_hpo_trial=50, only_source=False,
                 mapping_method='ottertune', prune_metrics=False):
        super().__init__(config_space, source_hpo_data, seed,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'mapping'
        self.mapping_method = mapping_method
        self.prune_metrics = prune_metrics
        self.source_dict = {}

        if mapping_method == 'dbtune':
            self.scaler = StandardScaler()
            self.binner = Bin(bin_start=1)

        self.extract_source()

        self.scale = True
        self.num_sample = 50
        self.iteration_id = 0

    def extract_source(self):
        for i, container in enumerate(self.source_hpo_data):
            X_scaled = convert_configurations_to_array(container.configurations)
            IM = np.vstack(container.get_internal_metrics())
            entry = {'X': X_scaled, 'IM': IM, 'pos': i}
            if self.mapping_method == 'ottertune':
                entry['configurations'] = container.configurations
            self.source_dict[container.task_id] = entry

        if self.mapping_method == 'dbtune':
            IM_all = np.vstack([item['IM'] for item in list(self.source_dict.values())])
            self.scaler.fit_transform(IM_all)
            self.binner.fit(IM_all)
            del IM_all

    def train(self, target_hpo_data: HistoryContainer):
        if self.mapping_method == 'ottertune':
            self._train_ottertune(target_hpo_data)
        else:
            self._train_dbtune(target_hpo_data)

    def _train_dbtune(self, target_hpo_data: HistoryContainer):
        # get target X, y, im
        target_X_scaled = convert_configurations_to_array(target_hpo_data.configurations)
        target_y = target_hpo_data.get_transformed_perfs()
        target_IM = np.vstack(target_hpo_data.get_internal_metrics())

        # Validate that source and target internal metrics have the same dimensionality
        source_dim = list(self.source_dict.values())[0]['IM'].shape[1]
        target_dim = target_IM.shape[1]
        if source_dim != target_dim:
            raise ValueError(
                f"Internal metrics dimension mismatch: source has {source_dim} metrics, "
                f"target has {target_dim}. Ensure both use the same internal_metrics_source "
                f"(innodb=65, prometheus=114)."
            )

        target_IM = self.scaler.transform(target_IM)
        target_IM = self.binner.transform(target_IM)

        scores = {}
        for task_id, item in list(self.source_dict.items()):
            predictions = np.empty_like(target_IM)
            source_X_scaled = item['X']
            source_IM_scaled = self.scaler.transform(item['IM'])
            for j, col in enumerate(source_IM_scaled.T):
                col = col.reshape(-1, 1)
                predictions[:, j] = gp_predict(source_X_scaled, col, target_X_scaled, task_id, j)
            predictions = self.binner.transform(predictions)
            dists = np.sqrt(np.sum(np.square(np.subtract(predictions, target_IM)), axis=1))
            scores[task_id] = np.mean(dists)

        best_score = np.inf
        best_task_id = None
        for task_id, similarity_score in list(scores.items()):
            if similarity_score < best_score:
                best_score = similarity_score
                best_task_id = task_id

        self._finalize_match(best_task_id, best_score, scores,
                             target_X_scaled, target_y, label='WorkloadMapping-DBTune')

    def _train_ottertune(self, target_hpo_data: HistoryContainer):
        """OtterTune workload mapping (Section 6.1).

        Compares internal metrics at exactly matching configurations between
        target and each source workload. The workload distance score is the
        average metric distance across all matched config pairs, using
        decile-binned Euclidean distance.
        """
        target_X_scaled = convert_configurations_to_array(target_hpo_data.configurations)
        target_y = target_hpo_data.get_transformed_perfs()
        target_IM = np.vstack(target_hpo_data.get_internal_metrics())

        # Validate dimensionality
        source_dim = list(self.source_dict.values())[0]['IM'].shape[1]
        target_dim = target_IM.shape[1]
        if source_dim != target_dim:
            raise ValueError(
                f"Internal metrics dimension mismatch: source has {source_dim} metrics, "
                f"target has {target_dim}. Ensure both use the same internal_metrics_source "
                f"(innodb=65, prometheus=114)."
            )

        # Optional metric pruning via FA+KMeans (Section 4.2)
        metric_indices = None
        if self.prune_metrics:
            wm_logger = logging.getLogger('autotune')
            all_IM = np.vstack([target_IM] + [item['IM'] for item in self.source_dict.values()])
            metric_indices = self._prune_metrics_fa_kmeans(all_IM)
            wm_logger.info('[WorkloadMapping-OtterTune] Pruned to %d metrics (from %d)',
                           len(metric_indices), source_dim)

        # Compute global decile bin edges from ALL source + target metrics
        all_IM_for_binning = np.vstack([target_IM] + [item['IM'] for item in self.source_dict.values()])
        if metric_indices is not None:
            all_IM_for_binning = all_IM_for_binning[:, metric_indices]
        global_bin_edges = np.percentile(all_IM_for_binning, np.linspace(0, 100, 11), axis=0)
        del all_IM_for_binning

        # Find best matching source via exact config comparison
        best_task_id, best_score, scores = self._match_exact(
            target_hpo_data.configurations, target_IM, metric_indices, global_bin_edges)

        self._finalize_match(best_task_id, best_score, scores,
                             target_X_scaled, target_y, label='WorkloadMapping-OtterTune')

    def _finalize_match(self, best_task_id, best_score, scores,
                        target_X_scaled, target_y, label):
        """Log match results, build surrogate from concatenated target + matched source data."""
        self.matched_context_id = best_task_id

        wm_logger = logging.getLogger('autotune')
        wm_logger.info('[%s] Iter %d: Matched context: %s (distance=%.4f)',
                       label, self.iteration_id, best_task_id, best_score)

        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        for rank, (tid, score) in enumerate(sorted_scores[:5], 1):
            wm_logger.info('[%s]   #%d: %s (distance=%.4f)', label, rank, tid, score)

        mapped_container = self.source_hpo_data[self.source_dict[best_task_id]['pos']]
        mapped_X_scaled = convert_configurations_to_array(mapped_container.configurations)
        mapped_y = mapped_container.get_transformed_perfs()
        new_X = np.vstack((target_X_scaled, mapped_X_scaled))
        new_y = np.concatenate((target_y, mapped_y))

        self.target_surrogate = self.build_single_surrogate(new_X, new_y, normalize='standardize')
        self.iteration_id += 1

    def _match_exact(self, target_configs, target_IM, metric_indices, global_bin_edges):
        """Find matching source by comparing metrics at shared configurations.

        For each source workload, find configs that match any target config
        (exact match in normalized config space). Compare internal metrics at
        those matching configs using decile-binned Euclidean distance.
        Score = average distance across all matched config pairs.

        Args:
            metric_indices: Column indices of pruned metrics (None = use all).
            global_bin_edges: Precomputed decile edges from ALL data (shape 11 x n_metrics).

        Returns (best_task_id, best_score, scores).
        """
        target_config_arrays = convert_configurations_to_array(target_configs)

        scores = {}
        for task_id, item in self.source_dict.items():
            source_config_arrays = item['X']
            source_IM = item['IM']

            # Find matching config pairs (exact match in config space array)
            matched_target_ims = []
            matched_source_ims = []
            for t_idx, t_arr in enumerate(target_config_arrays):
                for s_idx, s_arr in enumerate(source_config_arrays):
                    if np.allclose(t_arr, s_arr, atol=1e-6):
                        matched_target_ims.append(target_IM[t_idx])
                        matched_source_ims.append(source_IM[s_idx])
                        break  # one match per target config

            if len(matched_target_ims) == 0:
                continue

            matched_target = np.vstack(matched_target_ims)
            matched_source = np.vstack(matched_source_ims)

            if metric_indices is not None:
                matched_target = matched_target[:, metric_indices]
                matched_source = matched_source[:, metric_indices]

            # Bin using global edges and compute distance
            binned_target = self._apply_decile_bins(matched_target, global_bin_edges)
            binned_source = self._apply_decile_bins(matched_source, global_bin_edges)

            # Average Euclidean distance across matched pairs
            dists = np.sqrt(np.sum(np.square(binned_target - binned_source), axis=1))
            scores[task_id] = np.mean(dists)

        if not scores:
            raise ValueError(
                'No exact config matches found between target and any source workload. '
                'Ensure source histories include the default configuration.'
            )

        best_task_id = min(scores, key=scores.get)
        return best_task_id, scores[best_task_id], scores

    @staticmethod
    def _apply_decile_bins(vectors, bin_edges):
        """Bin each column of vectors into deciles (values 1-10) using precomputed edges.

        Args:
            vectors: np.ndarray of shape (n_samples, n_metrics)
            bin_edges: np.ndarray of shape (11, n_metrics) from np.percentile(..., linspace(0,100,11))

        Returns:
            np.ndarray of same shape with integer bin values 1-10
        """
        binned = np.empty_like(vectors, dtype=int)
        for j in range(vectors.shape[1]):
            binned[:, j] = np.clip(np.digitize(vectors[:, j], bin_edges[:, j], right=True), 1, 10)
        return binned

    def _prune_metrics_fa_kmeans(self, IM_all, max_samples=5000):
        """Prune redundant metrics using Factor Analysis + K-Means (OtterTune Section 4.2).

        Args:
            IM_all: np.ndarray of shape (n_observations, n_metrics) — all source + target metrics
            max_samples: Max observations to use for FA (subsample for scalability)

        Returns:
            np.ndarray of column indices of representative metrics to keep
        """
        from sklearn.decomposition import FactorAnalysis
        from sklearn.cluster import KMeans

        # Subsample if too many observations (FA only needs enough to capture correlations)
        if IM_all.shape[0] > max_samples:
            rng = np.random.RandomState(self.random_seed)
            indices = rng.choice(IM_all.shape[0], max_samples, replace=False)
            IM_all = IM_all[indices]

        # Transpose: rows=metrics, cols=observations
        IM_T = IM_all.T  # (n_metrics, n_observations)
        fa = FactorAnalysis()
        U = fa.fit_transform(IM_T)  # (n_metrics, n_components)

        k = self._select_k_pham_dimov_nguyen(U, random_state=self.random_seed)
        kmeans = KMeans(n_clusters=k, random_state=self.random_seed, n_init='auto')
        kmeans.fit(U)

        # Get metric index closest to each centroid
        indices = []
        for cluster_id in range(k):
            mask = kmeans.labels_ == cluster_id
            cluster_indices = np.where(mask)[0]
            centroid = kmeans.cluster_centers_[cluster_id]
            dists = np.linalg.norm(U[mask] - centroid, axis=1)
            indices.append(cluster_indices[np.argmin(dists)])

        return np.sort(indices)

    @staticmethod
    def _select_k_pham_dimov_nguyen(X, k_max=10, alpha=1.0, threshold=1.0, random_state=42):
        """Select number of clusters K using Pham, Dimov, and Nguyen (2005) method."""
        from sklearn.cluster import KMeans

        if isinstance(X, pd.DataFrame):
            X = X.values

        f_values = []
        for k in range(1, k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
            kmeans.fit(X)
            f_values.append(kmeans.inertia_)

        # Delta values
        delta_values = [0]  # delta(1) = 0 placeholder
        for k in range(2, k_max + 1):
            numerator = f_values[k - 2] - f_values[k - 1]
            denominator = f_values[k - 2]
            delta_k = numerator / denominator if denominator != 0 else 0
            delta_values.append(delta_k)

        # Ratio values
        ratio_values = [0, 0]  # ratio(1) and ratio(2) placeholders
        for k in range(3, k_max + 1):
            if delta_values[k - 2] > 0:
                ratio_k = delta_values[k - 1] / (delta_values[k - 2] ** alpha)
            else:
                ratio_k = 0
            ratio_values.append(ratio_k)

        best_k = 1
        for k in range(3, k_max + 1):
            if ratio_values[k - 1] >= threshold:
                best_k = k

        # Edge case if best_k remains 1
        if best_k == 1 and k_max >= 2:
            if delta_values[1] > 0.0:
                best_k = 2

        return best_k

    def predict(self, X: np.array):
        mu, var = self.target_surrogate.predict(X)
        return mu, var
