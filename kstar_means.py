import numpy as np
from typing import Tuple, Optional


class KStarMeans:
    """
    K*-Means: Parameter-free clustering algorithm using MDL principle.

    Based on: "K*-Means: A Parameter-free Clustering Algorithm"
    by Louis Mahon and Mirella Lapata (2025)
    """

    def __init__(self, patience: int = 5, convergence_threshold: float = 2.0,
                 max_iterations: int = 1000, random_state: Optional[int] = None):
        """
        Initialize K*-Means algorithm.

        Args:
            patience: Number of iterations without improvement before stopping
            convergence_threshold: Minimum cost improvement to continue
            max_iterations: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        self.patience = patience
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.random_state = random_state

        # Model state
        self.centroids_ = None  # Main cluster centroids (k x d)
        self.sub_centroids_ = None  # Sub-centroids for each cluster (k x 2 x d)
        self.labels_ = None  # Cluster assignments for each point
        self.n_clusters_ = None  # Final number of clusters
        self.cost_history_ = []  # MDL cost at each iteration

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray) -> 'KStarMeans':
        """
        Fit K*-Means model to data.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            self: Fitted estimator
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Initialize with single cluster containing all data
        self.centroids_ = np.mean(X, axis=0, keepdims=True)  # (1, d)
        self.labels_ = np.zeros(n_samples, dtype=int)

        # Initialize sub-centroids for the initial cluster
        self.sub_centroids_ = [self._init_sub_centroids(X)]

        # Tracking
        best_cost = float('inf')
        unimproved_count = 0
        iteration = 0

        while iteration < self.max_iterations:
            # Standard k-means step: assign + update
            self._kmeans_step(X)

            # Try to split clusters
            did_split = self._maybe_split(X)

            # If no split occurred, try to merge
            if not did_split:
                self._kmeans_step(X)
                self._maybe_merge(X)

            # Compute current cost
            current_cost = self._mdl_cost(X)
            self.cost_history_.append(current_cost)

            # Check for improvement
            if current_cost < best_cost - self.convergence_threshold:
                best_cost = current_cost
                unimproved_count = 0
            else:
                unimproved_count += 1

            # Check convergence
            if unimproved_count >= self.patience:
                break

            iteration += 1

        self.n_clusters_ = len(self.centroids_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            labels: Cluster labels for each sample
        """
        X = np.asarray(X, dtype=np.float64)
        distances = self._compute_distances(X, self.centroids_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit model and return cluster labels.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            labels: Cluster labels for each sample
        """
        self.fit(X)
        return self.labels_

    def _kmeans_step(self, X: np.ndarray) -> None:
        """
        Perform one iteration of standard k-means (assign + update).
        Updates both main clusters and sub-clusters.
        """
        n_samples = len(X)
        k = len(self.centroids_)

        # ASSIGN: Assign points to nearest centroid
        distances = self._compute_distances(X, self.centroids_)
        self.labels_ = np.argmin(distances, axis=1)

        # UPDATE: Update main centroids
        new_centroids = []
        new_sub_centroids = []

        for i in range(k):
            cluster_mask = self.labels_ == i
            cluster_points = X[cluster_mask]

            if len(cluster_points) > 0:
                # Update main centroid
                new_centroids.append(np.mean(cluster_points, axis=0))

                # Update sub-centroids
                sub_labels = self._assign_to_sub_centroids(
                    cluster_points, self.sub_centroids_[i]
                )

                new_sub_pair = []
                for sub_id in range(2):
                    sub_mask = sub_labels == sub_id
                    sub_points = cluster_points[sub_mask]
                    if len(sub_points) > 0:
                        new_sub_pair.append(np.mean(sub_points, axis=0))
                    else:
                        # Keep old centroid if no points assigned
                        new_sub_pair.append(self.sub_centroids_[i][sub_id])

                new_sub_centroids.append(np.array(new_sub_pair))
            else:
                # Keep old centroids if cluster is empty
                new_centroids.append(self.centroids_[i])
                new_sub_centroids.append(self.sub_centroids_[i])

        self.centroids_ = np.array(new_centroids)
        self.sub_centroids_ = new_sub_centroids

    def _maybe_split(self, X: np.ndarray) -> bool:
        """
        Check if any cluster should be split into sub-clusters.

        Returns:
            True if a split occurred, False otherwise
        """
        k = len(self.centroids_)
        n_samples = len(X)

        best_cost_change = 0.0
        split_at = -1

        # Check each cluster for potential split
        for i in range(k):
            cluster_mask = self.labels_ == i
            cluster_points = X[cluster_mask]

            if len(cluster_points) < 2:
                continue

            # Assign points to sub-centroids
            sub_labels = self._assign_to_sub_centroids(
                cluster_points, self.sub_centroids_[i]
            )

            # Compute costs
            main_q = self._compute_cluster_ss(cluster_points, self.centroids_[i])

            sub_q = 0.0
            for sub_id in range(2):
                sub_mask = sub_labels == sub_id
                sub_points = cluster_points[sub_mask]
                if len(sub_points) > 0:
                    sub_q += self._compute_cluster_ss(
                        sub_points, self.sub_centroids_[i][sub_id]
                    )

            # Cost change from splitting (negative is good)
            # Splitting increases index cost but decreases residual cost
            cost_change = (sub_q - main_q) + n_samples / (k + 1)

            if cost_change < best_cost_change:
                best_cost_change = cost_change
                split_at = i

        # Perform split if beneficial
        if split_at >= 0:
            self._split_cluster(X, split_at)
            return True

        return False

    def _maybe_merge(self, X: np.ndarray) -> None:
        """
        Check if the two closest clusters should be merged.
        """
        k = len(self.centroids_)

        if k < 2:
            return

        # Find closest pair of centroids
        i1, i2 = self._find_closest_centroid_pair()

        # Get points from both clusters
        mask1 = self.labels_ == i1
        mask2 = self.labels_ == i2
        points1 = X[mask1]
        points2 = X[mask2]
        merged_points = np.vstack([points1, points2])

        # Compute merged centroid
        merged_centroid = np.mean(merged_points, axis=0)

        # Compute costs
        main_q = self._compute_cluster_ss(merged_points, merged_centroid)
        sub_q = (self._compute_cluster_ss(points1, self.centroids_[i1]) +
                 self._compute_cluster_ss(points2, self.centroids_[i2]))

        # Cost change from merging (negative is good)
        # Merging decreases index cost but increases residual cost
        cost_change = (main_q - sub_q) - len(X) / k

        # Perform merge if beneficial
        if cost_change < 0:
            self._merge_clusters(X, i1, i2, merged_centroid)

    def _split_cluster(self, X: np.ndarray, cluster_idx: int) -> None:
        """
        Split a cluster into its two sub-clusters.
        """
        # Get points in the cluster
        cluster_mask = self.labels_ == cluster_idx
        cluster_points = X[cluster_mask]

        # Assign to sub-centroids
        sub_labels = self._assign_to_sub_centroids(
            cluster_points, self.sub_centroids_[cluster_idx]
        )

        # Promote sub-centroids to main centroids
        new_centroid1 = self.sub_centroids_[cluster_idx][0]
        new_centroid2 = self.sub_centroids_[cluster_idx][1]

        # Update centroids list
        centroids_list = list(self.centroids_)
        centroids_list[cluster_idx] = new_centroid1
        centroids_list.insert(cluster_idx + 1, new_centroid2)
        self.centroids_ = np.array(centroids_list)

        # Update labels
        cluster_point_indices = np.where(cluster_mask)[0]
        for local_idx, global_idx in enumerate(cluster_point_indices):
            if sub_labels[local_idx] == 0:
                self.labels_[global_idx] = cluster_idx
            else:
                self.labels_[global_idx] = cluster_idx + 1

        # Shift labels for clusters after the split
        self.labels_[self.labels_ > cluster_idx + 1] += 1

        # Initialize new sub-centroids for both new clusters
        mask1 = self.labels_ == cluster_idx
        mask2 = self.labels_ == cluster_idx + 1

        new_sub1 = self._init_sub_centroids(X[mask1])
        new_sub2 = self._init_sub_centroids(X[mask2])

        # Update sub-centroids list
        sub_centroids_list = self.sub_centroids_[:cluster_idx]
        sub_centroids_list.append(new_sub1)
        sub_centroids_list.append(new_sub2)
        sub_centroids_list.extend(self.sub_centroids_[cluster_idx + 1:])
        self.sub_centroids_ = sub_centroids_list

    def _merge_clusters(self, X: np.ndarray, i1: int, i2: int,
                        merged_centroid: np.ndarray) -> None:
        """
        Merge two clusters into one.
        """
        # Ensure i1 < i2 for consistent indexing
        if i1 > i2:
            i1, i2 = i2, i1

        # Update labels: merge i2 into i1
        self.labels_[self.labels_ == i2] = i1

        # Shift labels for clusters after i2
        self.labels_[self.labels_ > i2] -= 1

        # Update centroids: replace i1 with merged, remove i2
        centroids_list = list(self.centroids_)
        centroids_list[i1] = merged_centroid
        centroids_list.pop(i2)
        self.centroids_ = np.array(centroids_list)

        # Create new sub-centroids for merged cluster
        merged_mask = self.labels_ == i1
        merged_points = X[merged_mask]
        new_sub = self._init_sub_centroids(merged_points)

        # Update sub-centroids list
        sub_centroids_list = self.sub_centroids_[:i1]
        sub_centroids_list.append(new_sub)
        sub_centroids_list.extend(self.sub_centroids_[i1 + 1:i2])
        sub_centroids_list.extend(self.sub_centroids_[i2 + 1:])
        self.sub_centroids_ = sub_centroids_list

    def _init_sub_centroids(self, points: np.ndarray) -> np.ndarray:
        """
        Initialize two sub-centroids for a cluster using k-means++ strategy.

        Returns:
            Array of shape (2, n_features) with two sub-centroids
        """
        if len(points) < 2:
            # If cluster has < 2 points, create duplicate centroids
            centroid = np.mean(points, axis=0)
            return np.array([centroid, centroid])

        # k-means++ initialization for 2 centroids
        # Choose first centroid randomly
        idx1 = np.random.randint(len(points))
        centroid1 = points[idx1]

        # Choose second centroid with probability proportional to distance squared
        distances_sq = np.sum((points - centroid1) ** 2, axis=1)
        probabilities = distances_sq / np.sum(distances_sq)
        idx2 = np.random.choice(len(points), p=probabilities)
        centroid2 = points[idx2]

        return np.array([centroid1, centroid2])

    def _mdl_cost(self, X: np.ndarray) -> float:
        """
        Compute the Minimum Description Length cost.

        MDL = model_cost + index_cost + residual_cost

        where:
        - model_cost: bits to encode k centroids in d dimensions
        - index_cost: bits to encode cluster assignment for each point
        - residual_cost: bits to encode displacement from centroid
        """
        return self._compute_mdl_for_config(X, self.centroids_, self.labels_)

    def _compute_mdl_for_config(self, X: np.ndarray, centroids: np.ndarray,
                                labels: np.ndarray) -> float:
        """
        Compute MDL cost for a given clustering configuration.

        Based on Algorithm 1 from the K*-Means paper:
        MDL = model_cost + index_cost + residual_cost

        Args:
            X: Data matrix (n_samples, n_features)
            centroids: Cluster centroids (k, n_features)
            labels: Cluster assignments (n_samples,)

        Returns:
            Total MDL cost
        """
        n_samples, n_features = X.shape
        k = len(centroids)

        # Compute floating point precision from data
        # floatprecision = -log of the minimum distance between any values in X
        unique_values = np.unique(X.flatten())
        if len(unique_values) > 1:
            sorted_values = np.sort(unique_values)
            min_diff = np.min(np.diff(sorted_values))
            if min_diff > 0:
                floatprecision = -np.log2(min_diff)
            else:
                floatprecision = 32.0
        else:
            floatprecision = 32.0

        # Model cost: k centroids * d dimensions * floatcost
        data_range = np.ptp(X)  # max - min
        if data_range > 0 and floatprecision > 0:
            floatcost = data_range / floatprecision
        else:
            floatcost = 1.0

        model_cost = k * n_features * floatcost

        # Index cost: n points * log2(k) bits to encode cluster assignments
        index_cost = n_samples * np.log2(k) if k > 1 else 0.0

        # Residual cost: |X|*d*log(2π) + c/2
        # where c is the sum of squared distances
        total_ss = 0.0
        for i in range(k):
            cluster_mask = labels == i
            cluster_points = X[cluster_mask]
            if len(cluster_points) > 0:
                total_ss += self._compute_cluster_ss(cluster_points, centroids[i])

        residual_cost = n_samples * n_features * np.log(2 * np.pi) + total_ss / 2.0

        return model_cost + index_cost + residual_cost

    def _compute_cluster_ss(self, points: np.ndarray, centroid: np.ndarray) -> float:
        """
        Compute sum of squared distances from points to centroid.

        This is the Q(S) function from the paper.
        """
        if len(points) == 0:
            return 0.0
        return np.sum((points - centroid) ** 2)

    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances from points to centroids.

        Args:
            X: Data matrix (n_samples, n_features)
            centroids: Centroid matrix (n_centroids, n_features)

        Returns:
            distances: Distance matrix (n_samples, n_centroids)
        """
        # Using broadcasting: (n, 1, d) - (1, k, d) = (n, k, d)
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        return distances

    def _assign_to_sub_centroids(self, points: np.ndarray,
                                 sub_centroids: np.ndarray) -> np.ndarray:
        """
        Assign points to nearest sub-centroid (0 or 1).
        """
        distances = self._compute_distances(points, sub_centroids)
        return np.argmin(distances, axis=1)

    def _find_closest_centroid_pair(self) -> Tuple[int, int]:
        """
        Find the pair of centroids with minimum distance.

        Returns:
            (i1, i2): Indices of closest centroid pair
        """
        k = len(self.centroids_)
        min_dist = float('inf')
        closest_pair = (0, 1)

        for i in range(k):
            for j in range(i + 1, k):
                dist = np.linalg.norm(self.centroids_[i] - self.centroids_[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (i, j)

        return closest_pair

    def get_params(self) -> dict:
        """Get parameters of the estimator."""
        return {
            'patience': self.patience,
            'convergence_threshold': self.convergence_threshold,
            'max_iterations': self.max_iterations,
            'random_state': self.random_state
        }

    def set_params(self, **params) -> 'KStarMeans':
        """Set parameters of the estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

