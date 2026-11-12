"""E-means clustering."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import math
import warnings
from numbers import Integral, Real

import numpy as np

from sklearn.base import _fit_context
from sklearn.cluster._e_means_minibatch import _ema_update, _minibatch_statistics
from sklearn.cluster._k_means_common import CHUNK_SIZE
from sklearn.cluster._kmeans import _BaseKMeans, _labels_inertia_threadpool_limit
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import row_norms
from sklearn.utils.parallel import _get_threadpool_controller
from sklearn.utils.validation import (
    _is_arraylike_not_scalar,
    check_array,
    check_random_state,
    validate_data,
)

# The constants below are those of the reference implementation of e-means:
# https://github.com/facebookresearch/faiss/tree/main/contrib/e_means

# Coefficients (C, a, b, c) of the scaling laws of the batch size:
# batch_size = C * n_clusters**a * n_samples**b * n_epochs**c.
_BATCH_SIZE_LAWS = {
    "throughput": (0.035134, 0.134749, 0.876473, 0.813344),
    "quality": (0.00439175, 0.134749, 0.876473, 0.813344),
}

# Coefficients (C, a, b, c, d) of the scaling law of the initial learning rate:
# learning_rate_init = C * n_clusters**a * batch_size**b * n_samples**c
# * n_epochs**d, clipped to [1e-4, 1].
_LEARNING_RATE_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)

# The learning rate decreases from learning_rate_init to
# _FINAL_LEARNING_RATE_RATIO * learning_rate_init along a cosine schedule.
_FINAL_LEARNING_RATE_RATIO = 0.1

# Initial running weight of each cluster.
_INITIAL_RUNNING_WEIGHT = 2.0**-16

# Number of mini-batch steps between two reassignments of unused clusters.
_REASSIGNMENT_PERIOD = 5

# Relative perturbation of the coordinates of a center split in two.
_SPLIT_PERTURBATION = 1e-4

# The "quality" law, with batches 8 times smaller, is used up to this number of
# features, and the "throughput" law above.
_QUALITY_MAX_FEATURES = 16

# A cluster is unused when its running weight is below this fraction of the mean
# running weight.
_REASSIGNMENT_RATIO = 0.1

# Unused clusters are not reassigned up to this number of features.
_NO_REASSIGNMENT_MAX_FEATURES = 2


def _round_to_power_of_two(value):
    """Round a positive number to the closest power of two on a log scale."""
    if value <= 1:
        return 1
    low = 2 ** math.floor(math.log2(value))
    return low if value / low < 2 * low / value else 2 * low


def _batch_size_law(law, n_samples, n_clusters, n_epochs):
    """Batch size given by the scaling law named `law`."""
    coef, clusters_exp, samples_exp, epochs_exp = _BATCH_SIZE_LAWS[law]
    batch_size = _round_to_power_of_two(
        coef * n_clusters**clusters_exp * n_samples**samples_exp * n_epochs**epochs_exp
    )
    # The two laws were fitted with different caps on the batch size.
    if law == "throughput":
        batch_size = min(batch_size, _round_to_power_of_two(n_samples / 2))
    else:
        batch_size = min(batch_size, n_samples // 2)
    return max(1, min(batch_size, n_samples))


def _learning_rate_law(n_samples, n_clusters, n_epochs, batch_size):
    """Initial learning rate given by the scaling law for `batch_size`."""
    coef, clusters_exp, batch_exp, samples_exp, epochs_exp = _LEARNING_RATE_LAW
    learning_rate = (
        coef
        * n_clusters**clusters_exp
        * batch_size**batch_exp
        * n_samples**samples_exp
        * n_epochs**epochs_exp
    )
    return min(max(learning_rate, 1e-4), 1.0)


def _epoch_sizes(n_samples, n_epochs):
    """Number of samples seen in each pass over the data.

    The fractional part of `n_epochs` is a pass over a random subset of the
    samples, made before the last pass. When `n_epochs >= 1`, the last pass
    sees all the samples.
    """
    n_full_epochs = math.floor(n_epochs)
    n_partial = round((n_epochs - n_full_epochs) * n_samples)
    if n_full_epochs == 0:
        return [max(n_partial, 1)]
    partial = [n_partial] if n_partial > 0 else []
    return [n_samples] * (n_full_epochs - 1) + partial + [n_samples]


def _scheduled_learning_rate(learning_rate_init, step, n_steps):
    """Learning rate at `step`, following a cosine decay over `n_steps`."""
    cosine = 0.5 * (1 + math.cos(math.pi * min(step, n_steps) / max(n_steps, 1)))
    ratio = _FINAL_LEARNING_RATE_RATIO
    return learning_rate_init * (ratio + (1 - ratio) * cosine)


def _split_unused_clusters(centers, running_weights, random_state):
    """Reassign the unused clusters by splitting heavily used clusters in two.

    A cluster is unused when its running weight is below `_REASSIGNMENT_RATIO`
    times the mean running weight. Each unused cluster is reassigned to a
    donor cluster, sampled without replacement with a probability proportional
    to its running weight: the two clusters share the running weight of the
    donor, and their centers are the center of the donor, perturbed in opposite
    directions.

    Returns the number of reassigned clusters.
    """
    n_clusters, n_features = centers.shape
    threshold = _REASSIGNMENT_RATIO * running_weights.sum() / n_clusters
    unused = running_weights < threshold
    can_donate = ~unused & (running_weights > 0)
    n_reassigned = min(np.count_nonzero(unused), np.count_nonzero(can_donate))
    if n_reassigned == 0:
        return 0

    # Gumbel-top-k sampling of the donors.
    uniform = np.maximum(
        random_state.random_sample(n_clusters), np.finfo(np.float64).tiny
    )
    keys = np.full(n_clusters, -np.inf)
    keys[can_donate] = np.log(running_weights[can_donate]) - np.log(
        -np.log(uniform[can_donate])
    )
    donors = np.argsort(keys)[::-1][:n_reassigned]
    reassigned = np.flatnonzero(unused)[:n_reassigned]

    perturbation = np.full(n_features, _SPLIT_PERTURBATION, dtype=centers.dtype)
    perturbation[1::2] *= -1
    donor_centers = centers[donors] * (1 - perturbation)
    reassigned_centers = centers[donors] * (1 + perturbation)
    # The perturbation leaves a center at the origin unchanged: move one of its
    # two copies by the smallest representable amount to make them distinct.
    identical = (donor_centers == reassigned_centers).all(axis=1)
    reassigned_centers[identical, 0] = np.nextafter(
        reassigned_centers[identical, 0], np.inf
    )
    centers[donors] = donor_centers
    centers[reassigned] = reassigned_centers
    running_weights[donors] /= 2
    running_weights[reassigned] = running_weights[donors]
    return n_reassigned


class EMeans(_BaseKMeans):
    """E-means clustering.

    Mini-batch k-means where each center is the ratio of exponential moving
    averages of the sums and the numbers of the samples assigned to it. The
    learning rate follows a cosine decay, and the batch size and the initial
    learning rate are set by empirical scaling laws, so that the number of
    passes over the data, `n_epochs`, is in general the only parameter to set.

    Read more in the :ref:`User Guide <e_means>`.

    .. versionadded:: 1.10

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'random', 'k-means++'}, callable or array-like of shape \
            (n_clusters, n_features), default='random'
        Method for initialization:

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        'k-means++' : selects initial cluster centroids using sampling based on
        an empirical probability distribution of the points' contribution to the
        overall inertia. It is often more expensive than the clustering itself.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_epochs : float, default=5
        Number of passes over the dataset. Fractional values are supported:
        the fractional part is a pass over a random subset of the samples. The
        batch size, the learning rate and its decay depend on `n_epochs`.

    batch_size : int or "auto", default="auto"
        Number of samples per mini-batch. "auto" sets it by an empirical scaling
        law of the number of samples, clusters and epochs, with batches about 8
        times smaller when `n_features <= 16`. The initial learning rate is set
        from the batch size by another scaling law.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization, for
        the order of the samples in each epoch and for the reassignments. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    batch_size_ : int
        Batch size used during fit.

    n_steps_ : int
        Number of minibatches processed.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    KMeans : The classic implementation of the clustering method based on the
        Lloyd's algorithm. It consumes the whole set of input data at each
        iteration.
    MiniBatchKMeans : Alternative online implementation that does incremental
        updates of the centers positions using mini-batches.

    Notes
    -----
    E-means was introduced in [1]_, whose authors report that on embedding
    datasets, it reaches the inertia of 25 iterations of Lloyd's algorithm in 1.4
    to 2.6 passes over the data. This implementation follows the reference
    implementation in faiss:
    https://github.com/facebookresearch/faiss/tree/main/contrib/e_means

    The learning rate is the step size of the exponential moving averages. It
    decreases to a tenth of its initial value along a cosine schedule. Every 5
    mini-batches, the clusters whose running weight is below a tenth of the mean
    running weight are reassigned by splitting heavily used clusters in two,
    except when `n_features <= 2`.

    At the end of the last epoch, each center is set to the mean of the samples
    assigned to it during that epoch. Then, like :class:`MiniBatchKMeans`, an
    additional pass over the data assigns each sample to its closest center to
    compute `labels_` and `inertia_`.

    References
    ----------
    .. [1] :arxiv:`Bessa, S., Mazaré, P.-E., Szilvasy, G., Touvron, H.,
       Picard, D., Douze, M., & Jégou, H. (2026). "e-means: outperforming
       k-means (in one-tenth the epochs)". <2611.09501>`

    Examples
    --------
    >>> from sklearn.cluster import EMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> emeans = EMeans(n_clusters=2, random_state=0).fit(X)
    >>> emeans.labels_
    array([0, 0, 0, 1, 1, 1], dtype=int32)
    >>> emeans.predict([[0, 0], [12, 3]])
    array([0, 1], dtype=int32)
    >>> emeans.cluster_centers_
    array([[ 1.,  2.],
           [10.,  2.]])

    For a comparison of E-Means with K-Means and MiniBatchKMeans, see
    :ref:`sphx_glr_auto_examples_cluster_plot_e_means.py`.
    """

    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "init": [StrOptions({"k-means++", "random"}), callable, "array-like"],
        "n_epochs": [Interval(Real, 0, None, closed="neither")],
        "batch_size": [
            Interval(Integral, 1, None, closed="left"),
            StrOptions({"auto"}),
        ],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_clusters=8,
        *,
        init="random",
        n_epochs=5,
        batch_size="auto",
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _check_params_vs_input(self, X):
        n_samples, n_features = X.shape
        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples={n_samples} should be >= n_clusters={self.n_clusters}."
            )

        if self.batch_size == "auto":
            law = "quality" if n_features <= _QUALITY_MAX_FEATURES else "throughput"
            self._batch_size = _batch_size_law(
                law, n_samples, self.n_clusters, self.n_epochs
            )
        else:
            self._batch_size = min(self.batch_size, n_samples)

        self._learning_rate_init = _learning_rate_law(
            n_samples, self.n_clusters, self.n_epochs, self._batch_size
        )

    def _warn_mkl_vcomp(self, n_active_threads):
        """Warn when vcomp and mkl are both present"""
        warnings.warn(
            "EMeans is known to have a memory leak on "
            "Windows with MKL, when there are less chunks than "
            "available threads. You can prevent it by setting "
            f"batch_size >= {self._n_threads * CHUNK_SIZE} or by "
            "setting the environment variable "
            f"OMP_NUM_THREADS={n_active_threads}"
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Compute the centroids on X by chunking it into mini-batches.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = validate_data(self, X, dtype=[np.float64, np.float32], order="C")

        self._check_params_vs_input(X)
        random_state = check_random_state(self.random_state)
        # Unit weights, for the initialization and the computation of labels_
        # and inertia_.
        sample_weight = np.ones(X.shape[0], dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()
        n_samples = X.shape[0]

        # Validate init array
        init = self.init
        if _is_arraylike_not_scalar(init):
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        self._check_mkl_vcomp(X, self._batch_size)

        # Only the k-means++ initialization uses the squared norms of the samples.
        x_squared_norms = None
        if isinstance(init, str) and init == "k-means++":
            x_squared_norms = row_norms(X, squared=True)
        centers = self._init_centroids(
            X,
            x_squared_norms=x_squared_norms,
            init=init,
            random_state=random_state,
            sample_weight=sample_weight,
            n_threads=self._n_threads,
        )
        if callable(init):
            # The centers are updated in place: they must not be a view of X.
            centers = centers.copy()
        running_weights = np.full(
            self.n_clusters, _INITIAL_RUNNING_WEIGHT, dtype=X.dtype
        )

        epoch_sizes = _epoch_sizes(n_samples, self.n_epochs)
        n_steps = sum(math.ceil(size / self._batch_size) for size in epoch_sizes)
        reassign = X.shape[1] > _NO_REASSIGNMENT_MAX_FEATURES

        # Buffers for the statistics of a mini-batch.
        sums = np.empty_like(centers)
        counts = np.empty(self.n_clusters, dtype=X.dtype)
        labels = np.empty(self._batch_size, dtype=np.int32)
        # Statistics of the clusters during the last epoch.
        epoch_sums = np.zeros_like(centers)
        epoch_counts = np.zeros(self.n_clusters, dtype=X.dtype)

        step = 0
        with _get_threadpool_controller().limit(limits="sequential_blas_under_openmp"):
            for epoch, epoch_size in enumerate(epoch_sizes):
                is_last_epoch = epoch == len(epoch_sizes) - 1
                indices = random_state.permutation(n_samples)[:epoch_size]
                indices = indices.astype(np.intp, copy=False)

                for start in range(0, epoch_size, self._batch_size):
                    _minibatch_statistics(
                        X,
                        indices[start : start + self._batch_size],
                        centers,
                        sums,
                        counts,
                        labels,
                        self._n_threads,
                    )
                    learning_rate = _scheduled_learning_rate(
                        self._learning_rate_init, step, n_steps
                    )
                    _ema_update(centers, running_weights, counts, sums, learning_rate)
                    if is_last_epoch:
                        epoch_counts += counts
                        epoch_sums += sums
                    step += 1

                    if reassign and step % _REASSIGNMENT_PERIOD == 0:
                        _split_unused_clusters(centers, running_weights, random_state)

        # Set each center to the mean of the samples assigned to it during the
        # last epoch.
        assigned = epoch_counts > 0
        centers[assigned] = epoch_sums[assigned] / epoch_counts[assigned, None]

        self.cluster_centers_ = centers
        self._n_features_out = self.cluster_centers_.shape[0]
        self.batch_size_ = self._batch_size
        self.n_steps_ = n_steps

        self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(
            X, sample_weight, self.cluster_centers_, n_threads=self._n_threads
        )

        return self

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels_

    def fit_transform(self, X, y=None):
        """Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        return self.fit(X)._transform(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        return tags
