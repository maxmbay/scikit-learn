"""Tests for EMeans."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from sklearn.cluster import EMeans
from sklearn.cluster._e_means import (
    _learning_rate_law,
    _scheduled_learning_rate,
    _split_unused_clusters,
)
from sklearn.cluster._e_means_minibatch import _ema_update, _minibatch_statistics
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics.cluster import v_measure_score
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import assert_allclose, assert_array_equal


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_e_means_results(dtype):
    # Check that EMeans recovers well separated clusters, and that labels_ and
    # inertia_ are consistent with the final centers.
    X, y = make_blobs(
        n_samples=1000, n_features=3, centers=5, cluster_std=0.5, random_state=0
    )
    X = X.astype(dtype)

    emeans = EMeans(n_clusters=5, init="k-means++", random_state=0).fit(X)

    assert v_measure_score(y, emeans.labels_) == pytest.approx(1)
    assert emeans.cluster_centers_.dtype == dtype
    assert_array_equal(emeans.labels_, emeans.predict(X))
    assert emeans.inertia_ == pytest.approx(-emeans.score(X), rel=1e-5)


def test_e_means_full_batch_is_lloyd_iteration():
    # With a single epoch made of a single batch, the final centers are the means
    # of the samples assigned to each initial center, as after one iteration of
    # Lloyd's algorithm.
    X = np.random.RandomState(0).uniform(size=(100, 4))
    init = X[:3].copy()

    emeans = EMeans(n_clusters=3, init=init, n_epochs=1, batch_size=X.shape[0]).fit(X)

    labels = pairwise_distances_argmin(X, init)
    expected_centers = [X[labels == j].mean(axis=0) for j in range(3)]
    assert emeans.n_steps_ == 1
    assert_allclose(emeans.cluster_centers_, expected_centers)


def test_e_means_reference_implementation():
    # Check the centers against those of the reference implementation in faiss,
    # `NumpyEmeans(3, 6, nepochs=5, seed=0)` from `faiss.contrib.e_means`, on the
    # same data, from the same initial centers and with the same random numbers.
    rng = np.random.RandomState(0)
    X = rng.normal(size=(300, 3)) + 5 * rng.randint(3, size=(300, 1))
    X = X.astype(np.float32)

    class ReferenceRandomState(np.random.RandomState):
        # The reference implementation shuffles the samples of each epoch with a
        # RandomState seeded from its seed and the epoch, and draws the random
        # numbers used to reassign clusters from RandomState(seed).
        def __init__(self, seed):
            super().__init__(seed)
            self._seed = seed
            self._epoch = 0

        def permutation(self, n_samples):
            epoch_seed = (self._seed * 1000003 + self._epoch) % 2**32
            self._epoch += 1
            return np.random.RandomState(epoch_seed).permutation(n_samples)

    init = X[np.random.RandomState(0).permutation(X.shape[0])[:6]]
    emeans = EMeans(n_clusters=6, init=init, random_state=ReferenceRandomState(0))
    emeans.fit(X)

    # One cluster is reassigned during this run of the reference implementation.
    expected_centers = [
        [0.45666954, -1.0414114, -0.24914086],
        [6.292352, 5.3675885, 4.991632],
        [10.135136, 9.889491, 9.988427],
        [5.062959, 4.2198195, 4.6458964],
        [4.0048666, 5.6055803, 4.815979],
        [-0.5204016, 0.560384, -0.11574496],
    ]
    assert emeans.batch_size_ == 4
    assert _learning_rate_law(300, 6, 5, 4) == pytest.approx(0.0814347731973082)
    assert emeans.n_steps_ == 375
    assert_allclose(emeans.cluster_centers_, expected_centers)


@pytest.mark.parametrize(
    "n_samples, n_features, n_clusters, n_epochs, batch_size, "
    "expected_batch_size, expected_learning_rate",
    [
        # Values computed with the formulas given by the authors of e-means.
        (10_000, 2, 10, 5, "auto", 64, 0.16123550575909054),
        (20_000, 16, 50, 3, "auto", 128, 0.11061388834829086),
        (20_000, 17, 50, 3, "auto", 1024, 0.7192196997266359),
        (5_000, 17, 8, 2.5, "auto", 128, 0.49027924080125734),
        (5_000, 4, 8, 2.5, "auto", 16, 0.0754035147008455),
        (10_000, 2, 10, 5, 256, 256, 0.5616881924308813),
    ],
)
def test_e_means_scaling_laws(
    n_samples,
    n_features,
    n_clusters,
    n_epochs,
    batch_size,
    expected_batch_size,
    expected_learning_rate,
):
    # Check the batch size and the initial learning rate set by the scaling laws.
    # The batches are about 8 times smaller for data with 16 features or less.
    X = np.random.RandomState(0).uniform(size=(n_samples, n_features))
    emeans = EMeans(
        n_clusters=n_clusters, n_epochs=n_epochs, batch_size=batch_size
    ).fit(X)

    assert emeans.batch_size_ == expected_batch_size
    learning_rate = _learning_rate_law(
        n_samples, n_clusters, n_epochs, emeans.batch_size_
    )
    assert learning_rate == pytest.approx(expected_learning_rate)


@pytest.mark.parametrize(
    "n_samples, n_epochs, expected_n_steps",
    [(1000, 1, 10), (1000, 3, 30), (1000, 2.5, 25), (1050, 2, 22), (1000, 0.25, 3)],
)
def test_e_means_n_epochs(n_samples, n_epochs, expected_n_steps):
    # Check the number of steps for whole and fractional numbers of epochs. The
    # last batch of each epoch can be smaller than the others.
    X = np.random.RandomState(0).uniform(size=(n_samples, 2))
    emeans = EMeans(n_clusters=3, n_epochs=n_epochs, batch_size=100).fit(X)

    assert emeans.n_steps_ == expected_n_steps
    assert emeans.labels_.shape == (n_samples,)


def test_e_means_tiny_n_epochs():
    # Check that at least one sample is used when n_epochs is very small.
    X = np.random.RandomState(0).uniform(size=(100, 2))
    emeans = EMeans(n_clusters=3, n_epochs=1e-6).fit(X)

    assert emeans.n_steps_ == 1


def test_scheduled_learning_rate():
    # The learning rate follows a cosine decay to a tenth of its initial value.
    learning_rates = [_scheduled_learning_rate(0.5, step, 10) for step in range(11)]

    assert learning_rates[0] == pytest.approx(0.5)
    assert learning_rates[5] == pytest.approx(0.5 * 0.55)
    assert learning_rates[10] == pytest.approx(0.05)
    assert np.all(np.diff(learning_rates) < 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_threads", [1, 2])
def test_minibatch_statistics(dtype, n_threads):
    # Check the labels, the sums and the counts of a mini-batch against their
    # computation with numpy.
    rng = np.random.RandomState(0)
    n_samples, n_features, n_clusters = 2000, 5, 7
    X = rng.uniform(size=(n_samples, n_features)).astype(dtype)
    centers = rng.uniform(size=(n_clusters, n_features)).astype(dtype)
    # A mini-batch of 3 chunks, the last one being smaller than the others.
    indices = rng.permutation(n_samples)[:700].astype(np.intp)

    sums = np.full_like(centers, np.nan)
    counts = np.full(n_clusters, np.nan, dtype=dtype)
    labels = np.full(indices.shape[0], -1, dtype=np.int32)
    _minibatch_statistics(X, indices, centers, sums, counts, labels, n_threads)

    X_batch = X[indices]
    expected_labels = pairwise_distances_argmin(X_batch, centers)
    expected_sums = np.zeros_like(centers)
    np.add.at(expected_sums, expected_labels, X_batch)

    assert_array_equal(labels, expected_labels)
    assert_allclose(sums, expected_sums)
    assert_array_equal(counts, np.bincount(expected_labels, minlength=n_clusters))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ema_update(dtype):
    # Check the update against the exponential moving averages of the sums and
    # numbers of the samples of each cluster.
    centers = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=dtype)
    running_weights = np.array([2.0, 4.0], dtype=dtype)
    counts = np.array([2.0, 0.0], dtype=dtype)
    sums = np.array([[2.0, 4.0], [0.0, 0.0]], dtype=dtype)

    _ema_update(centers, running_weights, counts, sums, learning_rate=0.5)

    # The running sum of the first cluster is 0.5 * 0 + 0.5 * [2, 4] and its
    # running weight is 0.5 * 2 + 0.5 * 2. The second cluster has no sample.
    assert_allclose(running_weights, [2.0, 2.0])
    assert_allclose(centers, [[0.5, 1.0], [5.0, 5.0]])

    # With a learning rate of 1, the past of the updated clusters is forgotten
    # and the running weights of the other clusters are kept.
    _ema_update(centers, running_weights, counts, sums, learning_rate=1)

    assert_allclose(running_weights, [2.0, 2.0])
    assert_allclose(centers, [[1.0, 2.0], [5.0, 5.0]])


def test_split_unused_clusters():
    # Check that an unused cluster is reassigned by splitting the used one.
    centers = np.array([[1.0, 2.0, 3.0], [100.0, 100.0, 100.0]])
    running_weights = np.array([10.0, 1e-6])

    n_reassigned = _split_unused_clusters(
        centers, running_weights, np.random.RandomState(0)
    )

    assert n_reassigned == 1
    perturbation = np.array([1e-4, -1e-4, 1e-4])
    assert_allclose(centers[0], [1.0, 2.0, 3.0] * (1 - perturbation))
    assert_allclose(centers[1], [1.0, 2.0, 3.0] * (1 + perturbation))
    assert_allclose(running_weights, [5.0, 5.0])

    # No cluster is reassigned when the running weights are balanced.
    assert _split_unused_clusters(centers, running_weights, None) == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_split_unused_clusters_center_at_origin(dtype):
    # The relative perturbation does not move a center at the origin: check that
    # its two copies are made distinct.
    centers = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]], dtype=dtype)
    running_weights = np.array([10.0, 1e-6], dtype=dtype)

    _split_unused_clusters(centers, running_weights, np.random.RandomState(0))

    assert centers.dtype == dtype
    assert_array_equal(centers[0], 0)
    assert centers[1, 0] == np.nextafter(dtype(0), dtype(1))
    assert_array_equal(centers[1, 1:], 0)


@pytest.mark.parametrize("n_features, expected_n_used", [(3, 4), (2, 2)])
def test_e_means_reassignment(n_features, expected_n_used):
    # Check that clusters without samples are reassigned, except for data with 2
    # features or less.
    X, _ = make_blobs(n_samples=1000, n_features=n_features, centers=4, random_state=0)
    init = np.concatenate([X[:2], np.full((2, n_features), 1e3)])

    emeans = EMeans(n_clusters=4, init=init, random_state=0).fit(X)

    assert np.unique(emeans.labels_).size == expected_n_used


@pytest.mark.parametrize("init", ["random", "k-means++", "array", "callable"])
def test_e_means_init(init):
    X, y, centers = make_blobs(
        n_samples=300, centers=3, cluster_std=0.5, random_state=0, return_centers=True
    )
    if init == "array":
        init = centers
    elif init == "callable":

        def init(X, n_clusters, random_state):
            return centers

    emeans = EMeans(n_clusters=3, init=init, random_state=0).fit(X)

    assert v_measure_score(y, emeans.labels_) == pytest.approx(1)


def test_e_means_init_does_not_modify_inputs():
    # The centers are updated in place: check that neither the data nor the
    # initial centers are modified, even when a callable returns a view of X.
    X = np.random.RandomState(0).uniform(size=(100, 2))
    X_copy = X.copy()

    EMeans(n_clusters=3, init=lambda X, n_clusters, random_state: X[:3]).fit(X)
    assert_array_equal(X, X_copy)

    init = X[:3].copy()
    EMeans(n_clusters=3, init=init).fit(X)
    assert_array_equal(init, X_copy[:3])
    assert_array_equal(X, X_copy)


def test_e_means_n_threads(monkeypatch):
    # Check that the result does not depend on the number of threads: the
    # samples of a mini-batch are assigned in parallel but summed sequentially.
    X, _ = make_blobs(n_samples=3000, centers=10, n_features=5, random_state=0)
    # Use several threads even though the mini-batches are small.
    monkeypatch.setattr("sklearn.cluster._e_means._MIN_BATCH_WORK_FOR_THREADS", 0)
    emeans = EMeans(n_clusters=10, batch_size=1024, random_state=0)

    with threadpool_limits(limits=1, user_api="openmp"):
        centers_1 = emeans.fit(X).cluster_centers_
    with threadpool_limits(limits=2, user_api="openmp"):
        centers_2 = emeans.fit(X).cluster_centers_

    assert_array_equal(centers_1, centers_2)


def test_e_means_batch_n_threads(monkeypatch):
    # Check that the assignment of the samples of a mini-batch only uses several
    # threads when the mini-batches are large enough.
    X = np.random.RandomState(0).uniform(size=(1000, 2))
    n_threads_used = set()

    def minibatch_statistics(*args):
        # n_threads is the last argument.
        n_threads_used.add(args[-1])
        return _minibatch_statistics(*args)

    monkeypatch.setattr(
        "sklearn.cluster._e_means._minibatch_statistics", minibatch_statistics
    )
    EMeans(n_clusters=3, n_epochs=1).fit(X)
    assert n_threads_used == {1}

    n_threads_used.clear()
    monkeypatch.setattr("sklearn.cluster._e_means._MIN_BATCH_WORK_FOR_THREADS", 0)
    EMeans(n_clusters=3, n_epochs=1).fit(X)
    assert n_threads_used == {_openmp_effective_n_threads()}


def test_e_means_n_samples_lt_n_clusters():
    X = np.random.RandomState(0).uniform(size=(3, 2))

    with pytest.raises(ValueError, match="n_samples=3 should be >= n_clusters=4"):
        EMeans(n_clusters=4).fit(X)
