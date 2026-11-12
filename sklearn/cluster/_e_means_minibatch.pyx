# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from cython cimport floating
from cython.parallel cimport parallel, prange
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset

from sklearn.cluster._k_means_common import CHUNK_SIZE
from sklearn.utils._cython_blas cimport _gemm
from sklearn.utils._cython_blas cimport RowMajor, Trans, NoTrans
from sklearn.utils._typedefs cimport intp_t
from sklearn.utils.extmath import row_norms


def _minibatch_statistics(
        const floating[:, ::1] X,            # IN
        const intp_t[::1] indices,           # IN
        const floating[:, ::1] centers,      # IN
        floating[:, ::1] sums,               # OUT
        floating[::1] counts,                # OUT
        int[::1] labels,                     # OUT
        int n_threads):
    """Assign the samples of a mini-batch and sum them by cluster.

    The samples are assigned to their closest center in parallel, over chunks of
    the mini-batch. They are then summed sequentially, in the order of the
    mini-batch, so that the result does not depend on the number of threads.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=floating
        The observations to cluster.

    indices : ndarray of shape (batch_size,), dtype=intp
        The indices of the samples of the mini-batch.

    centers : ndarray of shape (n_clusters, n_features), dtype=floating
        The cluster centers.

    sums : ndarray of shape (n_clusters, n_features), dtype=floating
        Placeholder for the sums of the samples of the mini-batch assigned to
        each center.

    counts : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the number of samples of the mini-batch assigned to
        each center.

    labels : ndarray of shape (>= batch_size,), dtype=int
        Placeholder for the index of the closest center of each sample of the
        mini-batch.

    n_threads : int
        The number of threads to be used by openmp to assign the samples. At
        most one thread per chunk of the mini-batch is used.
    """
    cdef:
        int n_samples = indices.shape[0]
        int n_features = X.shape[1]
        int n_clusters = centers.shape[0]

        int i, k, label
        intp_t sample_idx

    memset(&sums[0, 0], 0, n_clusters * n_features * sizeof(floating))
    memset(&counts[0], 0, n_clusters * sizeof(floating))

    if n_samples == 0:
        return

    cdef:
        # Same chunks as for an iteration of Lloyd's algorithm.
        int n_samples_chunk = CHUNK_SIZE if n_samples > CHUNK_SIZE else n_samples
        int n_chunks = n_samples // n_samples_chunk
        int n_samples_rem = n_samples % n_samples_chunk
        int chunk_idx
        int start, end

        floating[::1] centers_squared_norms = row_norms(centers, squared=True)

        floating *X_chunk
        floating *pairwise_distances_chunk

    # count remainder chunk in total number of chunks
    n_chunks += n_samples != n_chunks * n_samples_chunk

    # number of threads should not be bigger than number of chunks
    n_threads = min(n_threads, n_chunks)

    with nogil, parallel(num_threads=n_threads):
        # thread local buffers
        X_chunk = <floating*> malloc(n_samples_chunk * n_features * sizeof(floating))
        pairwise_distances_chunk = <floating*> malloc(n_samples_chunk * n_clusters * sizeof(floating))

        for chunk_idx in prange(n_chunks, schedule='static'):
            start = chunk_idx * n_samples_chunk
            if chunk_idx == n_chunks - 1 and n_samples_rem > 0:
                end = start + n_samples_rem
            else:
                end = start + n_samples_chunk

            _assign_chunk(
                X,
                indices[start: end],
                centers,
                centers_squared_norms,
                labels[start: end],
                X_chunk,
                pairwise_distances_chunk)

        free(X_chunk)
        free(pairwise_distances_chunk)

    with nogil:
        for i in range(n_samples):
            sample_idx = indices[i]
            label = labels[i]

            counts[label] += 1
            for k in range(n_features):
                sums[label, k] += X[sample_idx, k]


cdef void _assign_chunk(
        const floating[:, ::1] X,                   # IN
        const intp_t[::1] indices,                  # IN
        const floating[:, ::1] centers,             # IN
        const floating[::1] centers_squared_norms,  # IN
        int[::1] labels,                            # OUT
        floating *X_chunk,                          # TMP
        floating *pairwise_distances) noexcept nogil:  # TMP
    """Assign the samples of a chunk of a mini-batch to their closest center."""
    cdef:
        int n_samples = indices.shape[0]
        int n_clusters = centers.shape[0]
        int n_features = centers.shape[1]

        floating sq_dist, min_sq_dist
        int i, j, label

    # Gather the samples of the chunk in a contiguous buffer, to compute their
    # distances to the centers with BLAS.
    for i in range(n_samples):
        memcpy(X_chunk + i * n_features, &X[indices[i], 0], n_features * sizeof(floating))

    # Instead of computing the full pairwise squared distances matrix,
    # ||X - C||² = ||X||² - 2 X.C^T + ||C||², we only need to store
    # the - 2 X.C^T + ||C||² term since the argmin for a given sample only
    # depends on the centers.
    # pairwise_distances = ||C||²
    for i in range(n_samples):
        for j in range(n_clusters):
            pairwise_distances[i * n_clusters + j] = centers_squared_norms[j]

    # pairwise_distances += -2 * X.dot(C.T)
    _gemm(RowMajor, NoTrans, Trans, n_samples, n_clusters, n_features,
          -2.0, X_chunk, n_features, &centers[0, 0], n_features,
          1.0, pairwise_distances, n_clusters)

    for i in range(n_samples):
        min_sq_dist = pairwise_distances[i * n_clusters]
        label = 0
        for j in range(1, n_clusters):
            sq_dist = pairwise_distances[i * n_clusters + j]
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j
        labels[i] = label


def _ema_update(
        floating[:, ::1] centers,            # INOUT
        floating[::1] running_weights,       # INOUT
        const floating[::1] counts,          # IN
        const floating[:, ::1] sums,         # IN
        double learning_rate):
    """Update the centers with the statistics of a mini-batch, in place.

    The running weight N_j and the running sum S_j of the samples assigned to
    cluster j are exponential moving averages of the number n_j and the sum s_j
    of the samples of the mini-batch assigned to it, and the center of the
    cluster is S_j / N_j:

        N_j <- (1 - learning_rate) * N_j + learning_rate * n_j
        S_j <- (1 - learning_rate) * S_j + learning_rate * s_j

    The centers of the clusters without samples in the mini-batch do not move.

    Parameters
    ----------
    centers : ndarray of shape (n_clusters, n_features), dtype=floating
        The cluster centers, S_j / N_j.

    running_weights : ndarray of shape (n_clusters,), dtype=floating
        The running weights of the clusters, N_j.

    counts : ndarray of shape (n_clusters,), dtype=floating
        The number of samples of the mini-batch assigned to each center, n_j.

    sums : ndarray of shape (n_clusters, n_features), dtype=floating
        The sums of the samples of the mini-batch assigned to each center, s_j.

    learning_rate : float
        The learning rate, in (0, 1].
    """
    cdef:
        int n_clusters = centers.shape[0]
        int n_features = centers.shape[1]

        # The computations are done with the precision of the data.
        floating alpha = learning_rate
        floating decay = 1 - learning_rate
        floating batch_weight, new_weight, step_size, shift
        int j, k

    for j in range(n_clusters):
        # With a learning rate of 1, the past of the clusters with samples in
        # the mini-batch is forgotten, and the running weights of the other
        # clusters are kept.
        if learning_rate < 1:
            running_weights[j] *= decay

        if counts[j] > 0:
            batch_weight = alpha * counts[j]
            if learning_rate < 1:
                new_weight = running_weights[j] + batch_weight
            else:
                new_weight = counts[j]
            running_weights[j] = new_weight

            # c_j <- c_j + step_size * (s_j / n_j - c_j)
            step_size = batch_weight / new_weight
            if step_size == 1:
                for k in range(n_features):
                    centers[j, k] = sums[j, k] / counts[j]
            else:
                for k in range(n_features):
                    shift = sums[j, k] / counts[j] - centers[j, k]
                    shift *= step_size
                    centers[j, k] += shift
