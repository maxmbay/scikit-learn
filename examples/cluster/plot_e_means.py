"""
==========================================================================
Comparison of the E-Means, K-Means and MiniBatchKMeans clustering algorithms
==========================================================================

:class:`~sklearn.cluster.EMeans`, :class:`~sklearn.cluster.KMeans` and
:class:`~sklearn.cluster.MiniBatchKMeans` minimize the same objective, the
inertia (see :ref:`k_means`), but update the cluster centers differently. This
example compares the inertia they reach for a given number of passes over the
data, and for a given fit time.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Generate the data
# -----------------
#
# We generate a mixture of 300 overlapping Gaussian blobs, to be grouped in 64
# clusters. With fewer clusters than blobs, the algorithms cannot recover the
# blobs and need many iterations to settle.

from sklearn.datasets import make_blobs

X, _ = make_blobs(
    n_samples=30_000, n_features=16, centers=300, cluster_std=4.0, random_state=0
)
n_clusters = 64

# %%
# Fit the models with increasing budgets
# --------------------------------------
#
# Each estimator has a parameter that sets its number of passes over the data:
#
# - `max_iter` for :class:`~sklearn.cluster.KMeans`, where a pass is an iteration
#   of Lloyd's algorithm. We set `tol=0` to disable its early stopping;
# - `max_iter` for :class:`~sklearn.cluster.MiniBatchKMeans`. We set
#   `max_no_improvement=None` to disable its early stopping;
# - `n_epochs` for :class:`~sklearn.cluster.EMeans`, which can be fractional.
#
# The three estimators start from samples picked at random, and are run once.

import time

from sklearn.cluster import EMeans, KMeans, MiniBatchKMeans

common_params = dict(n_clusters=n_clusters, init="random", random_state=0)
budgets = {
    "KMeans": [1, 2, 3, 5, 10, 20, 50, 100],
    "MiniBatchKMeans": [1, 2, 3, 5, 10],
    "EMeans": [0.5, 1, 2, 3, 5, 10],
}


def make_estimator(name, n_passes):
    if name == "KMeans":
        return KMeans(max_iter=n_passes, tol=0, n_init=1, **common_params)
    if name == "MiniBatchKMeans":
        return MiniBatchKMeans(
            max_iter=n_passes, max_no_improvement=None, n_init=1, **common_params
        )
    return EMeans(n_epochs=n_passes, **common_params)


inertias, fit_times = {}, {}
for name, n_passes_list in budgets.items():
    inertias[name], fit_times[name] = [], []
    for n_passes in n_passes_list:
        estimator = make_estimator(name, n_passes)
        tic = time.perf_counter()
        estimator.fit(X)
        fit_times[name].append(time.perf_counter() - tic)
        inertias[name].append(estimator.inertia_)

# %%
# Plot the inertia for each budget
# --------------------------------
#
# The inertia is divided by the one of :class:`~sklearn.cluster.KMeans` after
# 100 iterations.

import matplotlib.pyplot as plt
import numpy as np

reference_inertia = inertias["KMeans"][-1]
# Colors that remain distinct with a color vision deficiency.
styles = {
    "KMeans": dict(color="#0072B2", marker="o"),
    "MiniBatchKMeans": dict(color="#D55E00", marker="s"),
    "EMeans": dict(color="#009E73", marker="^"),
}

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
for name, n_passes_list in budgets.items():
    relative_inertia = np.array(inertias[name]) / reference_inertia
    ax.plot(n_passes_list, relative_inertia, label=name, **styles[name])
ax.axhline(1, color="gray", linewidth=0.8, zorder=0)
ax.set_xscale("log")
ax.grid(alpha=0.3)
ax.set_xlabel("Passes over the data")
ax.set_ylabel("Inertia relative to KMeans\nafter 100 iterations")
ax.set_title("Inertia for a given number of passes")
ax.legend()
plt.show()

# %%
# On this dataset, :class:`~sklearn.cluster.EMeans` needs 2 to 3 passes over the
# data to reach the inertia that :class:`~sklearn.cluster.KMeans` reaches after
# tens of iterations, and about 5 passes to reach its inertia after 100
# iterations. :class:`~sklearn.cluster.MiniBatchKMeans` makes most of its progress
# during its first pass, and then improves slowly.
#
# Compare the fit times
# ---------------------
#
# A pass does not take the same time for the three estimators. We print the fit
# time of the largest budget of :class:`~sklearn.cluster.KMeans` and
# :class:`~sklearn.cluster.MiniBatchKMeans`, and of the budget with which
# :class:`~sklearn.cluster.EMeans` reaches the reference inertia.

for name, n_passes in [("KMeans", 100), ("MiniBatchKMeans", 10), ("EMeans", 5)]:
    index = budgets[name].index(n_passes)
    print(
        f"{name} with {n_passes} passes: relative inertia "
        f"{inertias[name][index] / reference_inertia:.3f}, "
        f"fit time {fit_times[name][index]:.2f} s"
    )

# %%
# These fit times depend on the machine. In particular,
# :class:`~sklearn.cluster.KMeans` uses all the cores at each iteration, whereas
# the mini-batches of this example are small enough for
# :class:`~sklearn.cluster.EMeans` to process them with a single thread.
