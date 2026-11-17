"""
Region graphs, as E6 builds them.

`adjacency` mirrors the nested adjacency() inside
src/analysis/graph_relation_ablation.layer_c, and `gnn_train_rows`
reproduces the exact rows that function sees: panel rows with every lag and
every horizon target present, from the fold's training seasons minus the
most recent one (held out for early stopping). The research function is
nested inside layer_c and cannot be imported without running the
experiment, so it is mirrored here; the edge sets can be checked against
the experiment by re-running it.
"""

import numpy as np
import pandas as pd

from core import data as D

LAGS = [f"ili_asof_lag{k}" for k in range(1, 9)]
K = 3                     # NEIGHBOURS_K in the experiment
REL_KINDS = ["corr", "climate", "mass", "random", "full"]


def gnn_panel():
    core = D.core_panel()
    d = core.copy()
    for h in D.HORIZONS:
        k = core[["season", "hhs_region", "season_week_idx", "wili_final"]].copy()
        k["season_week_idx"] -= h
        d = d.merge(k.rename(columns={"wili_final": f"y{h}"}),
                    on=["season", "hhs_region", "season_week_idx"], how="left")
    need = LAGS + [f"y{h}" for h in D.HORIZONS]
    return d[d[need].notna().all(axis=1)].copy()


def gnn_train_rows(fold):
    P = gnn_panel()
    tr = dict((f, t) for f, t, _ in D.FOLDS)[fold]
    train_all = P[P.season.isin(tr)]
    val_season = tr[-1]
    train = train_all[train_all.season != val_season] if len(tr) > 1 else train_all
    return train if len(train) else train_all


def adjacency(kind, train, seed=0):
    """Binary directed adjacency (row r attends to columns). Unnormalised."""
    regions = D.REGIONS
    R = len(regions)
    ridx = {r: i for i, r in enumerate(regions)}
    A = np.zeros((R, R))
    if kind == "none":
        return np.eye(R)
    if kind == "full":
        return np.ones((R, R)) - np.eye(R)
    if kind == "corr":
        C = train.pivot_table(index="epiweek", columns="hhs_region",
                              values="ili_asof_lag1").corr()
        for r in regions:
            for s in C[r].drop(index=r).sort_values(ascending=False).index[:K]:
                A[ridx[r], ridx[s]] = 1
    elif kind == "random":
        rng = np.random.default_rng(500 + seed)
        for r in regions:
            for s in rng.choice([x for x in regions if x != r], K, replace=False):
                A[ridx[r], ridx[s]] = 1
    elif kind == "climate":
        prof = (train.groupby(["hhs_region", "season_week_idx"])
                [["temp_c", "abs_humidity"]].mean().reset_index())
        for v in ("temp_c", "abs_humidity"):
            prof[v] = (prof[v] - prof[v].mean()) / (prof[v].std() + 1e-9)
        piv = prof.pivot_table(index="hhs_region", columns="season_week_idx",
                               values=["temp_c", "abs_humidity"])
        piv = piv.reindex(regions).dropna(axis=1)
        V = piv.values.astype(np.float64)
        dist = np.sqrt(((V[:, None, :] - V[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        for i in range(R):
            for j in np.argsort(dist[i])[:K]:
                A[i, j] = 1
    elif kind == "mass":
        pop = train.groupby("hhs_region")["population"].first().reindex(regions)
        order = list(pop.sort_values(ascending=False).index)
        for r in regions:
            for s in [x for x in order if x != r][:K]:
                A[ridx[r], ridx[s]] = 1
    return A


def normalise(A):
    deg = A.sum(1, keepdims=True)
    deg[deg == 0] = 1
    return A / deg


def season_corr_graph(season):
    """Top-K correlation neighbours fitted on ONE season alone (Layer A view)."""
    core = D.core_panel()
    return adjacency("corr", core[core.season == season])


def edge_set(A):
    return {(i, j) for i in range(A.shape[0]) for j in range(A.shape[1])
            if A[i, j] and i != j}


def overlap(A, B):
    """Mean share of each node's K neighbours that survive (Layer A metric)."""
    o = []
    for i in range(A.shape[0]):
        a, b = set(np.flatnonzero(A[i])), set(np.flatnonzero(B[i]))
        if a:
            o.append(len(a & b) / len(a))
    return float(np.mean(o))
