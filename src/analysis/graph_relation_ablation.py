#!/usr/bin/env python3
"""
EpiSurge - Phase 3 / E6: is an epidemic graph better than chance?

Master Plan 4.3 specifies a tri-relational graph over the 10 HHS regions and
mandates a density-matched random-graph control (G5). This experiment runs
that control and answers the plan's third claim: WHICH relation between
regions carries forecasting information?

The answer, on this dataset, is none of them - and the interesting part is
that the mechanism is measurable without any model at all.

THREE LAYERS, WHICH AGREE
-------------------------
A. STABILITY (no model). Regions are strongly correlated within a season -
   mean pairwise r = 0.85 - so building adjacency from correlation looks
   obviously justified. But the correlation MATRIX does not persist between
   seasons (matrix-to-matrix r = +0.10), and the top-3 neighbours chosen on
   training seasons are the test season's true top-3 only ~35% of the time,
   against a 33.3% chance rate. A correlation graph fitted on history is,
   for the next season, statistically a random graph.

B. FEATURE SCREEN (gradient boosting). Adding neighbour wILI lags to B2
   makes forecasts WORSE, and correlation-chosen neighbours are
   consistently worse than randomly-chosen ones at identical column counts.

C. GNN (--with-gnn). A per-region GRU plus one graph-convolution layer,
   with architecture, parameter count, optimiser and seeds held fixed
   across four adjacencies: none / correlation / fully-connected /
   density-matched random. No relation beats no-graph, and correlation does
   not beat random, at any horizon.

WHY LAYER C IS CREDIBLE
-----------------------
A first implementation of the GNN scored 1.10 MAE at h=1 where plain
persistence gets 0.76. It had never learned to copy lag1, so its null told
us nothing about graphs - only that the model failed to train. Reporting
that as "graphs do not help" would have been wrong.

The model here predicts the RESIDUAL FROM PERSISTENCE, so zero output
already equals persistence and the network can only improve on it. It then
beats persistence by +18% to +50% at h=2..8, sign-consistent across folds.
A working model that still cannot use the graph is evidence; a broken one
is not. That distinction is the reason this file exists in this form.

WHAT IS NOT CLAIMED
-------------------
The GNN is NOT competitive with gradient boosting at short horizons
(-15.8% at h=1). With ~1,650 training rows over 10 nodes that is expected
and is reported as a limitation, not hidden. The contribution here is the
relation ablation and its mechanism, not the model's accuracy.

Inference is sign consistency across the pre-COVID folds, the rule standing
since E2. The fold testing 2022-23 is the COVID discontinuity and is
reported separately.

Inputs:
    data/processed/master_region_week.csv

Outputs:
    data/reports/graph_stability.csv
    data/reports/graph_feature_screen.csv
    data/reports/graph_gnn_ablation.csv        (only with --with-gnn)
    figures/fig_graph_stability.png
    figures/fig_graph_feature_screen.png
    figures/fig_graph_gnn_ablation.png         (only with --with-gnn)

Run (from the repository root, inside the venv):
    python src/analysis/graph_relation_ablation.py
    python src/analysis/graph_relation_ablation.py --with-gnn

Layer C needs PyTorch (a ~2.5 GB install). Layers A and B need nothing
beyond the project's existing requirements, and carry the finding on their
own.
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

_src = str(Path(__file__).resolve().parents[1])
if _src not in sys.path:
    sys.path.insert(0, _src)
_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from baselines import (                                           # noqa: E402
    FOLDS, HORIZONS, B2_FEATURES, B2_PARAMS,
    load, build_targets, usable,
)
import plotstyle                                                  # noqa: E402

REPORTS = Path("data/reports")
FIGURES = Path("figures")

LAGS = [f"ili_asof_lag{k}" for k in range(1, 9)]
NEIGHBOURS_K = 3                 # graph density: 3 of 9 possible neighbours
NEIGHBOUR_LAGS = [1, 2, 4]
COVID_FOLD_TEST = "2022-23"
SEED = 11


def pre_covid_folds():
    return [f for f, _, te in FOLDS if te != COVID_FOLD_TEST]


def sign_verdict(values):
    """A signed mean when every fold agrees, else FLIP. The standing rule."""
    v = [x for x in values if x == x]
    if not v:
        return "-"
    if all(x > 0 for x in v) or all(x < 0 for x in v):
        return f"{np.mean(v):+.1f}"
    return "FLIP"


# ──────────────────────────────────────────────────────────────────────
# LAYER A - stability of the correlation structure, no model involved
# ──────────────────────────────────────────────────────────────────────

def layer_a(core, regions):
    print("\n" + "=" * 76)
    print("LAYER A  Does the correlation structure persist across seasons?")
    print("=" * 76)

    piv = core.pivot_table(index=["season", "epiweek"], columns="hhs_region",
                           values="ili_asof_lag1")
    seasons = sorted(core["season"].unique())
    C = {s: piv.loc[s].corr() for s in seasons}
    iu = np.triu_indices(len(regions), 1)

    within = float(np.mean([np.abs(C[s].values[iu]).mean() for s in seasons]))
    print(f"\n  Mean pairwise |r| WITHIN a season: {within:.3f}")
    print("  Regions move together, so a correlation graph looks justified.")

    rows = []
    print("\n  Stability of the correlation MATRIX between season pairs:")
    for a, b in itertools.combinations(seasons, 2):
        r = float(np.corrcoef(C[a].values[iu], C[b].values[iu])[0, 1])
        rows.append({"kind": "matrix_persistence", "a": a, "b": b, "value": r})
        print(f"    {a} vs {b}:  r = {r:+.3f}")
    mat = np.mean([x["value"] for x in rows])
    print(f"    mean r = {mat:+.3f}")

    def top_k(cm):
        return {r: set(cm[r].drop(index=r)
                       .sort_values(ascending=False).index[:NEIGHBOURS_K])
                for r in regions}

    overlaps = []
    for a, b in itertools.combinations(seasons, 2):
        ta, tb = top_k(C[a]), top_k(C[b])
        o = float(np.mean([len(ta[r] & tb[r]) / NEIGHBOURS_K for r in regions]))
        overlaps.append(o)
        rows.append({"kind": "topk_overlap_seasons", "a": a, "b": b, "value": o})
    chance = NEIGHBOURS_K / (len(regions) - 1)
    print(f"\n  Top-{NEIGHBOURS_K} neighbour overlap between season pairs: "
          f"{100 * np.mean(overlaps):.1f}%   (chance {100 * chance:.1f}%)")

    print("\n  Per fold - do the neighbours chosen on TRAINING seasons turn out")
    print("  to be the test season's actual top-3?")
    hits = []
    for fold, train_seasons, test_season in FOLDS:
        sub = piv.loc[piv.index.get_level_values(0).isin(train_seasons)]
        a, b = top_k(sub.corr()), top_k(C[test_season])
        o = float(np.mean([len(a[r] & b[r]) / NEIGHBOURS_K for r in regions]))
        hits.append(o)
        rows.append({"kind": "train_to_test_overlap", "a": fold,
                     "b": test_season, "value": o})
        print(f"    {fold} (test {test_season}): {100 * o:5.1f}%")
    print(f"    mean {100 * np.mean(hits):.1f}% against {100 * chance:.1f}% chance")

    print("\n  READING: regions are correlated, but WHICH regions pair with")
    print("  which carries almost no information from one season to the next.")
    print("  A correlation graph fitted on history is, for the next season,")
    print("  statistically a random graph.")

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "graph_stability.csv", index=False)
    return df, within, float(mat), float(np.mean(hits)), chance


# ──────────────────────────────────────────────────────────────────────
# LAYER B - feature-level screen with gradient boosting
# ──────────────────────────────────────────────────────────────────────

def neighbour_frame(core, d, nbrs, tag):
    """Mean of each row's neighbours' as-of wILI, at several lags."""
    out = {}
    for lag in NEIGHBOUR_LAGS:
        p = core.pivot_table(index="epiweek", columns="hhs_region",
                             values=f"ili_asof_lag{lag}")
        vals = []
        for r, e in zip(d["hhs_region"], d["epiweek"]):
            ns = nbrs.get(r, [])
            if not ns or e not in p.index:
                vals.append(np.nan)
                continue
            vals.append(np.nanmean([p.at[e, n] for n in ns if n in p.columns]))
        out[f"{tag}_nb{lag}"] = vals
    return pd.DataFrame(out, index=d.index)


def layer_b(core, regions):
    print("\n" + "=" * 76)
    print("LAYER B  Does neighbour information help gradient boosting?")
    print("=" * 76)
    print("\n  Neighbour sets are chosen from TRAINING weeks only. The random")
    print("  arm has the same density, so it controls for column count.")

    rng = np.random.default_rng(SEED)
    piv = core.pivot_table(index="epiweek", columns="hhs_region",
                           values="ili_asof_lag1")
    labels = {
        "corr": f"+ {NEIGHBOURS_K} most-correlated neighbours",
        "rand": f"+ {NEIGHBOURS_K} RANDOM neighbours (control)",
        "all": "+ all other regions (mean)",
    }
    rows = []
    for h in HORIZONS:
        d0 = build_targets(core, h)
        for fold, train_seasons, test_season in FOLDS:
            train = usable(d0[d0["season"].isin(train_seasons)])
            test = usable(d0[d0["season"] == test_season])
            if train.empty or test.empty:
                continue
            sub = piv.loc[piv.index.isin(set(train["epiweek"]))]
            C = sub.corr()
            nbr = {
                "corr": {r: list(C[r].drop(index=r)
                                 .sort_values(ascending=False)
                                 .index[:NEIGHBOURS_K]) for r in regions},
                "rand": {r: list(rng.choice([x for x in regions if x != r],
                                            NEIGHBOURS_K, replace=False))
                         for r in regions},
                "all": {r: [x for x in regions if x != r] for r in regions},
            }
            y = test["y_target"].values

            def fit(extra_tr=None, extra_te=None):
                A = (pd.concat([train[B2_FEATURES].reset_index(drop=True),
                                extra_tr.reset_index(drop=True)], axis=1)
                     if extra_tr is not None else train[B2_FEATURES])
                B = (pd.concat([test[B2_FEATURES].reset_index(drop=True),
                                extra_te.reset_index(drop=True)], axis=1)
                     if extra_te is not None else test[B2_FEATURES])
                m = HistGradientBoostingRegressor(random_state=0, **B2_PARAMS)
                m.fit(A.values, train["y_target"].values)
                return float(np.abs(y - m.predict(B.values)).mean())

            rows.append({"horizon": h, "fold": fold, "test_season": test_season,
                         "arm": "B2 own history only", "mae": fit()})
            for tag, nb in nbr.items():
                rows.append({"horizon": h, "fold": fold,
                             "test_season": test_season, "arm": labels[tag],
                             "mae": fit(neighbour_frame(core, train, nb, tag),
                                        neighbour_frame(core, test, nb, tag))})
        print(f"  h={h} done")

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "graph_feature_screen.csv", index=False)
    pre = pre_covid_folds()

    print("\n  MAE (pre-COVID folds, mean)")
    print(df[df.fold.isin(pre)].pivot_table(index="arm", columns="horizon",
                                            values="mae").round(4).to_string())

    base = df[df.arm == "B2 own history only"].set_index(["horizon", "fold"])["mae"]
    print("\n  Skill vs own-history-only, sign-consistent across pre-COVID folds")
    out = []
    for arm in labels.values():
        line = {"arm": arm}
        for h in HORIZONS:
            vals = []
            for f in pre:
                s = df[(df.arm == arm) & (df.horizon == h) & (df.fold == f)]["mae"]
                if len(s):
                    b = base[(h, f)]
                    vals.append(100 * (b - s.iloc[0]) / b)
            line[f"h{h}"] = sign_verdict(vals)
        out.append(line)
    print(pd.DataFrame(out).to_string(index=False))
    print("\n  READING: neighbour information makes forecasts worse, and")
    print("  correlation-chosen neighbours are worse than random ones at the")
    print("  same column count - so it is not the extra parameters.")
    return df


# ──────────────────────────────────────────────────────────────────────
# LAYER C - the GNN (optional)
# ──────────────────────────────────────────────────────────────────────

def layer_c(core, regions, n_seeds):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("\n  [skip] LAYER C needs PyTorch, which is not installed.")
        print("         Layers A and B carry the finding without it.")
        print("         To include it:  pip install torch")
        return None

    print("\n" + "=" * 76)
    print("LAYER C  A trained spatio-temporal GNN, four adjacencies")
    print("=" * 76)
    torch.set_num_threads(4)

    R = len(regions)
    ridx = {r: i for i, r in enumerate(regions)}
    H = list(HORIZONS)

    d = core.copy()
    for h in H:
        k = core[["season", "hhs_region", "season_week_idx", "wili_final"]].copy()
        k["season_week_idx"] -= h
        d = d.merge(k.rename(columns={"wili_final": f"y{h}"}),
                    on=["season", "hhs_region", "season_week_idx"], how="left")
    need = LAGS + [f"y{h}" for h in H]
    P = d[d[need].notna().all(axis=1)].copy()
    print(f"\n  panel rows {len(P)}, {R} regions, {len(H)} horizons per head")

    def pack(df):
        keys = sorted(df.groupby(["season", "epiweek"]).groups.keys())
        W = len(keys)
        X = np.zeros((W, R, len(LAGS)), np.float32)
        Y = np.zeros((W, R, len(H)), np.float32)
        Bp = np.zeros((W, R, len(H)), np.float32)
        M = np.zeros((W, R), np.float32)
        WK = np.zeros((W, R, 2), np.float32)
        RI = np.zeros((W, R), np.int64)
        g = df.set_index(["season", "epiweek"])
        for i, key in enumerate(keys):
            for _, row in g.loc[[key]].iterrows():
                j = ridx[row["hhs_region"]]
                X[i, j] = row[LAGS].values.astype(np.float32)
                Y[i, j] = [row[f"y{h}"] for h in H]
                Bp[i, j] = row["ili_asof_lag1"]        # persistence anchor
                ang = 2 * np.pi * row["season_week_idx"] / 52.0
                WK[i, j] = [np.sin(ang), np.cos(ang)]
                RI[i, j] = j
                M[i, j] = 1
        t = [torch.tensor(a) for a in (X, Y, Bp, WK, M)]
        return t + [torch.tensor(RI)]

    def adjacency(kind, train_df, rng):
        if kind == "none":
            return torch.eye(R)
        A = np.zeros((R, R), np.float32)
        if kind == "full":
            A = np.ones((R, R), np.float32) - np.eye(R, dtype=np.float32)
        elif kind == "corr":
            C = train_df.pivot_table(index="epiweek", columns="hhs_region",
                                     values="ili_asof_lag1").corr()
            for r in regions:
                for s in (C[r].drop(index=r).sort_values(ascending=False)
                          .index[:NEIGHBOURS_K]):
                    A[ridx[r], ridx[s]] = 1
        elif kind == "random":
            for r in regions:
                for s in rng.choice([x for x in regions if x != r],
                                    NEIGHBOURS_K, replace=False):
                    A[ridx[r], ridx[s]] = 1
        deg = A.sum(1, keepdims=True)
        deg[deg == 0] = 1
        return torch.tensor(A / deg)

    class Net(nn.Module):
        """GRU over the lag window, one graph-conv layer, multi-horizon head.

        The head predicts the RESIDUAL from persistence, so an all-zero
        output already reproduces persistence.
        """

        def __init__(self, hid=48):
            super().__init__()
            self.gru = nn.GRU(1, hid, batch_first=True)
            self.emb = nn.Embedding(R, 8)
            self.wk = nn.Linear(2, 8)
            self.mix = nn.Linear(hid + 16, hid)
            self.self_ = nn.Linear(hid, hid)
            self.msg = nn.Linear(hid, hid)
            self.head = nn.Sequential(nn.ReLU(), nn.Linear(hid, hid),
                                      nn.ReLU(), nn.Linear(hid, len(H)))

        def forward(self, X, WK, RI, A):
            W, Rn, L = X.shape
            _, hN = self.gru(X.reshape(W * Rn, L, 1))
            h = hN[-1].reshape(W, Rn, -1)
            h = torch.relu(self.mix(torch.cat([h, self.emb(RI), self.wk(WK)], -1)))
            h = self.self_(h) + self.msg(torch.einsum("rs,wsh->wrh", A, h))
            return self.head(h)

    def fit_eval(kind, seed, train, val, test):
        rng = np.random.default_rng(500 + seed)
        torch.manual_seed(seed)
        A = adjacency(kind, train, rng)
        Xtr, Ytr, Btr, Wtr, Mtr, Rtr = pack(train)
        Xv, Yv, Bv, Wv, Mv, Rv = pack(val)
        Xte, Yte, Bte, Wte, Mte, Rte = pack(test)
        mu = Xtr[Mtr.bool()].mean()
        sd = Xtr[Mtr.bool()].std() + 1e-6

        def nz(x):
            return (x - mu) / sd

        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        best, bstate, patience = 1e9, None, 0
        for _ in range(400):
            net.train()
            opt.zero_grad()
            p = net(nz(Xtr), Wtr, Rtr, A)
            loss = (((p + Btr - Ytr).abs() * Mtr.unsqueeze(-1)).sum()
                    / (Mtr.sum() * len(H)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            net.eval()
            with torch.no_grad():
                pv = net(nz(Xv), Wv, Rv, A)
                v = (((pv + Bv - Yv).abs() * Mv.unsqueeze(-1)).sum()
                     / (Mv.sum() * len(H))).item()
            if v < best - 1e-4:
                best, patience = v, 0
                bstate = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                patience += 1
                if patience >= 60:
                    break
        net.load_state_dict(bstate)
        net.eval()
        with torch.no_grad():
            pt = net(nz(Xte), Wte, Rte, A) + Bte
        m = Mte.bool()
        return [float((pt[..., j] - Yte[..., j]).abs()[m].mean())
                for j in range(len(H))]

    rows = []
    for fold, train_seasons, test_season in FOLDS:
        train_all = P[P["season"].isin(train_seasons)]
        test = P[P["season"] == test_season]
        if train_all.empty or test.empty:
            continue
        # early stopping on the most recent TRAINING season, never on test
        val_season = train_seasons[-1]
        train = (train_all[train_all["season"] != val_season]
                 if len(train_seasons) > 1 else train_all)
        val = train_all[train_all["season"] == val_season]
        if train.empty:
            train, val = train_all, train_all

        for kind in ["none", "corr", "full", "random"]:
            for seed in range(n_seeds):
                e = fit_eval(kind, seed, train, val, test)
                for j, h in enumerate(H):
                    rows.append({"fold": fold, "test_season": test_season,
                                 "arm": kind, "seed": seed, "horizon": h,
                                 "mae": round(e[j], 4)})

        # references on identical rows
        for h in H:
            dd = build_targets(core, h)
            a = usable(dd[dd["season"].isin(train_seasons)])
            b = usable(dd[dd["season"] == test_season])
            rows.append({"fold": fold, "test_season": test_season,
                         "arm": "B0 persistence", "seed": 0, "horizon": h,
                         "mae": round(float(np.abs(
                             b["y_target"].values - b["ili_asof_lag1"].values
                         ).mean()), 4)})
            m = HistGradientBoostingRegressor(random_state=0, **B2_PARAMS)
            m.fit(a[B2_FEATURES].values, a["y_target"].values)
            rows.append({"fold": fold, "test_season": test_season,
                         "arm": "B2 boosting", "seed": 0, "horizon": h,
                         "mae": round(float(np.abs(
                             b["y_target"].values - m.predict(b[B2_FEATURES].values)
                         ).mean()), 4)})
        print(f"  {fold} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "graph_gnn_ablation.csv", index=False)
    pre = pre_covid_folds()

    print("\n  MAE (pre-COVID folds, mean over seeds)")
    print(df[df.fold.isin(pre)].pivot_table(index="arm", columns="horizon",
                                            values="mae").round(4).to_string())

    def skill(arm, ref):
        line = {}
        B = df[df.arm == ref].groupby(["horizon", "fold"])["mae"].mean()
        for h in H:
            vals = []
            for f in pre:
                s = df[(df.arm == arm) & (df.horizon == h) & (df.fold == f)]["mae"]
                if len(s) and (h, f) in B.index:
                    vals.append(100 * (B[(h, f)] - s.mean()) / B[(h, f)])
            line[f"h{h}"] = sign_verdict(vals)
        return line

    print("\n  Sign-consistent skill")
    print(pd.DataFrame([
        {"comparison": "GNN no-graph vs B0 persistence  [does it learn?]",
         **skill("none", "B0 persistence")},
        {"comparison": "GNN no-graph vs B2 boosting     [limitation]",
         **skill("none", "B2 boosting")},
        {"comparison": "GNN correlation vs no-graph",
         **skill("corr", "none")},
        {"comparison": "GNN fully-connected vs no-graph",
         **skill("full", "none")},
        {"comparison": "GNN random vs no-graph",
         **skill("random", "none")},
        {"comparison": "GNN correlation vs RANDOM  [G5 control]",
         **skill("corr", "random")},
    ]).to_string(index=False))

    print("\n  READING: the model learns - it beats persistence substantially")
    print("  at h>=2. With that working instrument, no relation beats")
    print("  no-graph and correlation does not beat a density-matched random")
    print("  graph. It does not beat gradient boosting at h=1; that is a")
    print("  stated limitation of 10 nodes and ~1,650 training rows.")
    return df


# ──────────────────────────────────────────────────────────────────────
# figures
# ──────────────────────────────────────────────────────────────────────

def fig_stability(stab, within, chance):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    m = stab[stab.kind == "matrix_persistence"]["value"].values
    axes[0].axhline(within, color=plotstyle.PALETTE[2], linewidth=2,
                    label=f"within a season (mean |r| = {within:.2f})")
    axes[0].bar(range(len(m)), m, color=plotstyle.PALETTE[0], width=0.72,
                label="between two seasons")
    axes[0].axhline(0, color=plotstyle.RULE, linewidth=0.8)
    axes[0].set_xticks([])
    axes[0].set_xlabel("season pair")
    axes[0].set_ylabel("correlation")
    axes[0].set_title("Regions correlate strongly, their correlation\n"
                      "STRUCTURE does not persist")
    axes[0].legend(fontsize=8)
    plotstyle.finish(axes[0])

    t = stab[stab.kind == "train_to_test_overlap"]
    axes[1].bar(t["a"], 100 * t["value"], color=plotstyle.PALETTE[0], width=0.6)
    axes[1].axhline(100 * chance, color=plotstyle.PALETTE[1], linestyle="--",
                    linewidth=1.8, label=f"chance ({100 * chance:.1f}%)")
    axes[1].set_ylabel("% of chosen neighbours that are\n"
                       "the test season's true top-3")
    axes[1].set_title("Neighbours chosen on history are\nno better than chance")
    axes[1].legend(fontsize=8)
    plotstyle.finish(axes[1])

    fig.tight_layout()
    return plotstyle.save(fig, "fig_graph_stability")


def fig_feature_screen(fs):
    import matplotlib.pyplot as plt
    pre = pre_covid_folds()
    g = (fs[fs.fold.isin(pre)].pivot_table(index="horizon", columns="arm",
                                           values="mae"))
    base = [c for c in g.columns if c.startswith("B2 own")][0]
    fig, ax = plt.subplots(figsize=(8.6, 5))
    for i, c in enumerate([c for c in g.columns if c != base]):
        ax.plot(g.index, 100 * (g[base] - g[c]) / g[base], marker="o",
                color=plotstyle.PALETTE[i % len(plotstyle.PALETTE)], label=c)
    plotstyle.zero_line(ax, "own history only", x=list(g.index)[0])
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("MAE skill vs own history only (%)")
    ax.set_title("Neighbour information makes incidence forecasts worse")
    ax.set_xticks(list(g.index))
    ax.legend(fontsize=8)
    plotstyle.finish(ax)
    fig.tight_layout()
    return plotstyle.save(fig, "fig_graph_feature_screen")


def fig_gnn(gnn):
    import matplotlib.pyplot as plt
    pre = pre_covid_folds()
    g = (gnn[gnn.fold.isin(pre)].pivot_table(index="horizon", columns="arm",
                                             values="mae"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    order = [("B0 persistence", plotstyle.GREY, ":"),
             ("B2 boosting", plotstyle.MODEL_COLOURS["B2"], "-"),
             ("none", plotstyle.PALETTE[0], "-"),
             ("corr", plotstyle.PALETTE[1], "--"),
             ("random", plotstyle.PALETTE[2], "--")]
    pretty = {"none": "GNN, no graph", "corr": "GNN, correlation graph",
              "random": "GNN, random graph"}
    for name, colour, ls in order:
        if name not in g.columns:
            continue
        axes[0].plot(g.index, g[name], marker="o", color=colour, linestyle=ls,
                     label=pretty.get(name, name))
    axes[0].set_xlabel("Forecast horizon (weeks)")
    axes[0].set_ylabel("MAE (wILI percentage points)")
    axes[0].set_title("The GNN learns: it beats persistence by a wide margin")
    axes[0].set_xticks(list(g.index))
    axes[0].legend(fontsize=8)
    plotstyle.finish(axes[0])

    if "none" in g.columns:
        for i, arm in enumerate(["corr", "full", "random"]):
            if arm not in g.columns:
                continue
            axes[1].plot(g.index, 100 * (g["none"] - g[arm]) / g["none"],
                         marker="o",
                         color=plotstyle.PALETTE[i % len(plotstyle.PALETTE)],
                         label=pretty.get(arm, f"GNN, {arm}"))
    plotstyle.zero_line(axes[1], "no graph", x=list(g.index)[0])
    axes[1].set_xlabel("Forecast horizon (weeks)")
    axes[1].set_ylabel("MAE skill vs no graph (%)")
    axes[1].set_title("No relation beats no graph, including random")
    axes[1].set_xticks(list(g.index))
    axes[1].legend(fontsize=8)
    plotstyle.finish(axes[1])

    fig.tight_layout()
    return plotstyle.save(fig, "fig_graph_gnn_ablation")


# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-gnn", action="store_true",
                    help="also train the GNN (needs PyTorch, ~10 min)")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plotstyle.apply()

    print("EpiSurge E6 - is an epidemic graph better than chance?")
    print("=" * 76)
    print("Master Plan 4.3's claim C3, with the mandated density-matched")
    print("random-graph control (G5).")

    core = load()
    regions = sorted(core["hhs_region"].unique())
    print(f"\nCore-season rows {len(core)}, regions {len(regions)}")

    stab, within, mat_r, hit, chance = layer_a(core, regions)
    fs = layer_b(core, regions)
    gnn = layer_c(core, regions, args.seeds) if args.with_gnn else None

    written = [fig_stability(stab, within, chance), fig_feature_screen(fs)]
    if gnn is not None:
        written.append(fig_gnn(gnn))
    for w in written:
        print(f"\n  wrote {w}")

    print("\n" + "=" * 76)
    print("CONCLUSION")
    print("=" * 76)
    print(f"  Within-season region correlation      mean |r| = {within:.3f}")
    print(f"  Cross-season persistence of structure mean r  = {mat_r:+.3f}")
    print(f"  Train-chosen neighbours that are right  {100 * hit:.1f}% "
          f"(chance {100 * chance:.1f}%)")
    print("\n  A correlation graph fitted on past seasons carries no more")
    print("  information about the next season than a random graph of the")
    print("  same density. The feature screen and, where run, the trained")
    print("  GNN both behave exactly as that predicts.")
    if gnn is None:
        print("\n  Layer C not run. Add --with-gnn for the trained-model"
              " confirmation.")


if __name__ == "__main__":
    main()
