# -*- coding: utf-8 -*-
"""make_figures.py — LAC-MRI 论文图件生成（读取 results_lac_mri/*.json + 重跑一次 E1 收集亚组覆盖）
用法: python make_figures.py            # 全部三张图
      python make_figures.py --no-rerun  # 不重跑 E1（跳过 Figure A 的亚组数据，用已存 json）
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lac_mri as L

def _get(d, *keys, default=np.nan):
    for k in keys:
        if k in d: return d[k]
    return default

NOMINAL = 0.90
OUT = Path("figures"); OUT.mkdir(exist_ok=True)
METHODS = list(L.METHODS.keys())   # 与 lac_mri 的方法注册表保持同步
COLORS = {"Global-CP": "#d62728", "Local-CP": "#ff9896", "CQR": "#1f77b4", "LAC": "#2ca02c",
          "Mondrian-CP": "#9467bd", "LAC-GBM": "#8c564b", "NCP (no Lepski)": "#e377c2",
          "Local-CP (tuned)": "#7f7f7f"}


def load_json(name):
    p = Path("results_lac_mri") / name
    return json.load(open(p)) if p.exists() else None


# ---------- Figure A: E1 亚组(H&Y)覆盖，LAC vs Global-CP ----------
def fig_subgroup(args):
    cfg = L.CFG_INST
    meta = L.load_ppmi_meta(cfg.meta_ppmi_subjects, cfg.meta_ppmi_ida, cfg.meta_ppmi_updrs)
    X, y, info = L.build_dataset(meta, "features_cache/PPMI_aal_features.csv", "updrs",
                                 cfg, group_filter=1)
    hy = pd.to_numeric(info["hy"], errors="coerce")
    hyq = pd.qcut(hy.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    subj_cov = {m: {q: [] for q in ["Q1", "Q2", "Q3", "Q4"]} for m in ["Global-CP", "LAC"]}
    from sklearn.model_selection import KFold
    for seed in [42, 43, 44]:
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
            for m in ["Global-CP", "LAC"]:
                lo, up = L.METHODS[m](X[tr], y[tr], X[te])
                hq = hyq.iloc[te].values
                for q in ["Q1", "Q2", "Q3", "Q4"]:
                    mq = hq == q
                    if mq.sum() >= 5:
                        subj_cov[m][q].append(float(((y[te][mq] >= lo[mq]) & (y[te][mq] <= up[mq])).mean()))
    means = {m: [np.mean(subj_cov[m][q]) for q in ["Q1", "Q2", "Q3", "Q4"]] for m in subj_cov}
    ses = {m: [np.std(subj_cov[m][q]) / np.sqrt(len(subj_cov[m][q])) for q in ["Q1", "Q2", "Q3", "Q4"]]
           for m in subj_cov}
    x = np.arange(4); w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(["Global-CP", "LAC"]):
        ax.bar(x + (i - 0.5) * w, means[m], w, yerr=ses[m], capsize=3,
               color=COLORS[m], label=m, alpha=0.9)
    ax.axhline(NOMINAL, ls="--", c="k", lw=1)
    ax.text(3.45, NOMINAL + 0.005, "nominal 0.90", fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(["Q1 (mild)", "Q2", "Q3", "Q4 (severe)"])
    ax.set_ylabel("Empirical coverage"); ax.set_ylim(0.75, 1.0)
    ax.set_title("Subgroup coverage by H&Y severity (E1, PPMI n=575)")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "figA_subgroup_coverage.png", dpi=200)
    print("[figA saved]", OUT / "figA_subgroup_coverage.png")


# ---------- Figure B: 跨实验覆盖退化 ----------
def fig_shift():
    e1 = load_json("e1_updrs.json")
    e3 = load_json("e3_GE.json")
    e4 = load_json("e4_neurocon.json")
    e5 = load_json("e5_shift.json")
    exps, cov = [], {m: [] for m in METHODS}
    def add(tag, src):
        if not src: return
        exps.append(tag)
        for m in METHODS:
            cov[m].append(_get(src[m], "coverage", "coverage_mean") if m in src else np.nan)
    add("E1 source\n(PPMI)", e1)
    add("E3 GE\nheld-out", e3)
    add("E4 NEUROCON\n(UPDRS)", e4)
    if e5:
        for m in METHODS:
            cov[m].append(e5["NEUROCON"][m]["coverage"])
        exps.append("E5 NEUROCON\n(age, vol)")
    x = np.arange(len(exps)); w = 0.2
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, m in enumerate(METHODS):
        ax.bar(x + (i - 1.5) * w, cov[m], w, color=COLORS[m], label=m)
    ax.axhline(NOMINAL, ls="--", c="k", lw=1)
    ax.text(len(exps) - 0.5, NOMINAL + 0.01, "nominal 0.90", fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(exps, fontsize=8)
    ax.set_ylabel("Empirical coverage"); ax.set_ylim(0, 1.05)
    ax.set_title("Coverage degradation under distribution shift")
    ax.legend(ncol=4, fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "figB_shift.png", dpi=200)
    print("[figB saved]", OUT / "figB_shift.png")


# ---------- Figure C: 覆盖-宽度权衡 ----------
def fig_pareto():
    e1 = load_json("e1_updrs.json"); e3 = load_json("e3_GE.json"); e4 = load_json("e4_neurocon.json")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m in METHODS:
        pts = []
        for src in [e1, e3, e4]:
            if src and m in src:
                pts.append((_get(src[m], "width", "width_mean"),
                            _get(src[m], "coverage", "coverage_mean")))
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "o-", color=COLORS[m], label=m, alpha=0.85)
            for (xx, yy) in pts:
                ax.annotate(f"{yy:.2f}", (xx, yy), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color=COLORS[m])
    ax.axhline(NOMINAL, ls="--", c="k", lw=1)
    ax.set_xlabel("Mean interval width"); ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage-width trade-off (E1 / E3 / E4)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "figC_pareto.png", dpi=200)
    print("[figC saved]", OUT / "figC_pareto.png")



def fig_fewshot():
    """Fig 6: few-shot target calibration 恢复曲线（需先跑 e7）。"""
    import numpy as np
    p = Path("results_lac_mri") / "e7_fewshot.json"
    if not p.exists():
        print("[fig6 skip] 先运行: python lac_mri.py e7")
        return
    curve = json.load(open(p))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m in ["Global-CP", "LAC"]:
        if m not in curve: continue
        ks = [d["k"] for d in curve[m]]
        cov = [d["coverage"] for d in curve[m]]
        se = [d.get("coverage_se", 0) for d in curve[m]]
        ax.errorbar(ks, cov, yerr=se, fmt="o-", color=COLORS[m], label=m, capsize=3)
    ax.axhline(0.90, ls="--", c="k", lw=1)
    ax.text(19, 0.905, "nominal 0.90", fontsize=8, ha="right")
    ax.set_xlabel("Target-site calibration subjects (k)")
    ax.set_ylabel("Empirical coverage (NEUROCON PD)")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Few-shot target calibration (E7, PPMI -> NEUROCON)")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "fig6_fewshot.png", dpi=200)
    print("[fig6 saved]", OUT / "fig6_fewshot.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerun", action="store_true")
    a = ap.parse_args()
    if not a.no_rerun:
        fig_subgroup(a)
    fig_shift()
    fig_pareto()
    fig_fewshot()

if __name__ == "__main__":
    main()