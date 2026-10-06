# -*- coding: utf-8 -*-
"""
LAC-MRI (revised per review, v3): Locally Adaptive Conformal Prediction
=======================================================================
管线结构（对应论文实验设计 E0-E10）：
  extract    : 从 T1 NIfTI + AAL atlas 提取 ROI 特征（均值/标准差/soft GM-mass proxy），缓存 CSV
  e0         : 合成机制验证（默认 shift=0，5 seeds 平均；已知真值 sigma；含 Oracle-NCP 上界）-> e0_synthetic.json
  e0sweep    : 合成扫描：异方差强度 x 偏移强度 x 残差尾部 -> e0_sweep.json
  e1         : PPMI 内部 5-fold CV 主实验（配对差异 CI + 非劣效检验）-> e1_updrs.json
  e2         : 尺度估计器对比（global/kNN/GBM/fixed-h/Lepski）-> e2_scale_comp.json
  pointpred  : 点预测基线性能（Mean / Demo / MRI-only / MRI+Demo / GBM）-> pointpred.json
  e3         : PPMI leave-vendor-out 外部验证（论文用 --sites GE）
  e4         : PPMI -> NEUROCON 独立数据集验证（UPDRS-OFF，探索性）
  e5/e6      : 跨人群年龄迁移 + MMD 特征级偏移
  e7         : few-shot 目标域校准（--protocol pooled|target；target 小样本自动退回全局尺度）
  e8grid     : ESS grid vs absolute-h grid 消融
  e9         : 多 alpha 校准曲线
  e10        : difficulty-error 分析 + 选择器诊断（带宽/ESS 分布、弃权比例）
  selftest   : 合成数据自检

v2 修复：
  - lepski_var: 局部样本不足时退回全局尺度（修 E7 target 协议 width=nan 的 bug）
  - e0: 默认 shift=0（机制验证配置），5 seeds 平均，避免单 seed 巧合
v3 新增（回应二轮评审）：
  - RLCP (Hore & Barber 2023) 基线（带宽随机化的局部保形）
  - e8demo: 距离压缩受控演示（fixed-mult 倍 median 带宽）-> e8_compressed_demo.json
  - e11: 真实 PPMI 残差注入合成异方差剂量-反应 -> e11_injection.json
  - e12: 信息高效版管线注入剂量-反应：LAC(72点,raw) / LAC-CF(230点,log) / LAC-ML(度量学习)
    -> e12_injection_cf.json
用法示例：
  python lac_mri_revised.py e0
  python lac_mri_revised.py e0sweep --ratios 1,2,4,8 --shifts 0,1.5 --tails normal,t3
  python lac_mri_revised.py e2 --features features_cache/PPMI_aal_features.csv
  python lac_mri_revised.py e1 --target updrs --folds 5 --seeds 3
  python lac_mri_revised.py e7 --site GE --protocol target --ks 10,15,20,30
依赖：numpy pandas scipy scikit-learn nibabel(仅 extract 需要)
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# 配置
# =============================================================================
@dataclass
class CFG:
    # ---- 路径（默认 M 盘，可用 --mri-root 覆盖）----
    mri_root: str = "M:/MRI"
    atlas_nii: str = "aal_extracted/aal/atlas/AAL.nii"
    meta_neurocon: str = "neurocon_patients.tsv"
    meta_taowu: str = "taowu_patients.tsv"
    meta_ppmi_subjects: str = "ppmi_subjects.csv"
    meta_ppmi_ida: str = "idaSearch_9_21_2026.xlsx"
    meta_ppmi_updrs: str = "PPMI-UPDRS.csv"
    feature_cache: str = "features_cache"

    # ---- 影像文件匹配 ----
    img_pattern: str = r"(.+?)\.(nii|nii\.gz)$"
    subject_regex: str = r"(\d+)"

    # ---- 方法超参数 ----
    alpha: float = 0.10          # 名义覆盖 0.90
    kappa: float = 1.0           # Lepski 临界值系数
    h_min: float = 0.2
    rho: float = 1.5
    n_grid: int = 10
    n_min_eff: int = 12          # 有效局部样本下限
    krr_lambda: float = 1e-3
    local_cp_h: float = 1.5
    grid_mode: str = "ess"       # "ess" / "absolute"
    random_state: int = 42

    # ---- E2 kNN 尺度 ----
    knn_k: int = 25

    # ---- 统计协议 ----
    ni_margin: float = 0.03      # 非劣效界值（覆盖差异，论文 Section 5）

    # ---- 协变量 ----
    use_age_sex_covariates: bool = True
    feat_suffix: str = ""

CFG_INST = CFG()

# =============================================================================
# 元数据加载
# =============================================================================
def _to_binary_sex(s):
    m = {"M": 0, "F": 1, "m": 0, "f": 1, 0: 0, 1: 1, "0": 0, "1": 1}
    return s.map(lambda x: m.get(x, np.nan))

def _resolve_meta(path, cfg, *rel_candidates):
    p = Path(path)
    if p.exists():
        return str(p)
    for c in rel_candidates:
        q = Path(cfg.mri_root) / c
        if q.exists():
            return str(q)
    hits = list(Path(cfg.mri_root).rglob(Path(path).name)) if Path(cfg.mri_root).exists() else []
    if hits:
        return str(hits[0])
    raise FileNotFoundError(f"找不到 {path}（已搜索工作目录与 {cfg.mri_root}），请用 --meta-xxx 指定完整路径")

def load_neurocon_meta(path):
    df = pd.read_csv(path, sep=None, engine="python")
    out = pd.DataFrame({
        "subj": df["code"].astype(str).str.extract(r"(\d+)")[0],
        "dataset": "NEUROCON",
        "group": df["status"].map({"PD": 1, "NORMAL": 0}),
        "age": pd.to_numeric(df["age"], errors="coerce"),
        "sex": _to_binary_sex(df["sex"]),
        "updrs": pd.to_numeric(df["UPDRS_motor_OFF"], errors="coerce"),
        "updrs_on": pd.to_numeric(df["UPDRS_motor_ON"], errors="coerce"),
        "hy": pd.to_numeric(df["H&Y"], errors="coerce"),
        "site": "NEUROCON-1",
        "scanner": "NEUROCON-1",
    })
    return out

def load_taowu_meta(path):
    df = pd.read_csv(path, sep=None, engine="python")
    group = df["name"].astype(str).str.startswith("patient").map({True: 1, False: 0})
    out = pd.DataFrame({
        "subj": df["name"].astype(str).str.extract(r"(\d+)")[0],
        "dataset": "TAOWU",
        "group": group,
        "age": pd.to_numeric(df["age"], errors="coerce"),
        "sex": _to_binary_sex(df["sex"]),
        "updrs": np.nan,
        "mmse": pd.to_numeric(df["mmse"], errors="coerce"),
        "duration": pd.to_numeric(df["disease_duration"], errors="coerce"),
        "hy": pd.to_numeric(df["H_Y"], errors="coerce"),
        "site": "TAOWU-1",
        "scanner": "TAOWU-1",
    })
    return out

def _norm_mfr(s):
    s = str(s).upper()
    if "SIEMENS" in s: return "SIEMENS"
    if "GE" in s: return "GE"
    if "PHILIPS" in s: return "PHILIPS"
    if "TOSHIBA" in s: return "TOSHIBA"
    return np.nan

def load_ppmi_meta(path_subjects, path_ida, path_updrs):
    a = pd.read_csv(path_subjects)
    a = a.rename(columns={"PATNO": "subj", "Group": "group_raw", "Manufacturer": "scanner"})
    a["subj"] = a["subj"].astype(str).str.extract(r"(\d+)")[0]
    a["group"] = a["group_raw"].map({"PD": 1, "Control": 0})
    a["scanner"] = a["scanner"].map(_norm_mfr)
    a = a.drop(columns=["group_raw"])

    b = pd.read_excel(path_ida, dtype=str)
    b = b[b["Subject ID"].astype(str).str.fullmatch(r"\d+").fillna(False)]
    b["subj"] = b["Subject ID"].astype(str)
    b["age"] = pd.to_numeric(b["Age"], errors="coerce")
    b["sex"] = b["Sex"].map({"M": 0, "F": 1})
    b = (b.dropna(subset=["age"]).sort_values("age")
           .drop_duplicates("subj", keep="first")[["subj", "age", "sex"]])

    with open(path_updrs, "rb") as _f:
        _head = _f.read(8)
    if _head[:4] == b"PK\x03\x04" or _head[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        try:
            c = pd.read_excel(path_updrs)
        except ImportError:
            raise SystemExit("[error] UPDRS 是 Excel 文件且缺少 xlrd/openpyxl，请另存为 CSV 后重试")
    else:
        c = pd.read_csv(path_updrs, sep=None, engine="python")
    c["subj"] = c["PATNO"].astype(str)
    c = c[c["EVENT_ID"] == "BL"].copy()
    c["updrs"] = pd.to_numeric(c["NP3TOT"], errors="coerce")
    c["hy"] = pd.to_numeric(c["NHY"], errors="coerce")
    c["pdstate"] = c["PDSTATE"].fillna("")
    c = c.dropna(subset=["updrs"])
    c = c.sort_values("pdstate", key=lambda s: s != "OFF")
    c = c.drop_duplicates("subj", keep="first")[["subj", "updrs", "hy"]]

    out = a.merge(b, on="subj", how="left").merge(c, on="subj", how="left")
    out["dataset"] = "PPMI"
    out["site"] = out["scanner"]
    out = out.dropna(subset=["group"])
    return out[["subj", "dataset", "group", "age", "sex", "updrs", "hy", "site", "scanner"]]

# =============================================================================
# AAL 特征提取（需要 nibabel）
# =============================================================================
def extract_aal_features(img_path, atlas_img, roi_ids, cache=None):
    import nibabel as nib
    from nibabel.processing import resample_from_to
    img = nib.load(str(img_path))
    same = (img.shape == atlas_img.shape) and np.allclose(img.affine, atlas_img.affine)
    if same:
        a_img = atlas_img
    else:
        if cache is None:
            cache = {}
        key = (tuple(img.shape), tuple(np.round(img.affine, 3).ravel()))
        if key not in cache:
            cache[key] = resample_from_to(atlas_img, img, order=0)
        a_img = cache[key]
    data = np.asarray(img.dataobj).astype(np.float64)
    atlas_labels = np.asarray(a_img.dataobj).astype(np.int32)
    brain = atlas_labels > 0
    mu, sd = data[brain].mean(), data[brain].std() + 1e-8
    data = (data - mu) / sd
    vox_vol = float(np.abs(np.linalg.det(img.affine[:3, :3])))
    if roi_ids is None:
        roi_ids = np.unique(atlas_labels[brain])
    feats, names = [], []
    for r in roi_ids:
        m = atlas_labels == r
        v = data[m]
        if v.size == 0:
            feats += [np.nan, np.nan, 0.0]
        else:
            # soft GM-mass proxy = ROI 内正归一化强度积分 x 体素物理体积
            # （注意：atlas 体素计数跨被试为常数，不能作体积特征；
            #   该量为归一化强度代理，非分割灰质体积 —— 论文 Section 3.5 已明确说明）
            feats += [float(v.mean()), float(v.std()), float(np.clip(v, 0, None).sum()) * vox_vol]
        names += [f"ROI{r}_mean", f"ROI{r}_std", f"ROI{r}_gmv"]
    return np.array(feats), names

def find_subject_images(root, cfg=CFG_INST):
    root = Path(root)
    pat = re.compile(cfg.img_pattern)
    sre = re.compile(cfg.subject_regex)
    out = {}
    for p in root.rglob("*"):
        if p.is_file() and pat.search(p.name):
            m = sre.search(p.stem)
            if m:
                sid = m.group(1)
                if sid not in out:
                    out[sid] = p
    return out

def cmd_extract(args, cfg=CFG_INST):
    import nibabel as nib
    root = Path(cfg.mri_root)
    atlas_img = nib.load(str(root / cfg.atlas_nii))
    img_root = root / args.dataset
    imgs = find_subject_images(img_root, cfg)
    print(f"[extract] {args.dataset}: found {len(imgs)} images under {img_root}")
    cache = {}
    cache_dir = Path(cfg.feature_cache); cache_dir.mkdir(exist_ok=True)
    out_csv = cache_dir / f"{args.dataset}_aal_features.csv"
    rows, names = [], None
    ok = 0
    for i, (sid, p) in enumerate(sorted(imgs.items())):
        try:
            f, names = extract_aal_features(p, atlas_img, None, cache)
            rows.append([sid] + f.tolist()); ok += 1
        except Exception as e:
            print(f"  [skip] {sid}: {e}")
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(imgs)} (ok={ok})")
    if not rows:
        print("[error] 0 subjects extracted —— 检查影像空间/模板路径/文件完整性")
        sys.exit(1)
    df = pd.DataFrame(rows, columns=["subj"] + names)
    df.to_csv(out_csv, index=False)
    print(f"[done] {ok}/{len(imgs)} subjects -> {out_csv}")

# =============================================================================
# 核与 KRR（RKHS 距离：dist^2 = 2(1-K)）
# =============================================================================
def rbf_kernel(X, Z=None, sigma=None):
    X = np.asarray(X, float)
    Z = X if Z is None else np.asarray(Z, float)
    d2 = cdist(X, Z, "sqeuclidean")
    if sigma is None:
        med = np.median(np.sqrt(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
        sigma = med if med > 0 else 1.0
    return np.exp(-d2 / (2 * sigma ** 2)), sigma

def rkhs_dist_from_kernel(K):
    return np.sqrt(np.clip(2.0 * (1.0 - K), 0.0, None))

def krr_fit(K, y, lam):
    n = K.shape[0]
    return np.linalg.solve(K + lam * n * np.eye(n), y)

def krr_predict(K_test_train, alpha):
    return K_test_train @ alpha

def _krr_pred(Xtr, ytr, Xq, cfg=CFG_INST):
    K, sig = rbf_kernel(Xtr)
    a = krr_fit(K, ytr, cfg.krr_lambda)
    Kq, _ = rbf_kernel(Xq, Xtr, sigma=sig)
    return krr_predict(Kq, a)

# =============================================================================
# 局部加权分位数 + Lepski 自适应
# =============================================================================
def conformal_level(m, alpha):
    """split conformal 水平：ceil((m+1)(1-alpha))/m，截断到 1。
    注意：有限样本保证要求 m >= (1-alpha)/alpha（alpha=0.10 时 m>=9）。"""
    if m < 1: return 1 - alpha
    return min(1.0, (int(np.ceil((m + 1) * (1 - alpha)))) / m)

def _neff(w):
    return float(w.sum() ** 2 / (np.sum(w ** 2) + 1e-12))

def weighted_quantile(scores, w, q):
    idx = np.argsort(scores)
    s, w = scores[idx], w[idx]
    cw = np.cumsum(w)
    tot = cw[-1]
    k = np.searchsorted(cw, q * tot, side="left")
    k = min(k, len(s) - 1)
    return s[k]

def make_grid(cfg=CFG_INST):
    return np.array([cfg.h_min * cfg.rho ** k for k in range(cfg.n_grid)])

def make_grid_neff(dist_row, n_min_eff, n_cal, gamma=1.6, n_max=12):
    targets = []
    t = float(n_min_eff)
    while t < n_cal and len(targets) < n_max:
        targets.append(t); t *= gamma
    if not targets or targets[-1] < n_cal:
        targets.append(float(n_cal))
    d = np.asarray(dist_row, float)
    lo, hi = max(d.min(), 1e-3), d.max() * 1.5 + 1e-3
    def neff(h):
        return _neff(np.exp(-d ** 2 / (2.0 * h ** 2)))
    grid = []
    for tg in targets:
        l, r = lo, hi
        for _ in range(40):
            mid = 0.5 * (l + r)
            if neff(mid) >= tg: r = mid
            else: l = mid
        grid.append(0.5 * (l + r))
    return np.array(sorted(set(round(g, 6) for g in grid)))

def local_weights(dist_row, h):
    return np.exp(-dist_row ** 2 / (2.0 * h ** 2))

def _weighted_density_at(scores, w, q):
    sw = w.sum()
    if sw <= 1:
        return np.nan
    mu = np.average(scores, weights=w)
    var = np.average((scores - mu) ** 2, weights=w)
    b = 1.06 * np.sqrt(var + 1e-12) * sw ** (-1 / 5)
    z = (q - scores) / b
    return float(np.exp(-0.5 * z ** 2).dot(w) / (sw * b * np.sqrt(2 * np.pi)))

def lepski_var(dist_row, r2, grid, kappa, n_min_eff):
    """尺度模式：局部统计量 = 加权平方残差均值。
    稳定性检验用数据驱动 SE（经验四阶矩），无高斯假设。
    局部样本不足（如 E7 target 协议的小校准集）时退回全局尺度而非 NaN。"""
    vs, sgs = [], []
    for g in grid:
        w = local_weights(dist_row, g); sw = w.sum()
        if sw < n_min_eff: vs.append(np.nan); sgs.append(np.inf); continue
        v = np.average(r2, weights=w)
        se = np.sqrt(max(np.sum(w ** 2 * (r2 - v) ** 2) / (w.sum() ** 2), 1e-16))
        vs.append(v); sgs.append(kappa * se)
    vs = np.array(vs); sgs = np.array(sgs)
    valid = np.isfinite(vs); bk = -1
    for k in np.where(valid)[0]:
        if all(abs(vs[k] - vs[j]) <= sgs[k] + sgs[j]
               for j in np.where(valid & (np.arange(len(grid)) <= k))[0]):
            bk = k
    if bk >= 0 and np.isfinite(vs[bk]):
        return grid[bk], vs[bk]
    # fallback：无有效局部带宽时退回全局残差方差（正确默认，保证有限覆盖）
    return grid[len(grid) // 2], float(np.mean(r2))

# =============================================================================
# 尺度估计器库（E2 对比：区分"异质性不存在"与"估计器找不到"）
# =============================================================================
def _knn_sigma(Xs, r2, Xq, k=None, cfg=CFG_INST):
    """kNN 固定有效样本数尺度：在 RKHS 距离下取最近的 k 个 scale 点。"""
    if k is None: k = cfg.knn_k
    Xs = np.asarray(Xs, float); Xq = np.asarray(Xq, float)
    Ks, sig = rbf_kernel(Xs)
    Kq, _ = rbf_kernel(Xq, Xs, sigma=sig)
    Dq = rkhs_dist_from_kernel(Kq)
    kk = min(k, len(r2))
    out = np.empty(len(Xq))
    for i in range(len(Xq)):
        idx = np.argpartition(Dq[i], kk)[:kk]
        out[i] = np.sqrt(max(np.mean(r2[idx]), 1e-8))
    return out

def _gbm_sigma(Xs, r2, Xq, seed=42):
    """GBM 残差尺度：在 scale 集上学 X -> E[r^2|X]，对测试点预测。"""
    g = GradientBoostingRegressor(random_state=seed).fit(np.asarray(Xs, float), np.asarray(r2, float))
    return np.sqrt(np.clip(g.predict(np.asarray(Xq, float)), 1e-8, None))

def _fixedh_sigma(Xs, r2, Xq, cfg=CFG_INST):
    """固定带宽核尺度：带宽 = D_scale 内 RKHS 距离中位数（无调参默认）。"""
    Xs = np.asarray(Xs, float); Xq = np.asarray(Xq, float)
    Ks, sig = rbf_kernel(Xs)
    Ds = rkhs_dist_from_kernel(Ks)
    h_fix = float(np.median(Ds[Ds > 0])) if np.any(Ds > 0) else 1.0
    Kq, _ = rbf_kernel(Xq, Xs, sigma=sig)
    Dq = rkhs_dist_from_kernel(Kq)
    out = np.empty(len(Xq))
    for i in range(len(Xq)):
        w = local_weights(Dq[i], h_fix)
        out[i] = np.sqrt(max(np.average(r2, weights=w), 1e-8))
    return out

def _global_sigma(Xs, r2, Xq):
    """全局常数尺度（异方差不存在时的正确模型）。"""
    return np.full(len(Xq), np.sqrt(max(np.mean(r2), 1e-8)))

SCALE_ESTIMATORS = {
    "GlobalScale-CP": _global_sigma,
    "KNN-CP (k=25)": _knn_sigma,
    "GBM-Scale-CP": _gbm_sigma,
    "NCP (fixed-h)": _fixedh_sigma,
}

# =============================================================================
# 方法封装：Global-CP / Local-CP / LAC / CQR / Mondrian
# =============================================================================
@dataclass
class CPResult:
    lower: np.ndarray
    upper: np.ndarray
    width: np.ndarray
    coverage: float
    subgroup_cov: dict = field(default_factory=dict)

def _scores(y, yhat):
    return np.abs(y - yhat)

def _split3(Xtr, ytr):
    """统一的三分割（fit / scale / cal）。所有归一化方法共用，保证数据预算公平。"""
    n = len(ytr)
    _p = np.random.RandomState(0).permutation(n); Xtr, ytr = Xtr[_p], ytr[_p]
    if n % 2 == 1: Xtr, ytr = Xtr[:-1], ytr[:-1]; n = len(ytr)
    n1 = n // 2; n2 = max(4, n // 4)
    return Xtr, ytr, n1, n2

def _ncp_generic(Xtr, ytr, Xte, scale_of, cfg=CFG_INST):
    """通用归一化 CP：scale_of(Xs, r2, Xq) 给出测试/校准点的局部尺度。
    Lepski 之外的任意尺度估计器走此通道，与 LAC 唯一的差别在 scale_of。"""
    Xtr, ytr, n1, n2 = _split3(Xtr, ytr)
    K, sig = rbf_kernel(Xtr)
    a = krr_fit(K[:n1, :n1], ytr[:n1], cfg.krr_lambda)
    def predict(Xq):
        Kq, _ = rbf_kernel(Xq, Xtr[:n1], sigma=sig)
        return krr_predict(Kq, a)
    Xs = Xtr[n1:n1 + n2]
    r2 = (ytr[n1:n1 + n2] - predict(Xs)) ** 2
    sig_cal = scale_of(Xs, r2, Xtr[n1 + n2:])
    A = np.abs(ytr[n1 + n2:] - predict(Xtr[n1 + n2:])) / (sig_cal + 1e-8)
    q = weighted_quantile(A, np.ones(len(A)), conformal_level(len(A), cfg.alpha))
    sig_te = scale_of(Xs, r2, Xte)
    yhat = predict(Xte)
    cfg._last_yhat = yhat; cfg._last_sigma = sig_te
    return yhat - q * sig_te, yhat + q * sig_te

def global_cp(Xtr, ytr, Xte, cfg=CFG_INST):
    n = len(ytr); h = n // 2
    _p = np.random.RandomState(0).permutation(n); Xtr, ytr = Xtr[_p], ytr[_p]
    if n % 2 == 1: Xtr, ytr = Xtr[:-1], ytr[:-1]; n = len(ytr)
    K, sig = rbf_kernel(Xtr)
    a = krr_fit(K[:h, :h], ytr[:h], cfg.krr_lambda)
    sc = _scores(ytr[h:], krr_predict(K[h:, :h], a))
    q = weighted_quantile(sc, np.ones(n - h), conformal_level(n - h, cfg.alpha))
    Kte, _ = rbf_kernel(Xte, Xtr[:h], sigma=sig)
    yhat = krr_predict(Kte, a)
    cfg._last_yhat = yhat
    return yhat - q, yhat + q

def local_cp(Xtr, ytr, Xte, h_fixed=None, cfg=CFG_INST):
    n = len(ytr); h = n // 2
    _p = np.random.RandomState(0).permutation(n); Xtr, ytr = Xtr[_p], ytr[_p]
    if n % 2 == 1: Xtr, ytr = Xtr[:-1], ytr[:-1]; n = len(ytr)
    n1 = n // 2; n2 = max(4, n // 4)
    K, sig = rbf_kernel(Xtr)
    a = krr_fit(K[:n1, :n1], ytr[:n1], cfg.krr_lambda)
    pred_cal = krr_predict(K[n1 + n2:, :n1], a)
    sc = np.abs(ytr[n1 + n2:] - pred_cal)
    if h_fixed is None:
        Dss = rkhs_dist_from_kernel(K[n1:n1 + n2, n1:n1 + n2])
        sc_s = np.abs(ytr[n1:n1 + n2] - krr_predict(K[n1:n1 + n2, :n1], a))
        hf = n2 // 2
        best, best_loss = None, np.inf
        for hh in make_grid(cfg):
            w = np.exp(-Dss[:hf, hf:] ** 2 / (2 * hh ** 2))
            q1 = np.array([weighted_quantile(sc_s[hf:], w[j], conformal_level(n2 - hf, cfg.alpha))
                           for j in range(hf)])
            loss = np.mean(np.maximum(cfg.alpha * (sc_s[:hf] - q1), (1 - cfg.alpha) * (q1 - sc_s[:hf])))
            if loss < best_loss: best_loss, best = loss, hh
        h_fixed = best
    Kte_c, _ = rbf_kernel(Xte, Xtr[n1 + n2:], sigma=sig)
    Dte = rkhs_dist_from_kernel(Kte_c)
    Kte_f, _ = rbf_kernel(Xte, Xtr[:n1], sigma=sig)
    yhat = krr_predict(Kte_f, a)
    lo = np.zeros(len(Xte)); up = np.zeros(len(Xte))
    m = len(sc)
    for i in range(len(Xte)):
        w = local_weights(Dte[i], h_fixed)
        if w.sum() < 1e-6:
            w = np.ones_like(w)
        q = weighted_quantile(sc, w, conformal_level(m, cfg.alpha))
        lo[i], up[i] = yhat[i] - q, yhat[i] + q
    cfg._last_yhat = yhat
    return lo, up

def _local_scale(Xq, Xs, r2, cfg):
    """Lepski 局部尺度 sqrt(v_hat)。同时记录每点被选带宽/ESS/网格位置（E10 选择器诊断）。
    局部样本不足时自动降低 N_min 阈值（E7 target 协议的小校准集场景）。"""
    Ks, sig = rbf_kernel(Xs)
    Kq, _ = rbf_kernel(Xq, Xs, sigma=sig)
    Dq = rkhs_dist_from_kernel(Kq)
    n_min_eff = min(cfg.n_min_eff, max(4, len(r2) // 2))
    out = np.zeros(len(Xq)); neff_out = np.zeros(len(Xq))
    h_out = np.zeros(len(Xq)); idx_out = np.zeros(len(Xq), int); glen_out = np.zeros(len(Xq), int)
    for i in range(len(Xq)):
        if getattr(cfg, "grid_mode", "ess") == "absolute":
            grid_i = make_grid(cfg)
        else:
            grid_i = make_grid_neff(Dq[i], n_min_eff, len(r2))
        h_sel, v = lepski_var(Dq[i], r2, grid_i, cfg.kappa, n_min_eff)
        out[i] = np.sqrt(max(v, 1e-8))
        neff_out[i] = _neff(local_weights(Dq[i], h_sel))
        h_out[i] = h_sel
        idx_out[i] = int(np.argmin(np.abs(grid_i - h_sel)))
        glen_out[i] = len(grid_i)
    cfg._last_neff = neff_out
    cfg._last_hsel = h_out
    cfg._last_hidx = idx_out
    cfg._last_gridlen = glen_out
    return out

def lac(Xtr, ytr, Xte, cfg=CFG_INST, base="krr"):
    """归一化局部自适应 conformal（三分割）。边际覆盖有限样本有效，不依赖高斯残差。"""
    Xtr, ytr, n1, n2 = _split3(Xtr, ytr)
    if base == "krr":
        K, sig = rbf_kernel(Xtr)
        a = krr_fit(K[:n1, :n1], ytr[:n1], cfg.krr_lambda)
        def predict(Xq):
            Kq, _ = rbf_kernel(Xq, Xtr[:n1], sigma=sig)
            return krr_predict(Kq, a)
    else:
        gbm = GradientBoostingRegressor(random_state=cfg.random_state).fit(Xtr[:n1], ytr[:n1])
        predict = gbm.predict
    r2 = (ytr[n1:n1 + n2] - predict(Xtr[n1:n1 + n2])) ** 2
    sig_cal = _local_scale(Xtr[n1 + n2:], Xtr[n1:n1 + n2], r2, cfg)
    A = np.abs(ytr[n1 + n2:] - predict(Xtr[n1 + n2:])) / (sig_cal + 1e-8)
    q = weighted_quantile(A, np.ones(len(A)), conformal_level(len(A), cfg.alpha))
    sig_te = _local_scale(Xte, Xtr[n1:n1 + n2], r2, cfg)
    yhat = predict(Xte)
    cfg._last_yhat = yhat; cfg._last_sigma = sig_te
    return yhat - q * sig_te, yhat + q * sig_te

def intrinsic_dim(X, k=10):
    D = cdist(np.asarray(X, float), np.asarray(X, float))
    np.fill_diagonal(D, np.inf)
    T = np.sort(D, axis=1)[:, :k]
    T = np.where(T <= 0, 1e-12, T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_hat = 1.0 / np.mean(np.log(T[:, -1:] / T[:, :-1]), axis=1)
    d_hat = d_hat[np.isfinite(d_hat) & (d_hat > 0)]
    return float(np.mean(d_hat)), (float(np.percentile(d_hat, 2.5)), float(np.percentile(d_hat, 97.5)))

def ncp_fixed(Xtr, ytr, Xte, cfg=CFG_INST):
    return _ncp_generic(Xtr, ytr, Xte, _fixedh_sigma, cfg)

def mondrian_cp(Xtr, ytr, Xte, cfg=CFG_INST):
    """Mondrian CP：分组完全由 D_fit 点预测值的四分位决定（阈值在校准前固定），
    不使用任何测试标签；校准样本 <10 的组回退全局分位数。可部署。"""
    n = len(ytr); h = n // 2
    _p = np.random.RandomState(0).permutation(n); Xtr, ytr = Xtr[_p], ytr[_p]
    if n % 2 == 1: Xtr, ytr = Xtr[:-1], ytr[:-1]; n = len(ytr)
    K, sig = rbf_kernel(Xtr)
    a = krr_fit(K[:h, :h], ytr[:h], cfg.krr_lambda)
    cal_pred = krr_predict(K[h:, :h], a)
    sc = np.abs(ytr[h:] - cal_pred)
    Kte, _ = rbf_kernel(Xte, Xtr[:h], sigma=sig)
    yhat = krr_predict(Kte, a)
    qs = np.quantile(cal_pred, [0.25, 0.5, 0.75])
    g_cal = np.digitize(cal_pred, qs); g_te = np.digitize(yhat, qs)
    q_glob = weighted_quantile(sc, np.ones(n - h), conformal_level(n - h, cfg.alpha))
    lo = np.zeros(len(Xte)); up = np.zeros(len(Xte))
    for g in range(4):
        m_te = g_te == g
        m_cal = g_cal == g
        if m_te.sum() == 0: continue
        q = weighted_quantile(sc[m_cal], np.ones(m_cal.sum()), conformal_level(m_cal.sum(), cfg.alpha)) if m_cal.sum() >= 10 else q_glob
        lo[m_te] = yhat[m_te] - q; up[m_te] = yhat[m_te] + q
    cfg._last_yhat = yhat
    return lo, up

def cqr(Xtr, ytr, Xte, cfg=CFG_INST):
    n = len(ytr); h = n // 2
    _p = np.random.RandomState(0).permutation(n); Xtr, ytr = Xtr[_p], ytr[_p]
    if n % 2 == 1: Xtr, ytr = Xtr[:-1], ytr[:-1]; n = len(ytr)
    lo_m = GradientBoostingRegressor(loss="quantile", alpha=cfg.alpha / 2, random_state=cfg.random_state)
    up_m = GradientBoostingRegressor(loss="quantile", alpha=1 - cfg.alpha / 2, random_state=cfg.random_state)
    lo_m.fit(Xtr[:h], ytr[:h]); up_m.fit(Xtr[:h], ytr[:h])
    sc = np.maximum(ytr[h:] - up_m.predict(Xtr[h:]), lo_m.predict(Xtr[h:]) - ytr[h:])
    q = weighted_quantile(sc, np.ones(n - h), conformal_level(n - h, cfg.alpha))
    yhat = 0.5 * (lo_m.predict(Xte) + up_m.predict(Xte))
    cfg._last_yhat = yhat
    lo = lo_m.predict(Xte) - q; up = up_m.predict(Xte) + q
    return lo, up

def rlcp(Xtr, ytr, Xte, cfg=CFG_INST, B=20):
    """RLCP (Hore & Barber 2023) 简化实现：带宽随机化的局部保形。
    每个测试点采样 B 个对数均匀带宽，各自做加权 conformal 分位数后对区间取平均；
    覆盖保证来自带宽随机化（RLCP 的局部保证对随机化成立）。"""
    Xtr, ytr, n1, n2 = _split3(Xtr, ytr)
    K, sig = rbf_kernel(Xtr)
    a = krr_fit(K[:n1, :n1], ytr[:n1], cfg.krr_lambda)
    def predict(Xq):
        Kq, _ = rbf_kernel(Xq, Xtr[:n1], sigma=sig)
        return krr_predict(Kq, a)
    sc = np.abs(ytr[n1 + n2:] - predict(Xtr[n1 + n2:]))
    m = len(sc)
    D = rkhs_dist_from_kernel(rbf_kernel(Xte, Xtr[n1 + n2:], sigma=sig)[0])
    dmax = float(max(D.max(), 1e-3))
    rng = np.random.RandomState(cfg.random_state)
    yhat = predict(Xte)
    lo_acc = np.zeros(len(Xte)); up_acc = np.zeros(len(Xte))
    for _ in range(B):
        h = float(np.exp(rng.uniform(np.log(dmax * 0.02 + 1e-8), np.log(dmax))))
        q = np.empty(len(Xte))
        for i in range(len(Xte)):
            w = local_weights(D[i], h)
            if w.sum() < 1e-6: w = np.ones_like(w)
            q[i] = weighted_quantile(sc, w, conformal_level(m, cfg.alpha))
        lo_acc += yhat - q; up_acc += yhat + q
    cfg._last_yhat = yhat
    return lo_acc / B, up_acc / B

METHODS = {"Global-CP": global_cp, "Local-CP (tuned)": lambda *a, **k: local_cp(*a, None, **k),
           "CQR": cqr, "Mondrian-CP": mondrian_cp, "NCP (no Lepski)": ncp_fixed, "LAC": lac,
           "LAC-GBM": lambda *a, **k: lac(*a, base="gbm", **k), "RLCP": rlcp}

# =============================================================================
# 指标与统计协议
# =============================================================================
def _wilson_ci(k, n, z=1.959964):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - hw), float(c + hw)

def _boot_ci(covered, B=1000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(covered)
    if n < 10: return (float("nan"), float("nan"))
    idx = rng.randint(0, n, (B, n))
    covs = covered[idx].mean(axis=1)
    return float(np.percentile(covs, 2.5)), float(np.percentile(covs, 97.5))

def metrics(y, lo, up, groups=None, alpha=None):
    covered = (y >= lo) & (y <= up)
    cov = float(covered.mean())
    width = float((up - lo).mean())
    lo_ci, hi_ci = _boot_ci(covered)
    alpha = alpha if alpha else 0.10
    iscore = float(np.mean((up - lo) + (2 / alpha) * np.maximum(lo - y, 0)
                           + (2 / alpha) * np.maximum(y - up, 0)))
    out = {"coverage": cov, "ace": abs(cov - (1 - alpha)), "width": width,
           "interval_score": iscore, "coverage_ci95": [lo_ci, hi_ci],
           "fairness": 0.0, "subgroups": {}}
    if groups is not None:
        for g in pd.unique(groups):
            m = groups == g
            if m.sum() >= 5:
                out["subgroups"][str(g)] = {"n": int(m.sum()),
                                            "coverage": float(((y[m] >= lo[m]) & (y[m] <= up[m])).mean())}
        if out["subgroups"]:
            out["fairness"] = float(max(abs(v["coverage"] - (1 - alpha)) for v in out["subgroups"].values()))
    return out

def _paired_diff_ci(fold_cov, mA, mB, B=2000, seed=0):
    """配对受试者级 cluster bootstrap：覆盖差异 covA-covB 的 95% CI。
    fold_cov: list of dict m -> (subj_array, covered_array)（同一 fold 内各方法对齐）。"""
    agg = defaultdict(lambda: [[], []])
    for fc in fold_cov:
        if mA not in fc or mB not in fc: continue
        sA, cA = fc[mA]; sB, cB = fc[mB]
        for s, ca in zip(sA, cA): agg[s][0].append(float(ca))
        for s, cb in zip(sB, cB): agg[s][1].append(float(cb))
    subj = [s for s in agg if agg[s][0] and agg[s][1]]
    if len(subj) < 10: return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(B):
        pick = [subj[j] for j in rng.randint(0, len(subj), len(subj))]
        a = np.mean([np.mean(agg[s][0]) for s in pick])
        b = np.mean([np.mean(agg[s][1]) for s in pick])
        diffs.append(a - b)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

# =============================================================================
# 数据组装
# =============================================================================
def build_dataset(meta, feat_csv, target, cfg=CFG_INST, group_filter=None):
    feats = pd.read_csv(feat_csv, dtype={"subj": str}, low_memory=False)
    feats["subj"] = feats["subj"].astype(str)
    _m = meta.assign(subj=meta["subj"].astype(str))
    _m["_k"] = pd.to_numeric(_m["subj"], errors="coerce")
    _fk = feats.drop(columns=["subj"]).copy()
    _fk["_k"] = pd.to_numeric(feats["subj"], errors="coerce")
    df = _m.merge(_fk, on="_k", how="inner").drop(columns=["_k"])
    print(f"  [merge] meta={len(_m)}, feats={len(feats)}, matched={len(df)}")
    if group_filter is not None:
        df = df[df["group"] == group_filter]
    df = df.dropna(subset=[target, "age"])
    y = df[target].astype(float).values
    base = [c for c in feats.columns if c != "subj"]
    if getattr(cfg, "feat_suffix", ""):
        base = [c for c in base if c.endswith(cfg.feat_suffix)]
        print(f"  [features] feat_suffix='{cfg.feat_suffix}' -> {len(base)} cols")
    X = df[base].values
    if cfg.use_age_sex_covariates:
        extra = pd.DataFrame(index=df.index)
        if target != "age":
            extra["age"] = (df["age"] - df["age"].mean()) / (df["age"].std() + 1e-12)
        if "sex" in df:
            extra["sex"] = df["sex"].fillna(0.5)
        if len(extra.columns):
            X = np.hstack([X, extra.values])
    ok = np.isfinite(X).all(axis=1)
    X, y, df = X[ok], y[ok], df.iloc[ok]
    info = df[["subj", "group", "site", "scanner"]].copy()
    if "hy" in df: info["hy"] = df["hy"]
    info["y_raw"] = df[target].astype(float).values
    return X, y, info

def _ppmi_meta(args, cfg=CFG_INST):
    return load_ppmi_meta(args.meta_subjects or cfg.meta_ppmi_subjects,
                          args.meta_ida or cfg.meta_ppmi_ida,
                          args.meta_updrs or cfg.meta_ppmi_updrs)

# =============================================================================
# E1 主实验（含配对差异 CI 与非劣效检验）
# =============================================================================
def run_cv(X, y, info, target_name, cfg=CFG_INST, folds=5, seeds=(42, 43, 44), subgroup_col="hy"):
    all_res = {m: [] for m in METHODS}
    pt = {m: {} for m in METHODS}
    fold_cov = []   # 每 fold 记录各方法 (subj, covered)，用于配对 bootstrap
    for seed in seeds:
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            xmu, xsd = X[tr].mean(0), X[tr].std(0) + 1e-12
            Xtr_s, Xte_s = (X[tr] - xmu) / xsd, (X[te] - xmu) / xsd
            mu, sd = float(y[tr].mean()), float(y[tr].std()) + 1e-12
            ytr_s = (y[tr] - mu) / sd
            fc = {}
            for m, fn in METHODS.items():
                lo, up = fn(Xtr_s, ytr_s, Xte_s)
                lo_r, up_r = lo * sd + mu, up * sd + mu
                r = metrics(y[te], lo_r, up_r,
                            info.iloc[te][subgroup_col].values if subgroup_col in info else None, cfg.alpha)
                cov_vec = (y[te] >= lo_r) & (y[te] <= up_r)
                wid_vec = up_r - lo_r
                fc[m] = (info.iloc[te]["subj"].values, cov_vec.astype(float))
                for s_j, c_j, w_j in zip(info.iloc[te]["subj"].values, cov_vec, wid_vec):
                    pt[m].setdefault(s_j, []).append((bool(c_j), float(w_j)))
                yhat = getattr(cfg, "_last_yhat", None)
                if yhat is not None:
                    ae = np.abs(y[te] - (yhat * sd + mu))
                    r["mae"] = float(ae.mean()); r["rmse"] = float(np.sqrt((ae ** 2).mean()))
                    r["width_err_corr"] = float(np.corrcoef(ae, wid_vec)[0, 1])
                all_res[m].append(r)
            fold_cov.append(fc)
    rngb = np.random.RandomState(0)
    cluster_ci = {}
    for m in METHODS:
        subj_items = list(pt[m].items())
        if not subj_items: continue
        means = []
        for _ in range(1000):
            samp = [subj_items[j] for j in rngb.randint(0, len(subj_items), len(subj_items))]
            cs, ns = 0, 0
            for _, lst in samp:
                for c_j, _ in lst: cs += c_j; ns += 1
            means.append(cs / max(ns, 1))
        cluster_ci[m] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    # 配对差异 CI（论文 Section 5 统计协议）
    paired = {}
    for mA, mB in [("LAC", "Global-CP"), ("Mondrian-CP", "LAC"), ("Mondrian-CP", "Global-CP")]:
        paired[f"{mA}-{mB}"] = _paired_diff_ci(fold_cov, mA, mB)
    summary = {}
    for m, rs in all_res.items():
        summary[m] = {
            "coverage_ci95_cluster": cluster_ci.get(m, [float("nan")] * 2),
            "coverage_mean": float(np.mean([r["coverage"] for r in rs])),
            "coverage_ci95": [float(np.percentile([r["coverage"] for r in rs], 2.5)),
                              float(np.percentile([r["coverage"] for r in rs], 97.5))],
            "coverage_se": float(np.std([r["coverage"] for r in rs]) / np.sqrt(len(rs))),
            "width_mean": float(np.mean([r["width"] for r in rs])),
            "ace_mean": float(np.mean([r.get("ace", np.nan) for r in rs])),
            "is_mean": float(np.mean([r.get("interval_score", np.nan) for r in rs])),
            "mae_mean": float(np.mean([r.get("mae", np.nan) for r in rs])),
            "width_err_corr": float(np.nanmean([r.get("width_err_corr", np.nan) for r in rs])),
            "fairness_mean": float(np.mean([r["fairness"] for r in rs])),
        }
    summary["_paired_diff_ci"] = paired
    print(f"\n=== E1 CV ({target_name}, folds={folds}, seeds={len(seeds)}) ===")
    print(f"{'method':<12}{'coverage':>18}{'width':>8}{'ACE':>7}{'IS':>7}{'MAE':>7}{'wErrCorr':>9}{'WGCD':>7}")
    for m, s in summary.items():
        if m.startswith("_"): continue
        ci = s.get("coverage_ci95", [float("nan")] * 2)
        cic = s.get("coverage_ci95_cluster", [float("nan")] * 2)
        print(f"{m:<12}{s['coverage_mean']:>8.3f} [{ci[0]:.3f},{ci[1]:.3f}] cl[{cic[0]:.3f},{cic[1]:.3f}]{s['width_mean']:>8.3f}"
              f"{s.get('ace_mean',float('nan')):>7.3f}{s.get('is_mean',float('nan')):>7.3f}"
              f"{s.get('mae_mean',float('nan')):>7.3f}{s.get('width_err_corr',float('nan')):>9.3f}{s['fairness_mean']:>7.3f}")
    print("\n[paired coverage-diff bootstrap 95% CI] (covA - covB, 受试者级)")
    for k, ci in paired.items():
        print(f"  {k:<28} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    ci_lg = paired.get("LAC-Global-CP", (float("nan"),) * 2)
    if np.isfinite(ci_lg[0]):
        verdict = "PASS" if ci_lg[0] > -cfg.ni_margin else "FAIL"
        print(f"[non-inferiority] LAC vs Global-CP, margin={cfg.ni_margin:.2f}: "
              f"lower bound {ci_lg[0]:+.3f} -> {verdict}（非显著不等于等效，见论文 Section 5）")
    return summary

def run_e1(args, cfg=CFG_INST):
    meta = _ppmi_meta(args, cfg)
    X, y, info = build_dataset(meta, args.features or f"features_cache/PPMI_aal_features.csv", args.target, cfg,
                               group_filter=1 if args.target == "updrs" else None)
    print(f"[E1] PPMI target={args.target}, n={len(y)}, PD only={args.target=='updrs'}")
    try:
        d_hat, d_ci = intrinsic_dim(X[:400])
        print(f"[E1] intrinsic dimension (Levina-Bickel, k=10): d_hat={d_hat:.2f}, 95% CI [{d_ci[0]:.2f}, {d_ci[1]:.2f}]")
    except Exception as e:
        print(f"[E1] intrinsic dim estimation skipped: {e}")
    summary = run_cv(X, y, info, args.target, cfg, args.folds, tuple(range(cfg.random_state, cfg.random_state + args.seeds)))
    _save(summary, f"e1_{args.target}.json")

# =============================================================================
# E2 尺度估计器对比（新）
# =============================================================================
def run_e2(args, cfg=CFG_INST):
    """E2: 多种合理尺度估计器在真实数据上是否都能找到可泛化的误差异质性。
    若全部找不到，才能支持'全局区间已足够'；若部分找到，说明结论依赖估计器选择。"""
    from scipy.stats import spearmanr
    meta = _ppmi_meta(args, cfg)
    X, y, info = build_dataset(meta, args.features or "features_cache/PPMI_aal_features.csv",
                               "updrs", cfg, group_filter=1)
    scale_fns = dict(SCALE_ESTIMATORS)
    recs = {m: [] for m in list(scale_fns) + ["LAC (Lepski)"]}
    mets = {m: [] for m in recs}
    for seed in [42, 43, 44]:
        kf = KFold(5, shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            xmu, xsd = X[tr].mean(0), X[tr].std(0) + 1e-12
            mu, sd = float(y[tr].mean()), float(y[tr].std()) + 1e-12
            Xtr_s, Xte_s = (X[tr] - xmu) / xsd, (X[te] - xmu) / xsd
            ytr_s = (y[tr] - mu) / sd
            for m, sfn in scale_fns.items():
                lo, up = _ncp_generic(Xtr_s, ytr_s, Xte_s, sfn, cfg)
                yhat, sig = cfg._last_yhat, cfg._last_sigma
                lo_r, up_r = lo * sd + mu, up * sd + mu
                mets[m].append(metrics(y[te], lo_r, up_r, alpha=cfg.alpha))
                ae = np.abs(y[te] - (yhat * sd + mu))
                for s_j, sj, aj in zip(info.iloc[te]["subj"], sig * sd, ae):
                    recs[m].append((s_j, float(sj), float(aj)))
            lo, up = lac(Xtr_s, ytr_s, Xte_s)
            yhat, sig = cfg._last_yhat, cfg._last_sigma
            mets["LAC (Lepski)"].append(metrics(y[te], lo * sd + mu, up * sd + mu, alpha=cfg.alpha))
            ae = np.abs(y[te] - (yhat * sd + mu))
            for s_j, sj, aj in zip(info.iloc[te]["subj"], sig * sd, ae):
                recs["LAC (Lepski)"].append((s_j, float(sj), float(aj)))
    summary = {}
    print("\n=== E2 scale-estimator comparison (PPMI E1 config, OOF, 3x5 folds) ===")
    print(f"{'method':<16}{'coverage':>9}{'ACE':>7}{'width':>8}{'rho(sig,|err|)':>15}")
    for m in recs:
        dfm = pd.DataFrame(recs[m], columns=["subj", "sigma", "ae"]).drop_duplicates(["subj"])
        rho = float(spearmanr(dfm["sigma"], dfm["ae"]).statistic)
        ms = mets[m]
        summary[m] = {"coverage": float(np.mean([r["coverage"] for r in ms])),
                      "ace": float(np.mean([r["ace"] for r in ms])),
                      "width": float(np.mean([r["width"] for r in ms])),
                      "spearman_sigma_ae": rho, "n_subj": int(dfm["subj"].nunique())}
        print(f"{m:<16}{summary[m]['coverage']:>9.3f}{summary[m]['ace']:>7.3f}"
              f"{summary[m]['width']:>8.2f}{rho:>15.3f}")
    _save(summary, "e2_scale_comp.json")

# =============================================================================
# 点预测基线（新）：覆盖率正确 != 影像有预测价值
# =============================================================================
def run_pointpred(args, cfg=CFG_INST):
    from sklearn.linear_model import LinearRegression
    meta = _ppmi_meta(args, cfg)
    feat = args.features or "features_cache/PPMI_aal_features.csv"
    X_all, y, info = build_dataset(meta, feat, "updrs", cfg, group_filter=1)   # MRI + demo
    saved = cfg.use_age_sex_covariates; cfg.use_age_sex_covariates = False
    X_mri, _, _ = build_dataset(meta, feat, "updrs", cfg, group_filter=1)      # MRI only
    cfg.use_age_sex_covariates = saved
    p_mri = X_mri.shape[1]
    n_demo = X_all.shape[1] - p_mri
    models = {
        "Mean": lambda Xtr, ytr, Xq: np.full(len(Xq), ytr.mean()),
        "Demo (age+sex)": (lambda Xtr, ytr, Xq:
                           LinearRegression().fit(Xtr[:, p_mri:], ytr).predict(Xq[:, p_mri:])),
        "MRI only (KRR)": lambda Xtr, ytr, Xq: _krr_pred(Xtr[:, :p_mri], ytr, Xq[:, :p_mri], cfg),
        "MRI+Demo (KRR)": lambda Xtr, ytr, Xq: _krr_pred(Xtr, ytr, Xq, cfg),
        "MRI+Demo (GBM)": lambda Xtr, ytr, Xq: GradientBoostingRegressor(
            random_state=cfg.random_state).fit(Xtr, ytr).predict(Xq),
    }
    res = {m: {"mae": [], "rmse": [], "r2": []} for m in models}
    for seed in [42, 43, 44]:
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(X_all):
            for m, fn in models.items():
                yhat = fn(X_all[tr], y[tr], X_all[te])
                res[m]["mae"].append(float(np.abs(y[te] - yhat).mean()))
                res[m]["rmse"].append(float(np.sqrt(((y[te] - yhat) ** 2).mean())))
                res[m]["r2"].append(float(1 - ((y[te] - yhat) ** 2).sum() /
                                          ((y[te] - y[tr].mean()) ** 2).sum()))
    print("\n=== Point-prediction performance (E1 config, 3x5 CV, UPDRS) ===")
    out = {}
    for m, d in res.items():
        out[m] = {k: float(np.mean(v)) for k, v in d.items()}
        print(f"{m:<18} MAE={out[m]['mae']:.2f} RMSE={out[m]['rmse']:.2f} R2={out[m]['r2']:.3f}")
    _save(out, "pointpred.json")

# =============================================================================
# 外部验证
# =============================================================================
def run_external(Xtr, ytr, infotr, Xte, yte, infote, tag, cfg=CFG_INST, subgroup_col="hy"):
    if len(yte) == 0:
        raise SystemExit(f"[error] {tag}: 测试集 0 行 —— 检查 [merge] 诊断行与 group 过滤条件")
    res = {}
    xmu, xsd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xtr_s, Xte_s = (Xtr - xmu) / xsd, (Xte - xmu) / xsd
    mu, sd = float(ytr.mean()), float(ytr.std()) + 1e-12
    ytr_s = (ytr - mu) / sd
    y_raw_sd = float(np.std(infote["y_raw"])) if "y_raw" in infote else float("nan")
    for m, fn in METHODS.items():
        lo, up = fn(Xtr_s, ytr_s, Xte_s)
        lo, up = lo * sd + mu, up * sd + mu
        r = metrics(yte, lo, up, infote[subgroup_col].values if subgroup_col in infote else None, cfg.alpha)
        yhat = getattr(cfg, "_last_yhat", None)
        if yhat is not None:
            ae = np.abs(yte - (yhat * sd + mu))
            r["mae"] = float(ae.mean()); r["rmse"] = float(np.sqrt((ae ** 2).mean()))
            r["width_err_corr"] = float(np.corrcoef(ae, up - lo)[0, 1])
        r["group_cov"] = {}
        for g in pd.unique(infote["group"]):
            mm = (infote["group"].values == g)
            if mm.sum() >= 5:
                r["group_cov"]["PD" if g == 1 else "HC"] = float(((yte[mm] >= lo[mm]) & (yte[mm] <= up[mm])).mean())
        res[m] = r
    print(f"\n=== {tag} ===  (y_raw_sd={y_raw_sd:.3f}; 小样本下 1 名患者 ≈ {100/len(yte):.1f} 个覆盖点)")
    print(f"{'method':<12}{'coverage':>18}{'width':>14}  group coverage")
    for m, r in res.items():
        wn = r["width"] / y_raw_sd if np.isfinite(y_raw_sd) and y_raw_sd > 0 else float("nan")
        r["width_norm"] = float(wn)
        ci = r.get("coverage_ci95", [float("nan")] * 2)
        wlo, whi = _wilson_ci(int(round(r['coverage'] * len(yte))), len(yte))
        r["coverage_wilson95"] = [wlo, whi]
        print(f"{m:<12}{r['coverage']:>8.3f} [{ci[0]:.3f},{ci[1]:.3f}] W[{wlo:.3f},{whi:.3f}]{r['width']:>8.3f} (norm {wn:.3f})  {r['group_cov']}")
    return res

def run_e3(args, cfg=CFG_INST):
    meta = _ppmi_meta(args, cfg)
    X, y, info = build_dataset(meta, args.features or f"features_cache/PPMI_aal_features.csv", args.target, cfg, group_filter=1 if args.target == "updrs" else None)
    sites = info["site"].value_counts()
    big = sites[sites >= args.min_site_n].index.tolist()
    held = args.sites.split(",") if args.sites else big[-1:]
    te = info["site"].isin(held).values
    print(f"[E3] held-out sites={held}, n_test={te.sum()}")
    res = run_external(X[~te], y[~te], info[~te], X[te], y[te], info[te], f"E3 leave-vendor-out {held}")
    _save(res, f"e3_{'-'.join(held)}.json")

def run_e4(args, cfg=CFG_INST):
    mp = _ppmi_meta(args, cfg)
    Xp, yp, ip = build_dataset(mp, args.features_ppmi, "updrs", cfg, group_filter=1)
    mn = load_neurocon_meta(_resolve_meta(args.meta_neurocon or cfg.meta_neurocon, cfg, "NEUROCON/neurocon_patients.tsv"))
    Xn, yn, inn = build_dataset(mn, args.features_neurocon, "updrs", cfg, group_filter=1)
    print(f"[E4] train PPMI PD n={len(yp)}, test NEUROCON PD n={len(yn)}（探索性：跨仪器标签 + 小样本）")
    res = run_external(Xp, yp, ip, Xn, yn, inn, "E4 PPMI->NEUROCON (UPDRS-OFF)", subgroup_col="hy")
    _save(res, "e4_neurocon.json")

def run_e5(args, cfg=CFG_INST):
    mp = _ppmi_meta(args, cfg)
    Xp, yp, ip = build_dataset(mp, args.features_ppmi, "age", cfg)
    outs = {}
    for name, loader, fcsv in [("NEUROCON", load_neurocon_meta, args.features_neurocon),
                               ("TAOWU", load_taowu_meta, args.features_taowu)]:
        meta_key = getattr(cfg, f"meta_{name.lower()}", name.lower() + "_patients.tsv")
        mn = loader(_resolve_meta(args.__dict__.get(f"meta_{name.lower()}") or meta_key, cfg,
                                  f"{name}/{meta_key}"))
        Xn, yn, inn = build_dataset(mn, fcsv, "age", cfg)
        res = run_external(Xp, yp, ip, Xn, yn, inn, f"E5 PPMI->{name} (age)", subgroup_col="group")
        try:
            saved_suffix = cfg.feat_suffix
            cfg.feat_suffix = ""
            Xp_f, _, _ = build_dataset(mp, args.features_ppmi, "age", cfg)
            Xn_f, _, _ = build_dataset(mn, fcsv, "age", cfg)
            cfg.feat_suffix = saved_suffix
            feats_full = pd.read_csv(fcsv, low_memory=False)
            base_cols = [c for c in feats_full.columns if c != "subj"]
            for sfx in ["_mean", "_std", "_gmv"]:
                idx = [i for i, c in enumerate(base_cols) if c.endswith(sfx)]
                if len(idx) < 2: continue
                X1, X2 = Xp_f[:, idx], Xn_f[:, idx]
                xmu, xsd = X1.mean(0), X1.std(0) + 1e-12
                print(f"  [shift] {name} {sfx}: MMD^2={mmd2((X1 - xmu) / xsd, (X2 - xmu) / xsd):.4f}")
        except Exception as _e:
            print(f"  [shift] skipped: {_e}")
        outs[name] = res
    _save(outs, "e5_shift.json")

def run_e6(args, cfg=CFG_INST):
    mp = _ppmi_meta(args, cfg)
    Xp, yp, ip = build_dataset(mp, args.features_ppmi, "age", cfg)
    mn = load_taowu_meta(_resolve_meta(args.meta_taowu or cfg.meta_taowu, cfg, "TaoWu/taowu_patients.tsv"))
    Xn, yn, inn = build_dataset(mn, args.features_taowu, "age", cfg)
    res = run_external(Xp, yp, ip, Xn, yn, inn, "E6 PPMI->TaoWu (age)", subgroup_col="group")
    _save(res, "e6_taowu_age.json")

# =============================================================================
# E0 合成机制验证（多 seed 平均） + 扫描实验（含 Oracle 上界）
# =============================================================================
def mmd2(X1, X2):
    X1 = np.asarray(X1, float); X2 = np.asarray(X2, float)
    Z = np.vstack([X1, X2])
    D = cdist(Z, Z, "sqeuclidean")
    med = np.median(np.sqrt(D[D > 0])) or 1.0
    K = np.exp(-D / (2 * med ** 2))
    n1, n2 = len(X1), len(X2)
    k11 = K[:n1, :n1].sum() - np.trace(K[:n1, :n1])
    k22 = K[n1:, n1:].sum() - np.trace(K[n1:, n1:])
    k12 = K[:n1, n1:].sum()
    return float(k11 / (n1 * (n1 - 1)) + k22 / (n2 * (n2 - 1)) - 2 * k12 / (n1 * n2))

def _synth_case(n_tr=300, n_te=200, p=30, ratio=4.0, tail="normal", shift=1.5, seed=0):
    """sigma(x) = sigma_lo * (1 + (ratio-1)*sigmoid(2*x0))：已知真值、任意点可算的平滑异方差。
    shift: 目标域沿潜变量第一维平移 -> 同时产生协变量偏移与 sigma 区域密度偏移。"""
    rng = np.random.RandomState(seed)
    W = rng.randn(2, p) / np.sqrt(2.0)
    beta = rng.randn(3) * 0.5
    def sig_of(X):
        g = 1.0 / (1.0 + np.exp(-2.0 * np.asarray(X)[:, 0]))
        return 0.3 * (1.0 + (ratio - 1.0) * g)
    def make(n, shift_z):
        Z = rng.randn(n, 2); Z[:, 0] += shift_z
        X = Z @ W + 0.1 * rng.randn(n, p)
        sig = sig_of(X)
        mu = X[:, :3] @ beta
        if tail == "normal":
            e = rng.randn(n)
        else:
            e = rng.standard_t(3, n) / np.sqrt(3.0)   # 单位方差 t3
        return X, mu + sig * e
    Xs, ys = make(n_tr, 0.0)
    Xt, yt = make(n_te, shift)
    return Xs, ys, sig_of, Xt, yt

def _oracle_ncp(Xtr, ytr, Xte, sig_of_std, cfg=CFG_INST):
    """Oracle 上界：归一化 CP 使用真值 sigma（不可部署，仅作性能上界参考）。"""
    Xtr, ytr, n1, n2 = _split3(Xtr, ytr)
    K, sig = rbf_kernel(Xtr)
    a = krr_fit(K[:n1, :n1], ytr[:n1], cfg.krr_lambda)
    def predict(Xq):
        Kq, _ = rbf_kernel(Xq, Xtr[:n1], sigma=sig)
        return krr_predict(Kq, a)
    A = np.abs(ytr[n1 + n2:] - predict(Xtr[n1 + n2:])) / (sig_of_std(Xtr[n1 + n2:]) + 1e-8)
    q = weighted_quantile(A, np.ones(len(A)), conformal_level(len(A), cfg.alpha))
    yhat = predict(Xte)
    return yhat - q * sig_of_std(Xte), yhat + q * sig_of_std(Xte)

def _e0_jobs(Xs_s, ys_s, Xt_s, xsd, xmu, sd, sig_of, cfg):
    return {
        "Global-CP": lambda: global_cp(Xs_s, ys_s, Xt_s),
        "NCP (fixed-h)": lambda: _ncp_generic(Xs_s, ys_s, Xt_s, _fixedh_sigma, cfg),
        "KNN-CP": lambda: _ncp_generic(Xs_s, ys_s, Xt_s, _knn_sigma, cfg),
        "GBM-Scale-CP": lambda: _ncp_generic(Xs_s, ys_s, Xt_s, _gbm_sigma, cfg),
        "LAC": lambda: lac(Xs_s, ys_s, Xt_s),
        "Oracle-NCP": lambda: _oracle_ncp(Xs_s, ys_s, Xt_s,
                                          lambda Xq: sig_of(Xq * xsd + xmu) / sd, cfg),
    }

def run_e0(args, cfg=CFG_INST):
    """E0 机制验证：默认 shift=0（论文正文配置），多 seeds 平均，报告 mean±sd。"""
    from scipy.stats import spearmanr
    acc = defaultdict(lambda: defaultdict(list))
    corrs = []
    for seed in range(args.seeds):
        Xs, ys, sig_of, Xt, yt = _synth_case(ratio=args.ratio, shift=args.shift,
                                             tail=args.tail, seed=seed)
        xmu, xsd = Xs.mean(0), Xs.std(0) + 1e-12
        mu, sd = ys.mean(), ys.std()
        Xs_s, Xt_s, ys_s = (Xs - xmu) / xsd, (Xt - xmu) / xsd, (ys - mu) / sd
        for m, fn in _e0_jobs(Xs_s, ys_s, Xt_s, xsd, xmu, sd, sig_of, cfg).items():
            lo, up = fn()
            r = metrics(yt, lo * sd + mu, up * sd + mu, alpha=cfg.alpha)
            acc[m]["coverage"].append(r["coverage"])
            acc[m]["ace"].append(r["ace"])
            acc[m]["width"].append(r["width"])
            if m == "LAC":
                sh = getattr(cfg, "_last_sigma", None)
                if sh is not None:
                    corrs.append(float(spearmanr(sh, sig_of(Xt) / sd).statistic))
    res = {}
    print(f"\n=== E0 synthetic (ratio={args.ratio}, shift={args.shift}, tail={args.tail}, "
          f"{args.seeds} seeds, mean±sd) ===")
    for m in acc:
        cov = np.array(acc[m]["coverage"]); ace = np.array(acc[m]["ace"]); wid = np.array(acc[m]["width"])
        res[m] = {"coverage_mean": float(cov.mean()), "coverage_sd": float(cov.std()),
                  "ace_mean": float(ace.mean()), "ace_sd": float(ace.std()),
                  "width_mean": float(wid.mean())}
        extra = ""
        if m == "LAC" and corrs:
            res[m]["sigma_recovery_corr_mean"] = float(np.mean(corrs))
            res[m]["sigma_recovery_corr_sd"] = float(np.std(corrs))
            extra = f"  sigma-corr={np.mean(corrs):.3f}±{np.std(corrs):.3f}"
        print(f"{m:<14} cov={cov.mean():.3f}±{cov.std():.3f} width={wid.mean():.3f} "
              f"ACE={ace.mean():.3f}±{ace.std():.3f}{extra}")
    _save(res, "e0_synthetic.json")

def run_e0sweep(args, cfg=CFG_INST):
    """E0 扫描：异方差强度(ratio) x 偏移强度(shift) x 残差尾部(tail)。
    回答'在什么条件下局部适应有收益'，并给出 Oracle 上界差距。"""
    from scipy.stats import spearmanr
    ratios = [float(r) for r in args.ratios.split(",")]
    shifts = [float(s) for s in args.shifts.split(",")]
    tails = args.tails.split(",")
    rows = []
    for ratio in ratios:
        for shift in shifts:
            for tail in tails:
                Xs, ys, sig_of, Xt, yt = _synth_case(ratio=ratio, tail=tail, shift=shift, seed=11)
                xmu, xsd = Xs.mean(0), Xs.std(0) + 1e-12
                mu, sd = ys.mean(), ys.std()
                Xs_s, Xt_s, ys_s = (Xs - xmu) / xsd, (Xt - xmu) / xsd, (ys - mu) / sd
                row = {"ratio": ratio, "shift": shift, "tail": tail}
                for m, fn in _e0_jobs(Xs_s, ys_s, Xt_s, xsd, xmu, sd, sig_of, cfg).items():
                    lo, up = fn()
                    r = metrics(yt, lo * sd + mu, up * sd + mu, alpha=cfg.alpha)
                    row[m] = {"coverage": r["coverage"], "ace": r["ace"], "width": r["width"]}
                    if m == "LAC":
                        sh = getattr(cfg, "_last_sigma", None)
                        if sh is not None:
                            rho = spearmanr(sh, sig_of(Xt) / sd).statistic
                            row[m]["sigma_recovery_corr"] = float(rho) if np.isfinite(rho) else None
                rows.append(row)
                lac_ace = row["LAC"]["ace"]; ora_ace = row["Oracle-NCP"]["ace"]
                sc = row["LAC"].get("sigma_recovery_corr")
                sc_str = "  const" if sc is None else f"{sc:.3f}"
                print(f"ratio={ratio:<3} shift={shift:<4} tail={tail:<7} "
                      f"GlobalACE={row['Global-CP']['ace']:.3f} LAC_ACE={lac_ace:.3f} "
                      f"OracleACE={ora_ace:.3f} LAC_sigcorr={sc_str}")
    _save(rows, "e0_sweep.json")

# =============================================================================
# E7 few-shot（双协议：pooled / target-only）
# =============================================================================
def run_e7(args, cfg=CFG_INST):
    """E7: 两个协议分开报告 ——
    (A) GE vendor-holdout (n=146) 主曲线（统计功效足够）；
    (B) NEUROCON (n=26) 探索性确认。
    --protocol pooled: k 个目标域受试者并入源域校准集（不自动保证目标域覆盖）；
    --protocol target : 仅用 k 个目标域受试者校准（k>=9 才有有限区间）。
    LAC 在小目标集上自动退回全局尺度（lepski_var fallback），不会产生 NaN。"""
    mp = _ppmi_meta(args, cfg)
    Xp, yp, ip = build_dataset(mp, args.features_ppmi, "updrs", cfg, group_filter=1)
    if args.site == "GE":
        ge = ip["scanner"].astype(str).values == "GE"
        Xn, yn, inn = Xp[ge], yp[ge], ip.iloc[ge]
        Xp, yp, ip = Xp[~ge], yp[~ge], ip.iloc[~ge]
        print(f"[E7-GE|{args.protocol}] source n={len(yp)} (non-GE PD), target n={len(yn)} (GE PD)")
    else:
        mn = load_neurocon_meta(_resolve_meta(args.meta_neurocon or cfg.meta_neurocon, cfg,
                                              "NEUROCON/neurocon_patients.tsv"))
        Xn, yn, inn = build_dataset(mn, args.features_neurocon, "updrs", cfg, group_filter=1)
    ks = [int(k) for k in args.ks.split(",")]
    R = args.repeats
    curve = {}
    for m in ["Global-CP", "LAC"]:
        fn = METHODS[m]
        row = []
        for k in ks:
            if args.protocol == "pooled":
                if 0 < k < 9:
                    print(f"  [warn] k={k}<9: 90% split conformal 无有限区间保证（m>=9 才成立）")
            else:
                if 0 < k < 9:
                    print(f"  [skip] target-only 协议 k={k}<9: 无有限区间保证")
                    continue
            if k >= len(yn): k = len(yn) - 5
            covs, widths = [], []
            rngk = np.random.RandomState(7)
            for rep in range(R):
                idx = rngk.permutation(len(yn))
                cal, te = idx[:k], idx[k:]
                if args.protocol == "pooled":
                    Xa = np.vstack([Xp, Xn[cal]]); ya = np.concatenate([yp, yn[cal]])
                else:
                    Xa, ya = Xn[cal], yn[cal]
                xmu, xsd = Xa.mean(0), Xa.std(0) + 1e-12
                mu, sd = ya.mean(), ya.std() + 1e-12
                lo, up = fn((Xa - xmu) / xsd, (ya - mu) / sd, (Xn[te] - xmu) / xsd)
                r = metrics(yn[te], lo * sd + mu, up * sd + mu, alpha=cfg.alpha)
                covs.append(r["coverage"]); widths.append(r["width"])
            row.append({"k": k, "coverage": float(np.mean(covs)),
                        "coverage_se": float(np.std(covs) / np.sqrt(len(covs))),
                        "width": float(np.mean(widths))})
            print(f"  {m} k={k:>2}: coverage={np.mean(covs):.3f}±{np.std(covs)/np.sqrt(len(covs)):.3f} width={np.mean(widths):.3f}")
        curve[m] = row
    print(f"\n=== E7 few-shot ({args.site}, protocol={args.protocol}) ===")
    _save(curve, f"e7_fewshot_{args.site.lower()}_{args.protocol}.json")

# =============================================================================
# E8/E9/E10
# =============================================================================
def run_e8grid(args, cfg=CFG_INST):
    meta = _ppmi_meta(args, cfg)
    X, y, info = build_dataset(meta, args.features or "features_cache/PPMI_aal_features.csv",
                               "updrs", cfg, group_filter=1)
    out = {}
    for mode in ["ess", "absolute"]:
        for kappa in [0.7, 1.5, 2.5]:
            cfg.grid_mode, cfg.kappa = mode, kappa
            sub = run_cv(X, y, info, f"updrs[{mode},k={kappa}]", cfg,
                         folds=3, seeds=(42, 43), subgroup_col="hy")
            out[f"{mode}_k{kappa}"] = {"LAC": sub["LAC"]}
    cfg.grid_mode, cfg.kappa = "ess", 1.0
    print("\n=== E8-grid ablation (kappa response) ===")
    for key, v in out.items():
        s = v["LAC"]
        print(f"{key:<16} coverage={s['coverage_mean']:.3f} width={s['width_mean']:.3f}")
    _save(out, "e8_grid_ablation.json")

def run_e9(args, cfg=CFG_INST):
    alphas = [float(a) for a in args.alphas.split(",")]
    meta = _ppmi_meta(args, cfg)
    X, y, info = build_dataset(meta, args.features or "features_cache/PPMI_aal_features.csv",
                               "updrs", cfg, group_filter=1)
    out = {}
    saved_alpha = cfg.alpha
    for al in alphas:
        cfg.alpha = al
        sub = run_cv(X, y, info, f"alpha={al}", cfg, folds=3, seeds=(42, 43), subgroup_col="hy")
        out[f"alpha_{al}"] = {m: {"nominal": 1 - al, "coverage": sub[m]["coverage_mean"],
                                  "coverage_ci95_cluster": sub[m]["coverage_ci95_cluster"]}
                              for m in ["Global-CP", "CQR", "LAC"]}
    cfg.alpha = saved_alpha
    print("\n=== E9 alpha calibration curve ===")
    for key, v in out.items():
        line = "  ".join(f"{m}:{d['coverage']:.3f}" for m, d in v.items())
        print(f"{key}: nominal={v['Global-CP']['nominal']:.2f}  {line}")
    _save(out, "e9_alpha.json")

def run_e10(args, cfg=CFG_INST):
    """E10: difficulty-error 分析 + 选择器行为诊断（带宽/ESS 分布、弃权比例、kappa 敏感性）。"""
    from scipy.stats import spearmanr
    meta = _ppmi_meta(args, cfg)
    X, y, info = build_dataset(meta, args.features or "features_cache/PPMI_aal_features.csv",
                               "updrs", cfg, group_filter=1)
    recs = []
    for seed in [42, 43, 44]:
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
            xmu, xsd = X[tr].mean(0), X[tr].std(0) + 1e-12
            mu, sd = float(y[tr].mean()), float(y[tr].std()) + 1e-12
            lo, up = lac((X[tr] - xmu) / xsd, (y[tr] - mu) / sd, (X[te] - xmu) / xsd)
            sig = getattr(cfg, "_last_sigma", None); yhat = getattr(cfg, "_last_yhat", None)
            neff = getattr(cfg, "_last_neff", None)
            hsel = getattr(cfg, "_last_hsel", None)
            hidx = getattr(cfg, "_last_hidx", None)
            glen = getattr(cfg, "_last_gridlen", None)
            if sig is None or yhat is None: continue
            for j in range(len(te)):
                recs.append({"subj": info.iloc[te[j]]["subj"], "seed": seed,
                             "sigma": float(sig[j] * sd), "ae": float(abs(y[te[j]] - (yhat[j] * sd + mu))),
                             "width": float((up[j] - lo[j]) * sd),
                             "covered": bool(y[te[j]] >= lo[j] * sd + mu and y[te[j]] <= up[j] * sd + mu),
                             "neff": float(neff[j]) if neff is not None else np.nan,
                             "hsel": float(hsel[j]) if hsel is not None else np.nan,
                             "at_max_grid": bool(hidx[j] == glen[j] - 1) if hidx is not None else False,
                             "hy": info.iloc[te[j]].get("hy", np.nan)})
    df = pd.DataFrame(recs).drop_duplicates(["subj", "seed"])
    rho_sig = spearmanr(df["sigma"], df["ae"]).statistic
    rho_wid = spearmanr(df["width"], df["ae"]).statistic
    df["bin"] = pd.qcut(df["sigma"].rank(method="first"), 3, labels=["low", "mid", "high"])
    table = df.groupby("bin", observed=True).agg(
        n=("subj", "count"), sigma_mean=("sigma", "mean"), mae=("ae", "mean"),
        width=("width", "mean"), coverage=("covered", "mean"), neff=("neff", "mean")).reset_index()
    # 选择器诊断
    frac_max = float(df["at_max_grid"].mean())
    ess_q = df["neff"].quantile([0.1, 0.5, 0.9]).values.tolist()
    print("\n=== E10 difficulty-error analysis + selector diagnostics (E1 OOF, LAC) ===")
    print(f"Spearman(sigma_hat, |err|) = {rho_sig:.3f}   Spearman(width, |err|) = {rho_wid:.3f}")
    print(f"selector: fraction at max grid scale (abstain) = {frac_max:.3f}")
    print(f"selector: ESS quantiles [q10,q50,q90] = [{ess_q[0]:.1f}, {ess_q[1]:.1f}, {ess_q[2]:.1f}]")
    print(table.round(3).to_string(index=False))
    _save({"rho_sigma_err": float(rho_sig), "rho_width_err": float(rho_wid),
           "frac_at_max_grid": frac_max, "ess_quantiles": ess_q,
           "table": table.to_dict("records")}, "e10_difficulty.json")

# =============================================================================
# 合成数据自检
# =============================================================================
def make_synthetic(n=400, p=40, seed=0):
    rng = np.random.RandomState(seed)
    nper = n // 2
    Z = np.vstack([np.array([[0., 0.]]) + 0.5 * rng.randn(nper, 2),
                   np.array([[5., 0.]]) + 0.5 * rng.randn(nper, 2)])
    W = rng.randn(2, p) * 0.5
    X = Z @ W + 0.1 * rng.randn(n, p)
    sev = np.repeat([0, 1], nper)
    y = 0.5 * sev + np.where(sev == 0, 0.25, 1.6) * rng.randn(n)
    y = (y - y.mean()) / y.std()
    return X, y, sev



def _oof_residuals(Xtr_s, ytr_s, cfg=CFG_INST, inner_folds=3):
    """内部 K-fold 交叉拟合残差：训练集每个点都获得一个未见过自身的预测残差。"""
    n = len(ytr_s)
    r = np.zeros(n)
    kf = KFold(inner_folds, shuffle=True, random_state=cfg.random_state)
    for tr2, te2 in kf.split(Xtr_s):
        K2, sig2 = rbf_kernel(Xtr_s[tr2])
        a2 = krr_fit(K2, ytr_s[tr2], cfg.krr_lambda)
        pred = krr_predict(rbf_kernel(Xtr_s[te2], Xtr_s[tr2], sigma=sig2)[0], a2)
        r[te2] = ytr_s[te2] - pred
    return r


def lac_cf(Xtr, ytr, Xte, cfg=CFG_INST):
    """信息高效版 LAC（E12）：(i) scale 集 = 全部训练点的交叉拟合残差
    (~n/2 点而非 n/4，有效样本 ~3x)；(ii) 对 log(r²) 做 Lepski 局部估计
    （log χ² 噪声有界且与尺度无关，重尾稳健，selector 不再被异常值钉死）。
    无泄漏：每个 scale 点的残差来自不含该点的内部模型。"""
    n = len(ytr)
    _p = np.random.RandomState(0).permutation(n); Xtr, ytr = Xtr[_p], ytr[_p]
    if n % 2 == 1: Xtr, ytr = Xtr[:-1], ytr[:-1]; n = len(ytr)
    n1 = n // 2
    K, sig = rbf_kernel(Xtr)
    a = krr_fit(K[:n1, :n1], ytr[:n1], cfg.krr_lambda)
    def predict(Xq):
        Kq, _ = rbf_kernel(Xq, Xtr[:n1], sigma=sig)
        return krr_predict(Kq, a)
    Xpool, ypool = Xtr[n1:], ytr[n1:]
    r2 = _oof_residuals(Xpool, ypool, cfg) ** 2
    def scale_of(Xq):
        Kq, _ = rbf_kernel(Xq, Xpool, sigma=sig)
        Dq = rkhs_dist_from_kernel(Kq)
        n_min = min(cfg.n_min_eff, max(4, len(r2) // 2))
        out = np.zeros(len(Xq))
        for i in range(len(Xq)):
            grid_i = make_grid_neff(Dq[i], n_min, len(r2)) if getattr(cfg, "grid_mode", "ess") == "ess" else make_grid(cfg)
            h_sel, v = lepski_var(Dq[i], np.log(r2 + 1e-8), grid_i, cfg.kappa, n_min)
            out[i] = np.exp(0.5 * v)
        return out
    sig_cal = scale_of(Xpool)
    A = np.abs(ypool - predict(Xpool)) / (sig_cal + 1e-8)
    q = weighted_quantile(A, np.ones(len(A)), conformal_level(len(A), cfg.alpha))
    sig_te = scale_of(Xte)
    yhat = predict(Xte)
    cfg._last_yhat = yhat; cfg._last_sigma = sig_te
    return yhat - q * sig_te, yhat + q * sig_te


def _metric_weights_shrunk(Xs, r2, lam=0.5, seed=42):
    """pilot 重要性收缩：w_f ∝ (imp_f + lam*mean(imp))，lam 大则趋向均匀度量。
    不用 CV-R2 硬门控 —— log r² 均值在 CV 下不可估（噪声地板 sd≈2.23），
    但重要性方向仍可学；收缩是稳健折中。"""
    g = GradientBoostingRegressor(random_state=seed).fit(
        np.asarray(Xs, float), np.log(np.asarray(r2, float) + 1e-8))
    imp = g.feature_importances_
    w = (imp + lam * imp.mean())
    w = w / (w.mean() + 1e-12)
    return np.sqrt(np.clip(w, 1e-3, None))


def lac_ml(Xtr, ytr, Xte, cfg=CFG_INST):
    """度量学习版 LAC（E12）：pilot GBM 在交叉拟合残差上学对角度量
    （重要性收缩版），核距离沿方差相关方向收缩后再做 Lepski log-scale。
    预期：中等以上信号（gain>=5）恢复率大幅提升；弱信号靠收缩回退近似均匀。"""
    n = len(ytr)
    _p = np.random.RandomState(0).permutation(n); Xtr, ytr = Xtr[_p], ytr[_p]
    if n % 2 == 1: Xtr, ytr = Xtr[:-1], ytr[:-1]; n = len(ytr)
    n1 = n // 2
    K, sig = rbf_kernel(Xtr)
    a = krr_fit(K[:n1, :n1], ytr[:n1], cfg.krr_lambda)
    def predict(Xq):
        Kq, _ = rbf_kernel(Xq, Xtr[:n1], sigma=sig)
        return krr_predict(Kq, a)
    Xpool, ypool = Xtr[n1:], ytr[n1:]
    r2 = _oof_residuals(Xpool, ypool, cfg) ** 2
    sw = _metric_weights_shrunk(Xpool, r2, lam=getattr(cfg, "ml_lam", 0.5))
    Xp_w, Xte_w = Xpool * sw, Xte * sw
    def scale_of(Xq_w):
        Kp, sigp = rbf_kernel(Xp_w)
        Dq = rkhs_dist_from_kernel(rbf_kernel(Xq_w, Xp_w, sigma=sigp)[0])
        n_min = min(cfg.n_min_eff, max(4, len(r2) // 2))
        out = np.zeros(len(Xq_w))
        for i in range(len(Xq_w)):
            gi = make_grid_neff(Dq[i], n_min, len(r2)) if getattr(cfg, "grid_mode", "ess") == "ess" else make_grid(cfg)
            h, v = lepski_var(Dq[i], np.log(r2 + 1e-8), gi, cfg.kappa, n_min)
            out[i] = np.exp(0.5 * v)
        return out
    sig_cal = scale_of(Xp_w)
    A = np.abs(ypool - predict(Xpool)) / (sig_cal + 1e-8)
    q = weighted_quantile(A, np.ones(len(A)), conformal_level(len(A), cfg.alpha))
    sig_te = scale_of(Xte_w); yhat = predict(Xte)
    cfg._last_yhat = yhat; cfg._last_sigma = sig_te
    return yhat - q * sig_te, yhat + q * sig_te


def run_e12(args, cfg=CFG_INST):
    """E12: 信息高效管线下的异方差注入剂量-反应。
    对比 Global-CP / NCP / LAC(72点,raw r²) / LAC-CF(~230点,log r²)。
    若 LAC-CF 的恢复 corr 显著高于 LAC，则 E11 的低恢复是实现层信息浪费，
    而非内禀噪声地板；同时与理论可检出性界（Remark~\ref{rem:detect}）对照。"""
    from scipy.stats import spearmanr
    meta = _ppmi_meta(args, cfg)
    X, y, info = build_dataset(meta, args.features or "features_cache/PPMI_aal_features.csv",
                               "updrs", cfg, group_filter=1)
    Xc = X - X.mean(0)
    if getattr(args, "mode", "pc1") == "sparse":
        # 稀疏局部方向：随机 5 个特征归一化组合（模拟"少数 ROI 驱动异方差"）
        rdirs = np.random.RandomState(0).randn(X.shape[1])
        keep = np.random.RandomState(1).choice(X.shape[1], 5, replace=False)
        wdir = np.zeros(X.shape[1]); wdir[keep] = rdirs[keep]
        wdir /= np.linalg.norm(wdir) + 1e-12
        t = Xc @ wdir
    else:
        _, _, vt = np.linalg.svd(Xc[:400], full_matrices=False)
        t = Xc @ vt[0]
    tnorm = (t - t.min()) / (np.ptp(t) + 1e-12)
    gains = [float(g) for g in args.gains.split(",")]
    names = ["Global-CP", "NCP (no Lepski)", "LAC", "LAC-CF", "LAC-ML"]
    fn_map = {"Global-CP": global_cp, "NCP (no Lepski)": ncp_fixed,
              "LAC": lac, "LAC-CF": lac_cf, "LAC-ML": lac_ml}
    print("\n=== E12 injection dose-response, information-efficient pipeline (3x5 folds) ===")
    print(f"{'gain':<6}{'method':<16}{'coverage':>9}{'ACE':>7}{'width':>8}{'sigma-corr':>12}{'rho_g':>12}")
    print("[theory] Gaussian log-chi2: SE(log-sigma) ~= 1.12/sqrt(N_eff); "
          "N_eff~40 -> detectable scale ratio ~ exp(2*0.18) = 1.4")
    out_all = {}
    for gain in gains:
        g = 1.0 + (gain - 1.0) * tnorm
        res = {m: {"cov": [], "ace": [], "width": [], "rho": []} for m in names}
        for seed in [42, 43, 44]:
            for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
                xmu, xsd = X[tr].mean(0), X[tr].std(0) + 1e-12
                mu, sd = float(y[tr].mean()), float(y[tr].std()) + 1e-12
                Xtr_s, Xte_s = (X[tr] - xmu) / xsd, (X[te] - xmu) / xsd
                ytr_s = (y[tr] - mu) / sd
                K, sig = rbf_kernel(Xtr_s)
                a = krr_fit(K, ytr_s, cfg.krr_lambda)
                pred_tr = krr_predict(K, a)
                pred_te = krr_predict(rbf_kernel(Xte_s, Xtr_s, sigma=sig)[0], a)
                r_tr = ytr_s - pred_tr
                r_te = (y[te] - mu) / sd - pred_te
                yinj_tr = pred_tr + r_tr * g[tr]
                yinj_te = pred_te + r_te * g[te]
                for m in names:
                    lo, up = fn_map[m](Xtr_s, yinj_tr, Xte_s)
                    r = metrics(yinj_te, lo, up, alpha=cfg.alpha)
                    res[m]["cov"].append(r["coverage"]); res[m]["ace"].append(r["ace"])
                    res[m]["width"].append(r["width"])
                    if m in ("LAC", "LAC-CF", "LAC-ML"):
                        res[m]["rho"].append(float(spearmanr(cfg._last_sigma, np.abs(r_te) * g[te]).statistic))
                        res[m].setdefault("rho_g", []).append(float(spearmanr(cfg._last_sigma, g[te]).statistic))
        out_all[str(gain)] = {}
        for m in names:
            o = {k: float(np.mean(v)) for k, v in res[m].items() if v}
            out_all[str(gain)][m] = o
            rho_s = (f"{o['rho']:>12.3f}" if "rho" in o else f"{'---':>12}")
            rho_g_s = (f"{o['rho_g']:>12.3f}" if "rho_g" in o else f"{'---':>12}")
            print(f"{gain:<6.1f}{m:<16}{o['cov']:>9.3f}{o['ace']:>7.3f}{o['width']:>8.2f}{rho_s}{rho_g_s}")
    _save(out_all, f"e12_injection_cf_{args.mode}.json")


def _synth_scaled(n_tr, n_te, p, d_signal, R, seed=0):
    """sigma(x)=exp(R*sigmoid(2*x0))，信号方向 x0 内嵌在 d_signal 维有效特征中，
    其余 p-d_signal 维为纯噪声维度（改变搜索难度 d）。"""
    rng = np.random.RandomState(seed)
    W = rng.randn(d_signal, p) / np.sqrt(d_signal)
    beta = rng.randn(3) * 0.5
    def make(n):
        Z = rng.randn(n, d_signal)
        X = Z @ W + 0.1 * rng.randn(n, p)
        g = 1.0 / (1.0 + np.exp(-2.0 * X[:, 0]))
        sig = np.exp(R * g) / np.exp(R * 0.5)   # 以 exp(R/2) 归一化,几何均值=1
        y = X[:, :3] @ beta + sig * rng.randn(n)
        return X, y, sig
    Xs, ys, ss = make(n_tr); Xt, yt, st = make(n_te)
    return Xs, ys, ss, Xt, yt, st


def run_e13(args, cfg=CFG_INST):
    """E13: 可检出性指数的实证标定（合成）。
    固定 R in {ln2,ln4,ln8}，扫描搜索维数 d 与 scale 集大小 m，记录恢复率；
    对每个 (d,m) 找出恢复率>=0.5 的最小 R（检测边界），再对 log(d/m) 拟合指数。
    若实证指数 ≈ 1/4，则支撑 Remark (d/m)^{1/4} 启发式界；若≈1/2 则支撑已知 link 界。"""
    from scipy.stats import spearmanr
    ds = [int(d) for d in args.dims.split(",")]
    ms = [int(m) for m in args.ms.split(",")]
    Rs = [float(r) for r in args.logranges.split(",")]   # log-range
    seeds = range(args.seeds)
    grid = {}
    print("\n=== E13 detectability-exponent calibration (synthetic) ===")
    print(f"{'d':>5}{'m':>6}{'R':>6}{'recovery':>12}")
    for d in ds:
        for m in ms:
            for R in Rs:
                recs = []
                for seed in seeds:
                    Xs, ys, ss, Xt, yt, st = _synth_scaled(4*m, 150, max(d, 10), min(d, 30), R, seed)
                    xmu, xsd = Xs.mean(0), Xs.std(0)+1e-12
                    mu, sd = ys.mean(), ys.std()
                    lo, up = lac((Xs-xmu)/xsd, (ys-mu)/sd, (Xt-xmu)/xsd)
                    rho = spearmanr(cfg._last_sigma, st/sd).statistic
                    recs.append(rho if np.isfinite(rho) else np.nan)
                rec = float(np.nanmean(recs))
                grid[(d, m, R)] = rec
                print(f"{d:>5}{m:>6}{R:>6.2f}{rec:>12.3f}")
    # 每 (d,m) 的检测边界 R_min：相邻点局部线性插值/外推解 rec(R)=0.5
    rows = []
    for d in ds:
        for m in ms:
            pts = sorted([(R, grid[(d, m, R)]) for R in Rs])
            Rv = np.array([p[0] for p in pts]); yv = np.array([p[1] for p in pts])
            r_min, tag = None, "interp"
            for i in range(len(pts) - 1):
                R0, y0 = pts[i]; R1, y1 = pts[i+1]
                if y0 < 0.5 <= y1 and y1 > y0:
                    r_min = R0 + (0.5 - y0) * (R1 - R0) / (y1 - y0); break
            if r_min is None:
                # 外推：用最近的两个点估计局部斜率
                if yv[0] >= 0.5:
                    s = (yv[1] - yv[0]) / (Rv[1] - Rv[0])
                    if s > 1e-6:
                        r_min = Rv[0] - (yv[0] - 0.5) / s; tag = "extrap-low"
                elif yv[-1] < 0.5:
                    s = (yv[-1] - yv[-2]) / (Rv[-1] - Rv[-2])
                    if s > 1e-6:
                        r_min = Rv[-1] + (0.5 - yv[-1]) / s; tag = "extrap-high"
            if r_min is not None:
                rows.append((d, m, r_min))
                print(f"  boundary: d={d}, m={m}, R_min={r_min:.3f} ({tag}), log10(d/m)={np.log10(d/m):.3f}")
    fit_rows = [(d, m, r) for d, m, r in rows if r > 0.05]
    if len(fit_rows) >= 3:
        # 双参数检验律: R_min = a * d^{1/4} / sqrt(m)
        xa = np.array([d**0.25/np.sqrt(m) for d, m, _ in fit_rows])
        ya = np.array([r for _, _, r in fit_rows])
        a = float((xa*ya).sum()/(xa*xa).sum())
        r2a = 1 - ((ya - a*xa)**2).sum()/((ya - ya.mean())**2).sum()
        # 单参折叠对照: R_min ~ (d/m)^b
        x2 = np.log10([d/m for d, m, _ in fit_rows])
        X2 = np.column_stack([np.ones(len(fit_rows)), x2])
        coef, *_ = np.linalg.lstsq(X2, np.log10(ya), rcond=None)
        r2b = 1 - ((np.log10(ya) - X2@coef)**2).sum()/((np.log10(ya) - np.log10(ya).mean())**2).sum()
        print(f"\n[fit A: detection law] R_min = {a:.2f} * d^0.25/sqrt(m),  R^2 = {r2a:.3f}")
        print(f"[fit B: one-param collapse] R_min ~ (d/m)^b,  b = {coef[1]:.3f},  R^2 = {r2b:.3f}")
    _save({"grid": {f"d{d}_m{m}_R{R:.2f}": v for (d, m, R), v in grid.items()},
           "boundaries": [{"d": d, "m": m, "R_min": r} for d, m, r in rows],
           "exponent": float(coef[1]) if len(rows) >= 3 else None},
          "e13_exponent.json")


def run_e14(args, cfg=CFG_INST):
    """E14: 重复测量降低噪声地板的合成验证。
    每个受试者 n_rep 次重复，scale 集用受试者内平均的 r^2（噪声方差 ~1/n_rep），
    检验恢复率是否随 n_rep 按 ~1/sqrt(n_rep) 改善。"""
    from scipy.stats import spearmanr
    reps = [int(r) for r in args.reps.split(",")]
    R = args.logrange
    recs_out = {}
    print("\n=== E14 repeated-measurement remedy (synthetic, R={:.2f}) ===".format(R))
    print(f"{'n_rep':>6}{'recovery':>12}{'theory 1/sqrt':>14}")
    base = None
    for nrep in reps:
        rr = []
        for seed in [0, 1]:
            rng = np.random.RandomState(seed)
            Xs, ys, ss, Xt, yt, st = _synth_scaled(4*72, 150, 30, 30, R, seed)
            xmu, xsd = Xs.mean(0), Xs.std(0)+1e-12
            mu, sd = ys.mean(), ys.std()
            Xs_s, Xt_s = (Xs-xmu)/xsd, (Xt-xmu)/xsd
            # 拟合点预测器
            K, sig = rbf_kernel(Xs_s)
            a = krr_fit(K, ys_s := (ys-mu)/sd, cfg.krr_lambda)
            pred = lambda Xq: krr_predict(rbf_kernel(Xq, Xs_s, sigma=sig)[0], a)
            # 每个 scale 点 nrep 次重复测量：r^2 取受试者内平均
            Xpool, ypool = Xs_s[2*72:], None
            n_pool = len(Xpool)
            pool_pred = pred(Xpool)
            r2_pool = np.zeros(n_pool)
            for j in range(n_pool):
                eps = rng.randn(nrep)
                r2_pool[j] = np.mean((ss[2*72+j]/sd * eps)**2)
            # 用 LAC 的局部尺度估计器（自定义 r2）
            def scale_of(Xq):
                Kq, _ = rbf_kernel(Xq, Xpool, sigma=sig)
                Dq = rkhs_dist_from_kernel(Kq)
                n_min = min(cfg.n_min_eff, max(4, len(r2_pool)//2))
                out = np.zeros(len(Xq))
                for i in range(len(Xq)):
                    gi = make_grid_neff(Dq[i], n_min, len(r2_pool))
                    h, v = lepski_var(Dq[i], r2_pool, gi, cfg.kappa, n_min)
                    out[i] = np.sqrt(max(v, 1e-8))
                return out
            sig_te = scale_of(Xt_s)
            rho = spearmanr(sig_te, st/sd).statistic
            rr.append(rho if np.isfinite(rho) else np.nan)
        rec = float(np.nanmean(rr))
        recs_out[str(nrep)] = rec
        if base is None:
            base = rec
        print(f"{nrep:>6}{rec:>12.3f}{(1-(1-base)/np.sqrt(nrep)):>14.3f}")
    _save(recs_out, "e14_replicates.json")


def run_e11(args, cfg=CFG_INST):
    """E11 (补充材料): 真实 PPMI 残差注入合成异方差的剂量-反应扫描。
    对每个 gain，将 OOF 残差乘以 g(x) in [1, gain]（g 由特征 PC1 无监督构造），
    观察 LAC 的 sigma-corr 随注入强度的变化 —— 标定该表征噪声地板下的可检出阈值。
    注意：真实残差 ~97% 为噪声（R²=0.035）且每受试者仅一次观测，弱注入可能不可检出。"""
    from scipy.stats import spearmanr
    meta = _ppmi_meta(args, cfg)
    X, y, info = build_dataset(meta, args.features or "features_cache/PPMI_aal_features.csv",
                               "updrs", cfg, group_filter=1)
    Xc = X - X.mean(0)
    if getattr(args, "mode", "pc1") == "sparse":
        # 稀疏局部方向：随机 5 个特征归一化组合（模拟"少数 ROI 驱动异方差"）
        rdirs = np.random.RandomState(0).randn(X.shape[1])
        keep = np.random.RandomState(1).choice(X.shape[1], 5, replace=False)
        wdir = np.zeros(X.shape[1]); wdir[keep] = rdirs[keep]
        wdir /= np.linalg.norm(wdir) + 1e-12
        t = Xc @ wdir
    else:
        _, _, vt = np.linalg.svd(Xc[:400], full_matrices=False)
        t = Xc @ vt[0]
    tnorm = (t - t.min()) / (np.ptp(t) + 1e-12)
    gains = [float(g) for g in args.gains.split(",")]
    names = ["Global-CP", "NCP (no Lepski)", "LAC"]
    out_all = {}
    print(f"\n=== E11 heteroscedasticity injection dose-response (3x5 folds) ===")
    print(f"{'gain':<6}{'method':<16}{'coverage':>9}{'ACE':>7}{'width':>8}{'sigma-corr':>12}{'rho_g':>12}")
    for gain in gains:
        g = 1.0 + (gain - 1.0) * tnorm
        res = {m: {"cov": [], "ace": [], "width": [], "rho": []} for m in names}
        for seed in [42, 43, 44]:
            for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
                xmu, xsd = X[tr].mean(0), X[tr].std(0) + 1e-12
                mu, sd = float(y[tr].mean()), float(y[tr].std()) + 1e-12
                Xtr_s, Xte_s = (X[tr] - xmu) / xsd, (X[te] - xmu) / xsd
                ytr_s = (y[tr] - mu) / sd
                K, sig = rbf_kernel(Xtr_s)
                a = krr_fit(K, ytr_s, cfg.krr_lambda)
                pred_tr = krr_predict(K, a)
                pred_te = krr_predict(rbf_kernel(Xte_s, Xtr_s, sigma=sig)[0], a)
                r_tr = ytr_s - pred_tr
                r_te = (y[te] - mu) / sd - pred_te
                yinj_tr = pred_tr + r_tr * g[tr]
                yinj_te = pred_te + r_te * g[te]
                for m in names:
                    lo, up = METHODS[m](Xtr_s, yinj_tr, Xte_s)
                    r = metrics(yinj_te, lo, up, alpha=cfg.alpha)
                    res[m]["cov"].append(r["coverage"]); res[m]["ace"].append(r["ace"])
                    res[m]["width"].append(r["width"])
                    if m == "LAC":
                        res[m]["rho"].append(float(spearmanr(cfg._last_sigma, np.abs(r_te) * g[te]).statistic))
                        res[m].setdefault("rho_g", []).append(float(spearmanr(cfg._last_sigma, g[te]).statistic))
        out_all[str(gain)] = {}
        for m in names:
            o = {k: float(np.mean(v)) for k, v in res[m].items() if v}
            out_all[str(gain)][m] = o
            rho_s = (f"{o['rho']:>12.3f}" if "rho" in o else f"{'---':>12}")
            rho_g_s = (f"{o['rho_g']:>12.3f}" if "rho_g" in o else f"{'---':>12}")
            print(f"{gain:<6.1f}{m:<16}{o['cov']:>9.3f}{o['ace']:>7.3f}{o['width']:>8.2f}{rho_s}{rho_g_s}")
    _save(out_all, f"e11_injection_{args.mode}.json")


def run_e8demo(args, cfg=CFG_INST):
    """E8 受控演示（主文/补充材料）：固定大核带宽（median 的 --fixed-mult 倍）使
    RKHS 距离压缩，对比 absolute-h 网格与 ESS 网格对 Lepski 常数 kappa 的敏感性。
    预期：压缩距离下 absolute 网格所有候选权重≈1 -> 选中尺度与 kappa 无关；
    ESS 网格按目标有效样本数反演 -> 保留 kappa 响应。"""
    from scipy.stats import spearmanr
    mult = args.fixed_mult
    orig_rbf = rbf_kernel
    def fixed_rbf(X, Z=None, sigma=None):
        X = np.asarray(X, float); Z = X if Z is None else np.asarray(Z, float)
        d2 = cdist(X, Z, "sqeuclidean")
        if sigma is None:
            med = np.median(np.sqrt(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
            sigma = mult * med
        return np.exp(-d2 / (2 * sigma ** 2)), sigma
    acc = defaultdict(list)
    try:
        globals()["rbf_kernel"] = fixed_rbf
        for seed in range(args.seeds):
            Xs, ys, sig_of, Xt, yt = _synth_case(ratio=4.0, shift=0.0, tail="normal", seed=seed)
            xmu, xsd = Xs.mean(0), Xs.std(0) + 1e-12
            mu, sd = ys.mean(), ys.std()
            Xs_s, Xt_s, ys_s = (Xs - xmu) / xsd, (Xt - xmu) / xsd, (ys - mu) / sd
            for mode in ["absolute", "ess"]:
                for kap in [0.7, 1.5, 2.5]:
                    cfg.grid_mode, cfg.kappa = mode, kap
                    lo, up = lac(Xs_s, ys_s, Xt_s)
                    r = metrics(yt, lo * sd + mu, up * sd + mu, alpha=cfg.alpha)
                    rho = float(spearmanr(cfg._last_sigma, sig_of(Xt) / sd).statistic)
                    acc[(mode, kap)].append((r["ace"], rho, float(cfg._last_neff.mean())))
    finally:
        globals()["rbf_kernel"] = orig_rbf
        cfg.grid_mode, cfg.kappa = "ess", 1.0
    print(f"\n=== E8 compressed-distance demo (kernel bandwidth = {mult}x median, "
          f"{args.seeds} seeds, mean±sd) ===")
    print(f"{'grid':<10}{'kappa':<7}{'ACE':<14}{'sigma-corr':<12}{'ESS(mean)'}")
    out = {}
    for k in [("absolute", 0.7), ("absolute", 1.5), ("absolute", 2.5),
              ("ess", 0.7), ("ess", 1.5), ("ess", 2.5)]:
        v = np.array(acc[k])
        out[f"{k[0]}_k{k[1]}"] = {"ace_mean": float(v[:, 0].mean()), "ace_sd": float(v[:, 0].std()),
                                  "sigma_corr": float(v[:, 1].mean()), "ess_mean": float(v[:, 2].mean())}
        print(f"{k[0]:<10}{k[1]:<7}{v[:,0].mean():.3f}±{v[:,0].std():.3f}   "
              f"{v[:,1].mean():.3f}       {v[:,2].mean():.1f}")
    _save(out, "e8_compressed_demo.json")

def cmd_selftest(_args, cfg=CFG_INST):
    X, y, sev = make_synthetic()
    info = pd.DataFrame({"subj": [f"s{i}" for i in range(len(sev))], "hy": sev,
                         "group": sev >= 2, "site": "syn", "scanner": "syn"})
    print(f"[selftest] synthetic n={len(y)}, subgroups={pd.unique(sev).tolist()}")
    summary = run_cv(X, y, info, "synthetic", cfg, folds=5, seeds=(42, 43))
    lac_cov = summary["LAC"]["coverage_mean"]; lac_fair = summary["LAC"]["fairness_mean"]
    g_cp_fair = summary["Global-CP"]["fairness_mean"]
    assert 0.80 < lac_cov < 0.98, f"LAC coverage {lac_cov:.3f} off nominal"
    assert lac_fair <= g_cp_fair + 0.06, f"LAC fairness {lac_fair:.3f} vs Global-CP {g_cp_fair:.3f}"
    print("[selftest] PASS (weak): pipeline correct; tune kappa/n_min_eff on real data")

def _save(obj, name):
    Path("results_lac_mri").mkdir(exist_ok=True)
    with open(Path("results_lac_mri") / name, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved] results_lac_mri/{name}")

# =============================================================================
# main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="LAC-MRI pipeline (revised v2)")
    ap.add_argument("--mri-root", default=CFG_INST.mri_root)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract"); p.add_argument("--dataset", required=True); p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("selftest"); p.set_defaults(fn=cmd_selftest)

    for name, fn in [("e1", run_e1), ("e2", run_e2), ("pointpred", run_pointpred),
                     ("e3", run_e3), ("e4", run_e4), ("e5", run_e5), ("e6", run_e6),
                     ("e0", run_e0), ("e0sweep", run_e0sweep), ("e7", run_e7),
                     ("e8grid", run_e8grid), ("e8demo", run_e8demo), ("e9", run_e9),
                     ("e10", run_e10), ("e11", run_e11), ("e12", run_e12),
                     ("e13", run_e13), ("e14", run_e14)]:
        p = sub.add_parser(name); p.set_defaults(fn=fn)
        p.add_argument("--meta-subjects", default=None)
        p.add_argument("--meta-ida", default=None)
        p.add_argument("--meta-updrs", default=None)
        p.add_argument("--features", default=None)
        p.add_argument("--features-ppmi", default="features_cache/PPMI_aal_features.csv")
        p.add_argument("--features-neurocon", default="features_cache/NEUROCON_aal_features.csv")
        p.add_argument("--features-taowu", default="features_cache/TAOWU_aal_features.csv")
        p.add_argument("--meta-neurocon", default=None)
        p.add_argument("--meta-taowu", default=None)
        if name in ("e1", "e3"):
            p.add_argument("--target", default="updrs", choices=["updrs", "age", "duration"])
        if name == "e1":
            p.add_argument("--folds", type=int, default=5); p.add_argument("--seeds", type=int, default=3)
        p.add_argument("--kappa", type=float, default=None)
        p.add_argument("--feat-suffix", default=None)
        p.add_argument("--n-min", type=int, default=None)
        if name == "e3":
            p.add_argument("--sites", default=None); p.add_argument("--min-site-n", type=int, default=30)
        if name == "e0":
            p.add_argument("--ratio", type=float, default=4.0)
            p.add_argument("--shift", type=float, default=0.0)   # 机制验证默认无偏移
            p.add_argument("--tail", default="normal", choices=["normal", "t3"])
            p.add_argument("--seeds", type=int, default=5)
        if name == "e0sweep":
            p.add_argument("--ratios", default="1,2,4,8")
            p.add_argument("--shifts", default="0,1.5")
            p.add_argument("--tails", default="normal,t3")
        if name == "e7":
            p.add_argument("--ks", default="0,10,15,20"); p.add_argument("--repeats", type=int, default=50)
            p.add_argument("--site", default="GE", choices=["NEUROCON", "GE"])
            p.add_argument("--protocol", default="pooled", choices=["pooled", "target"])
        if name == "e9":
            p.add_argument("--alphas", default="0.20,0.15,0.10,0.05")
        if name == "e8demo":
            p.add_argument("--fixed-mult", type=float, default=50.0)
            p.add_argument("--seeds", type=int, default=3)
        if name in ("e11", "e12"):
            p.add_argument("--gains", default="2,3,5,8")
            p.add_argument("--mode", default="pc1", choices=["pc1", "sparse"],
                           help="注入方向: pc1=稠密线性组合(最难) / sparse=随机5特征(局部)")

    args = ap.parse_args()
    CFG_INST.mri_root = args.mri_root
    if getattr(args, "kappa", None) is not None: CFG_INST.kappa = args.kappa
    if getattr(args, "n_min", None) is not None: CFG_INST.n_min_eff = args.n_min
    if getattr(args, "feat_suffix", None) is not None: CFG_INST.feat_suffix = args.feat_suffix
    t0 = time.time()
    args.fn(args)
    print(f"\n[elapsed] {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()