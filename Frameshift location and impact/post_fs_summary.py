#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_fs_summary.py

High-level summaries & stats from a single master_summary.tsv

INPUT
  master_summary.tsv (tab-delimited) with columns including at least:
    comparison, tag, gene, transcript, chrom, strand,
    window_start, window_end, direction,
    mfe_kcal, ensemble_kcal, mfe_freq,
    has_canonical, has_A6plus, has_U6plus, pqs_count, motif_hits,
    gc_frac, au_frac, cpg_obs, cpg_oe, entropy_nt, len_nt,
    cds_pos_mid_rel, cds_span_codons, cds_pos_start_rel, cds_pos_end_rel,
    aa_has_Px2, aa_has_Px3, aa_polyK6+, aa_polyR6+, aa_DE_to_P,
    rnafold_ok

USAGE
  python post_fs_summary.py ./post_fs_context/master_summary.tsv

OUTPUT (directory: post_fs_master_summary/)
  - combined/clean_master_summary.csv
      → same as input but with normalized timepoint/tag/direction and types

  - per_timepoint/global_dir_counts.csv
      → per timepoint × tag: counts and fractions of +1/-1

  - per_timepoint/global_feature_stats.csv
      → per timepoint × tag × direction: mean/median/IQR for numeric features

  - per_timepoint/up_vs_down_numeric_tests.csv
      → for each timepoint × feature: MW U-test + Cliff’s δ (UP vs DOWN)

  - per_timepoint/up_vs_down_motif_fisher.csv
      → for each timepoint × motif/aa-flag: Fisher test (UP vs DOWN)

  - per_timepoint/dir_vs_tag_fisher.csv
      → for each timepoint: 2×2 Fisher for direction (+1/-1) vs tag (UP/DOWN)

  - per_timepoint/dir_within_tag_numeric_tests.csv
      → for each timepoint × tag × feature: +1 vs -1 MW U + Cliff’s δ

  - per_timepoint/genes_per_timepoint.csv
      → gene-level summary: presence of UP/DOWN, +1/-1, mixed states, etc.

  - per_timepoint/dir_plus_vs_minus_numeric_tests.csv
      → for each timepoint × feature: +1 vs -1 MW U + Cliff’s δ (tags pooled)

  - per_timepoint/dir_plus_vs_minus_motif_fisher.csv
      → for each timepoint × motif/aa-flag: Fisher test (+1 vs -1, tags pooled)

  - per_timepoint/motif_type_counts_by_dir.csv
      → timepoint × direction × motif_type: raw counts/fractions from motif_hits

  - plots/<timepoint>_dir_ratio_UPvsDOWN.png
      → bar plot of +1/-1 counts in UP vs DOWN

  - plots/<timepoint>_mfe_box_UPvsDOWN.png
      → MFE boxplots by UP/DOWN

  - plots/<timepoint>_mfe_ecdf_UPvsDOWN.png
      → ECDFs of MFE by UP/DOWN
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- optional stats deps ----
_SCIPY_OK = True
_SM_OK = True
try:
    from scipy.stats import fisher_exact, mannwhitneyu
except Exception:
    _SCIPY_OK = False
try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    _SM_OK = False

# ----------------------------- helpers ---------------------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def ecdf(y):
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    if y.size == 0:
        return np.array([]), np.array([])
    x = np.sort(y)
    n = x.size
    f = np.arange(1, n+1) / n
    return x, f

def cliffs_delta(x, y):
    """
    Cliff's delta effect size in [-1, 1].
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    i = j = 0
    greater = less = 0
    while i < nx and j < ny:
        if x_sorted[i] > y_sorted[j]:
            greater += (nx - i)
            j += 1
        elif x_sorted[i] < y_sorted[j]:
            less += (ny - j)
            i += 1
        else:
            xv = x_sorted[i]
            i2 = i
            while i2 < nx and x_sorted[i2] == xv:
                i2 += 1
            j2 = j
            while j2 < ny and y_sorted[j2] == xv:
                j2 += 1
            i = i2
            j = j2
    den = nx * ny
    return (greater - less) / den if den > 0 else np.nan

def normalize_direction(series):
    """
    Normalize direction to '+1' / '-1' / other.
    Handles: 1, -1, '+1', '-1', '1', '-1', etc.
    """
    s = series.astype(str).str.strip()
    s = s.replace({
        "1": "+1",
        "+1": "+1",
        "-1": "-1"
    })
    return s

def parse_motif_hits(x):
    """
    Parse motif_hits column into a list of motif labels.
    Assumes ';' or ','-separated labels. Returns a clean list of strings.
    """
    if pd.isna(x):
        return []
    # Replace commas with semicolons and split
    tokens = str(x).replace(",", ";").split(";")
    tokens = [t.strip() for t in tokens if t.strip() not in ("", "NA", "None", "nan")]
    return tokens

# ----------------------------- plotting --------------------------------------

def plot_dir_ratio(df_counts_tp, out_png):
    """
    Side-by-side bars of +1 and -1 counts for UP vs DOWN for a given timepoint.
    expects df_counts_tp columns: tag, plus1_count, minus1_count
    """
    groups = ["UP", "DOWN"]
    counts = []
    for g in groups:
        sub = df_counts_tp[df_counts_tp["tag"] == g]
        if sub.empty:
            counts.append((0, 0))
        else:
            r = sub.iloc[0]
            counts.append((int(r.get("plus1_count", 0)),
                           int(r.get("minus1_count", 0))))

    idx = np.arange(len(groups))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(idx - w/2, [c[0] for c in counts], width=w, label="+1")
    ax.bar(idx + w/2, [c[1] for c in counts], width=w, label="-1")

    ax.set_xticks(idx)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Number of loci")
    ax.set_title("Frameshift direction counts (UP vs DOWN)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)

def plot_mfe_box(df_tp_loci, out_png):
    """
    Box plots of MFE by tag (UP vs DOWN).
    """
    groups = ["UP", "DOWN"]
    data = []
    for g in groups:
        vals = pd.to_numeric(
            df_tp_loci.loc[df_tp_loci["tag"] == g, "mfe_kcal"],
            errors="coerce"
        ).dropna().values
        data.append(vals)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.boxplot(data, labels=groups, showmeans=True, showfliers=False)
    ax.set_ylabel("MFE (kcal/mol)")
    ax.set_title("RNAfold MFE distribution (UP vs DOWN)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)

def plot_mfe_ecdf(df_tp_loci, out_png):
    """
    ECDF of MFE for UP vs DOWN groups.
    """
    groups = ["UP", "DOWN"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for g in groups:
        y = pd.to_numeric(
            df_tp_loci.loc[df_tp_loci["tag"] == g, "mfe_kcal"],
            errors="coerce"
        ).values
        x, f = ecdf(y)
        if x.size > 0:
            ax.plot(x, f, label=g)
    ax.set_xlabel("MFE (kcal/mol)")
    ax.set_ylabel("ECDF")
    ax.set_title("RNAfold MFE ECDF (UP vs DOWN)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)
    
def plot_feature_box_two_groups(df_tp, feature, group_col, groups, out_png,
                                ylabel=None, title=None):
    """
    Generic 2-group boxplot for a numeric feature.

    group_col: column name ('tag' or 'direction')
    groups:    list/tuple of two labels, e.g. ['UP', 'DOWN'] or ['+1', '-1']
    """
    data = []
    labels = []
    for g in groups:
        vals = pd.to_numeric(
            df_tp.loc[df_tp[group_col] == g, feature],
            errors="coerce"
        ).dropna().values
        if vals.size > 0:
            data.append(vals)
            labels.append(g)

    if len(data) < 2:
        return  # nothing to plot

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.boxplot(data, labels=labels, showmeans=True, showfliers=False)
    ax.set_ylabel(ylabel or feature)
    if title is None:
        title = f"{feature}: {groups[0]} vs {groups[1]}"
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)


# ----------------------------- main ------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python post_fs_master_summary.py master_summary.tsv")
        sys.exit(1)

    in_path = sys.argv[1]

    out_root   = ensure_dir("post_fs_master_summary")
    out_comb   = ensure_dir(os.path.join(out_root, "combined"))
    out_tp_dir = ensure_dir(os.path.join(out_root, "per_timepoint"))
    out_plots  = ensure_dir(os.path.join(out_root, "plots"))

    # ---------- 1) Load master_summary & normalize ----------
    try:
        df = pd.read_csv(in_path, sep="\t")
    except Exception:
        df = pd.read_csv(in_path)

    # Column compatibility
    if "comparison" not in df.columns and "comparison_tag" in df.columns:
        df["comparison"] = df["comparison_tag"]
    if "tag" not in df.columns and "group_tag" in df.columns:
        df["tag"] = df["group_tag"]

    # timepoint: part before "_vs_"
    df["timepoint"] = df["comparison"].astype(str).str.split("_vs_", n=1).str[0]
    df["tag"] = df["tag"].astype(str).str.upper()

    # Normalize direction to '+1' / '-1'
    if "direction" in df.columns:
        df["direction_raw"] = df["direction"]
        df["direction"] = normalize_direction(df["direction"])
    else:
        df["direction"] = np.nan

    # Standardize booleans
    bool_cols = [
        "has_canonical", "has_A6plus", "has_U6plus", "rnafold_ok",
        "aa_has_Px2", "aa_has_Px3", "aa_polyK6+", "aa_polyR6+", "aa_DE_to_P"
    ]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(bool)

    # Numeric features we want to compare
    numeric_features = [
        "mfe_kcal",
        "gc_frac", "au_frac",
        "cpg_obs", "cpg_oe",
        "entropy_nt",
        "len_nt",
        "cds_pos_mid_rel", "cds_span_codons",
        "cds_pos_start_rel", "cds_pos_end_rel"
    ]
    for c in numeric_features:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Save cleaned combined table
    df.to_csv(os.path.join(out_comb, "clean_master_summary.csv"), index=False)

    # ---------- 2) Global direction counts per timepoint × tag ----------
    def dir_counts(g):
        d = g["direction"].astype(str).str.strip()
        mask = d.isin(["+1", "-1"])
        plus1 = int((d[mask] == "+1").sum())
        minus1 = int((d[mask] == "-1").sum())
        total = plus1 + minus1
        return pd.Series({
            "n_loci": int(len(g)),
            "plus1_count": plus1,
            "minus1_count": minus1,
            "plus1_frac": plus1 / total if total else np.nan,
            "minus1_frac": minus1 / total if total else np.nan
        })

    counts_df = (
        df.groupby(["timepoint", "tag"], dropna=False)
        .apply(dir_counts)
        .reset_index()
        .sort_values(["timepoint", "tag"])
    )
    counts_df.to_csv(os.path.join(out_tp_dir, "global_dir_counts.csv"), index=False)

    # ---------- 3) Global numeric feature stats per timepoint × tag × direction ----------
    stats_rows = []
    for (tp, tag, direction), g in df.groupby(["timepoint", "tag", "direction"], dropna=False):
        for feat in numeric_features:
            if feat not in g.columns:
                continue
            x = g[feat].dropna()
            if x.empty:
                stats_rows.append({
                    "timepoint": tp,
                    "tag": tag,
                    "direction": direction,
                    "feature": feat,
                    "n": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                    "iqr": np.nan
                })
            else:
                q25 = x.quantile(0.25)
                q75 = x.quantile(0.75)
                stats_rows.append({
                    "timepoint": tp,
                    "tag": tag,
                    "direction": direction,
                    "feature": feat,
                    "n": int(x.size),
                    "mean": float(x.mean()),
                    "median": float(x.median()),
                    "q25": float(q25),
                    "q75": float(q75),
                    "iqr": float(q75 - q25)
                })

    feature_stats_df = pd.DataFrame(stats_rows)
    feature_stats_df.to_csv(os.path.join(out_tp_dir, "global_feature_stats.csv"),
                            index=False)

    # ---------- 4) UP vs DOWN numeric tests per timepoint ----------
    updown_rows = []
    for tp, sub in df.groupby("timepoint"):
        for feat in numeric_features:
            if feat not in sub.columns:
                continue
            up_vals = sub.loc[sub["tag"] == "UP", feat].dropna().values
            dn_vals = sub.loc[sub["tag"] == "DOWN", feat].dropna().values

            if len(up_vals) >= 3 and len(dn_vals) >= 3 and _SCIPY_OK:
                try:
                    U, p = mannwhitneyu(up_vals, dn_vals, alternative="two-sided")
                except Exception:
                    U, p = (np.nan, np.nan)
                d = cliffs_delta(up_vals, dn_vals)
            else:
                U, p, d = (np.nan, np.nan, np.nan)

            updown_rows.append({
                "timepoint": tp,
                "feature": feat,
                "n_UP": int(len(up_vals)),
                "n_DOWN": int(len(dn_vals)),
                "U_stat": U,
                "p_raw": p,
                "cliffs_delta_UP_minus_DOWN": d
            })

    updown_df = pd.DataFrame(updown_rows).sort_values(["timepoint", "feature"])
    if _SM_OK and "p_raw" in updown_df.columns:
        _, padj, _, _ = multipletests(updown_df["p_raw"].fillna(1.0),
                                      method="fdr_bh")
        updown_df["p_adj_bh"] = padj
    else:
        updown_df["p_adj_bh"] = np.nan

    updown_df.to_csv(os.path.join(out_tp_dir, "up_vs_down_numeric_tests.csv"),
                     index=False)

    # ---------- 5) UP vs DOWN motif / aa-feature Fisher tests ----------
    motif_bool_cols = [
        ("canonical", "has_canonical"),
        ("A6plus", "has_A6plus"),
        ("U6plus", "has_U6plus"),
        ("aa_has_Px2", "aa_has_Px2"),
        ("aa_has_Px3", "aa_has_Px3"),
        ("aa_polyK6plus", "aa_polyK6+"),
        ("aa_polyR6plus", "aa_polyR6+"),
        ("aa_DE_to_P", "aa_DE_to_P")
    ]

    motif_tests = []
    for tp, sub in df.groupby("timepoint"):
        D = sub[sub["tag"].isin(["UP", "DOWN"])].copy()
        if D.empty or not _SCIPY_OK:
            for label, col in motif_bool_cols:
                motif_tests.append({
                    "timepoint": tp,
                    "motif": label,
                    "UP_pos": np.nan, "UP_neg": np.nan,
                    "DOWN_pos": np.nan, "DOWN_neg": np.nan,
                    "odds_ratio": np.nan, "p_raw": np.nan
                })
            continue

        for label, col in motif_bool_cols:
            if col not in D.columns:
                motif_tests.append({
                    "timepoint": tp,
                    "motif": label,
                    "UP_pos": np.nan, "UP_neg": np.nan,
                    "DOWN_pos": np.nan, "DOWN_neg": np.nan,
                    "odds_ratio": np.nan, "p_raw": np.nan
                })
                continue

            up_pos = int(D.loc[D["tag"] == "UP", col].sum())
            up_tot = int((D["tag"] == "UP").sum())
            dn_pos = int(D.loc[D["tag"] == "DOWN", col].sum())
            dn_tot = int((D["tag"] == "DOWN").sum())
            up_neg = up_tot - up_pos
            dn_neg = dn_tot - dn_pos

            if up_tot > 0 and dn_tot > 0:
                try:
                    OR, p = fisher_exact([[up_pos, up_neg],
                                          [dn_pos, dn_neg]],
                                         alternative="two-sided")
                except Exception:
                    OR, p = (np.nan, np.nan)
            else:
                OR, p = (np.nan, np.nan)

            motif_tests.append({
                "timepoint": tp,
                "motif": label,
                "UP_pos": up_pos, "UP_neg": up_neg,
                "DOWN_pos": dn_pos, "DOWN_neg": dn_neg,
                "odds_ratio": OR, "p_raw": p
            })

    motif_tests_df = pd.DataFrame(motif_tests).sort_values(["timepoint", "motif"])
    if _SM_OK and "p_raw" in motif_tests_df.columns:
        _, padj, _, _ = multipletests(motif_tests_df["p_raw"].fillna(1.0),
                                      method="fdr_bh")
        motif_tests_df["p_adj_bh"] = padj
    else:
        motif_tests_df["p_adj_bh"] = np.nan

    motif_tests_df.to_csv(os.path.join(out_tp_dir, "up_vs_down_motif_fisher.csv"),
                          index=False)

    # ---------- 6) Direction (+1/-1) vs tag (UP/DOWN) Fisher per timepoint ----------
    dir_tests = []
    for tp, sub in df.groupby("timepoint"):
        D = sub[sub["tag"].isin(["UP", "DOWN"])].copy()
        if D.empty or not _SCIPY_OK:
            dir_tests.append({
                "timepoint": tp,
                "UP_plus1": np.nan, "UP_minus1": np.nan,
                "DOWN_plus1": np.nan, "DOWN_minus1": np.nan,
                "odds_ratio": np.nan, "p_raw": np.nan
            })
            continue

        D["direction"] = D["direction"].astype(str).str.strip()
        DD = D[D["direction"].isin(["+1", "-1"])]

        up_p = int(((DD["tag"] == "UP") & (DD["direction"] == "+1")).sum())
        up_m = int(((DD["tag"] == "UP") & (DD["direction"] == "-1")).sum())
        dn_p = int(((DD["tag"] == "DOWN") & (DD["direction"] == "+1")).sum())
        dn_m = int(((DD["tag"] == "DOWN") & (DD["direction"] == "-1")).sum())

        if (up_p + up_m) > 0 and (dn_p + dn_m) > 0:
            try:
                OR, p = fisher_exact([[up_p, up_m], [dn_p, dn_m]],
                                     alternative="two-sided")
            except Exception:
                OR, p = (np.nan, np.nan)
        else:
            OR, p = (np.nan, np.nan)

        dir_tests.append({
            "timepoint": tp,
            "UP_plus1": up_p, "UP_minus1": up_m,
            "DOWN_plus1": dn_p, "DOWN_minus1": dn_m,
            "odds_ratio": OR, "p_raw": p
        })

    dir_df = pd.DataFrame(dir_tests).sort_values("timepoint")
    if _SM_OK and "p_raw" in dir_df.columns:
        _, padj, _, _ = multipletests(dir_df["p_raw"].fillna(1.0),
                                      method="fdr_bh")
        dir_df["p_adj_bh"] = padj
    else:
        dir_df["p_adj_bh"] = np.nan

    dir_df.to_csv(os.path.join(out_tp_dir, "dir_vs_tag_fisher.csv"), index=False)

    # ---------- 7) +1 vs -1 numeric tests within each tag (UP/DOWN) ----------
    dir_within_rows = []
    for (tp, tag), sub in df.groupby(["timepoint", "tag"]):
        for feat in numeric_features:
            if feat not in sub.columns:
                continue
            d = sub[sub["direction"].isin(["+1", "-1"])].copy()
            plus_vals = d.loc[d["direction"] == "+1", feat].dropna().values
            minus_vals = d.loc[d["direction"] == "-1", feat].dropna().values

            if len(plus_vals) >= 3 and len(minus_vals) >= 3 and _SCIPY_OK:
                try:
                    U, p = mannwhitneyu(plus_vals, minus_vals, alternative="two-sided")
                except Exception:
                    U, p = (np.nan, np.nan)
                d_eff = cliffs_delta(plus_vals, minus_vals)
            else:
                U, p, d_eff = (np.nan, np.nan, np.nan)

            dir_within_rows.append({
                "timepoint": tp,
                "tag": tag,
                "feature": feat,
                "n_plus1": int(len(plus_vals)),
                "n_minus1": int(len(minus_vals)),
                "U_stat": U,
                "p_raw": p,
                "cliffs_delta_plus_minus": d_eff
            })

    dir_within_df = pd.DataFrame(dir_within_rows).sort_values(
        ["timepoint", "tag", "feature"]
    )
    if _SM_OK and "p_raw" in dir_within_df.columns:
        _, padj, _, _ = multipletests(dir_within_df["p_raw"].fillna(1.0),
                                      method="fdr_bh")
        dir_within_df["p_adj_bh"] = padj
    else:
        dir_within_df["p_adj_bh"] = np.nan

    dir_within_df.to_csv(
        os.path.join(out_tp_dir, "dir_within_tag_numeric_tests.csv"),
        index=False
    )

    # ---------- 8) Gene-level summary ----------
    gene_rows = []
    for (tp, gene), g in df.groupby(["timepoint", "gene"]):
        tags = set(g["tag"].dropna().astype(str))
        dirs = set(g["direction"].dropna().astype(str))

        gene_rows.append({
            "timepoint": tp,
            "gene": gene,
            "n_loci": int(len(g)),
            "tags_present": ",".join(sorted(tags)),
            "directions_present": ",".join(sorted(dirs)),
            "has_UP": "UP" in tags,
            "has_DOWN": "DOWN" in tags,
            "has_both_UP_and_DOWN": ("UP" in tags) and ("DOWN" in tags),
            "has_plus1": "+1" in dirs,
            "has_minus1": "-1" in dirs,
            "has_both_directions": ("+1" in dirs) and ("-1" in dirs)
        })

    gene_df = pd.DataFrame(gene_rows).sort_values(["timepoint", "gene"])
    gene_df.to_csv(os.path.join(out_tp_dir, "genes_per_timepoint.csv"),
                   index=False)

    # ---------- 9) Simple plots: direction ratio & MFE UP vs DOWN ----------
    timepoints = sorted(df["timepoint"].dropna().unique().tolist())

    for tp in timepoints:
        sub_counts = counts_df[counts_df["timepoint"] == tp]
        if not sub_counts.empty:
            plot_dir_ratio(
                sub_counts,
                os.path.join(out_plots, f"{tp}_dir_ratio_UPvsDOWN.png")
            )

        sub_loci = df[(df["timepoint"] == tp) &
                      (df["tag"].isin(["UP", "DOWN"]))]
        if not sub_loci.empty and "mfe_kcal" in sub_loci.columns:
            plot_mfe_box(
                sub_loci,
                os.path.join(out_plots, f"{tp}_mfe_box_UPvsDOWN.png")
            )
            plot_mfe_ecdf(
                sub_loci,
                os.path.join(out_plots, f"{tp}_mfe_ecdf_UPvsDOWN.png")
            )

    # ---------- 10) +1 vs -1 numeric tests per timepoint (tags pooled) ----------
    dir_tp_rows = []
    for tp, sub in df.groupby("timepoint"):
        D = sub[sub["direction"].isin(["+1", "-1"])].copy()
        if D.empty:
            continue
        for feat in numeric_features:
            if feat not in D.columns:
                continue
            plus_vals = D.loc[D["direction"] == "+1", feat].dropna().values
            minus_vals = D.loc[D["direction"] == "-1", feat].dropna().values

            if len(plus_vals) >= 3 and len(minus_vals) >= 3 and _SCIPY_OK:
                try:
                    U, p = mannwhitneyu(plus_vals, minus_vals, alternative="two-sided")
                except Exception:
                    U, p = (np.nan, np.nan)
                d_eff = cliffs_delta(plus_vals, minus_vals)
            else:
                U, p, d_eff = (np.nan, np.nan, np.nan)

            dir_tp_rows.append({
                "timepoint": tp,
                "feature": feat,
                "n_plus1": int(len(plus_vals)),
                "n_minus1": int(len(minus_vals)),
                "U_stat": U,
                "p_raw": p,
                "cliffs_delta_plus_minus": d_eff
            })

    dir_tp_df = pd.DataFrame(dir_tp_rows).sort_values(["timepoint", "feature"])
    if not dir_tp_df.empty and _SM_OK and "p_raw" in dir_tp_df.columns:
        _, padj, _, _ = multipletests(dir_tp_df["p_raw"].fillna(1.0),
                                      method="fdr_bh")
        dir_tp_df["p_adj_bh"] = padj
    else:
        dir_tp_df["p_adj_bh"] = np.nan

    dir_tp_df.to_csv(
        os.path.join(out_tp_dir, "dir_plus_vs_minus_numeric_tests.csv"),
        index=False
    )
    
    
        # ---------- 11) +1 vs -1 numeric tests (pooled tags, per timepoint) ----------
    plusminus_rows = []
    for tp, sub in df.groupby("timepoint"):
        d = sub[sub["direction"].isin(["+1", "-1"])].copy()
        if d.empty:
            continue
        for feat in numeric_features:
            if feat not in d.columns:
                continue
            plus_vals = d.loc[d["direction"] == "+1", feat].dropna().values
            minus_vals = d.loc[d["direction"] == "-1", feat].dropna().values
            if len(plus_vals) >= 3 and len(minus_vals) >= 3 and _SCIPY_OK:
                try:
                    U, p = mannwhitneyu(plus_vals, minus_vals, alternative="two-sided")
                except Exception:
                    U, p = (np.nan, np.nan)
                d_eff = cliffs_delta(plus_vals, minus_vals)
            else:
                U, p, d_eff = (np.nan, np.nan, np.nan)

            plusminus_rows.append({
                "timepoint": tp,
                "feature": feat,
                "n_plus1": int(len(plus_vals)),
                "n_minus1": int(len(minus_vals)),
                "U_stat": U,
                "p_raw": p,
                "cliffs_delta_plus_minus": d_eff
            })

    plusminus_df = pd.DataFrame(plusminus_rows).sort_values(
        ["timepoint", "feature"]
    )
    if _SM_OK and "p_raw" in plusminus_df.columns:
        _, padj, _, _ = multipletests(plusminus_df["p_raw"].fillna(1.0),
                                      method="fdr_bh")
        plusminus_df["p_adj_bh"] = padj
    else:
        plusminus_df["p_adj_bh"] = np.nan

    plusminus_path = os.path.join(out_tp_dir, "dir_plus_vs_minus_numeric_tests.csv")
    plusminus_df.to_csv(plusminus_path, index=False)


    # ---------- 12) +1 vs -1 motif/AA Fisher tests per timepoint (tags pooled) ----------
    dir_motif_rows = []
    for tp, sub in df.groupby("timepoint"):
        D = sub[sub["direction"].isin(["+1", "-1"])].copy()
        if D.empty or not _SCIPY_OK:
            for label, col in motif_bool_cols:
                dir_motif_rows.append({
                    "timepoint": tp,
                    "motif": label,
                    "plus1_pos": np.nan, "plus1_neg": np.nan,
                    "minus1_pos": np.nan, "minus1_neg": np.nan,
                    "odds_ratio": np.nan, "p_raw": np.nan
                })
            continue

        for label, col in motif_bool_cols:
            if col not in D.columns:
                dir_motif_rows.append({
                    "timepoint": tp,
                    "motif": label,
                    "plus1_pos": np.nan, "plus1_neg": np.nan,
                    "minus1_pos": np.nan, "minus1_neg": np.nan,
                    "odds_ratio": np.nan, "p_raw": np.nan
                })
                continue

            p_pos = int(D.loc[D["direction"] == "+1", col].sum())
            p_tot = int((D["direction"] == "+1").sum())
            m_pos = int(D.loc[D["direction"] == "-1", col].sum())
            m_tot = int((D["direction"] == "-1").sum())
            p_neg = p_tot - p_pos
            m_neg = m_tot - m_pos

            if p_tot > 0 and m_tot > 0:
                try:
                    OR, p = fisher_exact([[p_pos, p_neg],
                                          [m_pos, m_neg]],
                                         alternative="two-sided")
                except Exception:
                    OR, p = (np.nan, np.nan)
            else:
                OR, p = (np.nan, np.nan)

            dir_motif_rows.append({
                "timepoint": tp,
                "motif": label,
                "plus1_pos": p_pos, "plus1_neg": p_neg,
                "minus1_pos": m_pos, "minus1_neg": m_neg,
                "odds_ratio": OR, "p_raw": p
            })

    dir_motif_df = pd.DataFrame(dir_motif_rows).sort_values(["timepoint", "motif"])
    if _SM_OK and "p_raw" in dir_motif_df.columns:
        _, padj, _, _ = multipletests(dir_motif_df["p_raw"].fillna(1.0),
                                      method="fdr_bh")
        dir_motif_df["p_adj_bh"] = padj
    else:
        dir_motif_df["p_adj_bh"] = np.nan

    dir_motif_df.to_csv(
        os.path.join(out_tp_dir, "dir_plus_vs_minus_motif_fisher.csv"),
        index=False
    )

    # ---------- 13) Motif content & type from motif_hits (per timepoint × direction) ----------
    if "motif_hits" in df.columns:
        motif_long = []
        for idx, row in df.iterrows():
            tp = row.get("timepoint")
            direction = str(row.get("direction"))
            if direction not in ["+1", "-1"]:
                continue
            motifs = parse_motif_hits(row.get("motif_hits"))
            if not motifs:
                # still record "none" so we know how many loci had no motif
                motif_long.append({
                    "timepoint": tp,
                    "direction": direction,
                    "motif_type": "NONE"
                })
            else:
                for m in motifs:
                    motif_long.append({
                        "timepoint": tp,
                        "direction": direction,
                        "motif_type": m
                    })

        motif_long_df = pd.DataFrame(motif_long)
        if not motif_long_df.empty:
            summary_rows = []
            for (tp, direction), g in motif_long_df.groupby(["timepoint", "direction"]):
                total = int(g.shape[0])
                for motif_type, gg in g.groupby("motif_type"):
                    n = int(gg.shape[0])
                    summary_rows.append({
                        "timepoint": tp,
                        "direction": direction,
                        "motif_type": motif_type,
                        "count": n,
                        "fraction": n / total if total else np.nan
                    })
            motif_type_summary = pd.DataFrame(summary_rows).sort_values(
                ["timepoint", "direction", "motif_type"]
            )
        else:
            motif_type_summary = pd.DataFrame(
                columns=["timepoint", "direction", "motif_type", "count", "fraction"]
            )

        motif_type_summary.to_csv(
            os.path.join(out_tp_dir, "motif_type_counts_by_dir.csv"),
            index=False
        )
        
            # ---------- 14) Top effect-size features & quick boxplots ----------
    topN = 5

    # (a) UP vs DOWN – rank by |Cliff's δ|
    updown_top = updown_df.copy()
    updown_top["abs_delta"] = updown_top["cliffs_delta_UP_minus_DOWN"].abs()
    updown_top = (
        updown_top.sort_values(["timepoint", "abs_delta"], ascending=[True, False])
                  .groupby("timepoint", as_index=False)
                  .head(topN)
    )
    updown_top.to_csv(
        os.path.join(out_tp_dir, "top_effects_up_vs_down_numeric.csv"),
        index=False
    )

    # Make boxplots for these top contrasts
    for _, row in updown_top.iterrows():
        tp = row["timepoint"]
        feat = row["feature"]
        sub_tp = df[(df["timepoint"] == tp) & (df["tag"].isin(["UP", "DOWN"]))]
        if sub_tp.empty:
            continue
        out_png = os.path.join(
            out_plots,
            f"{tp}_UPvsDOWN_{feat}_box.png"
        )
        plot_feature_box_two_groups(
            sub_tp, feature=feat,
            group_col="tag", groups=["UP", "DOWN"],
            out_png=out_png,
            ylabel=feat,
            title=f"{feat} (UP vs DOWN, {tp})"
        )

    # (b) +1 vs -1 pooled – rank by |Cliff's δ|
    plusminus_top = plusminus_df.copy()
    plusminus_top["abs_delta"] = plusminus_top["cliffs_delta_plus_minus"].abs()
    plusminus_top = (
        plusminus_top.sort_values(["timepoint", "abs_delta"], ascending=[True, False])
                     .groupby("timepoint", as_index=False)
                     .head(topN)
    )
    plusminus_top.to_csv(
        os.path.join(out_tp_dir, "top_effects_plus_vs_minus_numeric.csv"),
        index=False
    )

    # Boxplots for top +1 vs -1 contrasts (pooled tags)
    for _, row in plusminus_top.iterrows():
        tp = row["timepoint"]
        feat = row["feature"]
        sub_tp = df[(df["timepoint"] == tp) &
                    (df["direction"].isin(["+1", "-1"]))]
        if sub_tp.empty:
            continue
        out_png = os.path.join(
            out_plots,
            f"{tp}_plus_vs_minus_{feat}_box.png"
        )
        plot_feature_box_two_groups(
            sub_tp, feature=feat,
            group_col="direction", groups=["+1", "-1"],
            out_png=out_png,
            ylabel=feat,
            title=f"{feat} (+1 vs -1, {tp})"
        )


    print("Done. Outputs written to:", out_root)
    
        # ---------- 15) Detailed mixed-direction genes ----------
    mixed_rows = []
    for (tp, gene), g in df.groupby(["timepoint", "gene"]):
        dirs = g["direction"].astype(str).str.strip()
        tags = g["tag"].astype(str).str.strip()

        plus_n = int((dirs == "+1").sum())
        minus_n = int((dirs == "-1").sum())
        if plus_n == 0 or minus_n == 0:
            continue  # only keep true mixed-direction genes

        up_n = int((tags == "UP").sum())
        down_n = int((tags == "DOWN").sum())
        total_dir = plus_n + minus_n

        mixed_rows.append({
            "timepoint": tp,
            "gene": gene,
            "total_loci": int(len(g)),
            "total_dir_loci": total_dir,
            "plus1_count": plus_n,
            "minus1_count": minus_n,
            "plus1_frac": plus_n / total_dir if total_dir else np.nan,
            "UP_loci": up_n,
            "DOWN_loci": down_n,
            "has_UP_and_DOWN": (up_n > 0 and down_n > 0),
        })

    mixed_df = pd.DataFrame(mixed_rows)
    mixed_df = mixed_df.sort_values(
        ["timepoint", "total_dir_loci", "plus1_frac"],
        ascending=[True, False, True]
    )
    mixed_df.to_csv(
        os.path.join(out_tp_dir, "mixed_direction_genes_detail.csv"),
        index=False
    )
    
        # ---------- 16) Motif class summary by direction & timepoint ----------
    if "motif_super" in df.columns:
        motif_rows = []
        for (tp, direction, msuper), g in df.groupby(
                ["timepoint", "direction", "motif_super"], dropna=False):
            n = len(g)
            motif_rows.append({
                "timepoint": tp,
                "direction": direction,
                "motif_super": msuper,
                "n_loci": int(n)
            })

        motif_df = pd.DataFrame(motif_rows)
        if not motif_df.empty:
            motif_df["total_in_tp_dir"] = (
                motif_df.groupby(["timepoint", "direction"])["n_loci"]
                        .transform("sum")
            )
            motif_df["frac_in_tp_dir"] = (
                motif_df["n_loci"] / motif_df["total_in_tp_dir"]
            )
            motif_df = motif_df.sort_values(
                ["timepoint", "direction", "n_loci"],
                ascending=[True, True, False]
            )
            motif_df.to_csv(
                os.path.join(out_tp_dir, "motif_super_counts_by_dir.csv"),
                index=False
            )




if __name__ == "__main__":
    main()

