#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_fs_context.py  —  Context + RNAfold analysis for frameshift loci.

Features
--------
1) Quantify and visualize the +1/–1 ratio per comparison (and per UP/DOWN tag)
2) Extract fixed-size context around each locus (strand-aware), write FASTA:
     post_fs_context/<cmp>/contexts_all.fa
     post_fs_context/<cmp>/contexts_plus1.fa
     post_fs_context/<cmp>/contexts_minus1.fa
3) Scan canonical and extended motifs (slippery, poly-A/U, proline-rich)
4) Run RNAfold on the locus window (MFE, ensemble energy, MFE frequency)
5) Compute G-quadruplex hits, sequence composition, CDS position, AA-context
6) Export FASTAs for MEME / downstream motif discovery

USAGE
-----
  python post_fs_context.py config.yaml frameshift_loci/loci_calls

Inputs
------
  - config.yaml  (must contain: gtf, genome_fasta; optional: max_workers, flank_nt, rnafold_temp)
  - loci_dir containing TSVs from frameshift_loci.py (e.g., H6_vs_Sham_UP_loci.tsv)

Outputs (under post_fs_context/)
--------------------------------
  plots/direction_ratio_per_comparison.png
  <cmp>/contexts_all.fa
  <cmp>/contexts_plus1.fa
  <cmp>/contexts_minus1.fa
  <cmp>/motif_rnafold.tsv
  master_summary.tsv
"""

import os, sys, re, glob, math, random, tempfile
import yaml, logging, subprocess as sp
import numpy as np
import pandas as pd
import pysam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import isnan

# ----------------------------- Logging ---------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("post_fs_context")

# ----------------------------- IO schema -------------------------------------

UNIFIED_OUT_COLS = [
    "comparison","tag","gene","transcript","chrom","strand",
    "window_start","window_end","direction",
    # RNAfold structure/ensemble
    "mfe_kcal","ensemble_kcal","mfe_freq",
    # motifs, G4, composition, cds-pos
    "has_canonical","has_A6plus","has_U6plus","pqs_count","motif_hits",
    "gc_frac","au_frac","cpg_obs","cpg_oe","entropy_nt","len_nt",
    "cds_pos_mid_rel","cds_span_codons","cds_pos_start_rel","cds_pos_end_rel",
    # aa-context
    "aa_has_Px2","aa_has_Px3","aa_polyK6+","aa_polyR6+","aa_DE_to_P",
    # housekeeping
    "rnafold_ok"
]

# ----------------------------- IO helpers ------------------------------------

def read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _infer_cmp_and_tag(tsv_path):
    """
    Examples:
      H6_vs_Sham_UP_loci.tsv   -> comparison="H6_vs_Sham", tag="UP"
      H6_vs_Sham_DOWN_loci.tsv -> comparison="H6_vs_Sham", tag="DOWN"
      H6_vs_Sham_ALL_loci.tsv  -> comparison="H6_vs_Sham", tag="ALL"
      H6_vs_Sham_loci.tsv      -> comparison="H6_vs_Sham", tag="ALL"
    """
    base = os.path.basename(tsv_path).replace(".tsv", "")
    if base.endswith("_loci"):
        base = base[:-5]
    m = re.match(r"(.+_vs_.+?)_(UP|DOWN|ALL)$", base)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"(.+_vs_.+)$", base)
    if m:
        return m.group(1), "ALL"
    return base, "ALL"

def load_loci_tsvs(loci_dir):
    tsvs = sorted(glob.glob(os.path.join(loci_dir, "*_vs_*_loci.tsv")))
    if not tsvs:
        raise FileNotFoundError(f"No loci TSV files found in {loci_dir}")
    dfs = []
    req = {"gene","transcript","chrom","strand",
           "genome_start","genome_end","direction",
           "locus_codon_start","locus_codon_end"}
    for t in tsvs:
        df = pd.read_csv(t, sep="\t")
        missing = req - set(df.columns)
        if missing:
            logger.warning(f"{os.path.basename(t)} missing cols {missing}; proceeding.")
        cmp_name, tag = _infer_cmp_and_tag(t)
        df["comparison"] = cmp_name
        df["tag"] = tag
        df["_src_file"] = os.path.basename(t)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out["direction"] = out["direction"].astype(str).replace(
        {"1":"+1", "+1":"+1", "-1":"-1"}
    )
    return out

def cds_pos_features(row):
    """Return dict with positional context if codon indices exist; else {}."""
    try:
        n_cod = int(row.get("n_codons"))
        i = int(row.get("locus_codon_start"))
        j = int(row.get("locus_codon_end"))
        if n_cod > 0 and 0 <= i <= j < n_cod:
            mid = (i + j) / 2.0
            return {
                "cds_pos_mid_rel": float(mid / n_cod),
                "cds_span_codons": int(j - i + 1),
                "cds_pos_start_rel": float(i / n_cod),
                "cds_pos_end_rel": float(j / n_cod),
            }
    except Exception:
        pass
    return {}

# ----------------------------- FASTA helpers ---------------------------------

def revcomp(seq):
    tr = str.maketrans("ACGTUacgtuNn", "TGCAAugcaaNn")
    rc = seq.translate(tr)[::-1]
    return rc.replace("T","U").replace("t","u")

def to_rna(seq):
    return seq.upper().replace("T","U")

def fetch_window(fa: pysam.FastaFile, chrom, start, end, strand, flank=20):
    """
    start/end are 0-based half-open genomic coords of the called locus window.
    Extend by ±flank; clamp to reference bounds. Return RNA alphabet (U).
    """
    if chrom not in fa.references:
        raise ValueError(f"Chromosome {chrom} not in FASTA")
    rlen = fa.get_reference_length(chrom)
    g_start = max(0, int(start) - int(flank))
    g_end   = min(rlen, max(int(end) + int(flank), int(start) + 1))
    seq = fa.fetch(chrom, g_start, g_end)
    if strand == "+":
        return to_rna(seq), g_start, g_end
    else:
        return revcomp(seq), g_start, g_end

# ----------------------------- Motif scanning --------------------------------

def scan_slippery_extended(seq_rna):
    """
    Returns list[(class, pos, kmer)] including:
      - Canonical −1 heptamers: XXXYYYZ; viral-like (e.g., GGGAAAC)
      - Homopolymers A6+/U6+
      - +1 candidates: >=3 consecutive Pro codons, >=3 consecutive AAA codons
    """
    s = seq_rna
    hits = []
    L = len(s)

    # −1 canonical-ish heptamers
    for i in range(0, L - 6):
        h = s[i:i+7]
        X, Y, Z = h[:3], h[3:6], h[6]
        if X[0] == X[1] == X[2] and Y[0] == Y[1] == Y[2] and (Y[0] in "AU") and (Z in "ACU"):
            hits.append(("canon_heptamer", i, h))
        if X in ("GGG", "AAA", "UUU") and Y in ("AAA", "UUU") and Z in "ACU":
            hits.append(("viral_like", i, h))

    # +1 crude proxies — require runs of >=3 consecutive codons
    pro_codons = {"CCC", "CCU", "CCA", "CCG"}
    run_len_codons = 3
    run_len_nt = 3 * run_len_codons

    # scan in nt-space but enforce codon contiguity: cod1, cod2, cod3 = i, i+3, i+6
    for i in range(0, L - run_len_nt + 1):
        cod1 = s[i:i+3]
        cod2 = s[i+3:i+6]
        cod3 = s[i+6:i+9]

        # Proline run (PPP…)
        if cod1 in pro_codons and cod2 in pro_codons and cod3 in pro_codons:
            hits.append(("+1_proline_like", i, s[i:i+run_len_nt]))

        # AAA run (AAA-AAA-AAA…)
        if cod1 == "AAA" and cod2 == "AAA" and cod3 == "AAA":
            hits.append(("+1_AAA_like", i, s[i:i+run_len_nt]))

    # Homopolymers (nt-level, unchanged)
    for base, tag, runlen in (("A", "A6+", 6), ("U", "U6+", 6)):
        run = base * runlen
        idx = s.find(run)
        while idx != -1:
            hits.append((tag, idx, s[idx:idx+runlen]))
            idx = s.find(run, idx+1)

    return hits


def seq_comp_features(rna):
    s = rna.upper()
    L = max(1, len(s))
    A = s.count("A"); U = s.count("U"); G = s.count("G"); C = s.count("C")
    gc = (G + C) / L
    au = (A + U) / L
    cpg = s.count("CG")
    exp_cpg = (C / L) * (G / L) * L
    cpg_oe = (cpg / exp_cpg) if exp_cpg > 0 else float("nan")
    freqs = [A/L, U/L, G/L, C/L]
    ent = -sum(f * math.log2(f) for f in freqs if f > 0)
    return {
        "len_nt": L,
        "gc_frac": gc,
        "au_frac": au,
        "cpg_obs": cpg,
        "cpg_oe": cpg_oe,
        "entropy_nt": ent
    }

AA_MAP = {
    "UUU":"F","UUC":"F","UUA":"L","UUG":"L",
    "CUU":"L","CUC":"L","CUA":"L","CUG":"L",
    "AUU":"I","AUC":"I","AUA":"I","AUG":"M",
    "GUU":"V","GUC":"V","GUA":"V","GUG":"V",
    "UCU":"S","UCC":"S","UCA":"S","UCG":"S","AGU":"S","AGC":"S",
    "CCU":"P","CCC":"P","CCA":"P","CCG":"P",
    "ACU":"T","ACC":"T","ACA":"T","ACG":"T",
    "GCU":"A","GCC":"A","GCA":"A","GCG":"A",
    "UAU":"Y","UAC":"Y","UAA":"*","UAG":"*",
    "CAU":"H","CAC":"H","CAA":"Q","CAG":"Q",
    "AAU":"N","AAC":"N","AAA":"K","AAG":"K",
    "GAU":"D","GAC":"D","GAA":"E","GAG":"E",
    "UGU":"C","UGC":"C","UGA":"*","UGG":"W",
    "CGU":"R","CGC":"R","CGA":"R","CGG":"R","AGA":"R","AGG":"R",
    "GGU":"G","GGC":"G","GGA":"G","GGG":"G"
}

def aa_context_features_from_codons(codon_tokens):
    aa = "".join(AA_MAP.get(c, "X") for c in codon_tokens if len(c) == 3)
    out = {}
    for r in (2, 3):
        out[f"aa_has_Px{r}"] = ("P" * r in aa)
    out["aa_polyK6+"] = ("K" * 6 in aa)
    out["aa_polyR6+"] = ("R" * 6 in aa)
    out["aa_DE_to_P"] = any(
        aa[i] in "DE" and i + 1 < len(aa) and aa[i + 1] == "P"
        for i in range(len(aa) - 1)
    )
    return out

def pqs_hits(seq_rna, pattern):
    try:
        return len(re.findall(pattern, seq_rna))
    except re.error:
        return 0

def codon_tokens_from_seq_centered(seq_rna):
    toks = []
    L = (len(seq_rna) // 3) * 3
    s = seq_rna[:L]
    for i in range(0, L, 3):
        toks.append(s[i:i+3])
    return toks

# ----------------------------- RNAfold wrapper -------------------------------

def parse_rnafold_stdout(text):
    mfe = np.nan
    ens = np.nan
    mfe_freq = np.nan
    for line in text.splitlines():
        t = line.strip()
        # MFE line (typical: "... ( -12.34)")
        if ")" in t and "(" in t:
            try:
                val = t.rsplit("(", 1)[-1].strip(") ").replace("kcal/mol","").strip()
                mfe = float(val)
            except Exception:
                pass
        tl = t.lower()
        # Ensemble energy
        if "free energy of ensemble" in tl or "ensemble free energy" in tl:
            try:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", t.replace("−","-"))
                if nums:
                    ens = float(nums[-1])
            except Exception:
                pass
        if "frequency of mfe structure" in tl:
            try:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", t)
                if nums:
                    mfe_freq = float(nums[-1])
            except Exception:
                pass
    return mfe, ens, mfe_freq

def run_rnafold(seq_id, seq_rna, temp_c=37.0, timeout_s=30, extra_args=None):
    cmd = ["RNAfold", "-T", str(float(temp_c)), "-p", "--noPS", "--noDP"]
    if extra_args:
        cmd.extend(extra_args)
    fasta = f">{seq_id}\n{seq_rna}\n"
    try:
        p = sp.run(
            cmd,
            input=fasta.encode("utf-8"),
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            timeout=timeout_s,
            check=False
        )
    except sp.TimeoutExpired:
        return {
            "ok": False,
            "mfe_kcal": np.nan,
            "ensemble_kcal": np.nan,
            "mfe_freq": np.nan,
            "stdout": "",
            "stderr": "TIMEOUT"
        }
    out = p.stdout.decode("utf-8", errors="ignore")
    err = p.stderr.decode("utf-8", errors="ignore")
    mfe, ens, mfe_freq = parse_rnafold_stdout(out)
    return {
        "ok": (p.returncode == 0),
        "mfe_kcal": mfe,
        "ensemble_kcal": ens,
        "mfe_freq": mfe_freq,
        "stdout": out,
        "stderr": err
    }

# ----------------------------- Worker ----------------------------------------

def worker_one_locus(args):
    (cmp_name, tag, row_idx, row, fa_path, flank, rf_temp,
     do_rnafold, save_raw, cmp_dir, ctx_cfg) = args

    fa = pysam.FastaFile(fa_path)
    try:
        chrom = str(row["chrom"])
        strand = str(row["strand"])
        gstart = int(row["genome_start"])
        gend   = int(row["genome_end"])
        gene   = str(row.get("gene","NA"))
        tx     = str(row.get("transcript","NA"))
        direction = str(row.get("direction","NA"))

        flank = ctx_cfg["flank_nt"]
        max_window = ctx_cfg.get("max_window_nt", 2 * flank + 1)

        locus_mid = (gstart + gend) // 2
        half = max_window // 2
        win_start = locus_mid - half
        win_end   = locus_mid + (max_window - half)

        seq_rna, win_s, win_e = fetch_window(
            fa, chrom, win_start, win_end, strand, flank=0
        )

        # Positional context (uses codon indices if present)
        pos_ctx = cds_pos_features(row)

        # Composition
        comp = seq_comp_features(seq_rna)

        # Motifs (nt-space)
        hits = scan_slippery_extended(seq_rna)
        motif_summary = ";".join(
            [f"{t}@{pos}:{mer}" for (t,pos,mer) in hits]
        ) if hits else ""
        has_canon = any(h[0] == "canon_heptamer" for h in hits)
        has_A6 = any(h[0] == "A6+" for h in hits)
        has_U6 = any(h[0] == "U6+" for h in hits)

        # Optional AA-context if codon tokens present
        aa_ctx = {}
        if ctx_cfg["motifs"].get("aa_context", True):
            if "codon_tokens" in row and isinstance(row["codon_tokens"], str):
                codons = [c for c in row["codon_tokens"].split(",") if len(c)==3]
            else:
                codons = codon_tokens_from_seq_centered(seq_rna)
            aa_ctx = aa_context_features_from_codons(codons)

        # RNAfold on the full window
        rf = {"ok": False, "mfe_kcal": np.nan, "ensemble_kcal": np.nan,
              "mfe_freq": np.nan, "stdout": "", "stderr": ""}
        if do_rnafold:
            rf = run_rnafold(
                f"{cmp_name}|{tag}|{gene}|{tx}",
                seq_rna,
                temp_c=rf_temp
            )
            if save_raw:
                out_dir = ensure_dir(os.path.join(cmp_dir, "rnafold"))
                base = re.sub(
                    r"[^A-Za-z0-9_.-]+", "_",
                    f"{cmp_name}_{tag}_{gene}_{tx}_{row_idx}"
                )[:240]
                with open(os.path.join(out_dir, base + ".out"), "w") as f:
                    f.write(rf.get("stdout",""))
                if rf.get("stderr"):
                    with open(os.path.join(out_dir, base + ".err"), "w") as f:
                        f.write(rf.get("stderr",""))

        # G-quad propensity
        if ctx_cfg.get("gquad",{}).get("enable", True):
            pat = ctx_cfg["gquad"].get(
                "pattern",
                r"(G{3,})[ACGU]{1,7}(G{3,})[ACGU]{1,7}(G{3,})[ACGU]{1,7}(G{3,})"
            )
            pqs_count = int(pqs_hits(seq_rna, pat))
        else:
            pqs_count = 0

        fasta_rec = (
            f">{gene}|{tx}|{direction}|{tag}|idx{row_idx}\n"
            f"{seq_rna}\n"
        )

        templ = {k: (np.nan) for k in UNIFIED_OUT_COLS}
        templ.update({
           "comparison": cmp_name, "tag": tag,
           "gene": gene, "transcript": tx,
           "chrom": chrom, "strand": strand,
           "window_start": int(win_s), "window_end": int(win_e),
           "direction": direction,
           "rnafold_ok": False, "has_canonical": False,
           "has_A6plus": False, "has_U6plus": False, "pqs_count": 0,
           "motif_hits": ""
        })
        out = dict(templ)

        out.update({
            "comparison": cmp_name,
            "tag": tag,
            "gene": gene,
            "transcript": tx,
            "chrom": chrom,
            "strand": strand,
            "window_start": int(win_s),
            "window_end": int(win_e),
            "direction": direction,
            # RNAfold
            "mfe_kcal": float(rf["mfe_kcal"]),
            "ensemble_kcal": float(rf["ensemble_kcal"]),
            "mfe_freq": float(rf.get("mfe_freq", np.nan)),
            "rnafold_ok": bool(rf["ok"]),
            # motifs & composition
            "has_canonical": bool(has_canon),
            "has_A6plus": bool(has_A6),
            "has_U6plus": bool(has_U6),
            "motif_hits": motif_summary,
            "pqs_count": pqs_count,
            # cds position
            **pos_ctx,
            # composition
            **comp,
            # aa-context
            **aa_ctx,
            "_fasta": fasta_rec,
            "_seq": seq_rna
        })

        return out
    finally:
        try:
            fa.close()
        except Exception:
            pass

# ----------------------------- Plots -----------------------------------------

def plot_direction_ratio(df_all, out_png):
    cnt = df_all.groupby(
        ["comparison","tag","direction"], dropna=False
    )["gene"].count().reset_index(name="n")
    if cnt.empty:
        logger.warning("No data for direction ratio plot.")
        return
    piv = cnt.pivot_table(
        index=["comparison","tag"],
        columns="direction",
        values="n",
        fill_value=0
    ).reset_index()
    piv["label"] = piv["comparison"] + " [" + piv["tag"] + "]"

    fig, ax = plt.subplots(figsize=(max(8, 0.32*len(piv)), 4))
    x = np.arange(len(piv))
    w = 0.35
    y1 = piv["+1"].values if "+1" in piv else np.zeros(len(piv))
    y2 = piv["-1"].values if "-1" in piv else np.zeros(len(piv))
    ax.bar(x - w/2, y1, width=w, label="+1")
    ax.bar(x + w/2, y2, width=w, label="-1")
    ax.set_xticks(x)
    ax.set_xticklabels(piv["label"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("Number of loci")
    ax.set_title("+1 / −1 frameshift loci per comparison (split by UP/DOWN if available)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)

# ----------------------------- Main ------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python post_fs_context.py config.yaml frameshift_loci/loci_calls")
        sys.exit(1)

    cfg = read_yaml(sys.argv[1])
    loci_dir = sys.argv[2]

    # ---- Context config -----------------------------------------------------
    ctx = cfg.get("context", {})
    flank_nt = int(ctx.get("flank_nt", cfg.get("flank_nt", 20)))
    max_window_nt = int(ctx.get("max_window_nt", 2 * flank_nt + 1))
    ctx_cfg = {
        "flank_nt": flank_nt,
        "max_window_nt": max_window_nt,
        "run_rnafold": bool(ctx.get("run_rnafold", cfg.get("run_rnafold", True))),
        "gquad": ctx.get("gquad", {"enable": True}),
        "motifs": ctx.get("motifs", {
            "expand": True,
            "aa_context": True,
            "kmer_discovery": {
                "enable": True,
                "nt_k":[6,7,8],
                "codon_k":[2,3],
                "export_fasta": True
            }
        })
    }
    flank = ctx_cfg["flank_nt"]
    rnafold_temp = float(ctx.get("rnafold_temp_c", cfg.get("rnafold_temp_c", 37.0)))
    do_rnafold = bool(ctx_cfg["run_rnafold"])
    save_raw = bool(ctx.get("save_rnafold_raw", cfg.get("save_rnafold_raw", False)))

    # ---- Required paths -----------------------------------------------------
    gtf = cfg.get("gtf")
    fa_path = cfg.get("genome_fasta")
    if not gtf or not os.path.isfile(gtf):
        raise FileNotFoundError("GTF path not found in config.yaml")
    if not fa_path or not os.path.isfile(fa_path):
        raise FileNotFoundError("genome_fasta path not found in config.yaml")

    # ---- Output dirs & general knobs ---------------------------------------
    out_root = ensure_dir("post_fs_context")
    plots_dir = ensure_dir(os.path.join(out_root, "plots"))
    max_workers = int(cfg.get("max_workers", 8))

    # ---- Load loci ----------------------------------------------------------
    df = load_loci_tsvs(loci_dir)
    if df.empty:
        logger.error("No loci found; exiting.")
        sys.exit(1)

    # ---- Direction ratio plot ----------------------------------------------
    out_ratio_png = os.path.join(plots_dir, "direction_ratio_per_comparison.png")
    try:
        plot_direction_ratio(df, out_ratio_png)
        logger.info(f"+1/−1 ratio plot → {out_ratio_png}")
    except Exception as e:
        logger.warning(f"Direction ratio plot failed: {e}")

    # ---- Process per (comparison, tag) -------------------------------------
    master_rows = []

    for (cmp_name, tag), sub in df.groupby(["comparison","tag"], sort=False):
        cmp_dir = ensure_dir(os.path.join(out_root, f"{cmp_name}_{tag}"))
        fasta_all = os.path.join(cmp_dir, "contexts_all.fa")
        fasta_p1  = os.path.join(cmp_dir, "contexts_plus1.fa")
        fasta_m1  = os.path.join(cmp_dir, "contexts_minus1.fa")
        summary_tsv = os.path.join(cmp_dir, "motif_rnafold.tsv")
        ensure_dir(os.path.join(cmp_dir, "rnafold"))

        # Build worker args
        args = []
        sub = sub.reset_index(drop=True)
        for i, row in sub.iterrows():
            args.append((
                cmp_name, tag, int(i), row, fa_path,
                flank, rnafold_temp, do_rnafold, save_raw, cmp_dir, ctx_cfg
            ))

        rows = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(worker_one_locus, a): a[2] for a in args}
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    res = fut.result()
                    rows.append(res)
                except Exception as e:
                    logger.error(f"[{cmp_name}][{tag}] locus idx {idx} failed: {e}")

        if not rows:
            logger.warning(f"[{cmp_name}][{tag}] no rows processed.")
            continue

        # Write FASTAs (all, +1, -1)
        n_all = n_p1 = n_m1 = 0
        with open(fasta_all, "w") as fa_all, \
             open(fasta_p1,  "w") as fa_p1,  \
             open(fasta_m1,  "w") as fa_m1:
            for r in rows:
                fa_all.write(r["_fasta"]); n_all += 1
                if str(r["direction"]) == "+1":
                    fa_p1.write(r["_fasta"]); n_p1 += 1
                elif str(r["direction"]) == "-1":
                    fa_m1.write(r["_fasta"]); n_m1 += 1

        # Per-comparison TSV (subset to available columns)
        df_rows = pd.DataFrame(rows)
        cols_here = [c for c in UNIFIED_OUT_COLS if c in df_rows.columns]
        df_rows[cols_here].to_csv(summary_tsv, sep="\t", index=False)
        logger.info(
            f"[{cmp_name}][{tag}] FASTA(all/+1/−1) → {fasta_all} | {fasta_p1} | {fasta_m1} "
            f"(N={n_all}/{n_p1}/{n_m1})"
        )
        logger.info(f"[{cmp_name}][{tag}] Motif/RNAfold summary → {summary_tsv}")

        # Optional FASTA export for MEME (per comparison)
        kd = ctx_cfg.get("motifs",{}).get("kmer_discovery",{})
        if kd.get("export_fasta", True):
            meme_dir = ensure_dir(os.path.join(cmp_dir, "meme_fastas"))
            with open(os.path.join(meme_dir, "plus1.fa"), "w") as fp, \
                 open(os.path.join(meme_dir, "minus1.fa"), "w") as fm:
                for r in rows:
                    if str(r["direction"]) == "+1":
                        fp.write(r["_fasta"])
                    elif str(r["direction"]) == "-1":
                        fm.write(r["_fasta"])

        master_rows.extend(rows)

    # ---- Master summary -----------------------------------------------------
    if master_rows:
        master_tsv = os.path.join(out_root, "master_summary.tsv")
        mdf = pd.DataFrame(master_rows)
        cols_here = [c for c in UNIFIED_OUT_COLS if c in mdf.columns]
        mdf[cols_here].to_csv(master_tsv, sep="\t", index=False)
        logger.info(f"Master summary → {master_tsv}")

    logger.info("All done.")

if __name__ == "__main__":
    main()

