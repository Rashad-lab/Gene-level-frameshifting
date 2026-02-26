#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fs_feature_deepdive.py

Deep-dive feature analysis for frameshift loci, building on master_summary.tsv.

Implements:

1) Codon enrichment in frameshift loci vs matched background windows
   (within same CDS, same window size), split by:
     - comparison (e.g. H1_vs_Sham)
     - tag (UP / DOWN)
     - direction (+1 / -1 / pooled)

   Outputs:
   - codon_enrichment_frameshift_vs_bg.csv
   - codon_enrichment_up_vs_down.csv

2) Amino acid context:
   - Detect runs of >=3 hydrophobic amino acids (A, G, I, L, M, F, P, W, V)
   - Compare FS vs matched background per comparison/tag/direction
   - Compare UP vs DOWN within FS set

   Outputs:
   - hydrophobic_run_summary_FS_vs_bg.csv
   - hydrophobic_run_summary_UP_vs_DOWN.csv

3) Positional meta-analysis:
   - Binned distributions of cds_pos_mid_rel per comparison/tag/direction

   Outputs:
   - positional_meta_binned.csv

4) Modeling +1/-1 frameshift impact:
   - For each locus, simulate translation in +1 and -1 frame from locus onward
   - Classify outcome as PTC, SCRT_like, no_stop_in_CDS, or NA
   - Quantify delta of stop position (shifted vs canonical) in codons

   Outputs:
   - locus_ptc_scrt_annotation.csv

5) Ribosome metrics (lightweight stub):
   - For any columns whose names contain 'pause' or 'psite', summarize by
     comparison/tag/direction (mean, median, IQR).

   Outputs:
   - ribo_metric_summary.csv

Usage
-----

python fs_feature_deepdive.py \
    --master ./post_fs_context/master_summary.tsv \
    --cds-fasta ./Mus_musculus.GRCm39.cds.all.fa \ # CDS fasta from Gencode
    --outdir fs_feature_deepdive

Requirements
-----------
- Python 3
- pandas, numpy
- scipy (optional: for Fisher/MWU; otherwise p-values will be NaN)
- statsmodels (optional: for BH correction)
"""

import os
import argparse
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# Optional stats dependencies
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


# ---------------- Helpers ---------------- #

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_fasta(path: str):
    """Simple FASTA reader -> dict[id] = seq (uppercased, no gaps)."""
    seqs = {}
    cur_id = None
    buf = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(buf).upper().replace("U", "T")
                cur_id = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if cur_id is not None:
        seqs[cur_id] = "".join(buf).upper().replace("U", "T")
    return seqs

def load_ensembl_cds(path: str):
    """
    Parse Ensembl CDS FASTA.

    Returns:
      seq_by_tid_full:   {transcript_id_with_version: seq}
      seq_by_tid_novers: {transcript_id_without_version: seq}
      longest_tid_by_gene_id:  {gene_id (as in header): tid_full_of_longest}
      longest_tid_by_symbol:   {gene_symbol: tid_full_of_longest}
      meta: list of per-transcript metadata dicts
    """
    meta = []
    cur_header = None
    cur_seq = []

    def flush_record():
        nonlocal cur_header, cur_seq
        if cur_header is None:
            return
        head = cur_header[1:].strip()  # remove '>'
        fields = head.split()
        tid_full = fields[0]
        tid_novers = tid_full.split(".")[0]

        gene_id = None
        gene_symbol = None
        for tok in fields[1:]:
            if tok.startswith("gene:"):
                gene_id = tok.split(":", 1)[1]
            elif tok.startswith("gene_symbol:"):
                gene_symbol = tok.split(":", 1)[1]

        seq = "".join(cur_seq).upper().replace("U", "T")
        meta.append({
            "transcript_id_full": tid_full,
            "transcript_id": tid_novers,
            "gene_id": gene_id,
            "gene_symbol": gene_symbol,
            "seq": seq
        })
        cur_header = None
        cur_seq = []

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                flush_record()
                cur_header = line
            else:
                cur_seq.append(line)
    flush_record()

    # Build lookups
    seq_by_tid_full = {}
    seq_by_tid_novers = {}
    longest_tid_by_gene_id = {}
    longest_tid_by_symbol = {}

    # track best length
    best_len_gene_id = {}
    best_len_symbol = {}

    for r in meta:
        tidf = r["transcript_id_full"]
        tid0 = r["transcript_id"]
        seq = r["seq"]
        L = len(seq)

        seq_by_tid_full[tidf] = seq
        seq_by_tid_novers[tid0] = seq

        gid = r["gene_id"]
        gsym = r["gene_symbol"]

        if gid:
            if gid not in best_len_gene_id or L > best_len_gene_id[gid]:
                best_len_gene_id[gid] = L
                longest_tid_by_gene_id[gid] = tidf

        if gsym:
            if gsym not in best_len_symbol or L > best_len_symbol[gsym]:
                best_len_symbol[gsym] = L
                longest_tid_by_symbol[gsym] = tidf

    return {
        "seq_by_tid_full": seq_by_tid_full,
        "seq_by_tid_novers": seq_by_tid_novers,
        "longest_tid_by_gene_id": longest_tid_by_gene_id,
        "longest_tid_by_symbol": longest_tid_by_symbol,
        "meta": meta,
    }

def get_cds_for_row(row, cds_data):
    """
    Given a master_summary row and cds_data from load_ensembl_cds,
    return (chosen_transcript_id_full_or_novers, cds_seq) or (None, None) if not found.
    """
    seq_by_tid_full = cds_data["seq_by_tid_full"]
    seq_by_tid_novers = cds_data["seq_by_tid_novers"]
    longest_by_gid = cds_data["longest_tid_by_gene_id"]
    longest_by_sym = cds_data["longest_tid_by_symbol"]

    # 1) Transcript-based lookup
    tid = str(row.get("transcript", "")).strip()
    if tid and tid.lower() != "nan":
        # Try exact ID (with version)
        if tid in seq_by_tid_full:
            return tid, seq_by_tid_full[tid]
        # Try without version
        tid0 = tid.split(".")[0]
        if tid0 in seq_by_tid_novers:
            return tid0, seq_by_tid_novers[tid0]

    # 2) Use 'gene' column:
    #    - If it looks like an ENS* id -> treat as Ensembl gene_id
    #    - Otherwise, treat it as gene symbol and use longest transcript per symbol
    gene_val = row.get("gene", None)
    if isinstance(gene_val, str):
        g = gene_val.strip()
        if g:
            if g.startswith("ENS"):
                # Treat as Ensembl gene_id
                gid_full = g
                gid0 = g.split(".")[0]

                # First try exact key as stored in FASTA meta
                if gid_full in longest_by_gid:
                    tid_full = longest_by_gid[gid_full]
                    return tid_full, seq_by_tid_full[tid_full]

                # Then try versionless
                if gid0 in longest_by_gid:
                    tid_full = longest_by_gid[gid0]
                    return tid_full, seq_by_tid_full[tid_full]
            else:
                # Treat 'gene' as gene symbol (e.g. Shisal1)
                if g in longest_by_sym:
                    tid_full = longest_by_sym[g]
                    return tid_full, seq_by_tid_full[tid_full]

    # 3) Alternative symbol columns, if present
    gsym = None
    for col in ["gene_symbol", "symbol", "GeneSymbol"]:
        if col in row.index:
            if not pd.isna(row[col]):
                gsym = str(row[col]).strip()
            break

    if gsym:
        if gsym in longest_by_sym:
            tid_full = longest_by_sym[gsym]
            return tid_full, seq_by_tid_full[tid_full]

    # If everything fails:
    return None, None



def normalize_direction(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({
        "1": "+1",
        "+1": "+1",
        "-1": "-1"
    })
    return s


HYDROPHOBIC_AA = set(list("AGILMFVWP"))  # your definition

CODON_TABLE = {
    # Standard code, T instead of U
    "TTT": "F", "TTC": "F",
    "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I",
    "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y",
    "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H",
    "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S",
    "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

STOP_CODONS = {"TAA", "TAG", "TGA"}


def translate_codons(codons):
    aas = []
    for c in codons:
        if len(c) != 3:
            break
        aas.append(CODON_TABLE.get(c, "X"))
    return "".join(aas)


def max_hydrophobic_run(aa_seq: str) -> int:
    """Return the length of the longest run of hydrophobic AAs."""
    m = 0
    cur = 0
    for a in aa_seq:
        if a in HYDROPHOBIC_AA:
            cur += 1
            if cur > m:
                m = cur
        else:
            cur = 0
    return m


def cliffs_delta(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    greater = less = 0
    for xv in x:
        greater += np.sum(y < xv)
        less += np.sum(y > xv)
    den = nx * ny
    return (greater - less) / den if den > 0 else np.nan


# ---------------- Window extraction ---------------- #

def infer_fs_codon_index(row, cds_len_codons):
    """
    Approximate the codon index (0-based) of the FS locus from cds_pos_mid_rel.
    """
    rel = row.get("cds_pos_mid_rel", np.nan)
    if pd.isna(rel):
        return None
    rel = float(rel)
    rel = min(max(rel, 0.0), 1.0)
    # Map [0,1] to [0, cds_len_codons-1]
    idx = int(round(rel * max(cds_len_codons - 1, 0)))
    return idx


def get_window_indices(center_idx, span_codons, cds_len_codons):
    """Return (start_idx, end_idx) for a codon window around center_idx."""
    if span_codons is None or math.isnan(span_codons):
        span = 27
    else:
        span = int(round(span_codons))
        if span < 3:
            span = 3
    half = span // 2
    start = max(0, center_idx - half)
    end = start + span
    if end > cds_len_codons:
        end = cds_len_codons
        start = max(0, end - span)
    return start, end


def extract_codon_window(cds_seq, start_idx, end_idx):
    codons = []
    for i in range(start_idx, end_idx):
        codon = cds_seq[i*3:(i+1)*3]
        if len(codon) == 3:
            codons.append(codon)
    return codons


def generate_matched_control_centers(center_idx, span_codons, cds_len_codons,
                                     min_sep_codons=5):
    """
    Return list of possible control window centers in the same CDS:
    - same window size
    - center not within +/- min_sep_codons of FS center
    """
    if span_codons is None or math.isnan(span_codons):
        span = 27
    else:
        span = int(round(span_codons))
        if span < 3:
            span = 3
    half = span // 2
    valid_centers = []
    for c in range(half, cds_len_codons - half):
        if abs(c - center_idx) < min_sep_codons:
            continue
        valid_centers.append(c)
    return valid_centers


# ---------------- PTC / SCRT modeling ---------------- #

def find_canonical_stop_nt(cds_seq):
    """Return nucleotide position of first in-frame stop (frame 0) or last codon."""
    n_codons = len(cds_seq) // 3
    for i in range(n_codons):
        codon = cds_seq[i*3:(i+1)*3]
        if codon in STOP_CODONS:
            return i * 3
    # If no stop codon, treat last codon as "stop"
    return (n_codons - 1) * 3 if n_codons > 0 else None


def find_shifted_stop_nt(cds_seq, start_nt):
    """
    From nucleotide start_nt, walk in steps of 3 in that frame, return
    NT position of first stop codon in that frame, or None if none.
    """
    L = len(cds_seq)
    if start_nt is None or start_nt < 0 or start_nt >= L:
        return None
    pos = start_nt
    while pos + 3 <= L:
        codon = cds_seq[pos:pos+3]
        if codon in STOP_CODONS:
            return pos
        pos += 3
    return None


def classify_shift_outcome(canonical_stop_nt, shifted_stop_nt):
    if canonical_stop_nt is None or shifted_stop_nt is None:
        if shifted_stop_nt is None:
            return "no_stop_in_CDS", np.nan
        return "NA", np.nan
    delta_codons = (shifted_stop_nt - canonical_stop_nt) / 3.0
    if shifted_stop_nt < canonical_stop_nt:
        return "PTC", delta_codons
    elif shifted_stop_nt > canonical_stop_nt:
        return "SCRT_like", delta_codons
    else:
        return "same_stop", delta_codons


# ---------------- Main analysis ---------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Deep feature analysis for frameshift loci"
    )
    ap.add_argument("--master", required=True, help="master_summary.tsv")
    ap.add_argument("--cds-fasta", required=True, help="CDS FASTA (transcript IDs)")
    ap.add_argument("--outdir", default="fs_feature_deepdive",
                    help="Output directory (default: fs_feature_deepdive)")
    ap.add_argument("--max-matched", type=int, default=20,
                    help="Max matched control windows per locus (default: 20)")
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)

    # Load master summary
    df = pd.read_csv(args.master, sep="\t")
    if "comparison" not in df.columns:
        raise ValueError("master_summary.tsv must have 'comparison' column")

    if "tag" not in df.columns:
        raise ValueError("master_summary.tsv must have 'tag' column (UP/DOWN)")

    if "direction" not in df.columns:
        raise ValueError("master_summary.tsv must have 'direction' column")

    if "cds_pos_mid_rel" not in df.columns or "cds_span_codons" not in df.columns:
        raise ValueError("master_summary.tsv must have 'cds_pos_mid_rel' and 'cds_span_codons'")

    # Normalize
    df["tag"] = df["tag"].astype(str).str.upper()
    df["direction"] = normalize_direction(df["direction"])

    # Timepoint label (H1, H6, H24, etc.)
    df["timepoint"] = df["comparison"].astype(str).str.split("_vs_", n=1).str[0]

    # Load CDS sequences
    cds_data = load_ensembl_cds(args.cds_fasta)
    
    mapped = 0
    checked = 0
    for _, r in df.head(100).iterrows():
        tid_used, seq = get_cds_for_row(r, cds_data)
        checked += 1
        if seq is not None:
            mapped += 1
    print(f"[DEBUG] Mapped {mapped} / {checked} of first rows to a CDS")

    # Containers for outputs
    codon_rows_fs_vs_bg = []
    codon_rows_up_vs_down = []
    hydrophobic_rows_fs_bg = []
    hydrophobic_rows_up_down = []
    positional_rows = []
    locus_ptc_rows = []
    ribo_rows = []

    # Identify candidate ribo metric columns
    ribo_cols = [c for c in df.columns
                 if ("pause" in c.lower()) or ("psite" in c.lower())]

    rng = np.random.default_rng(1)

    comparisons = sorted(df["comparison"].unique())
    directions_to_do = ["all", "+1", "-1"]

    for comp in comparisons:
        df_comp = df[df["comparison"] == comp].copy()
        timepoint = df_comp["timepoint"].iloc[0]

        # Positional meta: record cds_pos_mid_rel distribution
        for tag in ["UP", "DOWN"]:
            for direction in ["+1", "-1"]:
                sub = df_comp[(df_comp["tag"] == tag) &
                              (df_comp["direction"] == direction)]
                for _, row in sub.iterrows():
                    positional_rows.append({
                        "comparison": comp,
                        "timepoint": timepoint,
                        "tag": tag,
                        "direction": direction,
                        "cds_pos_mid_rel": row["cds_pos_mid_rel"]
                    })

        # Per-locus PTC/SCRT modeling
        for _, row in df_comp.iterrows():
            tid_used, seq = get_cds_for_row(row, cds_data)
            if seq is None:
                continue
            n_codons = len(seq) // 3
            fs_idx = infer_fs_codon_index(row, n_codons)
            if fs_idx is None:
                continue
            canonical_stop_nt = find_canonical_stop_nt(seq)

            plus1_outcome = "NA"
            plus1_delta = np.nan
            minus1_outcome = "NA"
            minus1_delta = np.nan

            # +1 frame
            start_nt_plus = fs_idx * 3 + 1
            shifted_stop_plus = find_shifted_stop_nt(seq, start_nt_plus)
            if shifted_stop_plus is not None or canonical_stop_nt is not None:
                plus1_outcome, plus1_delta = classify_shift_outcome(
                    canonical_stop_nt, shifted_stop_plus
                )

            # -1 frame
            start_nt_minus = fs_idx * 3 - 1
            shifted_stop_minus = find_shifted_stop_nt(seq, start_nt_minus)
            if shifted_stop_minus is not None or canonical_stop_nt is not None:
                minus1_outcome, minus1_delta = classify_shift_outcome(
                    canonical_stop_nt, shifted_stop_minus
                )

            locus_ptc_rows.append({
                **row.to_dict(),
                "canonical_stop_nt": canonical_stop_nt,
                "plus1_outcome": plus1_outcome,
                "plus1_delta_stop_codons": plus1_delta,
                "minus1_outcome": minus1_outcome,
                "minus1_delta_stop_codons": minus1_delta
            })

        # Ribo metric summaries (if any columns found)
        if ribo_cols:
            for tag in ["UP", "DOWN"]:
                for direction in ["+1", "-1"]:
                    sub = df_comp[(df_comp["tag"] == tag) &
                                  (df_comp["direction"] == direction)]
                    if sub.empty:
                        continue
                    for col in ribo_cols:
                        vals = pd.to_numeric(sub[col], errors="coerce").dropna()
                        if vals.empty:
                            continue
                        ribo_rows.append({
                            "comparison": comp,
                            "timepoint": timepoint,
                            "tag": tag,
                            "direction": direction,
                            "metric": col,
                            "n": len(vals),
                            "mean": float(vals.mean()),
                            "median": float(vals.median()),
                            "q25": float(vals.quantile(0.25)),
                            "q75": float(vals.quantile(0.75)),
                        })

        # ---------- Codon & hydrophobic analysis with matched controls ---------- #
        for direction_filter in directions_to_do:
            if direction_filter == "all":
                df_dir = df_comp
            else:
                df_dir = df_comp[df_comp["direction"] == direction_filter]

            if df_dir.empty:
                continue

            for tag in ["UP", "DOWN"]:
                df_tag = df_dir[df_dir["tag"] == tag]
                if df_tag.empty:
                    continue

                # Collect frameshift windows and matched background (per locus)
                fs_codons = Counter()
                bg_codons = Counter()
                fs_hydro_max = []
                bg_hydro_max = []
                fs_hydro_has_run = 0
                bg_hydro_has_run = 0

                for _, row in df_tag.iterrows():
                    tid_used, seq = get_cds_for_row(row, cds_data)
                    if seq is None:
                        continue
                    n_codons = len(seq) // 3
                    if n_codons < 5:
                        continue

                    fs_idx = infer_fs_codon_index(row, n_codons)
                    if fs_idx is None:
                        continue

                    span = row.get("cds_span_codons", np.nan)
                    win_start, win_end = get_window_indices(fs_idx, span, n_codons)
                    fs_window_codons = extract_codon_window(seq, win_start, win_end)
                    if not fs_window_codons:
                        continue

                    # Frameshift codons
                    for c in fs_window_codons:
                        fs_codons[c] += 1

                    fs_aa = translate_codons(fs_window_codons)
                    fs_mrun = max_hydrophobic_run(fs_aa)
                    fs_hydro_max.append(fs_mrun)
                    if fs_mrun >= 3:
                        fs_hydro_has_run += 1

                    # Matched controls within same CDS
                    centers = generate_matched_control_centers(
                        fs_idx, span, n_codons, min_sep_codons=5
                    )
                    if not centers:
                        continue
                    rng.shuffle(centers)
                    centers = centers[:args.max_matched]
                    for c_idx in centers:
                        bg_start, bg_end = get_window_indices(c_idx, span, n_codons)
                        bg_codons_win = extract_codon_window(seq, bg_start, bg_end)
                        if not bg_codons_win:
                            continue
                        for c in bg_codons_win:
                            bg_codons[c] += 1
                        aa_bg = translate_codons(bg_codons_win)
                        mrun_bg = max_hydrophobic_run(aa_bg)
                        bg_hydro_max.append(mrun_bg)
                        if mrun_bg >= 3:
                            bg_hydro_has_run += 1

                # Skip if nothing collected
                if sum(fs_codons.values()) == 0 or sum(bg_codons.values()) == 0:
                    continue

                # Codon enrichment FS vs matched background
                total_fs = sum(fs_codons.values())
                total_bg = sum(bg_codons.values())
                rows = []
                for codon in sorted(set(list(fs_codons.keys()) + list(bg_codons.keys()))):
                    c_fs = fs_codons.get(codon, 0)
                    c_bg = bg_codons.get(codon, 0)
                    fs_freq = c_fs / total_fs
                    bg_freq = c_bg / total_bg
                    OR = np.nan
                    p = np.nan
                    if _SCIPY_OK:
                        try:
                            OR, p = fisher_exact([[c_fs, total_fs - c_fs],
                                                  [c_bg, total_bg - c_bg]],
                                                 alternative="two-sided")
                        except Exception:
                            OR, p = (np.nan, np.nan)
                    log2_enrich = np.log2((fs_freq + 1e-9) / (bg_freq + 1e-9))
                    rows.append({
                        "comparison": comp,
                        "timepoint": timepoint,
                        "tag": tag,
                        "direction_group": direction_filter,
                        "codon": codon,
                        "fs_count": c_fs,
                        "bg_count": c_bg,
                        "fs_freq": fs_freq,
                        "bg_freq": bg_freq,
                        "log2_enrich_fs_vs_bg": log2_enrich,
                        "odds_ratio": OR,
                        "p_raw": p
                    })

                df_cod = pd.DataFrame(rows)
                if _SM_OK and not df_cod.empty:
                    _, padj, _, _ = multipletests(
                        df_cod["p_raw"].fillna(1.0),
                        method="fdr_bh"
                    )
                    df_cod["p_adj_bh"] = padj
                else:
                    df_cod["p_adj_bh"] = np.nan

                codon_rows_fs_vs_bg.extend(df_cod.to_dict(orient="records"))

                # Hydrophobic FS vs BG summary
                fs_max = np.array(fs_hydro_max, float) if fs_hydro_max else np.array([])
                bg_max = np.array(bg_hydro_max, float) if bg_hydro_max else np.array([])

                if fs_max.size > 0 and bg_max.size > 0:
                    fs_n = fs_max.size
                    bg_n = bg_max.size
                    fs_has = fs_hydro_has_run
                    bg_has = bg_hydro_has_run

                    # Fisher for has_run>=3
                    p_fisher = np.nan
                    OR_run = np.nan
                    if _SCIPY_OK:
                        try:
                            OR_run, p_fisher = fisher_exact(
                                [[fs_has, fs_n - fs_has],
                                 [bg_has, bg_n - bg_has]],
                                alternative="two-sided"
                            )
                        except Exception:
                            OR_run, p_fisher = (np.nan, np.nan)

                    # MWU & Cliff's delta for max run length
                    p_mwu = np.nan
                    U = np.nan
                    d_eff = np.nan
                    if _SCIPY_OK and fs_n >= 3 and bg_n >= 3:
                        try:
                            U, p_mwu = mannwhitneyu(
                                fs_max, bg_max, alternative="two-sided"
                            )
                        except Exception:
                            U, p_mwu = (np.nan, np.nan)
                        d_eff = cliffs_delta(fs_max, bg_max)

                    hydrophobic_rows_fs_bg.append({
                        "comparison": comp,
                        "timepoint": timepoint,
                        "tag": tag,
                        "direction_group": direction_filter,
                        "fs_n": fs_n,
                        "bg_n": bg_n,
                        "fs_has_run3": fs_has,
                        "bg_has_run3": bg_has,
                        "fs_frac_run3": fs_has / fs_n if fs_n else np.nan,
                        "bg_frac_run3": bg_has / bg_n if bg_n else np.nan,
                        "odds_ratio_run3": OR_run,
                        "p_fisher_run3": p_fisher,
                        "U_stat_max_run": U,
                        "p_mwu_max_run": p_mwu,
                        "cliffs_delta_max_run": d_eff,
                        "fs_mean_max_run": float(fs_max.mean()),
                        "bg_mean_max_run": float(bg_max.mean()),
                    })

        # ---------- Codon + hydrophobic: UP vs DOWN within FS ---------- #
        for direction_filter in directions_to_do:
            if direction_filter == "all":
                df_dir = df_comp
            else:
                df_dir = df_comp[df_comp["direction"] == direction_filter]

            df_up = df_dir[df_dir["tag"] == "UP"]
            df_dn = df_dir[df_dir["tag"] == "DOWN"]
            if df_up.empty or df_dn.empty:
                continue

            # UP
            up_codons = Counter()
            up_maxruns = []
            # DOWN
            dn_codons = Counter()
            dn_maxruns = []

            for tag, df_tag, cod_counter, maxruns in [
                ("UP", df_up, up_codons, up_maxruns),
                ("DOWN", df_dn, dn_codons, dn_maxruns),
            ]:
                for _, row in df_tag.iterrows():
                    tid_used, seq = get_cds_for_row(row, cds_data)
                    if seq is None:
                        continue
                    n_codons = len(seq) // 3
                    fs_idx = infer_fs_codon_index(row, n_codons)
                    if fs_idx is None:
                        continue
                    span = row.get("cds_span_codons", np.nan)
                    win_start, win_end = get_window_indices(fs_idx, span, n_codons)
                    cods = extract_codon_window(seq, win_start, win_end)
                    if not cods:
                        continue
                    for c in cods:
                        cod_counter[c] += 1
                    aa = translate_codons(cods)
                    maxruns.append(max_hydrophobic_run(aa))

            if sum(up_codons.values()) == 0 or sum(dn_codons.values()) == 0:
                continue

            total_up = sum(up_codons.values())
            total_dn = sum(dn_codons.values())
            rows_updn = []
            for codon in sorted(set(list(up_codons.keys()) + list(dn_codons.keys()))):
                c_up = up_codons.get(codon, 0)
                c_dn = dn_codons.get(codon, 0)
                up_freq = c_up / total_up
                dn_freq = c_dn / total_dn
                OR = np.nan
                p = np.nan
                if _SCIPY_OK:
                    try:
                        OR, p = fisher_exact([[c_up, total_up - c_up],
                                              [c_dn, total_dn - c_dn]],
                                             alternative="two-sided")
                    except Exception:
                        OR, p = (np.nan, np.nan)
                log2_enrich = np.log2((up_freq + 1e-9) / (dn_freq + 1e-9))
                rows_updn.append({
                    "comparison": comp,
                    "timepoint": timepoint,
                    "direction_group": direction_filter,
                    "codon": codon,
                    "up_count": c_up,
                    "down_count": c_dn,
                    "up_freq": up_freq,
                    "down_freq": dn_freq,
                    "log2_enrich_UP_vs_DOWN": log2_enrich,
                    "odds_ratio": OR,
                    "p_raw": p
                })

            df_cod_updn = pd.DataFrame(rows_updn)
            if _SM_OK and not df_cod_updn.empty:
                _, padj, _, _ = multipletests(
                    df_cod_updn["p_raw"].fillna(1.0),
                    method="fdr_bh"
                )
                df_cod_updn["p_adj_bh"] = padj
            else:
                df_cod_updn["p_adj_bh"] = np.nan
            codon_rows_up_vs_down.extend(df_cod_updn.to_dict(orient="records"))

            # Hydrophobic UP vs DOWN
            up_max = np.array(up_maxruns, float) if up_maxruns else np.array([])
            dn_max = np.array(dn_maxruns, float) if dn_maxruns else np.array([])
            if up_max.size > 0 and dn_max.size > 0:
                up_has = int((up_max >= 3).sum())
                dn_has = int((dn_max >= 3).sum())
                up_n = up_max.size
                dn_n = dn_max.size

                p_fisher = np.nan
                OR_run = np.nan
                if _SCIPY_OK:
                    try:
                        OR_run, p_fisher = fisher_exact(
                            [[up_has, up_n - up_has],
                             [dn_has, dn_n - dn_has]],
                            alternative="two-sided"
                        )
                    except Exception:
                        OR_run, p_fisher = (np.nan, np.nan)

                p_mwu = np.nan
                U = np.nan
                d_eff = np.nan
                if _SCIPY_OK and up_n >= 3 and dn_n >= 3:
                    try:
                        U, p_mwu = mannwhitneyu(up_max, dn_max, alternative="two-sided")
                    except Exception:
                        U, p_mwu = (np.nan, np.nan)
                    d_eff = cliffs_delta(up_max, dn_max)

                hydrophobic_rows_up_down.append({
                    "comparison": comp,
                    "timepoint": timepoint,
                    "direction_group": direction_filter,
                    "up_n": up_n,
                    "down_n": dn_n,
                    "up_has_run3": up_has,
                    "down_has_run3": dn_has,
                    "up_frac_run3": up_has / up_n if up_n else np.nan,
                    "down_frac_run3": dn_has / dn_n if dn_n else np.nan,
                    "odds_ratio_run3": OR_run,
                    "p_fisher_run3": p_fisher,
                    "U_stat_max_run": U,
                    "p_mwu_max_run": p_mwu,
                    "cliffs_delta_max_run": d_eff,
                    "up_mean_max_run": float(up_max.mean()),
                    "down_mean_max_run": float(dn_max.mean()),
                })

    # ------------- Write outputs ------------- #

    # PTC/SCRT table
    if locus_ptc_rows:
        pd.DataFrame(locus_ptc_rows).to_csv(
            os.path.join(outdir, "locus_ptc_scrt_annotation.csv"),
            index=False
        )

    # Positional meta
    if positional_rows:
        pd.DataFrame(positional_rows).to_csv(
            os.path.join(outdir, "positional_meta_binned_raw.csv"),
            index=False
        )

    # Codon enrichment FS vs matched background
    if codon_rows_fs_vs_bg:
        pd.DataFrame(codon_rows_fs_vs_bg).to_csv(
            os.path.join(outdir, "codon_enrichment_frameshift_vs_bg.csv"),
            index=False
        )

    # Codon enrichment UP vs DOWN
    if codon_rows_up_vs_down:
        pd.DataFrame(codon_rows_up_vs_down).to_csv(
            os.path.join(outdir, "codon_enrichment_up_vs_down.csv"),
            index=False
        )

    # Hydrophobic FS vs BG
    if hydrophobic_rows_fs_bg:
        pd.DataFrame(hydrophobic_rows_fs_bg).to_csv(
            os.path.join(outdir, "hydrophobic_run_summary_FS_vs_bg.csv"),
            index=False
        )

    # Hydrophobic UP vs DOWN
    if hydrophobic_rows_up_down:
        pd.DataFrame(hydrophobic_rows_up_down).to_csv(
            os.path.join(outdir, "hydrophobic_run_summary_UP_vs_DOWN.csv"),
            index=False
        )

    # Ribo metrics
    if ribo_rows:
        pd.DataFrame(ribo_rows).to_csv(
            os.path.join(outdir, "ribo_metric_summary.csv"),
            index=False
        )

    print("Done. Outputs in:", outdir)


if __name__ == "__main__":
    main()

