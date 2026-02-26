#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frameshift_loci.py  —  Region-agnostic frameshift locus caller with optional mini-plots.
Refactored with validation, codon-per-exon indexing, faster lookups, diagnostics,
optional caching, stronger window stats (G-test), and richer QC outputs.
Run by:
python frameshift_loci.py config.yaml
Inputs:
  * config.yaml
  * GTF (UCSC/RefSeq-like; features include CDS/start_codon/stop_codon)
  * Genome FASTA (not required for locus calling; kept for future use)
  * BAMs per group (sorted, indexed)
  * Per-BAM P-site offset files, one per BAM:
      psite_offsets/<sample>_psite.txt (cols: read_length, corrected_offset)
      psite_offsets/<sample>_psite.meta (optional JSON: {"chosen_ext":"5end"|"3end"})
  * Significant genes CSV per comparison (columns: gene_id (or gene/symbol), logFC, p)

Outputs (under frameshift_loci/):
  logs/
  loci_calls/<cmp>_loci.tsv
  loci_calls/<cmp>.bed12
  per_gene_plots/<cmp>/<gene>.png           (if make_plots: true)
  tmp/                                       (optional count cache)

Config additions (all optional; sensible defaults):
  make_plots: true|false
  cache_counts: true|false
  gtest_alpha: 0.05
  min_window_reads: 30
  transcript_choice: "longest_cds" | "NM_only" | "whitelist"
  transcript_whitelist: "/path/tx_list.txt"
  name_normalization: true|false
  psite_name_map: { "1hr1": "1hr1", ... }

Author: (Sherif Rashad)
"""

import os
import sys
import yaml
import math
import gzip
import json
import time
import glob
import pysam
import numpy as np
import pandas as pd
import logging
import itertools
from collections import defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess as sp
from typing import Optional, Tuple, List


# Try SciPy for p-values; fallback to LLR only if not installed
try:
    from scipy.stats import chi2
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- Logging ---------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("frameshift_loci")


# ----------------------------- Data models -----------------------------------
CDSBlock = namedtuple("CDSBlock", ["chrom","start","end"])  # half-open [start,end)
TranscriptModel = namedtuple("TranscriptModel", [
    "gene", "transcript", "chrom", "strand",
    "cds_blocks",            # list[CDSBlock] in transcript/genomic order
    "cds_len_nt",            # length of CDS in nt
    "codon_map",             # list[(chrom, start, end)] per codon index (0-based, each 3nt span)
    "block_meta"             # NEW: per-block codon arrays for O(log n) lookup
])

# =============================================================================
# Utility helpers
# =============================================================================

def read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def is_gz(path):
    return path.endswith(".gz")

def open_textmaybe(path):
    return gzip.open(path, "rt") if is_gz(path) else open(path, "r")

def moving_average(v, k=1):
    """Simple (2k+1)-bin smoothing; returns float array."""
    if k <= 0 or len(v) == 0:
        return v
    v = np.asarray(v, float)
    out = v.copy()
    for i in range(len(v)):
        a = max(0, i-k)
        b = min(len(v), i+k+1)
        out[i] = np.mean(v[a:b])
    return out
    
    # ============================ Sequence helpers ================================

def _to_rna(seq: str) -> str:
    return seq.upper().replace("T","U")

def _revcomp_rna(seq: str) -> str:
    tr = str.maketrans("ACGTUacgtuNn", "UGCAAugcaaNn")
    return seq.translate(tr)[::-1].upper()

def fetch_window_seq(fa, chrom: str, start: int, end: int, strand: str,
                     flank_nt: int) -> Tuple[str, int, int]:
    """
    Fetch ±flank_nt around [start,end) (0-based, half-open).
    Returns (RNA_seq, win_start, win_end) in transcript sense:
      - '+' strand → sequence as-is (to RNA)
      - '-' strand → reverse-complement to RNA
    """
    g_start = max(0, start - flank_nt)
    g_end   = max(end + flank_nt, start + 1)
    raw = fa.fetch(chrom, g_start, g_end).upper()
    if strand == "+":
        return _to_rna(raw), g_start, g_end
    else:
        return _revcomp_rna(raw), g_start, g_end


# ----------------------------- validation --------------------------------

def validate_inputs(cfg, log_dir):
    """Validate config & inputs early; dump frozen run-config to logs for provenance."""
    problems = []

    # GTF
    gtf = cfg.get("gtf")
    if not gtf or not os.path.isfile(gtf):
        problems.append(f"GTF not found: {gtf}")

    # Groups & BAMs
    groups = cfg.get("groups", {})
    if not groups:
        problems.append("No groups defined in config.")

    # Ensure each group has at least one BAM; BAM & BAI exist
    for g, bams in (groups or {}).items():
        if not bams:
            problems.append(f"Group '{g}' has no BAMs.")
            continue
        for b in bams:
            if not os.path.isfile(b):
                problems.append(f"BAM not found: {b}")
            bai = b + ".bai"
            bai_alt = b.replace(".bam", ".bai")
            if not (os.path.isfile(bai) or os.path.isfile(bai_alt)):
                problems.append(f"BAI index missing for BAM: {b}")

    # Genome fasta if needed
    need_fa = (cfg.get("emit_context",{}).get("enable_sequences", False) or
               cfg.get("structure_scan",{}).get("enable", False))
    if need_fa:
        fa = cfg.get("genome_fasta")
        if not fa or not os.path.isfile(fa):
            problems.append(f"genome_fasta required by emit_context/structure_scan not found: {fa}")
    
    # Comparisons exist and reference valid groups
    comps = cfg.get("comparisons", [])
    if not comps:
        problems.append("No comparisons defined in config.")
    else:
        group_names = set(groups.keys())
        for c in comps:
            case = c.get("case"); ctrl = c.get("control"); csvp = c.get("sig_genes_csv")
            if case not in group_names:
                problems.append(f"Comparison case='{case}' not found in groups.")
            if ctrl not in group_names:
                problems.append(f"Comparison control='{ctrl}' not found in groups.")
            if not csvp or not os.path.isfile(csvp):
                problems.append(f"sig_genes_csv missing or not found: {csvp} for {case} vs {ctrl}")

    # P-site dir
    psite_dir = cfg.get("psite_offsets_dir", "psite_offsets")
    if not os.path.isdir(psite_dir):
        problems.append(f"P-site offsets directory not found: {psite_dir}")

    if problems:
        for p in problems:
            logger.error(p)
        sys.exit(2)

    # Dump frozen config to logs for provenance
    try:
        cfg_out = os.path.join(log_dir, f"run_config_{int(time.time())}.json")
        with open(cfg_out, "w") as f:
            json.dump(cfg, f, indent=2)
        logger.info(f"Wrote run configuration to {cfg_out}")
    except Exception as e:
        logger.warning(f"Could not write run configuration JSON: {e}")

# ----------------------------- NEW: name normalization ------------------------

def normalize_gene_name(name: str) -> str:
    """Make gene IDs robust across sources: strip version suffixes, unify case minimally."""
    if name is None:
        return "NA"
    n = str(name).strip()
    # strip transcript or gene version suffix "NM_xxx.y" -> "NM_xxx"
    if "." in n:
        parts = n.split(".")
        # only drop if trailing piece is purely numeric
        if parts[-1].isdigit():
            n = ".".join(parts[:-1])
    return n

# ----------------------------- in-frame codon analysis --------------------------------

CODON_TABLE = set([
    "AAA","AAC","AAG","AAU","ACA","ACC","ACG","ACU","AGA","AGC","AGG","AGU",
    "AUA","AUC","AUG","AUU","CAA","CAC","CAG","CAU","CCA","CCC","CCG","CCU",
    "CGA","CGC","CGG","CGU","CUA","CUC","CUG","CUU","GAA","GAC","GAG","GAU",
    "GCA","GCC","GCG","GCU","GGA","GGC","GGG","GGU","GUA","GUC","GUG","GUU",
    "UAA","UAC","UAG","UAU","UCA","UCC","UCG","UCU","UGA","UGC","UGG","UGU",
    "UUA","UUC","UUG","UUU"
])

def codons_in_frame_around(tm: TranscriptModel,
                           locus_i: int, locus_j: int,
                           flank_codons: int,
                           fa) -> Tuple[List[str], int, int]:
    """
    Return in-frame codon tokens around locus midpoint within CDS:
      - Determine mid codon index m = (i+j)//2
      - Take [m-flank_codons, m+flank_codons], clamp to [0, n_codons-1]
      - Fetch genomic sequence block(s) and return token list in transcript order (RNA alphabet)
    Also returns (first_codon_idx, last_codon_idx).
    """
    n = len(tm.codon_map)
    if n == 0:
        return [], 0, -1
    m = int((locus_i + locus_j) // 2)
    a = max(0, m - flank_codons)
    b = min(n - 1, m + flank_codons)

    # Assemble contiguous genomic slices spanning codons a..b in transcript order
    chrom = tm.chrom
    if tm.strand == "+":
        start = tm.codon_map[a][1]
        end   = tm.codon_map[b][2]
        dna = fa.fetch(chrom, start, end).upper()
        rna = _to_rna(dna)
    else:
        # minus strand → fetch genomic then reverse-complement to transcript sense
        start = tm.codon_map[b][1]
        end   = tm.codon_map[a][2]
        dna = fa.fetch(chrom, start, end).upper()
        rna = _revcomp_rna(dna)

    # Tokenize in triplets; ensure length is multiple of 3 by trimming end
    L = (len(rna) // 3) * 3
    rna = rna[:L]
    toks = [rna[k:k+3] for k in range(0, len(rna), 3)]
    # Validate codons (stay in-frame only)
    toks = [c if c in CODON_TABLE else "NNN" for c in toks]
    return toks, a, b

# ----------------------------- Optional, cheap stall-codon flags --------------------------------

def flag_stall_codon_motifs(codons: List[str],
                            proline_runs=(2,3),
                            polybasic_min=6,
                            de_to_p=True) -> dict:
    """
    Quick in-frame codon motif flags.
    Returns dict of booleans + counts.
    """
    aa = []
    # Minimal 1-letter AA mapping for fast checks
    aa_map = {
        "CCU":"P","CCC":"P","CCA":"P","CCG":"P",
        "AAU":"N","AAC":"N","GAU":"D","GAC":"D",
        "GAA":"E","GAG":"E",
        "GGU":"G","GGC":"G","GGA":"G","GGG":"G",
        "AAA":"K","AAG":"K",
        "CGU":"R","CGC":"R","CGA":"R","CGG":"R","AGA":"R","AGG":"R",
        # others collapsed
    }
    for c in codons:
        aa.append(aa_map.get(c, "X"))

    # Proline runs
    pro_flags = {}
    for r in proline_runs:
        pro_flags[f"has_Px{r}"] = ("P"*r in "".join(aa))

    # Polybasic K/R runs
    k_run = "K"*polybasic_min
    r_run = "R"*polybasic_min
    has_Krun = (k_run in "".join(aa)) if polybasic_min > 1 else False
    has_Rrun = (r_run in "".join(aa)) if polybasic_min > 1 else False

    # D/E → P transition (any acidic immediately followed by Pro)
    has_DE_to_P = False
    if de_to_p:
        for i in range(len(aa)-1):
            if aa[i] in ("D","E") and aa[i+1] == "P":
                has_DE_to_P = True
                break

    out = dict(
        **pro_flags,
        has_polyK=has_Krun,
        has_polyR=has_Rrun,
        has_DE_to_P=has_DE_to_P
    )
    return out

# ----------------------------- Optional downstream structure (ΔG) via RNAfold --------------------------------

def rnafold_min_dG(seq_rna: str, rnafold_bin="RNAfold", do_pf=True, no_ps=True, timeout_s=20) -> Tuple[float, Optional[str]]:
    """
    Run RNAfold on RNA seq, return (mfe_kcal, stdout) or (nan, None) on failure.
    """
    cmd = [rnafold_bin]
    if do_pf:
        cmd.append("-p")
    if no_ps:
        cmd.extend(["--noPS","--noDP"])
    try:
        p = sp.run(cmd, input=(seq_rna+"\n").encode("utf-8"),
                   stdout=sp.PIPE, stderr=sp.PIPE, timeout=timeout_s, check=False)
        out = p.stdout.decode("utf-8", errors="ignore")
    except Exception:
        return float("nan"), None

    mfe = float("nan")
    for line in out.splitlines():
        line=line.strip()
        if line.endswith(")") and "(" in line:
            # e.g. "....(((...))).... (-12.30)"
            try:
                val = line.split("(")[-1].strip(")")
                mfe = float(val)
            except Exception:
                pass
    return mfe, out

def downstream_structure_scores(seq_rna: str, offsets=(6,18), win=30,
                                rnafold_bin="RNAfold") -> dict:
    """
    Slide windows starting at each nt in [offsets[0]..offsets[1]] downstream;
    record minimum ΔG across those windows.
    Return {"down_min_dG": float, "down_min_dG_at": int}
    """
    a, b = offsets
    best = float("inf"); best_pos = None
    for off in range(a, b+1):
        if off + win <= len(seq_rna):
            sub = seq_rna[off:off+win]
            dG, _ = rnafold_min_dG(sub, rnafold_bin=rnafold_bin, do_pf=True, no_ps=True)
            if not np.isnan(dG) and dG < best:
                best = dG; best_pos = off
    if best is float("inf"):
        return {"down_min_dG": np.nan, "down_min_dG_at": None}
    return {"down_min_dG": float(best), "down_min_dG_at": int(best_pos)}


# =============================================================================
# GTF parsing → per-gene transcript model with per-exon codon indexing
# =============================================================================

def build_transcript_models(gtf_path, transcript_choice="longest_cds", nm_prefer=True,
                            whitelist_path=None, name_norm=True):
    """
    Build per-gene transcript models; choose transcript per 'transcript_choice':
      - "longest_cds" (default; prefer NM_ tie)
      - "NM_only"     (choose longest among NM_ only; fallback to longest if none)
      - "whitelist"   (choose the transcript present in whitelist; if multiple, prefer NM_; else fallback)
    Returns: dict normalized_gene_symbol -> TranscriptModel
    """
    logger.info(f"Parsing GTF: {gtf_path}")
    transcripts = defaultdict(lambda: {
        "gene": None, "transcript": None, "chrom": None, "strand": None, "cds_blocks": []
    })

    whitelist = set()
    if transcript_choice == "whitelist" and whitelist_path and os.path.isfile(whitelist_path):
        with open(whitelist_path) as f:
            for line in f:
                wl = line.strip()
                if wl:
                    whitelist.add(wl)

    with open_textmaybe(gtf_path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            chrom, source, feature, start, end, score, strand, frame, attrs = fields
            if feature != "CDS":
                continue
            # Enforce legal strands and skip garbage early
            if strand not in {"+", "-"}:
                logger.warning(f"Skipping CDS with non-canonical strand '{strand}' on {chrom}:{start}-{end}")
                continue
            start = int(start) - 1  # GTF is 1-based inclusive; convert to 0-based half-open
            end = int(end)          # half-open
            a = _parse_attributes(attrs)
            tx = a.get("transcript_id") or a.get("transcriptId") or a.get("transcript")
            gene = a.get("gene_name") or a.get("gene_id") or a.get("gene")
            if not tx or not gene:
                tx = tx or a.get("transcript_id")
                gene = gene or a.get("gene_id") or "NA"
            if name_norm:
                gene = normalize_gene_name(gene)
                tx = normalize_gene_name(tx)
            rec = transcripts[tx]
            rec["gene"] = gene
            rec["transcript"] = tx
            rec["chrom"] = chrom
            rec["strand"] = strand
            rec["cds_blocks"].append(CDSBlock(chrom, start, end))

    # consolidate to gene -> choose transcript per policy
    by_gene = defaultdict(list)
    for tx, rec in transcripts.items():
        blks = sorted(rec["cds_blocks"], key=lambda b: b.start)
        if rec["strand"] == "-":
            blks = sorted(blks, key=lambda b: b.end, reverse=True)
        length = sum(b.end - b.start for b in blks)
        by_gene[rec["gene"]].append( (length, tx, rec["chrom"], rec["strand"], blks) )

    def choose_tx(items):
        if transcript_choice == "NM_only":
            nms = [t for t in items if t[1].startswith("NM_")]
            if nms:
                items = nms
        elif transcript_choice == "whitelist" and whitelist:
            wl = [t for t in items if t[1] in whitelist]
            if wl:
                items = wl
        items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
        best_len = items_sorted[0][0]
        tied = [t for t in items_sorted if t[0] == best_len]
        nm = [t for t in tied if t[1].startswith("NM_")]
        return nm[0] if nm else tied[0]

    gene_models = {}
    for gene, items in by_gene.items():
        length, tx, chrom, strand, blks = choose_tx(items)

        # Build codon map (3-nt bins along CDS) and per-block codon arrays (NEW)
        codon_map = []
        nt_accum = 0
        block_meta = []  # list of dict for each CDS block: starts[], ends[], idx[]
        if strand == "+":
            ordered = sorted(blks, key=lambda b: b.start)
            for b in ordered:
                seg_len = b.end - b.start
                pos = b.start
                starts_this, ends_this, idx_this = [], [], []
                while seg_len > 0:
                    take = min(seg_len, 3 - (nt_accum % 3))
                    nt_accum += take
                    pos += take
                    seg_len -= take
                    if nt_accum % 3 == 0:
                        codon_end = pos
                        codon_start = codon_end - 3
                        codon_map.append((b.chrom, codon_start, codon_end))
                        starts_this.append(codon_start)
                        ends_this.append(codon_end)
                        idx_this.append(len(codon_map)-1)
                block_meta.append({
                    "chrom": b.chrom,
                    "start": b.start,
                    "end": b.end,
                    "codon_starts": np.array(starts_this, dtype=np.int64),
                    "codon_ends":   np.array(ends_this,   dtype=np.int64),
                    "codon_idx":    np.array(idx_this,    dtype=np.int64)
                })
        else:
            ordered = sorted(blks, key=lambda b: b.end, reverse=True)
            for b in ordered:
                seg_len = b.end - b.start
                pos = b.end
                starts_this, ends_this, idx_this = [], [], []
                while seg_len > 0:
                    take = min(seg_len, 3 - (nt_accum % 3))
                    nt_accum += take
                    pos -= take
                    seg_len -= take
                    if nt_accum % 3 == 0:
                        codon_start = pos
                        codon_end = pos + 3
                        codon_map.append((b.chrom, codon_start, codon_end))
                        starts_this.append(codon_start)
                        ends_this.append(codon_end)
                        idx_this.append(len(codon_map)-1)
                    # IMPORTANT: sort codon_starts ascending so binary search works
                    if starts_this:
                        order = np.argsort(starts_this)
                        starts_arr = np.array(starts_this, dtype=np.int64)[order]
                        ends_arr   = np.array(ends_this,   dtype=np.int64)[order]
                        idx_arr    = np.array(idx_this,    dtype=np.int64)[order]
                    else:
                        starts_arr = np.array([], dtype=np.int64)
                        ends_arr   = np.array([], dtype=np.int64)
                        idx_arr    = np.array([], dtype=np.int64)
                        block_meta.append({
                    "chrom": b.chrom,
                    "start": b.start,
                    "end": b.end,
                    "codon_starts": np.array(starts_this, dtype=np.int64),
                    "codon_ends":   np.array(ends_this,   dtype=np.int64),
                    "codon_idx":    np.array(idx_this,    dtype=np.int64)
                })

        cds_len_nt = sum(b.end - b.start for b in blks)
        gene_models[gene] = TranscriptModel(
            gene=gene, transcript=tx, chrom=chrom, strand=strand,
            cds_blocks=blks, cds_len_nt=cds_len_nt, codon_map=codon_map, block_meta=block_meta
        )
    logger.info(f"Built transcript models for {len(gene_models)} genes.")
    return gene_models

def _parse_attributes(attr_field):
    out = {}
    parts = [p.strip() for p in attr_field.strip().split(";") if p.strip()]
    for p in parts:
        if " " not in p:
            continue
        k, v = p.split(" ", 1)
        v = v.strip().strip('"')
        out[k] = v
    return out

# =============================================================================
# P-site offsets (per-BAM) + diagnostics
# =============================================================================

def load_psite_for_bam(psite_dir, bam_path, default_chosen_ext="5end", psite_name_map=None):
    """
    Read per-BAM P-site offsets from <prefix>_psite.txt (2 cols).
    If <prefix>_psite.meta is absent, use default_chosen_ext from config.
    """
    sample_base = os.path.splitext(os.path.basename(bam_path))[0]
    prefix = psite_name_map.get(sample_base, sample_base) if psite_name_map else sample_base

    txt = os.path.join(psite_dir, f"{prefix}_psite.txt")
    meta = os.path.join(psite_dir, f"{prefix}_psite.meta")  # optional

    if not os.path.isfile(txt):
        raise FileNotFoundError(f"P-site offset file not found for {sample_base} (prefix {prefix}): {txt}")

    df = pd.read_csv(txt, sep="\t")
    if {"read_length","corrected_offset"}.issubset(df.columns):
        offsets = dict(zip(df["read_length"].astype(int), df["corrected_offset"].astype(int)))
    else:
        offsets = dict(zip(df.iloc[:,0].astype(int), df.iloc[:,1].astype(int)))

    chosen_ext = default_chosen_ext
    if os.path.isfile(meta):
        try:
            with open(meta, "r") as f:
                md = json.load(f)
                chosen_ext = md.get("chosen_ext", default_chosen_ext)
        except Exception:
            pass

    return offsets, chosen_ext, prefix

# =============================================================================
# BAM → P-site codon frame counts (fast lookup + diagnostics + optional cache)
# =============================================================================

def psite_for_read(read, chosen_ext, offset):
    if read.is_reverse:
        if chosen_ext == "5end":
            p = read.reference_end - offset
        else:
            p = read.reference_start + offset
    else:
        if chosen_ext == "5end":
            p = read.reference_start + offset
        else:
            p = read.reference_end - offset
    return p

def block_binary_search_codon(block_meta_entry, p):
    starts = block_meta_entry["codon_starts"]
    ends   = block_meta_entry["codon_ends"]
    if starts.size == 0:
        return None

    # Detect order of codon_starts
    if starts[0] <= starts[-1]:
        # ascending
        i = np.searchsorted(starts, p, side="right") - 1
    else:
        # descending: reverse-view search
        rev = starts[::-1]
        j = np.searchsorted(rev, p, side="right") - 1
        i = starts.size - 1 - j

    if 0 <= i < starts.size and starts[i] <= p < ends[i]:
        return int(block_meta_entry["codon_idx"][i])
    return None


def quick_cds_read_count(bamfile, tm, max_needed):
    """Fast pre-gate: count reads overlapping CDS; bail early if threshold reached."""
    n = 0
    for b in tm.cds_blocks:
        try:
            for _ in bamfile.fetch(b.chrom, b.start, b.end):
                n += 1
                if n >= max_needed:
                    return n
        except Exception:
            continue
    return n

def _self_test_strand_frame_mapping():
    """
    Minimal internal test:
      - build toy + and - strand transcript models with a single CDS block
      - for every codon and every offset (0,1,2) compute a synthetic P-site
      - verify that block_binary_search_codon returns the right codon index
        and that 'within % 3' equals the expected frame.
    """
    def _build_test_tm(chrom, start, end, strand):
        blks = [CDSBlock(chrom, start, end)]
        codon_map = []
        block_meta = []
        nt_accum = 0

        if strand == "+":
            ordered = sorted(blks, key=lambda b: b.start)
            for b in ordered:
                seg_len = b.end - b.start
                pos = b.start
                starts_this, ends_this, idx_this = [], [], []
                while seg_len > 0:
                    take = min(seg_len, 3 - (nt_accum % 3))
                    nt_accum += take
                    pos += take
                    seg_len -= take
                    if nt_accum % 3 == 0:
                        codon_end = pos
                        codon_start = codon_end - 3
                        codon_map.append((b.chrom, codon_start, codon_end))
                        starts_this.append(codon_start)
                        ends_this.append(codon_end)
                        idx_this.append(len(codon_map) - 1)
                block_meta.append({
                    "chrom": b.chrom,
                    "start": b.start,
                    "end": b.end,
                    "codon_starts": np.array(starts_this, dtype=np.int64),
                    "codon_ends":   np.array(ends_this,   dtype=np.int64),
                    "codon_idx":    np.array(idx_this,    dtype=np.int64),
                })
        else:
            ordered = sorted(blks, key=lambda b: b.end, reverse=True)
            for b in ordered:
                seg_len = b.end - b.start
                pos = b.end
                starts_this, ends_this, idx_this = [], [], []
                while seg_len > 0:
                    take = min(seg_len, 3 - (nt_accum % 3))
                    nt_accum += take
                    pos -= take
                    seg_len -= take
                    if nt_accum % 3 == 0:
                        codon_start = pos
                        codon_end = pos + 3
                        codon_map.append((b.chrom, codon_start, codon_end))
                        starts_this.append(codon_start)
                        ends_this.append(codon_end)
                        idx_this.append(len(codon_map) - 1)

                # same fix as main code: sort by start ascending
                if starts_this:
                    order = np.argsort(starts_this)
                    starts_arr = np.array(starts_this, dtype=np.int64)[order]
                    ends_arr   = np.array(ends_this,   dtype=np.int64)[order]
                    idx_arr    = np.array(idx_this,    dtype=np.int64)[order]
                else:
                    starts_arr = np.array([], dtype=np.int64)
                    ends_arr   = np.array([], dtype=np.int64)
                    idx_arr    = np.array([], dtype=np.int64)

                block_meta.append({
                    "chrom": b.chrom,
                    "start": b.start,
                    "end": b.end,
                    "codon_starts": starts_arr,
                    "codon_ends":   ends_arr,
                    "codon_idx":    idx_arr,
                })

        cds_len_nt = sum(b.end - b.start for b in blks)
        return TranscriptModel(
            gene="TEST", transcript=f"TEST_{strand}",
            chrom=chrom, strand=strand,
            cds_blocks=blks,
            cds_len_nt=cds_len_nt,
            codon_map=codon_map,
            block_meta=block_meta,
        )

    def _check_tm(tm: TranscriptModel):
        for block in tm.block_meta:
            for cs, ce, ci in zip(block["codon_starts"],
                                  block["codon_ends"],
                                  block["codon_idx"]):
                for off in (0, 1, 2):
                    # pick a genomic P-site that corresponds to this frame
                    if tm.strand == "+":
                        p = cs + off
                        within = p - cs
                    else:
                        p = (ce - 1) - off
                        within = (ce - 1) - p

                    idx = block_binary_search_codon(block, p)
                    if idx != ci:
                        raise AssertionError(
                            f"{tm.strand} strand: expected codon {ci}, "
                            f"got {idx} for p={p}"
                        )
                    frame = within % 3
                    if frame != off:
                        raise AssertionError(
                            f"{tm.strand} strand: expected frame {off}, "
                            f"got {frame} for p={p}"
                        )

    tm_plus  = _build_test_tm("chrT", 100, 112, "+")
    tm_minus = _build_test_tm("chrT", 200, 212, "-")

    _check_tm(tm_plus)
    _check_tm(tm_minus)

    print("Self-test OK: plus/minus strand frame mapping passes.")


def count_frames_per_codon_for_group(
    tm: TranscriptModel, group_bams, psite_dir, min_read_length, max_read_length,
    default_chosen_ext="5end", psite_name_map=None, min_reads_gate=0,
    cache_root=None, diagnostics_acc=None):
    """
    Returns arrays (n_codons x 3) of counts aggregated across all BAMs in the group.
    - min_reads_gate: if >0, do a quick pre-pass per BAM; skip deep parse if both case/ctrl later < gate.
    - cache_root: if provided, caches per-(bam,gene) counts to .npy and reuses on reruns.
    - diagnostics_acc: optional dict to append per-BAM missing-offset fractions.
    """
    n_codons = len(tm.codon_map)
    counts = np.zeros((n_codons, 3), dtype=np.int64)

    for bam in group_bams:
        per_bam = np.zeros((n_codons, 3), dtype=np.int64)
        bamfile = pysam.AlignmentFile(bam, "rb")
        

        # Optional quick gate
        if min_reads_gate > 0:
            qn = quick_cds_read_count(bamfile, tm, min_reads_gate)
            if qn < min_reads_gate:
                bamfile.close()
                continue
            # re-open to reset iterator
            bamfile.close()
            bamfile = pysam.AlignmentFile(bam, "rb")

        # Optional cache
        cache_used = False
        if cache_root:
            key = f"{os.path.splitext(os.path.basename(bam))[0]}__{tm.gene}.npy"
            cache_path = os.path.join(cache_root, key)
            if os.path.isfile(cache_path):
                try:
                    arr = np.load(cache_path)
                    if arr.shape == (n_codons, 3):
                        counts += arr
                        cache_used = True
                        bamfile.close()
                        continue
                except Exception:
                    pass

        offsets, chosen_ext, sample = load_psite_for_bam(psite_dir, bam, default_chosen_ext=default_chosen_ext, psite_name_map=psite_name_map)

        # Diagnostics for missing offset lengths
        seen, miss = 0, 0

        # Iterate CDS blocks only; per-block binary search
        for block in tm.block_meta:
            chrom, bstart, bend = block["chrom"], block["start"], block["end"]
            try:
                it = bamfile.fetch(chrom, bstart, bend)
            except Exception:
                continue
            # Collect arrays to enable partial vectorization
            p_sites = []
            within_frames = []
            codon_indices = []

            for read in it:
                if read.is_unmapped or read.mapping_quality < 1 or read.is_secondary or read.is_supplementary:
                    continue
                try:
                    if read.has_tag("NH") and read.get_tag("NH") > 1:
                        continue
                except Exception:
                    pass
                rlen = read.infer_query_length(always=True)
                if rlen is None or rlen < min_read_length or rlen > max_read_length:
                    continue
                seen += 1
                off = offsets.get(int(rlen))
                if off is None:
                    miss += 1
                    continue
                p = psite_for_read(read, chosen_ext, off)
                if p < bstart or p >= bend:
                    continue
                idx = block_binary_search_codon(block, p)
                if idx is None:
                    continue
                # within-codon offset (strand-aware)
                cchrom, cs, ce = tm.codon_map[idx]
                within = (p - cs) if tm.strand == "+" else ((ce - 1) - p)
                frame = within % 3
                p_sites.append(p)
                within_frames.append(frame)
                codon_indices.append(idx)

            if codon_indices:
                codon_indices = np.asarray(codon_indices, dtype=np.int64)
                within_frames = np.asarray(within_frames, dtype=np.int64)
                # Vectorized accumulation
                np.add.at(per_bam, (codon_indices, within_frames), 1)
            counts += per_bam

        if diagnostics_acc is not None and seen > 0:
            frac = miss / float(seen)
            diagnostics_acc.setdefault("missing_offset_frac", []).append((os.path.basename(bam), frac))
            if frac > 0.20:
                logger.warning(f"{os.path.basename(bam)}: {miss}/{seen} reads ({frac:.1%}) had lengths missing from P-site offsets.")

        # Save cache
        if cache_root and not cache_used:
            try:
                np.save(cache_path, per_bam)
            except Exception:
                pass

        bamfile.close()

    return counts  # n_codons x 3

# =============================================================================
# HMM (3 states)
# =============================================================================

class ThreeStateHMM:
    """
    States: 0 -> F0, 1 -> F+1, 2 -> F-1 (represented as dominant 0,+1,+2 respectively)
    Emissions: per state, multinomial probs over (0,+1,+2)
    Transitions: high self-transition; symmetric small switch prob
    """
    def __init__(self, pi=None, A=None, emissions=None):
        self.pi = pi if pi is not None else np.array([0.9, 0.05, 0.05], dtype=float)
        if A is None:
            stay = 0.995
            move = (1.0 - stay) / 2.0
            self.A = np.array([
                [stay, move, move],
                [move, stay, move],
                [move, move, stay]
            ], dtype=float)
        else:
            self.A = A.astype(float)
        if emissions is None:
            self.emissions = np.array([
                [0.90, 0.05, 0.05],  # F0
                [0.10, 0.80, 0.10],  # F+1
                [0.10, 0.10, 0.80],  # F-1
            ], dtype=float)
        else:
            self.emissions = emissions.astype(float)
      

    def emission_loglik(self, counts):
        T = counts.shape[0]
        logE = np.zeros((T, 3), dtype=float)
        ps = np.clip(self.emissions, 1e-6, 1.0)
        logps = np.log(ps)
        for s in range(3):
            logE[:, s] = (counts * logps[s, :]).sum(axis=1)
        return logE

    def viterbi(self, counts):
        T = counts.shape[0]
        if T == 0:
            return np.array([], dtype=int), 0.0
        logE = self.emission_loglik(counts)
        logA = np.log(self.A)
        logPi = np.log(self.pi)

        dp = np.zeros((T, 3), dtype=float)
        ptr = np.zeros((T, 3), dtype=int)
        dp[0, :] = logPi + logE[0, :]
        ptr[0, :] = -1
        for t in range(1, T):
            for s in range(3):
                prev_vals = dp[t-1, :] + logA[:, s]
                j = int(np.argmax(prev_vals))
                dp[t, s] = prev_vals[j] + logE[t, s]
                ptr[t, s] = j
        last_state = int(np.argmax(dp[T-1, :]))
        logprob = float(dp[T-1, last_state])

        states = np.zeros(T, dtype=int)
        states[T-1] = last_state
        for t in range(T-2, -1, -1):
            states[t] = ptr[t+1, states[t+1]]
        return states, logprob

# =============================================================================
# Locus calling helpers
# =============================================================================

def first_offframe_run(states, min_persist=6):
    """Find first run where state != 0 with length >= min_persist."""
    T = len(states)
    i = 0
    while i < T:
        if states[i] == 0:
            i += 1; continue
        s = states[i]
        j = i
        while j+1 < T and states[j+1] == s:
            j += 1
        if (j - i + 1) >= min_persist:
            return (i, j, s)
        i = j + 1
    return None

def stroke_vs_sham_window_counts(counts_case, counts_ctrl, i, j):
    """Return (case_vec, ctrl_vec) where each is [F0,F+1,F-1] over codon window [i..j]."""
    c = counts_case[i:j+1, :].sum(axis=0)
    s = counts_ctrl[i:j+1, :].sum(axis=0)
    return c, s

def g_test_3x2(case_vec, ctrl_vec):
    """G-test (LLR) for 3x2 table; returns (G, pval or None if SciPy missing). df=2."""
    O = np.vstack([case_vec, ctrl_vec]).astype(float)  # 2x3
    colsums = O.sum(axis=0, keepdims=True)
    rowsums = O.sum(axis=1, keepdims=True)
    total = O.sum()
    if total <= 0 or (colsums == 0).any() or (rowsums == 0).any():
        return 0.0, None
    E = rowsums @ (colsums / total)  # 2x3
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(O > 0, O * np.log(O / E), 0.0)
    G = 2.0 * np.nansum(term)
    if SCIPY_OK:
        p = chi2.sf(G, df=2)
        return float(G), float(p)
    else:
        return float(G), None

def mask_counts(counts, i, j, pad):
    """Zero out counts in window ±pad codons to allow searching another locus."""
    out = counts.copy()
    a = max(0, i - pad); b = min(counts.shape[0]-1, j + pad)
    out[a:b+1, :] = 0
    return out

def extract_cds_window_seq(
    tm: TranscriptModel,
    win_start_codon: int,
    win_end_codon: int,
    flank_nt: int = 0,
    fasta: pysam.FastaFile = None
) -> Tuple[str, int, int]:
    """
    Codon-stitched CDS window extractor.

    Returns:
        (window_seq_rna, genomic_start, genomic_end)

    - The window is defined in CODON SPACE: [win_start_codon .. win_end_codon].
    - We NEVER pull a contiguous genomic region; instead we fetch each codon (3 nt)
      from cds codon_map and stitch them in transcript order.
    - flank_nt is converted to an approximate flank in codons:
        flank_codons = ceil(flank_nt / 3)
      and applied in codon space, so we never drag in introns.

    Notes:
    - window_seq_rna is always in transcript 5'→3' sense with U instead of T,
      regardless of genomic strand.
    - genomic_start / genomic_end are the min/max genomic coordinates that
      contributed to this stitched window (useful for BED or sanity checks),
      but the actual sequence ignores introns.
    """
    if fasta is None:
        raise ValueError("extract_cds_window_seq requires an open FASTA handle (fasta).")

    codon_map = tm.codon_map
    n_codons = len(codon_map)
    if n_codons == 0:
        return "", 0, 0

    # --- Normalize and clip codon window ---
    win_start_codon = max(0, int(win_start_codon))
    win_end_codon   = min(n_codons - 1, int(win_end_codon))
    if win_start_codon > win_end_codon:
        win_start_codon, win_end_codon = win_end_codon, win_start_codon

    # --- Convert flank_nt → flank_codons (stay in codon space) ---
    flank_codons = 0
    if flank_nt and flank_nt > 0:
        flank_codons = int(math.ceil(float(flank_nt) / 3.0))

    codon_start = max(0, win_start_codon - flank_codons)
    codon_end   = min(n_codons - 1, win_end_codon + flank_codons)

    dna_chunks: List[str] = []
    global_start: Optional[int] = None
    global_end: Optional[int] = None

    if tm.strand == "+":
        # Transcript 5'→3' is low → high genomic coordinates.
        for k in range(codon_start, codon_end + 1):
            _, cs, ce = codon_map[k]  # (chrom, start, end)
            piece = fasta.fetch(tm.chrom, cs, ce).upper()
            dna_chunks.append(piece)
            if global_start is None or cs < global_start:
                global_start = cs
            if global_end is None or ce > global_end:
                global_end = ce
        dna = "".join(dna_chunks)
        rna = _to_rna(dna)
    else:
        # Transcript 5'→3' is high → low genomic coordinates.
        # Fetch codons in REVERSE index order, then reverse-complement once.
        for k in range(codon_end, codon_start - 1, -1):
            _, cs, ce = codon_map[k]
            piece = fasta.fetch(tm.chrom, cs, ce).upper()
            dna_chunks.append(piece)
            if global_start is None or cs < global_start:
                global_start = cs
            if global_end is None or ce > global_end:
                global_end = ce
        dna = "".join(dna_chunks)
        rna = _revcomp_rna(dna)

    if global_start is None or global_end is None:
        # Should not happen if n_codons > 0, but guard anyway.
        global_start, global_end = 0, 0

    # rna length is guaranteed to be 3 * number_of_codons_in_window
    return rna, int(global_start), int(global_end)


# =============================================================================
# Per-gene analysis
# =============================================================================

def worker_analyze_gene(args):
    (
        gene, tm, case_bams, ctrl_bams, psite_dir,
        min_rl, max_rl,
        min_persist, smoothing_k,
        allow_multiple, mask_codons,
        out_plot_cmp,
        default_chosen_ext, psite_name_map,
        pre_gate_reads, cache_root,
        make_plots,
        gtest_alpha, min_window_reads,
        diagnostics_store,
        min_reads_group,
        aw_flank_up, aw_flank_down, aw_use_min_seed,
    ) = args

    loci = analyze_gene_for_comparison(
        gene, tm, case_bams, ctrl_bams, psite_dir,
        min_rl, max_rl,
        min_persist, smoothing_k,
        allow_multiple, mask_codons,
        out_plot_cmp,
        default_chosen_ext, psite_name_map,
        pre_gate_reads, cache_root,
        make_plots,
        gtest_alpha, min_window_reads,
        diagnostics_store,
        aw_flank_up, aw_flank_down, aw_use_min_seed,
    )

    # gene-level gate
    kept = [
        loc for loc in loci
        if loc["n_reads_case"] >= min_reads_group and loc["n_reads_ctrl"] >= min_reads_group
    ]
    return kept


def analyze_gene_for_comparison(
    gene, tm, case_bams, ctrl_bams, psite_dir,
    min_read_length, max_read_length,
    min_persist_codons, smoothing_k,
    allow_multiple_loci, mask_codons_each_side,
    out_plot_dir,
    default_chosen_ext, psite_name_map,
    pre_gate_reads, cache_root,
    make_plots,
    gtest_alpha, min_window_reads,
    diagnostics_store,
    aw_flank_up, aw_flank_down, aw_use_min_seed,
):
    """Returns list of dicts (one per locus) + optional plot saved to out_plot_dir/gene.png"""
    
    logger.debug(f"Analyzing {tm.gene} ({tm.transcript}), strand={tm.strand}")
    # Per-group counts (with diagnostics + cache + pre-gate)
    diag_case = {}
    diag_ctrl = {}

    counts_case = count_frames_per_codon_for_group(
        tm, case_bams, psite_dir, min_read_length, max_read_length,
        default_chosen_ext=default_chosen_ext, psite_name_map=psite_name_map,
        min_reads_gate=pre_gate_reads, cache_root=cache_root, diagnostics_acc=diag_case
    )
    counts_ctrl = count_frames_per_codon_for_group(
        tm, ctrl_bams, psite_dir, min_read_length, max_read_length,
        default_chosen_ext=default_chosen_ext, psite_name_map=psite_name_map,
        min_reads_gate=pre_gate_reads, cache_root=cache_root, diagnostics_acc=diag_ctrl
    )

    diagnostics_store["per_bam_missing_offsets"].extend(diag_case.get("missing_offset_frac", []))
    diagnostics_store["per_bam_missing_offsets"].extend(diag_ctrl.get("missing_offset_frac", []))

    # Coverage gate (avoid ultra-low)
    total_case = int(counts_case.sum())
    total_ctrl = int(counts_ctrl.sum())
    if total_case < 1 or total_ctrl < 1:
        return []

    # Smoothing on *proportions* for plotting; HMM uses raw counts
    def frame_props(counts):
        tot = counts.sum(axis=1, keepdims=True) + 1e-9
        props = counts / tot
        if smoothing_k > 0:
            props = np.vstack([
                moving_average(props[:,0], smoothing_k),
                moving_average(props[:,1], smoothing_k),
                moving_average(props[:,2], smoothing_k),
            ]).T
        return props

    props_case = frame_props(counts_case)
    props_ctrl = frame_props(counts_ctrl)

    hmm = ThreeStateHMM()
    remaining_case = counts_case.copy()
    loci = []
    tried = 0
    max_loci = 3 if allow_multiple_loci else 1
    reported_sham_windows = []

    while tried < max_loci and remaining_case.sum() > 0:
        states, lp = hmm.viterbi(remaining_case)
        found = first_offframe_run(states, min_persist=min_persist_codons)
        if not found:
            break
        i, j, s = found  # s: 1 -> F+1, 2 -> F-1
        # ─── Define slip seed and analysis window in codon space ─────
        # Seed = minimal off-frame core
        if aw_use_min_seed:
            seed_start = i
            seed_end   = min(j, i + min_persist_codons - 1)
        else:
            # use full HMM run as seed
            seed_start = i
            seed_end = seed_start + min_persist_codons - 1

        # Analysis window = ± flank codons around the seed
        n_codons = len(tm.codon_map)
        win_codon_start = max(0, seed_start - aw_flank_up)
        win_codon_end   = min(n_codons - 1, seed_end + aw_flank_down)

        # We'll still keep i,j as the "HMM run" boundaries, but we also store the
        # seed and analysis window explicitly in the locus record below.
        case_vec, ctrl_vec = stroke_vs_sham_window_counts(counts_case, counts_ctrl, win_codon_start, win_codon_end)
        win_reads_case = int(case_vec.sum())
        win_reads_ctrl = int(ctrl_vec.sum())
        win_reads_total = win_reads_case + win_reads_ctrl

        # Minimum coverage requirement inside window
        if win_reads_total < max(min_window_reads, 0):
            remaining_case = mask_counts(remaining_case, i, j, mask_codons_each_side)
            tried += 1
            continue

        # Direction dominance & G-test
        # s = HMM state: 1 (= F+1), 2 (= F-1)
        if tm.strand == "+":
            direction = "+1" if s == 1 else "-1"
        
        G, pval = g_test_3x2(case_vec, ctrl_vec)
        pval_pass = (pval is not None and pval <= gtest_alpha)

        # Determine case vs sham dominance by proportions
        def dom_prop(v, sstate):
            if sstate == 1:  # +1
                return float(v[1] / max(v.sum(), 1e-9))
            else:            # -1
                return float(v[2] / max(v.sum(), 1e-9))

        p_case_dom = dom_prop(case_vec, s)
        p_ctrl_dom = dom_prop(ctrl_vec, s)
        dscore = float(p_case_dom - p_ctrl_dom)

        keep = False
        if pval is not None:
            keep = pval_pass and (dscore > 0.0)
        else:
            # If SciPy missing, fallback to diff-only
            keep = (dscore > 0.0)

        # If not case-dominant, optionally record as present in Sham (QC) and continue
        present_in = "Case" if keep else "Sham"
        if not keep:
            reported_sham_windows.append((i, j, direction, G, pval, p_case_dom, p_ctrl_dom))
            remaining_case = mask_counts(remaining_case, i, j, mask_codons_each_side)
            tried += 1
            continue

        # Save locus
        gstart = min(tm.codon_map[i][1], tm.codon_map[j][1])
        gend   = max(tm.codon_map[i][2], tm.codon_map[j][2])
        mid_codon = int((i + j) // 2)
        gmid = int((tm.codon_map[mid_codon][1] + tm.codon_map[mid_codon][2]) // 2)

        # Gene-level QC
        off_cds_case = float((counts_case[:,1].sum() + counts_case[:,2].sum()) / max(total_case,1))
        off_cds_ctrl = float((counts_ctrl[:,1].sum() + counts_ctrl[:,2].sum()) / max(total_ctrl,1))
        win_off_case = float((case_vec[1] + case_vec[2]) / max(case_vec.sum(),1))
        win_off_ctrl = float((ctrl_vec[1] + ctrl_vec[2]) / max(ctrl_vec.sum(),1))

        locus = {
            "gene": tm.gene,
            "transcript": tm.transcript,
            "chrom": tm.chrom,
            "strand": tm.strand,
            "cds_len_nt": tm.cds_len_nt,
            "n_codons": len(tm.codon_map),

            # HMM run boundaries
            "locus_codon_start": int(i),
            "locus_codon_end": int(j),
            "locus_codon_peak": mid_codon,
            "slip_codon": int(i),

            # NEW: slip seed core (minimal off-frame region)
            "seed_codon_start": int(seed_start),
            "seed_codon_end": int(seed_end),

            # NEW: analysis window in codon space (for stall / structure / 3-nt decay)
            "window_codon_start": int(win_codon_start),
            "window_codon_end": int(win_codon_end),

            "genome_start": int(gstart),
            "genome_end": int(gend),
            "genome_mid": gmid,
            "direction": direction,
            "persistence_codons": int(j - i + 1),
            "dominant_prop_case": float(p_case_dom),
            "dominant_prop_ctrl": float(p_ctrl_dom),
            "stroke_minus_sham_diff": float(dscore),
            "G_stat": float(G),                               # NEW
            "p_value": (None if pval is None else float(pval)),
            "method": "HMM",
            "n_reads_case": total_case,                       # NEW: total CDS reads (gene-level)
            "n_reads_ctrl": total_ctrl,                       # NEW
            "win_reads_case": win_reads_case,                 # NEW: window reads
            "win_reads_ctrl": win_reads_ctrl,                 # NEW
            "off_frame_frac_cds_case": off_cds_case,          # NEW
            "off_frame_frac_cds_ctrl": off_cds_ctrl,          # NEW
            "off_frame_frac_win_case": win_off_case,          # NEW
            "off_frame_frac_win_ctrl": win_off_ctrl           # NEW
        }
        loci.append(locus)
        
        try:
            # Access runtime options 
            emit_ctx = globals().get("_EMIT_CTX", {})
            struct_cfg = globals().get("_STRUCT_SCAN", {})
            stall_cfg = globals().get("_STALL_SCAN", {})
            fasta_path = globals().get("_GENOME_FA_PATH", None)

            # 7a) sequences & codon tokens
            if emit_ctx.get("enable_sequences", False) and fasta_path:
                fa = pysam.FastaFile(fasta_path)  # open per-process
                try:
                    seq_rna, w_s, w_e = extract_cds_window_seq(
                        tm,
                        win_codon_start,
                        win_codon_end,
                        flank_nt=int(emit_ctx.get("flank_nt", 20)),
                        fasta=fa
                    )
                                   
                    locus["window_seq_rna"] = seq_rna
                    locus["window_start_ext"] = int(w_s)
                    locus["window_end_ext"] = int(w_e)

                    if emit_ctx.get("enable_codon_tokens", False):
                        flank_c = int(emit_ctx.get("flank_codons", 9))
                        codons, ca, cb = codons_in_frame_around(
                            tm, win_codon_start, win_codon_end,
                            flank_codons=flank_c, fa=fa
                        )
                        locus["codon_tokens"] = ",".join(codons)
                        locus["codon_win_first_idx"] = int(ca)
                        locus["codon_win_last_idx"]  = int(cb)
                finally:
                    try: fa.close()
                    except Exception: pass

            # 7b) stall codon flags
            if stall_cfg.get("enable", False) and "codon_tokens" in locus:
                toks = locus["codon_tokens"].split(",")
                flags = flag_stall_codon_motifs(
                    toks,
                    proline_runs=tuple(stall_cfg.get("proline_runs",[2,3])),
                    polybasic_min=int(stall_cfg.get("polybasic_min",6)),
                    de_to_p=bool(stall_cfg.get("de_to_p", True))
                )
                for k,v in flags.items():
                    locus[k] = bool(v)

            # 7c) downstream structure scan (RNAfold)
            if struct_cfg.get("enable", False) and ("window_seq_rna" in locus):
                # Build a downstream sequence aligned to transcript sense:
                # use window centered on the locus; downstream in RNA sense is to the right.
                seq = locus["window_seq_rna"]
                offA, offB = struct_cfg.get("offset_nt",[6,18])
                win = int(struct_cfg.get("window_nt",30))
                rnafold_bin = struct_cfg.get("rnafold_bin","RNAfold")
                scores = downstream_structure_scores(seq, offsets=(int(offA), int(offB)), win=win,
                                                     rnafold_bin=rnafold_bin)
                locus.update(scores)

        except Exception as _e:
            # We keep locus even if enrichments fail
            logger.debug(f"Context enrichment failed for {tm.gene}: {_e}")
        

        # Mask and search again
        remaining_case = mask_counts(remaining_case, i, j, mask_codons_each_side)
        tried += 1

    # Mini-plot
    if make_plots and len(props_case) > 0:
        try:
            ensure_dir(out_plot_dir)
            fig, ax = plt.subplots(figsize=(9, 3.2))
            x = np.arange(len(props_case))
            ax.plot(x, props_case[:,0], label="Case F0", linewidth=1.5)
            ax.plot(x, props_case[:,1], label="Case F+1", linewidth=1.0)
            ax.plot(x, props_case[:,2], label="Case F-1", linewidth=1.0)
            ax.plot(x, props_ctrl[:,0], linestyle="--", label="Sham F0", linewidth=1.0)
            ax.plot(x, props_ctrl[:,1], linestyle="--", label="Sham F+1", linewidth=1.0)
            ax.plot(x, props_ctrl[:,2], linestyle="--", label="Sham F-1", linewidth=1.0)
            # Shade loci kept
            for loc in loci:
                si, sj = loc["locus_codon_start"], loc["locus_codon_end"]
                ax.axvspan(si, sj, alpha=0.15, ymin=0, ymax=1)
            ax.set_title(f"{tm.gene} ({tm.transcript}) frame proportions across CDS")
            ax.set_xlabel("CDS codon index")
            ax.set_ylabel("Proportion")
            ax.set_ylim(0, 1.0)
            ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.3))
            plt.tight_layout()
            outpng = os.path.join(out_plot_dir, f"{tm.gene}.png")
            plt.savefig(outpng, dpi=220)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Plot failed for {tm.gene}: {e}")

    return loci

# =============================================================================
# Significant gene lists per comparison
# =============================================================================

def load_sig_genes(sig_csv, p_col="padj", logfc_col="logFC", p_thr=0.05, lfc_thr=1.0, name_norm=True):
    df = pd.read_csv(sig_csv)
    cols = {c.lower(): c for c in df.columns}
    pcol = cols.get(p_col.lower(), None)
    lfcc = cols.get(logfc_col.lower(), None)
    gcol = None
    for cand in ["gene_id", "gene", "symbol"]:
        if cand in cols:
            gcol = cols[cand]
            break
    if pcol is None or lfcc is None or gcol is None:
        raise ValueError(f"CSV {sig_csv} must contain gene_id (or gene/symbol), logFC, p columns.")
    filt = df[(df[pcol] <= p_thr) & (np.abs(df[lfcc]) >= lfc_thr)]
    genes = [str(x) for x in filt[gcol].tolist()]
    if name_norm:
        genes = [normalize_gene_name(x) for x in genes]
    genes = sorted(list(set(genes)))
    return genes

def load_sig_genes_split(sig_csv, p_col="padj", logfc_col="logFC",
                         p_thr=0.05, lfc_thr=1.0, name_norm=True):
    """
    Return two sets of genes: UP (logFC >= lfc_thr) and DOWN (logFC <= -lfc_thr),
    both with p <= p_thr. Names normalized to match GTF if name_norm=True.
    """
    df = pd.read_csv(sig_csv)
    cols = {c.lower(): c for c in df.columns}
    pcol = cols.get(p_col.lower())
    lfcc = cols.get(logfc_col.lower())
    gcol = None
    for cand in ["gene_id", "gene", "symbol"]:
        if cand in cols:
            gcol = cols[cand]
            break
    if pcol is None or lfcc is None or gcol is None:
        raise ValueError(f"CSV {sig_csv} must contain gene_id (or gene/symbol), logFC, p columns.")

    df = df[df[pcol] <= p_thr].copy()
    up_df   = df[df[lfcc] >=  lfc_thr]
    down_df = df[df[lfcc] <= -lfc_thr]

    def _norm_list(x):
        xs = [str(v) for v in x]
        if name_norm:
            xs = [normalize_gene_name(v) for v in xs]
        return sorted(set(xs))

    up_genes   = _norm_list(up_df[gcol].tolist())
    down_genes = _norm_list(down_df[gcol].tolist())
    return up_genes, down_genes

# =============================================================================
# BED12 writer for locus windows
# =============================================================================

def write_bed12(loci, out_bed, name_field="gene"):
    with open(out_bed, "w") as f:
        for i, loc in enumerate(loci):
            chrom = loc["chrom"]
            start = loc["genome_start"]
            end   = loc["genome_end"]
            name  = f'{loc[name_field]}|{loc["direction"]}|{loc["locus_codon_start"]}-{loc["locus_codon_end"]}'
            score = 0
            strand = loc["strand"]
            thickStart = start
            thickEnd = end
            rgb = "255,0,0" if loc["direction"] == "+1" else "0,0,255"
            blockCount = 1
            blockSizes = f"{end-start},"
            blockStarts = "0,"
            f.write("\t".join(map(str, [
                chrom, start, end, name, score, strand,
                thickStart, thickEnd, rgb, blockCount, blockSizes, blockStarts
            ])) + "\n")

# =============================================================================
# Main driver
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python frameshift_loci.py config.yaml  "
              "or: python frameshift_loci.py --selftest")
        sys.exit(1)
        
    # optional internal sanity check
    if sys.argv[1] == "--selftest":
        _self_test_strand_frame_mapping()
        return
    cfg = read_yaml(sys.argv[1])
    

    out_root = ensure_dir("frameshift_loci")
    log_dir  = ensure_dir(os.path.join(out_root, "logs"))
    loci_dir = ensure_dir(os.path.join(out_root, "loci_calls"))
    plot_dir = ensure_dir(os.path.join(out_root, "per_gene_plots"))
    tmp_dir  = ensure_dir(os.path.join(out_root, "tmp"))
        # ── Cache config toggles into module globals so worker processes can read them
    global _EMIT_CTX, _STRUCT_SCAN, _STALL_SCAN, _GENOME_FA_PATH
    _EMIT_CTX = cfg.get("emit_context", {}) or {}
    _STRUCT_SCAN = cfg.get("structure_scan", {}) or {}
    _STALL_SCAN = cfg.get("stall_codon_scan", {}) or {}
    _GENOME_FA_PATH = cfg.get("genome_fasta", None)  # used by per-process FASTA open


    # Early validation + provenance dump (NEW)
    validate_inputs(cfg, log_dir)

    # Params
    psite_dir = cfg.get("psite_offsets_dir", "psite_offsets")
    default_chosen_ext = cfg.get("default_chosen_ext", "5end")
    psite_name_map = cfg.get("psite_name_map", {}) or {}

    gtf = cfg["gtf"]
    genome_fa = cfg.get("genome_fasta", None)
    groups = cfg["groups"]
    comparisons = cfg["comparisons"]

    # knobs
    min_rl = cfg.get("min_read_length", 25)
    max_rl = cfg.get("max_read_length", 36)
    min_reads_group = cfg.get("min_total_reads_per_gene_per_group", 20)
    pre_gate_reads = min_reads_group  # NEW: use same threshold for quick pre-gate
    min_persist = cfg.get("min_persistence_codons", 6)
    smoothing_k = cfg.get("codon_bin_smoothing", 1)
    allow_multiple = cfg.get("allow_multiple_loci", True)
    mask_nt = cfg.get("locus_refit_mask_nt", 90)
    mask_codons = max(1, mask_nt // 3)
    split_by_direction = bool(cfg.get("split_by_direction", False))
    
    # NEW: analysis window configuration
    aw_cfg = cfg.get("analysis_window", {}) or {}
    aw_flank_up = int(aw_cfg.get("flank_up_codons", 10))
    aw_flank_down = int(aw_cfg.get("flank_down_codons", 10))
    aw_use_min_seed = bool(aw_cfg.get("use_min_seed", True))

    p_thr = cfg.get("fs_p_cut", 0.05)
    lfc_thr = cfg.get("fs_lfc_cut", 1.0)

    max_workers = cfg.get("max_workers", 8)
    make_plots = bool(cfg.get("make_plots", True))
    cache_counts = bool(cfg.get("cache_counts", False))
    cache_root = tmp_dir if cache_counts else None

    gtest_alpha = float(cfg.get("gtest_alpha", 0.05))          # NEW
    min_window_reads = int(cfg.get("min_window_reads", 30))    # NEW

    name_norm = bool(cfg.get("name_normalization", True))
    transcript_choice = cfg.get("transcript_choice", "longest_cds")
    whitelist_path = cfg.get("transcript_whitelist", None)

    # Build transcript models
    gene_models = build_transcript_models(
        gtf,
        transcript_choice=transcript_choice,
        nm_prefer=True,
        whitelist_path=whitelist_path,
        name_norm=name_norm
    )
    
    # ─── Strand sanity check ─────────────────────────────────────────────
    from collections import Counter
    strand_counts = Counter(tm.strand for tm in gene_models.values())
    logger.info(f"Strand distribution in transcript models: {strand_counts}")

    # For each comparison
    for cmp in comparisons:
        case = cmp["case"]
        ctrl = cmp["control"]
        sig_csv = cmp["sig_genes_csv"]
        logger.info(f"Comparison: {case} vs {ctrl}  | sig list: {sig_csv}")

        case_bams = groups[case]
        ctrl_bams = groups[ctrl]

        if split_by_direction:
            up_genes, down_genes = load_sig_genes_split(
                sig_csv, p_thr=p_thr, lfc_thr=lfc_thr, name_norm=name_norm
            )
            dir_sets = [("UP", up_genes), ("DOWN", down_genes)]
        else:
            sig_genes = load_sig_genes(sig_csv, p_thr=p_thr, lfc_thr=lfc_thr, name_norm=name_norm)
            dir_sets = [("ALL", sig_genes)]

        for tag, gene_list in dir_sets:
            logger.info(f"{case} vs {ctrl} [{tag}]: {len(gene_list)} significant genes before GTF match")

            # Filter by available models
            genes_run = [g for g in gene_list if g in gene_models]
            missing = set(gene_list) - set(genes_run)
            if missing:
                logger.warning(f"{len(missing)} [{tag}] genes not in GTF; e.g., {list(itertools.islice(missing,5))} ...")

            out_tsv = os.path.join(loci_dir, f"{case}_vs_{ctrl}_{tag}_loci.tsv")
            out_bed = os.path.join(loci_dir, f"{case}_vs_{ctrl}_{tag}.bed12")
            out_plot_cmp = ensure_dir(os.path.join(plot_dir, f"{case}_vs_{ctrl}", tag))

            loci_all = []
            diagnostics_store = {"per_bam_missing_offsets": []}

            args_list = [
                (
                    g,
                    gene_models[g],
                    case_bams,
                    ctrl_bams,
                    psite_dir,
                    min_rl,
                    max_rl,
                    min_persist,
                    smoothing_k,
                    allow_multiple,
                    mask_codons,
                    out_plot_cmp,
                    default_chosen_ext,
                    psite_name_map,
                    pre_gate_reads,
                    cache_root,
                    make_plots,
                    gtest_alpha,
                    min_window_reads,
                    diagnostics_store,
                    min_reads_group,
                    aw_flank_up, aw_flank_down, aw_use_min_seed,
                )
                for g in genes_run
            ]

            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(worker_analyze_gene, a): a[0] for a in args_list}
                for fut in as_completed(futs):
                    gene = futs[fut]
                    try:
                        res = fut.result()
                        if res:
                            loci_all.extend(res)
                    except Exception as e:
                        logger.error(f"Gene {gene} [{tag}] failed: {e}")

            if loci_all:
                loci_df = pd.DataFrame(loci_all).sort_values(["gene", "locus_codon_start"])
                # --- Deduplicate loci that have identical windows and stats ---
                key_cols = [
                        "gene", "transcript",
                        "window_codon_start", "window_codon_end",
                        "dominant_prop_case", "dominant_prop_ctrl",
                        "G_stat", "p_value"
                ]

                if all(k in loci_df.columns for k in key_cols):
                        loci_df = (
                        loci_df
                        .sort_values("persistence_codons")
                        .drop_duplicates(subset=key_cols, keep="first")
                        )
                loci_all = loci_df.to_dict(orient="records")
                loci_df.to_csv(out_tsv, sep="\t", index=False)
                logger.info(f"Wrote loci table: {out_tsv} ({len(loci_df)} rows)")
                write_bed12(loci_all, out_bed)
                logger.info(f"Wrote BED12: {out_bed}")
            else:
                logger.warning(f"No loci detected for {case} vs {ctrl} [{tag}]")

            if diagnostics_store["per_bam_missing_offsets"]:
                dpath = os.path.join(log_dir, f"missing_offsets_{case}_vs_{ctrl}_{tag}.tsv")
                pd.DataFrame(diagnostics_store["per_bam_missing_offsets"],
                             columns=["bam", "missing_offset_fraction"]).to_csv(dpath, sep="\t", index=False)
                logger.info(f"Wrote P-site offset diagnostics: {dpath}")
                
                            # Optional FASTA dump from source step
            if (_EMIT_CTX.get("enable_sequences", False) and _EMIT_CTX.get("save_window_fasta", False)):
                fasta_path = os.path.join(loci_dir, f"{case}_vs_{ctrl}_{tag}_contexts.fa")
                with open(fasta_path, "w") as f:
                    for loc in loci_all:
                        if "window_seq_rna" not in loc:
                            continue
                        sid = f"{case}_vs_{ctrl}_{tag}|{loc['gene']}|{loc['transcript']}|{loc['chrom']}:{loc['window_start_ext']}-{loc['window_end_ext']}({loc['strand']})|{loc['direction']}"
                        f.write(f">{sid}\n{loc['window_seq_rna']}\n")
                logger.info(f"Wrote contexts FASTA: {fasta_path}")


    logger.info("Done.")


if __name__ == "__main__":
    main()

