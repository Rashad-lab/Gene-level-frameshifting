#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fs_impact_analyzer.py

Analyze the impact of detected frameshift (FS) loci on coding sequences.

Transcript-aware:
  * Uses the transcript ID in master_summary when available.
  * Falls back to choosing the transcript whose CDS exons cover the FS locus.
  * All sequences and impact calls are done per transcript, not per gene.

Inputs
------
  --config  : YAML with at least:
                genome_fasta: path to genome FASTA (indexed)
                gtf         : path to GTF (RefSeq/GENCODE-style, with CDS/start_codon/stop_codon)
  --master  : post_fs_context/master_summary.tsv
              (must have: gene, chrom, strand, window_start, window_end, direction;
               optionally: transcript, comparison, tag)
  --outdir  : output directory (default: fs_impact)
  --n-workers : number of worker processes
  
Usage: 
python fs_impact_analyzer.py      --config config.yaml      --master post_fs_context/master_summary.tsv      --outdir fs_impact      --n-workers 6


For each FS locus:
  1) Select a transcript model:
       - Prefer the transcript in master_summary
       - Fallback: transcript whose CDS exons cover the FS position
  2) Map FS locus to CDS coordinate using genomic position (window midpoint).
  3) Build frameshifted CDS + AA sequence (CDS + 3'UTR context).
  4) Flag:
       - is_ptc  : frameshift introduces a stop upstream of canonical stop (PTC)
       - is_scrt : frameshift bypasses canonical stop and terminates downstream (SCRT-like extension)
       - is_nmd  : PTC is >~50 nt upstream of last CDS exon-exon junction

Outputs
-------
  <outdir>/impact_summary.tsv
  <outdir>/canonical_cds_rna.fa      (per gene–transcript)
  <outdir>/frameshift_cds_rna.fa     (per FS event–transcript)
  <outdir>/canonical_protein.fa      (per gene–transcript)
  <outdir>/frameshift_protein.fa     (per FS event–transcript)
"""

import os
import sys
import argparse
from collections import defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml
import numpy as np
import pandas as pd
import pysam

# AA translation table (RNA codons)
AA_MAP = {
    "UUU": "F","UUC": "F","UUA": "L","UUG": "L",
    "CUU": "L","CUC": "L","CUA": "L","CUG": "L",
    "AUU": "I","AUC": "I","AUA": "I","AUG": "M",
    "GUU": "V","GUC": "V","GUA": "V","GUG": "V",
    "UCU": "S","UCC": "S","UCA": "S","UCG": "S","AGU": "S","AGC": "S",
    "CCU": "P","CCC": "P","CCA": "P","CCG": "P",
    "ACU": "T","ACC": "T","ACA": "T","ACG": "T",
    "GCU": "A","GCC": "A","GCA": "A","GCG": "A",
    "UAU": "Y","UAC": "Y","UAA": "*","UAG": "*",
    "CAU": "H","CAC": "H","CAA": "Q","CAG": "Q",
    "AAU": "N","AAC": "N","AAA": "K","AAG": "K",
    "GAU": "D","GAC": "D","GAA": "E","GAG": "E",
    "UGU": "C","UGC": "C","UGA": "*","UGG": "W",
    "CGU": "R","CGC": "R","CGA": "R","CGG": "R","AGA": "R","AGG": "R",
    "GGU": "G","GGC": "G","GGA": "G","GGG": "G"
}

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def read_yaml(path: str):
    with open(path, "r") as fh:
        return yaml.safe_load(fh)

def to_rna(seq: str) -> str:
    return seq.upper().replace("T", "U")

def revcomp_dna(seq: str) -> str:
    tr = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
    return seq.translate(tr)[::-1]

def translate_rna(rna: str) -> str:
    aa = []
    L = (len(rna) // 3) * 3
    for i in range(0, L, 3):
        cod = rna[i:i+3]
        aa.append(AA_MAP.get(cod, "X"))
    return "".join(aa)

# ---------------------------------------------------------------------------
# GTF parsing and transcript models
# ---------------------------------------------------------------------------

GtfRow = namedtuple("GtfRow", ["chrom","source","feature","start","end","score","strand","frame","attrs"])

def parse_gtf_attributes(attr_str: str) -> dict:
    out = {}
    for part in attr_str.strip().split(";"):
        part = part.strip()
        if not part:
            continue
        if " " not in part:
            continue
        key, val = part.split(" ", 1)
        val = val.strip().strip('"')
        out[key] = val
    return out

def load_gtf_minimal(gtf_path: str):
    """
    Parse GTF; keep only CDS, 3UTR, start_codon, stop_codon.
    Returns dict: gene_name -> transcript_id -> dict of raw features.
    """
    gene_models = defaultdict(lambda: defaultdict(lambda: {
        "chrom": None,
        "strand": None,
        "cds_exons": [],   # list of (start0, end0) 0-based half-open
        "utr3_exons": [],  # list of (start0, end0)
        "start_codons": [],
        "stop_codons": []
    }))

    with open(gtf_path, "r") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, source, feature, start, end, score, strand, frame, attrs = parts
            if feature not in ("CDS", "3UTR", "start_codon", "stop_codon"):
                continue
            attrs_dict = parse_gtf_attributes(attrs)
            gene_name = attrs_dict.get("gene_name") or attrs_dict.get("gene_id")
            transcript_id = attrs_dict.get("transcript_id")
            if not gene_name or not transcript_id:
                continue
            start = int(start)
            end = int(end)
            # Convert to 0-based half-open
            start0 = start - 1
            end0 = end

            mtx = gene_models[gene_name][transcript_id]
            mtx["chrom"] = chrom
            mtx["strand"] = strand

            if feature == "CDS":
                mtx["cds_exons"].append((start0, end0))
            elif feature == "3UTR":
                mtx["utr3_exons"].append((start0, end0))
            elif feature == "start_codon":
                mtx["start_codons"].append((start0, end0))
            elif feature == "stop_codon":
                mtx["stop_codons"].append((start0, end0))

    # Sort exons in transcription order (strand-aware)
    for gene_name, txs in gene_models.items():
        for tx_id, tx in txs.items():
            strand = tx["strand"]
            if strand == "+":
                tx["cds_exons"].sort(key=lambda x: x[0])
                tx["utr3_exons"].sort(key=lambda x: x[0])
            else:
                tx["cds_exons"].sort(key=lambda x: x[0], reverse=True)
                tx["utr3_exons"].sort(key=lambda x: x[0], reverse=True)
    return gene_models

def build_transcript_models(gtf_path: str):
    """
    Build per-transcript CDS models.

    Returns:
      models[gene_name][transcript_id] = {
        "gene": gene_name,
        "tx": transcript_id,
        "chrom": chrom,
        "strand": strand,
        "cds_exons": [(start0,end0),...], transcription order
        "utr3_exons": [(start0,end0),...],
        "canonical_cds_len_nt": int,
        "exon_cum_len": [0, len(exon0), len(exon0)+len(exon1), ...],
        "last_junction_cds_pos_nt": int or None
      }
    """
    raw = load_gtf_minimal(gtf_path)
    models = defaultdict(dict)

    for gene_name, txs in raw.items():
        for tx_id, tx in txs.items():
            cds_exons = tx["cds_exons"]
            if not cds_exons:
                # transcript has no CDS; skip
                continue
            strand = tx["strand"]
            chrom = tx["chrom"]
            utr3_exons = tx["utr3_exons"]

            exon_lens = [e[1] - e[0] for e in cds_exons]
            canonical_cds_len_nt = int(sum(exon_lens))
            if canonical_cds_len_nt <= 0:
                continue

            exon_cum_len = [0]
            for L in exon_lens[:-1]:
                exon_cum_len.append(exon_cum_len[-1] + L)

            if len(cds_exons) >= 2:
                last_junction_cds_pos_nt = exon_cum_len[-1]  # start of last CDS exon
            else:
                last_junction_cds_pos_nt = None

            models[gene_name][tx_id] = {
                "gene": gene_name,
                "tx": tx_id,
                "chrom": chrom,
                "strand": strand,
                "cds_exons": cds_exons,
                "utr3_exons": utr3_exons,
                "canonical_cds_len_nt": canonical_cds_len_nt,
                "exon_cum_len": exon_cum_len,
                "last_junction_cds_pos_nt": last_junction_cds_pos_nt
            }

    # Drop genes with no CDS-bearing transcripts
    models = {g: txs for g, txs in models.items() if len(txs) > 0}
    return models

# ---------------------------------------------------------------------------
# Mapping and impact
# ---------------------------------------------------------------------------

def genomic_to_cds_nt_index(model, gpos: int):
    """
    Map genomic position (0-based) to CDS nt index (0-based from CDS start),
    using exon structure in transcription order.
    Returns None if gpos not inside CDS.
    """
    cds_exons = model["cds_exons"]
    strand = model["strand"]
    exon_cum = model["exon_cum_len"]

    for idx, (start0, end0) in enumerate(cds_exons):
        if strand == "+":
            if start0 <= gpos < end0:
                offset = gpos - start0
                return exon_cum[idx] + offset
        else:
            # minus strand, exons ordered 5'->3' which is high->low genomic
            if start0 <= gpos < end0:
                # on minus, 5' is at end0-1
                offset_from_5prime = (end0 - 1) - gpos
                return exon_cum[idx] + offset_from_5prime
    return None

def fetch_cds_and_utr3_rna(model, fa: pysam.FastaFile):
    chrom = model["chrom"]
    strand = model["strand"]
    cds_exons = model["cds_exons"]
    utr3_exons = model["utr3_exons"]

    cds_seq = []
    for start0, end0 in cds_exons:
        dna = fa.fetch(chrom, start0, end0)
        cds_seq.append(dna)
    cds_dna = "".join(cds_seq)

    utr3_seq = []
    for start0, end0 in utr3_exons:
        dna = fa.fetch(chrom, start0, end0)
        utr3_seq.append(dna)
    utr3_dna = "".join(utr3_seq)

    if strand == "-":
        cds_dna = revcomp_dna(cds_dna)
        utr3_dna = revcomp_dna(utr3_dna)

    cds_rna = to_rna(cds_dna)
    utr3_rna = to_rna(utr3_dna) if utr3_dna else ""
    return cds_rna, utr3_rna

def frameshift_orf(cds_plus_utr_rna: str, fs_cds_nt: int, direction: str):
    """
    Given CDS+3'UTR RNA sequence and FS position in CDS nt space,
    return (fs_rna, fs_aa, stop_nt_pos_from_cds_start or None).

    direction: "+1" or "-1"
    """
    L = len(cds_plus_utr_rna)
    if fs_cds_nt < 0 or fs_cds_nt >= L:
        return "", "", None

    if direction == "+1":
        start_nt = fs_cds_nt + 1
    elif direction == "-1":
        start_nt = fs_cds_nt - 1
    else:
        # treat unknown as no shift, but this should not happen
        start_nt = fs_cds_nt

    start_nt = max(0, min(start_nt, L - 1))
    fs_rna = cds_plus_utr_rna[start_nt:]

    aa = []
    stop_pos = None  # nt index from CDS start where STOP codon begins
    L3 = (len(fs_rna) // 3) * 3
    for i in range(0, L3, 3):
        cod = fs_rna[i:i+3]
        a = AA_MAP.get(cod, "X")
        aa.append(a)
        if a == "*":
            stop_pos = start_nt + i  # global nt index in CDS+UTR
            break
    aa_seq = "".join(aa)
    if stop_pos is None:
        # no in-frame stop
        return fs_rna, aa_seq, None
    else:
        # truncate RNA at stop codon end for consistency
        stop_end = (stop_pos - start_nt) + 3
        fs_rna_trunc = fs_rna[:stop_end]
        return fs_rna_trunc, aa_seq, stop_pos

# ---------------------------------------------------------------------------
# Transcript selection helpers
# ---------------------------------------------------------------------------

def pick_model_for_row(models_for_gene: dict, row) -> tuple:
    """
    Select which transcript model to use for this FS row.

    Preference:
      1) Use row["transcript"] if present in models_for_gene and chrom matches.
      2) Otherwise, choose transcript whose CDS exons contain FS midpoint
         (ignoring strand from master_summary; we trust GTF strand).
         If multiple, take the one with longest CDS.
      3) If none, return (None, None).
    """
    chrom = str(row["chrom"])
    ws = int(row["window_start"])
    we = int(row["window_end"])
    fs_mid = (ws + we) // 2

    # 1) direct transcript match: make sure chromosome agrees
    tx = str(row.get("transcript", "NA"))
    if tx in models_for_gene:
        model = models_for_gene[tx]
        if model["chrom"] == chrom:
            return tx, model
        # if chrom mismatches, fall through to genomic fallback

    # 2) fallback: any transcript whose CDS covers fs_mid on this chromosome
    candidates = []
    for tx_id, model in models_for_gene.items():
        if model["chrom"] != chrom:
            continue
        for start0, end0 in model["cds_exons"]:
            if start0 <= fs_mid < end0:
                candidates.append((tx_id, model))
                break

    if not candidates:
        return None, None

    # pick one with longest CDS
    candidates.sort(key=lambda x: x[1]["canonical_cds_len_nt"], reverse=True)
    tx_id, model = candidates[0]
    return tx_id, model

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker_one_gene(args):
    (gene, models_for_gene, fa_path, df_gene) = args
    fa = pysam.FastaFile(fa_path)
    try:
        out_rows = []
        fs_rna_records = []
        fs_aa_records = []
        canon_rna_records = {}  # key: (gene, tx) -> (header, seq)
        canon_aa_records = {}   # key: (gene, tx) -> (header, seq)

        # cache CDS/UTR/AA per transcript to avoid re-fetch
        tx_seq_cache = {}

        for idx, row in df_gene.iterrows():
            chrom = str(row["chrom"])
            strand = str(row["strand"])
            comparison = str(row.get("comparison", "NA"))
            tag = str(row.get("tag", "NA"))
            direction = str(row.get("direction", "NA"))
            row_tx = str(row.get("transcript", "NA"))

            ws = int(row["window_start"])
            we = int(row["window_end"])
            fs_mid = (ws + we) // 2  # 0-based genomic

            # pick transcript model
            model_tx_id, model = pick_model_for_row(models_for_gene, row)

            if model is None:
                # no usable transcript model (FS likely outside CDS in all isoforms)
                out_rows.append({
                    "gene": gene,
                    "transcript": row_tx,
                    "model_tx": "NA",
                    "comparison": comparison,
                    "tag": tag,
                    "direction": direction,
                    "chrom": chrom,
                    "strand": strand,
                    "window_start": ws,
                    "window_end": we,
                    "fs_event_id": f"{gene}_idx{idx}",
                    "fs_cds_nt": np.nan,
                    "fs_stop_nt_from_cds_start": np.nan,
                    "canonical_cds_len_nt": np.nan,
                    "canonical_aa_len": np.nan,
                    "is_ptc": False,
                    "is_scrt": False,
                    "is_nmd": False,
                    "note": "No transcript CDS covers FS (FS outside CDS in all isoforms)"
                })
                continue

            # get CDS + UTR sequences for this transcript (from cache or FASTA)
            if model_tx_id not in tx_seq_cache:
                cds_rna, utr3_rna = fetch_cds_and_utr3_rna(model, fa)
                cds_plus_utr = cds_rna + utr3_rna
                canonical_aa_full = translate_rna(cds_rna)
                if "*" in canonical_aa_full:
                    canonical_len_aa = canonical_aa_full.index("*")
                else:
                    canonical_len_aa = len(canonical_aa_full)
                tx_seq_cache[model_tx_id] = (
                    cds_rna,
                    utr3_rna,
                    cds_plus_utr,
                    canonical_aa_full,
                    canonical_len_aa,
                )

                # record canonical sequences (one per gene–transcript)
                canon_rna_header = f"{gene}|{model_tx_id}|canonical"
                canon_aa_header = f"{gene}|{model_tx_id}|canonical"
                canon_rna_records[(gene, model_tx_id)] = (canon_rna_header, cds_rna)
                canon_aa_records[(gene, model_tx_id)] = (canon_aa_header, canonical_aa_full)

            cds_rna, utr3_rna, cds_plus_utr, canonical_aa_full, canonical_len_aa = tx_seq_cache[model_tx_id]
            canonical_cds_len_nt = model["canonical_cds_len_nt"]
            last_junction_nt = model["last_junction_cds_pos_nt"]

            fs_cds_nt = genomic_to_cds_nt_index(model, fs_mid)
            if fs_cds_nt is None:
                # FS is outside the CDS of this chosen transcript
                out_rows.append({
                    "gene": gene,
                    "transcript": row_tx,
                    "model_tx": model_tx_id,
                    "comparison": comparison,
                    "tag": tag,
                    "direction": direction,
                    "chrom": chrom,
                    "strand": strand,
                    "window_start": ws,
                    "window_end": we,
                    "fs_event_id": f"{gene}_idx{idx}",
                    "fs_cds_nt": np.nan,
                    "fs_stop_nt_from_cds_start": np.nan,
                    "canonical_cds_len_nt": canonical_cds_len_nt,
                    "canonical_aa_len": canonical_len_aa,
                    "is_ptc": False,
                    "is_scrt": False,
                    "is_nmd": False,
                    "note": "FS outside CDS of selected transcript"
                })
                continue

            if direction not in {"+1", "-1"}:
                out_rows.append({
                    "gene": gene,
                    "transcript": row_tx,
                    "model_tx": model_tx_id,
                    "comparison": comparison,
                    "tag": tag,
                    "direction": direction,
                    "chrom": chrom,
                    "strand": strand,
                    "window_start": ws,
                    "window_end": we,
                    "fs_event_id": f"{gene}_idx{idx}",
                    "fs_cds_nt": fs_cds_nt,
                    "fs_stop_nt_from_cds_start": np.nan,
                    "canonical_cds_len_nt": canonical_cds_len_nt,
                    "canonical_aa_len": canonical_len_aa,
                    "is_ptc": False,
                    "is_scrt": False,
                    "is_nmd": False,
                    "note": "Unknown direction"
                })
                continue

            # compute frameshifted ORF
            fs_rna, fs_aa, stop_nt_pos = frameshift_orf(
                cds_plus_utr, fs_cds_nt, direction
            )

            if stop_nt_pos is None:
                is_ptc = False
                is_scrt = False
                is_nmd = False
            else:
                is_ptc = stop_nt_pos < canonical_cds_len_nt
                is_scrt = stop_nt_pos > canonical_cds_len_nt
                if is_ptc and last_junction_nt is not None:
                    is_nmd = stop_nt_pos <= (last_junction_nt - 50)
                else:
                    is_nmd = False

            out_rows.append({
                "gene": gene,
                "transcript": row_tx,
                "model_tx": model_tx_id,
                "comparison": comparison,
                "tag": tag,
                "direction": direction,
                "chrom": chrom,
                "strand": strand,
                "window_start": ws,
                "window_end": we,
                "fs_event_id": f"{gene}_idx{idx}",
                "fs_cds_nt": fs_cds_nt,
                "fs_stop_nt_from_cds_start": float(stop_nt_pos) if stop_nt_pos is not None else np.nan,
                "canonical_cds_len_nt": canonical_cds_len_nt,
                "canonical_aa_len": canonical_len_aa,
                "is_ptc": bool(is_ptc),
                "is_scrt": bool(is_scrt),
                "is_nmd": bool(is_nmd),
                "note": ""
            })

            # record frameshifted sequences for FASTA export
            if fs_rna:
                header = f"{gene}|{model_tx_id}|{comparison}|{tag}|{direction}|idx{idx}"
                fs_rna_records.append((header, fs_rna))
                fs_aa_records.append((header, fs_aa))

        # flatten canonical records to lists
        canon_rna_list = list(canon_rna_records.values())
        canon_aa_list = list(canon_aa_records.values())

        return {
            "gene": gene,
            "rows": out_rows,
            "canonical_rna_records": canon_rna_list,
            "canonical_aa_records": canon_aa_list,
            "fs_rna_records": fs_rna_records,
            "fs_aa_records": fs_aa_records
        }

    finally:
        fa.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="config.yaml with genome_fasta and gtf")
    ap.add_argument("--master", required=True,
                    help="post_fs_context/master_summary.tsv")
    ap.add_argument("--outdir", default="fs_impact",
                    help="Output directory")
    ap.add_argument("--n-workers", type=int, default=4,
                    help="Number of worker processes")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    fa_path = cfg.get("genome_fasta")
    gtf_path = cfg.get("gtf")
    if not fa_path or not os.path.isfile(fa_path):
        raise SystemExit(f"genome_fasta not found: {fa_path}")
    if not gtf_path or not os.path.isfile(gtf_path):
        raise SystemExit(f"gtf not found: {gtf_path}")

    outdir = ensure_dir(args.outdir)

    # Load master summary
    df = pd.read_csv(args.master, sep="\t")
    required_cols = {"gene", "chrom", "strand", "window_start", "window_end", "direction"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"{args.master} missing required columns: {missing}")

    # Default values for optional columns
    if "comparison" not in df.columns:
        df["comparison"] = "NA"
    if "tag" not in df.columns:
        df["tag"] = "NA"
    if "transcript" not in df.columns:
        df["transcript"] = "NA"

    # Normalize direction labels
    df["direction"] = df["direction"].astype(str).str.strip().replace({
        "1": "+1", "+1": "+1", "-1": "-1"
    })

    # Build transcript models
    print("Building transcript CDS models from GTF...")
    tx_models = build_transcript_models(gtf_path)
    print(f"Built models for {len(tx_models)} genes with CDS transcripts.")

    # Subset to genes that have both FS loci and transcript models
    genes_with_fs = sorted(df["gene"].dropna().unique().tolist())
    genes_to_run = [g for g in genes_with_fs if g in tx_models]

    print(f"Processing {len(genes_to_run)} genes with FS loci and CDS models...")

    all_rows = []
    canon_rna_records = []
    canon_aa_records = []
    fs_rna_records_all = []
    fs_aa_records_all = []

    work_items = []
    for g in genes_to_run:
        models_for_gene = tx_models[g]
        df_g = df[df["gene"] == g].copy()
        work_items.append((g, models_for_gene, fa_path, df_g))

    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futs = {ex.submit(worker_one_gene, wi): wi[0] for wi in work_items}
        for fut in as_completed(futs):
            gene = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"Gene {gene} failed: {e}", file=sys.stderr)
                continue
            all_rows.extend(res["rows"])
            canon_rna_records.extend(res["canonical_rna_records"])
            canon_aa_records.extend(res["canonical_aa_records"])
            fs_rna_records_all.extend(res["fs_rna_records"])
            fs_aa_records_all.extend(res["fs_aa_records"])

    # Write summary TSV
    if all_rows:
        df_out = pd.DataFrame(all_rows)
        df_out.to_csv(os.path.join(outdir, "impact_summary.tsv"), sep="\t", index=False)
        print(f"Impact summary → {os.path.join(outdir, 'impact_summary.tsv')}")
    else:
        print("No impact rows produced.")

    # Write FASTAs
    def write_fasta(path, records):
        if not records:
            return
        # deduplicate by header (keep first occurrence)
        seen = set()
        dedup = []
        for h, s in records:
            if h in seen:
                continue
            seen.add(h)
            dedup.append((h, s))

        with open(path, "w") as fh:
            for header, seq in dedup:
                fh.write(f">{header}\n")
                # wrap at 60 nt/aa per line
                for i in range(0, len(seq), 60):
                    fh.write(seq[i:i+60] + "\n")

    if canon_rna_records:
        write_fasta(os.path.join(outdir, "canonical_cds_rna.fa"), canon_rna_records)
    if canon_aa_records:
        write_fasta(os.path.join(outdir, "canonical_protein.fa"), canon_aa_records)
    if fs_rna_records_all:
        write_fasta(os.path.join(outdir, "frameshift_cds_rna.fa"), fs_rna_records_all)
    if fs_aa_records_all:
        write_fasta(os.path.join(outdir, "frameshift_protein.fa"), fs_aa_records_all)

    print("All done.")

if __name__ == "__main__":
    main()

