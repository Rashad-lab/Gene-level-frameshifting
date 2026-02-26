#!/usr/bin/env python3
"""
create_annotation.py
---------------------

Input
-----
    mm39.ncbiRefSeq.gtf   (or any NCBI‐style RefSeq GTF)

Output (4 files + 1 QC file)
------
    annotation_CDS.bed        # raw CDSs  (0-based BED, identical to your awk)
    filtered_CDS.bed          # CDSs after 18-/15-nt trimming
    stop_codons.bed           # final 3-bp stop codon interval per transcript
    non_overlapping_3UTR.bed  # 100-bp downstream UTR minus CDS overlap
    overlaps_check.txt        # intersects filtered_CDS vs 3'UTR (sanity check)

Python ≥3.8, no external deps except (optionally) pybedtools for the subtract/intersect bits.

how to run:
chmod +x make_filtered_beds.py
./create_annotaion.py mm39.ncbiRefSeq.gtf 
"""

from collections import defaultdict, namedtuple
import sys, pathlib, re, tempfile, subprocess

# ---------- helpers ---------------------------------------------------------

GtfRow = namedtuple(
    "GtfRow",
    "chrom source feature start end score strand frame attr_raw "
)  # thin wrapper, ints for start/end

def parse_attr(attr_raw):
    """Return dict of key→value parsed from ninth column."""
    return {
        m.group(1): m.group(2)
        for m in re.finditer(r'(\S+) "([^"]+)"', attr_raw)
    }

def bed_line(chrom, start0, end, name, strand, extra="."):
    """Return a BED6 string (0-based, half-open)."""
    return f"{chrom}\t{start0}\t{end}\t{name}\t{extra}\t{strand}\n"

def write_lines(path, lines):
    with open(path, "w") as fh:
        fh.writelines(lines)

# ---------- 1. load GTF -----------------------------------------------------

gtf_in = sys.argv[1] if len(sys.argv) > 1 else "mm39.ncbiRefSeq.gtf"

transcripts = defaultdict(lambda: {"CDS": [], "start_codon": [], "stop_codon": []})

with open(gtf_in) as fh:
    for line in fh:
        if line.startswith("#") or not line.strip():
            continue
        cols = line.rstrip("\n").split("\t")
        row = GtfRow(cols[0], cols[1], cols[2],
                     int(cols[3]), int(cols[4]),
                     cols[5], cols[6], cols[7], cols[8])
        attrs = parse_attr(row.attr_raw)
        key = f'{attrs.get("gene_id","NA")};{attrs.get("transcript_id","NA")}'
        if row.feature in ("CDS", "start_codon", "stop_codon"):
            transcripts[key][row.feature].append(
                (row.chrom, row.start, row.end, attrs.get("gene_name", "NA"), row.strand)
            )

# ---------- 2. raw CDS BED --------------------------------------------------

raw_cds_lines = []
for cds_list in transcripts.values():
    for chrom, st, en, gname, strand in cds_list["CDS"]:
        raw_cds_lines.append(bed_line(chrom, st - 1, en, gname, strand))
write_lines("annotation_CDS.bed", raw_cds_lines)

# ---------- 3. trim first+last CDS exon -------------------------------------

trimmed_cds_lines = []

for key, feat in transcripts.items():
    cds = feat["CDS"]
    if not cds:
        continue
    # sort per genomic coordinate
    cds_sorted = sorted(cds, key=lambda x: x[1])  # by start
    strand = cds_sorted[0][4]

    # identify first/last in *transcript* order
    if strand == "+":
        first_idx, last_idx = 0, -1
    else:
        first_idx, last_idx = -1, 0  # reversed gene orientation

    # copy to mutable list
    cds_adj = [list(x) for x in cds_sorted]

    # --- trim 18 nt from first CDS exon (from start_codon side) --------------
    chrom, st, en, gname, _ = cds_adj[first_idx]
    if strand == "+":
        st += 18
        if st >= en:  # exon too short, drop it
            cds_adj[first_idx] = None
        else:
            cds_adj[first_idx][1] = st
    else:
        en -= 18
        if en <= st:
            cds_adj[first_idx] = None
        else:
            cds_adj[first_idx][2] = en

    # --- trim 15 nt from last CDS exon (from stop_codon side) ----------------
    chrom, st, en, gname, _ = cds_adj[last_idx] if cds_adj[last_idx] else (None,) * 5
    if chrom is not None:
        if strand == "+":
            en -= 15
            if en <= st:
                cds_adj[last_idx] = None
            else:
                cds_adj[last_idx][2] = en
        else:
            st += 15
            if st >= en:
                cds_adj[last_idx] = None
            else:
                cds_adj[last_idx][1] = st

    # write remaining CDS exons
    for item in cds_adj:
        if item:
            chrom, st, en, gname, strand = item
            trimmed_cds_lines.append(bed_line(chrom, st - 1, en, gname, strand))

write_lines("filtered_CDS.bed", trimmed_cds_lines)

# ---------- 4. stop-codon BED & 3'UTR seed ----------------------------------

stop_lines   = []
utr_seed     = []  # un-subtracted 100 nt windows

for key, feat in transcripts.items():
    cds = feat["CDS"]
    if not cds:
        continue
    cds_sorted = sorted(cds, key=lambda x: x[1])
    strand = cds_sorted[0][4]
    gene_name = cds_sorted[0][3]
    chrom, first_start, first_end, *_ = cds_sorted[0]
    chrom, last_start,  last_end,  *_ = cds_sorted[-1]

    if strand == "+":
        # stop codon at last_end (1-based gtf)
        stop_start0 = last_end - 3        # 0-based
        stop_end    = last_end
        utr_start0  = last_end            # immediately after CDS
        utr_end     = last_end + 100
    else:
        stop_start0 = last_start - 1      # last_start is 1-based; 0-based interval is (start-4, start-1]
        stop_end    = last_start + 2
        utr_start0  = last_start - 100 - 1
        utr_end     = last_start - 1

        # keep coords ordered low→high for BED
        stop_start0, stop_end = sorted((stop_start0, stop_end))
        utr_start0,  utr_end  = sorted((utr_start0, utr_end))

    stop_lines.append(bed_line(chrom, stop_start0, stop_end, gene_name, strand))
    utr_seed  .append(bed_line(chrom, utr_start0,  utr_end,  gene_name, strand))

write_lines("stop_codons.bed", stop_lines)
write_lines("initial_3UTR.bed", utr_seed)

# ---------- 5. subtract CDS from UTR seed & QC intersect --------------------

# (If pybedtools isn’t available you can shell-out to bedtools; both snippets below)

try:
    import pybedtools
    cds_bt   = pybedtools.BedTool("filtered_CDS.bed")
    utr_bt   = pybedtools.BedTool("initial_3UTR.bed")
    utr_clean = utr_bt.subtract(cds_bt)
    utr_clean.saveas("non_overlapping_3UTR.bed")

    # QC: any remaining overlaps?
    overlaps = utr_clean.intersect(cds_bt, wa=True, wb=True)
    overlaps.saveas("overlaps_check.txt")

except ModuleNotFoundError:
    # fall back to CLI bedtools
    tmpdir = tempfile.gettempdir()
    clean = pathlib.Path("non_overlapping_3UTR.bed")
    subprocess.run(
        ["bedtools", "subtract", "-a", "initial_3UTR.bed",
         "-b", "filtered_CDS.bed"],
        check=True, stdout=open(clean, "w"))
    subprocess.run(
        ["bedtools", "intersect", "-a", str(clean), "-b", "filtered_CDS.bed",
         "-wa", "-wb"],
        check=True, stdout=open("overlaps_check.txt", "w"))

print("✅  Finished.  BEDs written:\n"
      "   annotation_CDS.bed\n   filtered_CDS.bed\n"
      "   stop_codons.bed\n   non_overlapping_3UTR.bed\n"
      "   overlaps_check.txt")
