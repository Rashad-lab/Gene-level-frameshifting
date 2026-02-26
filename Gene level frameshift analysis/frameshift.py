#!/usr/bin/env python3
import os
import sys
import yaml
import logging
import pandas as pd
import pysam
import numpy as np
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from intervaltree import Interval, IntervalTree

# =============================================================================
# Setup Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Config & Annotation Loading
# =============================================================================
def load_config(config_path):
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

def load_annotation(annotation_file):
    """
    Load BED annotation file.
    Assumes BED with at least 6 columns: chrom, start, end, gene, ., strand.
    Returns a dictionary keyed by chromosome with a list of gene records.
    """
    annotation = {}
    try:
        with open(annotation_file, 'r') as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 6:
                    continue
                chrom, start, end, gene, _, strand = fields[:6]
                start, end = int(start), int(end)
                rec = {"chrom": chrom, "start": start, "end": end, "gene": gene, "strand": strand}
                if chrom not in annotation:
                    annotation[chrom] = []
                annotation[chrom].append(rec)
        logger.info(f"Loaded annotation for {len(annotation)} chromosomes from {annotation_file}")
        return annotation
    except Exception as e:
        logger.error(f"Error reading annotation file: {e}")
        sys.exit(1)

# =============================================================================
# Building Interval Trees for Start Codons
# =============================================================================
def build_start_codon_trees(annotation):
    """
    Build an interval tree for each chromosome representing the start codon.
    For plus-strand genes, we use the first base (start).
    For minus-strand genes, we use the last base (end-1, because BED is 0-based).
    Returns a dictionary: {chrom: IntervalTree}
    """
    trees = {}
    for chrom, genes in annotation.items():
        tree = IntervalTree()
        for rec in genes:
            if rec["strand"] == "+":
                tree.addi(rec["start"], rec["start"] + 1, rec)
            elif rec["strand"] == "-":
                tree.addi(rec["end"] - 1, rec["end"], rec)
        trees[chrom] = tree
    logger.info("Built start codon interval trees for annotation")
    return trees

# =============================================================================
# Helper Functions for Offset Calculation
# =============================================================================
def tempoff(v_dist):
    """
    Given a list of candidate distances (v_dist), compute the mode that is higher
    than its immediate neighbors. Mimics the RiboWaltz tempoff() function.
    """
    if not v_dist:
        return None
    counts = Counter(v_dist)
    all_keys = list(range(min(v_dist), max(v_dist)+1))
    freq = {k: counts.get(k, 0) for k in all_keys}
    # Sort candidates in descending order of frequency
    sorted_candidates = sorted(all_keys, key=lambda x: freq[x], reverse=True)
    for candidate in sorted_candidates:
        left = freq.get(candidate - 1, 0)
        right = freq.get(candidate + 1, 0)
        if freq[candidate] > left and freq[candidate] > right:
            return candidate
    return None

def adj_off(v_dist, add, bestoff):
    """
    Correction step to adjust the temporary offset. Builds a frequency profile for
    the range [min(v_dist)-2, max(v_dist)+add] and finds local maxima.
    Returns the candidate (local maximum) closest to bestoff.
    """
    if not v_dist:
        return bestoff
    min_val = min(v_dist)
    max_val = max(v_dist)
    keys = list(range(min_val - 2, max_val + add + 1))
    freq = {k: 0 for k in keys}
    for val in v_dist:
        if val in freq:
            freq[val] += 1
    # Adjust the first two bins to avoid edge issues
    if min_val in freq:
        base_val = freq[min_val]
        freq[min_val - 2] = base_val + 1
        freq[min_val - 1] = base_val + 1
    locmax = []
    # Find local maxima: candidate k is a local max if freq[k] > freq[k-1] and > freq[k+1]
    for k in keys[1:-1]:
        if freq[k] > freq.get(k-1, 0) and freq[k] > freq.get(k+1, 0):
            locmax.append(k)
    if not locmax:
        return bestoff
    # Choose the candidate with minimum absolute difference to bestoff
    adj = min(locmax, key=lambda x: abs(x - bestoff))
    return adj

# =============================================================================
# Enhanced P-site Offset Calculation
# =============================================================================
def calculate_psite_offsets(bam_path, min_len, max_len, start_tree_by_chrom, flanking=6, use_start=True):
    """
    For each read overlapping a gene's start codon, compute two distances:
      - site_dist_end5: difference between the read's 5' end and the CDS start.
      - site_dist_end3: difference between the read's 3' end and the CDS start.
    Only reads with sufficient flanking (site_dist_end5 <= -flanking and site_dist_end3 >= flanking - 1)
    are retained.
    
    For each read length (within min_len and max_len), candidate offsets are computed using
    tempoff() on the occupancy profiles from the 5' and 3' ends.
    
    Then, a global decision is made (using an "auto" approach) to choose the optimal extremity
    (either 5' or 3') based on aggregated percentages. Finally, a correction step (adj_off) is applied.
    
    Returns a tuple:
      (final_offsets, chosen_extremity, occupancy_info)
    where final_offsets is a dict mapping read_length -> corrected offset,
    chosen_extremity is either "5end" or "3end", and occupancy_info is a dataframe with details.
    """
    occ_5 = defaultdict(list)
    occ_3 = defaultdict(list)
    counts_by_length = Counter()
    total_accepted = 0

    try:
        bamfile = pysam.AlignmentFile(bam_path, "rb")
    except Exception as e:
        logger.error(f"Error opening BAM file {bam_path}: {e}")
        return {}, None, None

    # Iterate over reads and collect occupancy information for reads that overlap a start codon
    for read in bamfile.fetch(until_eof=True):
        if read.is_unmapped or read.mapping_quality < 1:
            continue
        try:
            if read.has_tag("NH") and read.get_tag("NH") > 1:
                continue
        except Exception:
            pass

        read_length = read.infer_query_length(always=True)
        if read_length is None or read_length < min_len or read_length > max_len:
            continue

        chrom = bamfile.get_reference_name(read.reference_id)
        if chrom not in start_tree_by_chrom:
            continue

        # For simplicity, we treat plus-strand reads here.
        # (For negative strand, analogous logic with reversed coordinates should be applied.)
        if not read.is_reverse:
            cds_candidates = list(start_tree_by_chrom[chrom].overlap(read.reference_start, read.reference_end))
            # Filter to only include reads that fully cover the start codon
            valid = []
            for iv in cds_candidates:
                gene_rec = iv.data
                # Check strand compatibility
                if gene_rec["strand"] != "+":
                    continue
                cds_start = gene_rec["start"]
                # For plus strand, 5' end = read.reference_start, 3' end = read.reference_end
                site_dist_end5 = read.reference_start - cds_start
                site_dist_end3 = read.reference_end - cds_start
                # Apply flanking filter
                if site_dist_end5 <= -flanking and site_dist_end3 >= (flanking - 1):
                    valid.append((site_dist_end5, site_dist_end3))
            if len(valid) != 1:
                continue  # only use unambiguous reads
            sd5, sd3 = valid[0]
        else:
            # For negative strand, assume annotation is set such that cds_start is the start codon (which is at gene end)
            cds_candidates = list(start_tree_by_chrom[chrom].overlap(read.reference_start, read.reference_end))
            valid = []
            for iv in cds_candidates:
                gene_rec = iv.data
                if gene_rec["strand"] != "-":
                    continue
                # For negative strand, define:
                # 5' end is read.reference_end and 3' end is read.reference_start.
                cds_start = gene_rec["end"]  # using gene end as the start codon for negative strand
                site_dist_end5 = read.reference_end - cds_start
                site_dist_end3 = read.reference_start - cds_start
                if site_dist_end5 >= flanking and site_dist_end3 <= -(flanking - 1):
                    valid.append((site_dist_end5, site_dist_end3))
            if len(valid) != 1:
                continue
            sd5, sd3 = valid[0]

        occ_5[read_length].append(sd5)
        occ_3[read_length].append(sd3)
        counts_by_length[read_length] += 1
        total_accepted += 1

    bamfile.close()
    if total_accepted == 0:
        logger.warning(f"No reads passed the flanking filter in {bam_path}")
        return {}, None, None

    # For each read length, compute candidate offsets from the 5' and 3' occupancy lists
    candidate_info = []
    for rl in sorted(occ_5.keys()):
        cand5 = tempoff(occ_5[rl])
        cand3 = tempoff(occ_3[rl])
        pct = (counts_by_length[rl] / total_accepted) * 100
        candidate_info.append({
            "length": rl,
            "candidate_5": cand5,
            "candidate_3": cand3,
            "percentage": pct
        })
    occupancy_df = pd.DataFrame(candidate_info)

    # Group by candidate offset values to aggregate percentages
    group5 = occupancy_df.dropna(subset=["candidate_5"]).groupby("candidate_5")["percentage"].sum()
    group3 = occupancy_df.dropna(subset=["candidate_3"]).groupby("candidate_3")["percentage"].sum()
    best5_offset = group5.idxmax() if not group5.empty else None
    best5_pct = group5.max() if not group5.empty else 0
    best3_offset = group3.idxmax() if not group3.empty else None
    best3_pct = group3.max() if not group3.empty else 0
    min_rl = min(occupancy_df["length"]) if not occupancy_df.empty else None

    # Decide on extremity: mimic R conditions.
    # For simplicity, if best3_pct > best5_pct and best3_offset <= (min_rl - 2), choose "3end"; else "5end".
    if best3_offset is not None and ((best3_pct > best5_pct and best3_offset <= (min_rl - 2)) or (best3_pct > best5_pct)):
        chosen_ext = "3end"
        global_best = best3_offset
    else:
        chosen_ext = "5end"
        global_best = best5_offset if best5_offset is not None else best3_offset

    # Now, apply correction (adj_off) for each read length using the chosen extremity
    final_offsets = {}
    for rl in occupancy_df["length"]:
        if chosen_ext == "3end":
            corrected = adj_off(occ_3[rl], add=0, bestoff=global_best)
        else:
            corrected = abs(adj_off(occ_5[rl], add=1, bestoff=global_best))
        final_offsets[rl] = corrected

    logger.info(f"Calculated P-site offsets for {bam_path} using {chosen_ext} with global best offset {global_best}")
    return final_offsets, chosen_ext, occupancy_df

# =============================================================================
# Frameshift Counting Using Corrected Offsets
# =============================================================================
def count_frameshift(bam_path, offsets, chosen_ext, annotation, min_len, max_len):
    """
    Count reads per gene into in-frame (0) and off-frame (+1, +2) bins,
    using the computed P-site offsets.
    For each read:
      - Determine its read length and get the corresponding offset.
      - For plus strand:
            if chosen_ext=="5end": psite = read.reference_start + offset
            if chosen_ext=="3end": psite = read.reference_end - offset
      - For minus strand, analogous adjustments are made.
      - Find which CDS (from the annotation) covers the computed P-site.
      - Compute the reading frame.
    Returns a dictionary:
      { gene: { 'frame0': count, 'frame+1': count, 'frame+2': count } }
    """
    counts = {}
    try:
        bamfile = pysam.AlignmentFile(bam_path, "rb")
    except Exception as e:
        logger.error(f"Error opening BAM file {bam_path}: {e}")
        return counts

    for read in bamfile.fetch(until_eof=True):
        if read.is_unmapped or read.mapping_quality < 1:
            continue
        try:
            if read.has_tag("NH") and read.get_tag("NH") > 1:
                continue
        except Exception:
            pass

        read_length = read.infer_query_length(always=True)
        if read_length is None or read_length < min_len or read_length > max_len:
            continue

        offset = offsets.get(read_length, None)
        if offset is None:
            continue

        # Compute P-site position based on chosen extremity and strand.
        if not read.is_reverse:
            if chosen_ext == "5end":
                psite = read.reference_start + offset
            else:  # "3end"
                psite = read.reference_end - offset
        else:
            # For negative strand, the logic is reversed.
            if chosen_ext == "5end":
                psite = read.reference_end - offset
            else:
                psite = read.reference_start + offset

        chrom = bamfile.get_reference_name(read.reference_id)
        if chrom not in annotation:
            continue

        # Find the gene whose CDS covers the psite.
        for gene_rec in annotation[chrom]:
            if gene_rec["start"] <= psite < gene_rec["end"]:
                if gene_rec["strand"] == "+":
                    frame = (psite - gene_rec["start"]) % 3
                else:
                    frame = (gene_rec["end"] - psite - 1) % 3
                gene = gene_rec["gene"]
                if gene not in counts:
                    counts[gene] = {"frame0": 0, "frame+1": 0, "frame+2": 0}
                if frame == 0:
                    counts[gene]["frame0"] += 1
                elif frame == 1:
                    counts[gene]["frame+1"] += 1
                elif frame == 2:
                    counts[gene]["frame+2"] += 1
                break  # avoid ambiguous counting
    bamfile.close()
    logger.info(f"Counted frameshift reads for {bam_path}")
    return counts

# =============================================================================
# Processing a Single BAM File
# =============================================================================
def process_bam_file(bam_path, config, annotation):
    """
    Process one BAM file:
      1. Build start codon trees.
      2. Calculate P-site offsets (using both 5' and 3' occupancy profiles,
         flanking filter, candidate offset determination, and correction).
      3. Count reads per gene in each frame (using the corrected offsets).
      4. Compute the frameshift score for each gene.
      5. Save the offsets and gene counts/FS scores to CSV.
    Returns a tuple (sample_name, sample_csv_path).
    """
    try:
        sample_name = os.path.splitext(os.path.basename(bam_path))[0]
        logger.info(f"Processing sample: {sample_name}")
        
        min_len = config["min_read_length"]
        max_len = config["max_read_length"]
        output_dir = config["output_dir"]
        
        psite_dir = os.path.join(output_dir, "psite_offsets")
        sample_out_dir = os.path.join(output_dir, "sample_counts")
        os.makedirs(psite_dir, exist_ok=True)
        os.makedirs(sample_out_dir, exist_ok=True)
        
        start_tree_by_chrom = build_start_codon_trees(annotation)
        
        # Enhanced P-site offset calculation
        offsets, chosen_ext, occ_df = calculate_psite_offsets(bam_path, min_len, max_len, start_tree_by_chrom, flanking=6, use_start=True)
        offset_file = os.path.join(psite_dir, f"{sample_name}_psite.txt")
        with open(offset_file, 'w') as f:
            f.write("read_length\tcorrected_offset\n")
            for rl, off in offsets.items():
                f.write(f"{rl}\t{off}\n")
        logger.info(f"Saved P-site offsets to {offset_file}")
        
        # Count frameshift reads using the computed offsets and chosen extremity
        gene_counts = count_frameshift(bam_path, offsets, chosen_ext, annotation, min_len, max_len)
        
        # Compute Frameshift Score (FS = (frame+1 + frame+2) / frame0)
        data = []
        for gene, counts in gene_counts.items():
            in_frame = counts["frame0"]
            off_frame = counts["frame+1"] + counts["frame+2"]
            fs_score = off_frame / in_frame if in_frame > 0 else np.nan
            data.append({
                "Gene": gene,
                "Counts_in_0_frame": in_frame,
                "Counts_in_+1_frame": counts["frame+1"],
                "Counts_in_+2_frame": counts["frame+2"],
                "Frameshift_score": fs_score
            })
        df = pd.DataFrame(data)
        sample_csv = os.path.join(sample_out_dir, f"{sample_name}_FS.csv")
        df.to_csv(sample_csv, index=False)
        logger.info(f"Saved frameshift analysis for {sample_name} to {sample_csv}")
        return sample_name, sample_csv
    except Exception as e:
        logger.error(f"Error processing {bam_path}: {e}")
        return None

# =============================================================================
# Merge and filter Sample Results
# =============================================================================
def merge_sample_results(sample_csv_paths, output_dir):

    merged_df = None

    # We will collect the names of frameshift and total count columns
    # so we can do filtering and final assembly easily.
    fs_columns = []
    total_count_columns = []

    for sample_name, csv_path in sample_csv_paths:
        df = pd.read_csv(csv_path)

        # Ensure the CSV has the columns we need
        required_cols = {
            "Gene", 
            "Counts_in_0_frame", 
            "Counts_in_+1_frame", 
            "Counts_in_+2_frame", 
            "Frameshift_score"
        }
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"File {csv_path} is missing one or more required columns: "
                f"{required_cols - set(df.columns)}"
            )

        # Compute total counts for filtering
        df["Total_Counts"] = (
            df["Counts_in_0_frame"] + 
            df["Counts_in_+1_frame"] + 
            df["Counts_in_+2_frame"]
        )

        # Rename columns to keep them unique across samples
        fs_col_name = f"{sample_name}_FS"
        total_col_name = f"{sample_name}_TotalCounts"

        df.rename(
            columns={
                "Frameshift_score": fs_col_name,
                "Total_Counts": total_col_name
            },
            inplace=True
        )

        # Keep only the columns we need for merging
        # (Gene, frameshift score, total counts)
        df = df[["Gene", fs_col_name, total_col_name]]

        # Merge into the main dataframe
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="Gene", how="outer")

        fs_columns.append(fs_col_name)
        total_count_columns.append(total_col_name)

    # -------------------------------------------------------------------------
    # 1) Filter: Keep only genes that have >= 10 total counts in EVERY sample
    # -------------------------------------------------------------------------
    condition = (merged_df[total_count_columns] >= 10).all(axis=1)
    merged_df = merged_df[condition].copy()

    # -------------------------------------------------------------------------
    # 2) Replace missing (NaN) frameshift scores with 10
    # -------------------------------------------------------------------------
    for col in fs_columns:
        merged_df[col] = merged_df[col].fillna(10)

    # -------------------------------------------------------------------------
    # Create the final FS matrix with just Gene + FS columns
    # -------------------------------------------------------------------------
    final_cols = ["Gene"] + fs_columns
    final_df = merged_df[final_cols]

    # Save the merged and filtered results
    merged_csv = os.path.join(output_dir, "merged_FS_matrix.csv")
    final_df.to_csv(merged_csv, index=False)
    logger.info(f"Merged FS score matrix saved to {merged_csv}")

    return merged_csv


# =============================================================================
# Main Function
# =============================================================================
def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python frameshift_analysis.py config.yaml")
        sys.exit(1)
    config_path = sys.argv[1]
    config = load_config(config_path)
    
    os.makedirs(config["output_dir"], exist_ok=True)
    annotation = load_annotation(config["annotation_file"])
    
    # Collect all BAM files from the config
    sample_files = []
    for group, files in config["samples"].items():
        for file in files:
            if os.path.exists(file):
                sample_files.append(file)
            else:
                logger.warning(f"BAM file not found: {file}")
    if not sample_files:
        logger.error("No BAM files found. Exiting.")
        sys.exit(1)
    
    sample_results = []
    with ProcessPoolExecutor(max_workers=config["max_threads"]) as executor:
        futures = {executor.submit(process_bam_file, bam, config, annotation): bam for bam in sample_files}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                sample_results.append(result)
    if not sample_results:
        logger.error("No sample results obtained. Exiting.")
        sys.exit(1)
    
    merge_sample_results(sample_results, config["output_dir"])
    logger.info("Frameshift analysis pipeline completed successfully.")

if __name__ == "__main__":
    main()
