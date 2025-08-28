# CALCULATE AND PLOT AFS FROM VCF AND ARG (SEPARATE FOR SNP AND TE)
# denominator of AFS here is unmasked chromosome length
# Last updated by Regina August 28, 2025

import allel
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from Bio import SeqIO
import tskit

#--- Parameters ---#
# sample number
num_samples = 26

# SNP mutation rate
snp_mutation_rate = 3.3e-8

# TE mutation rate
te_mutation_rate = 3.3e-11

# Chromosome length - for chromosome 1
chrom_length = 308452471

#--- Input file paths ---#
# mut file from Relate - has mutation ages, positions, whether site flipped ref/alt
mut_file = "/group/jrigrp10/tebeiliu/te_evo/all_linked_sv_manisha/for_regina/te.snp.combined.chr.1.1196.masked.relate.mut"

# VCF containing only SNPs
snp_vcf_file = "/group/jrigrp10/tebeiliu/te_evo/all_linked_sv_manisha/for_regina/new_nam.1.format.addb73.vcf"

# VCF containing only TEs
te_vcf_file = "/group/jrigrp10/tebeiliu/te_evo/all_linked_sv_manisha/for_regina/all.linked.svequalte.chr.1.recode.vcf"

# ts file
trees_file = "/home/reginaf/beibei_afs/chr1.1196_1.1196_out.trees"

# Mask file
mask_file = "/group/jrigrp10/tebeiliu/te_evo/all_linked_sv_manisha/for_regina/mask_genome/1.fa"

#--- Load in files ---#
mut = pd.read_csv(mut_file, sep=';', low_memory=False)

snp_vcf = allel.read_vcf(snp_vcf_file)
te_vcf = allel.read_vcf(te_vcf_file)

ts = tskit.load(trees_file)

#--- Find flipped sites ---#
flipped = mut[mut['is_flipped'] == 1]
print(f"Number of flipped sites: {len(flipped.snp)}")

snp_flipped = list(set(snp_vcf['variants/POS']) & set(flipped.pos_of_snp))
print(f"Number of flipped sites in the SNP VCF: {len(snp_flipped)}")

te_flipped = list(set(te_vcf['variants/POS']) & set(flipped.pos_of_snp))
print(f"Number of flipped sites in the TE VCF: {len(te_flipped)}")

print(f"All of the flipped mutations are accounted for? {len(snp_flipped) + len(te_flipped) == len(flipped.snp)}")

#--- Find sites not mapped by Relate to drop ---#
# These are the sites that did not map
drop_not_mapped = mut[mut['is_not_mapping'] == 1]
print(f"Number of sites not mapped: {len(drop_not_mapped.snp)}")
drop_not_mapped_pos = drop_not_mapped['pos_of_snp']

#--- length ---#
# Loads in fasta file and counts number of "passing" which were not masked positions
for seq_record in SeqIO.parse(mask_file, "fasta"):
    print(seq_record.id)
    print(repr(seq_record.seq))
    print(len(seq_record))
    print(seq_record.count('N'))
    length = seq_record.count('P')

print(length)

#--- Calculate AFS on tree sequence ---#
# Calculate AFS on tree sequence - POLARIZED = TRUE FOR UNFOLDED, span_normalise = FALSE so that it doesn't divide by seq length (we want to do this on our own with our unmasked length)
afs = ts.allele_frequency_spectrum(mode='branch', span_normalise=False, polarised=True)

# SNP Branch AFS
afs1 = afs * snp_mutation_rate
normalized_branch_afs_snp = afs1 / length
print(normalized_branch_afs_snp)

# TE Branch AFS
afs2 = afs * te_mutation_rate
normalized_branch_afs_te = afs2 / length
print(normalized_branch_afs_te)

#--- Flip sites and calculate AFS from VCF ---#
def process_vcf(input_vcf, mut, flipped_pos, is_te=False):
    # Drop sites that did not map to a branch
    drop_mask = ~np.isin(input_vcf['variants/POS'], drop_not_mapped_pos)
    the_vcf = {
        'variants/POS': input_vcf['variants/POS'][drop_mask],
        'variants/ID': input_vcf['variants/ID'][drop_mask],
        'variants/REF': input_vcf['variants/REF'][drop_mask],
        'variants/ALT': input_vcf['variants/ALT'][drop_mask],
        'calldata/GT': input_vcf['calldata/GT'][drop_mask, :]
    }
    
    # Retain sites in Relate output files; separate SNP and TE
    if is_te:
        has_age_pos = mut[mut["rs-id"].str.contains("chr")]["pos_of_snp"]
    else:
        has_age_pos = mut[mut["rs-id"].str.contains("snp")]["pos_of_snp"]

    has_age = np.isin(the_vcf['variants/POS'], has_age_pos)
    not_flipped = ~np.isin(the_vcf['variants/POS'], flipped_pos)

    # --- Flip based on Relate "is_flipped" ---
    flip_mask = np.isin(the_vcf['variants/POS'], flipped_pos)
    flipped_the_vcf = {
        'variants/POS': the_vcf['variants/POS'][flip_mask],
        'variants/ID': the_vcf['variants/ID'][flip_mask],
        'variants/REF': the_vcf['variants/REF'][flip_mask],
        'variants/ALT': the_vcf['variants/ALT'][flip_mask],
        'calldata/GT': the_vcf['calldata/GT'][flip_mask, :]
    }

    # Check for accidental hets (this data should have none)
    flat = flipped_the_vcf['calldata/GT'].reshape(-1, 2)
    unique, counts = np.unique(flat, axis=0, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"{u.tolist()}: {c}")

    # Reverse REF and ALT
    newAlt = [[item] for item in flipped_the_vcf['variants/REF']]
    max_alt_alleles = 3
    newAlt2 = [row + [''] * (max_alt_alleles - len(row)) for row in newAlt]
    newRef = np.array([row[0] for row in flipped_the_vcf['variants/ALT']])
    flipped_GT = np.select([flipped_the_vcf['calldata/GT'] == 0, flipped_the_vcf['calldata/GT'] == 1], [1, 0], flipped_the_vcf['calldata/GT'])

    flipped_vcf = {
        'variants/POS': flipped_the_vcf['variants/POS'],
        'variants/ID': flipped_the_vcf['variants/ID'],
        'variants/REF': newRef,
        'variants/ALT': newAlt,
        'calldata/GT': flipped_GT
    }

    # --- Unflipped section ---
    unflip_mask = not_flipped & has_age
    unflipped = {
        'variants/POS': the_vcf['variants/POS'][unflip_mask],
        'variants/ID': the_vcf['variants/ID'][unflip_mask],
        'variants/REF': the_vcf['variants/REF'][unflip_mask],
        'variants/ALT': the_vcf['variants/ALT'][unflip_mask],
        'calldata/GT': the_vcf['calldata/GT'][unflip_mask, :]
    }

    # --- Calculate AFS ---
    chrom_length = length
    
    # First for unflipped
    genotypes = allel.GenotypeArray(unflipped['calldata/GT'])
    counts = genotypes.count_alleles()
    derived_counts = counts[:, 1]  
    afs_unflipped = allel.sfs(derived_counts)
    norm_afs_unflipped = afs_unflipped / chrom_length

    # Now for flipped
    genotypes = allel.GenotypeArray(flipped_vcf['calldata/GT'])
    counts = genotypes.count_alleles()
    derived_counts = counts[:, 1]
    afs_flipped = allel.sfs(derived_counts)
    norm_afs_flipped = afs_flipped / chrom_length

    normalized_afs = norm_afs_unflipped + norm_afs_flipped

    print(normalized_afs)

    return normalized_afs

snp_afs = process_vcf(snp_vcf, mut, snp_flipped, is_te=False)

te_afs = process_vcf(te_vcf, mut, te_flipped, is_te=True)

# normalize so as to not care about the mutation rate
norm_snp_afs = snp_afs/(sum(snp_afs))
norm_te_afs = te_afs/(sum(te_afs))

norm_branch_afs = afs/(sum(afs))

#--- Plot ---#
# unnormalized
freq_branch = np.arange(1, 26)          
freq_snp = np.arange(1, 26)               
freq_te = freq_snp

# get rid of the 0s (we don't actually have any heterozygotes here due to all inbreds)
snp_afs = snp_afs[::2]
te_afs = te_afs[::2]
norm_snp_afs = norm_snp_afs[::2]
norm_te_afs = norm_te_afs[::2]

plt.scatter(freq_branch, normalized_branch_afs_snp[1:26], c='#233d4d',  marker='o', label='ARG', s=8)
plt.scatter(freq_snp, snp_afs[1:26], c='#fe7f2d',  marker='s', label='SNP', s=8)
plt.scatter(freq_te, te_afs[1:26], c='#fcca46', marker='^', label='TE', s=8)
plt.xlabel("Derived allele frequency")
plt.ylabel("# of variants / base")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("/home/reginaf/beibei_afs/AFS_august282025.jpg", dpi=600)
plt.clf()

# normalized
plt.scatter(freq_branch, norm_branch_afs[1:26], c='#233d4d', marker='o', label='ARG', s=8)
plt.scatter(freq_snp, norm_snp_afs[1:26], c='#fe7f2d', marker='s', label='SNP', s=8)
plt.scatter(freq_te, norm_te_afs[1:26], c='#fcca46', marker='^', label='TE', s=8)
plt.xlabel("Derived allele frequency")
plt.ylabel("Density")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("/home/reginaf/beibei_afs/norm_AFS_august282025.jpg", dpi=600)
plt.clf()

