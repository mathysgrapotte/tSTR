#!/usr/bin/env python3

import argparse
from src.data.fasta import FastaSnpIntroduction

def get_args():
    parser = argparse.ArgumentParser(description="Launch individual fasta")
    parser.add_argument("-f", "--fasta_file", help="Fasta file")
    parser.add_argument("-b", "--bed_file", help="Bed file")
    parser.add_argument("-v", "--vcf_file", help="Vcf file")
    parser.add_argument("-o", "--output", help="Output file", default="output.fasta")
    return parser.parse_args()

def main(fasta_file, bed_file, vcf_file, output):
    fasta = FastaSnpIntroduction(fasta_file, bed_file, vcf_file)
    fasta.introduce_snps()
    fasta.write_fasta_with_snps(output)

if __name__ == "__main__":
    args = get_args()
    main(args.fasta_file, args.bed_file, args.vcf_file, args.output)