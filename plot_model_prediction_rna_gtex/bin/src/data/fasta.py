from abc import ABC
import numpy as np
import copy

import sys
sys.path.append("..")

from src.data.vcf import VcfGtex
from src.data.bed import Bed

class Fasta(ABC):
    """
    This class is the master class for all fasta related classes.
    """

    # init function for the fasta class, takes as input the path to the fasta file
    def __init__(self):
        self.sequences = []
        self.sequence_names = []
        self.tags = []

    def write_fasta(self, path_to_write):
        """
        This function is a helper function for writing a fasta file taking as input self. 
        Fasta should be the following format : 
        
        >sequence_name|tag
        sequence

        Self has to contain sequences, tags and names. 
        """
        
        # check if sequences, sequences_names and tags are populated; if an error is thrown, print the length of each list
        assert len(self.sequences) == len(self.sequence_names) == len(self.tags), f"The sequences, sequence_names and tags lists should have the same length. The sequences list is of length {len(self.sequences)}, the sequence_names list is of length {len(self.sequence_names)} and the tags list is of length {len(self.tags)}"
        # check if there is at least one sequence
        assert len(self.sequences) > 0, "There should be at least one sequence"

        with open(path_to_write, 'w') as fasta_file:
            for sequence, sequence_name, tag in zip(self.sequences, self.sequence_names, self.tags):
                fasta_file.write(f">{sequence_name}|{tag}\n")
                fasta_file.write(sequence + '\n')

    def write_fasta_without_tags(self, path_to_write):
        """
        This function is a helper function for writing a fasta file taking as input self. 
        Fasta should be the following format : 
        
        >sequence_name
        sequence

        Self has to contain sequences and names. 
        """
        
        # check if sequences and sequences_names are populated; if an error is thrown, print the length of each list
        assert len(self.sequences) == len(self.sequence_names), f"The sequences and sequence_names lists should have the same length. The sequences list is of length {len(self.sequences)} and the sequence_names list is of length {len(self.sequence_names)}"
        # check if there is at least one sequence
        assert len(self.sequences) > 0, "There should be at least one sequence"

        with open(path_to_write, 'w') as fasta_file:
            for sequence, sequence_name in zip(self.sequences, self.sequence_names):
                fasta_file.write(f">{sequence_name}\n")
                fasta_file.write(sequence + '\n')

    def load_fasta(self, path_to_read):
        """
        This function is a helper function for loading a fasta file taking as input self. 
        Fasta should be the following format : 
        
        >sequence_name|tag
        sequence

        Self has to contain sequences, tags and names. 
        """
        with open(path_to_read, 'r') as fasta_file:
            for line in fasta_file:
                if line[0] == '>':
                    # this is a sequence name
                    sequence_name, tag = line[1:].split('|')
                    self.sequence_names.append(sequence_name)
                    # convert tag to float
                    self.tags.append(float(tag.strip()))
                else:
                    # this is a sequence
                    # check if the sequence contains 'n' or 'N' and if so, remove the previously added tag and sequence names
                    if 'n' in line or 'N' in line:
                        self.sequence_names.pop()
                        self.tags.pop()        
                    else:                      
                        self.sequences.append(line.strip())

    def truncate_sequences(self, length):
        """
        This function truncates the sequences to the given interval length, the interval is positionned at exactly the middle of the sequence, given that the sequence is longer than the interval """

        # check if the sequence is longer than the interval
        assert len(self.sequences[0]) > length, f"The sequence is shorter than the interval length. The sequence is of length {len(self.sequences[0])} and the interval length is {length}"

        # get the middle position of the sequence
        middle_position = len(self.sequences[0]) // 2

        # truncate the sequence to the interval length
        self.sequences = [sequence[middle_position - length // 2:middle_position + length // 2 + 1] for sequence in self.sequences]

    def load_fasta_without_tags(self, path_to_read):
        """
        This function is a helper function for loading a fasta file taking as input self. 
        Fasta should be the following format : 
        
        >sequence_name
        sequence

        Self has to contain sequences and names. 
        """
        with open(path_to_read, 'r') as fasta_file:
            for line in fasta_file:
                if line[0] == '>':
                    # this is a sequence name
                    sequence_name = line[1:].strip()
                    self.sequence_names.append(sequence_name)
                else:
                    # this is a sequence

                    # check if the sequence contains 'n' or 'N' and if so, remove the previously added sequence name
                    if 'n' in line or 'N' in line:
                        self.sequence_names.pop()                   
                    else:                     
                        self.sequences.append(line.strip())

    def get_sequence_length(self):
        """ This function returns the length of the sequences in the fasta file. """
        return len(self.sequences[0])

class FastaSnpIntroduction(Fasta):
    """
    This class is for handling modifying fasta files according to SNPs
    """

    def __init__(self, fasta_file, bed_file, vcf_file):
        self.fasta_data = Fasta()
        self.fasta_data.load_fasta(fasta_file)
        self.fasta_data.truncate_sequences(100)
        self.vcf_data = VcfGtex(vcf_file)
        self.bed_data = Bed(bed_file)

        self.sequences_with_snps = []
        self.names_with_snps = []
        self.tags_with_snps = []

    def write_fasta_with_snps(self, path_to_write):
        """
        This function is a helper function for writing a fasta file taking as input self. 
        Fasta should be the following format : 
        
        >sequence_name|tag
        sequence

        Self has to contain sequences, tags and names. 
        """
        with open(path_to_write, 'w') as fasta_file:
            for sequence, sequence_name, tag in zip(self.sequences_with_snps, self.names_with_snps, self.tags_with_snps):
                fasta_file.write(f">{sequence_name}|{tag}\n")
                fasta_file.write(sequence + '\n')

    def introduce_snps(self):
        """
        This function introduces snps to the fasta file
        """
        # get the individuals from the vcf file
        individuals = self.vcf_data.vcf_data.columns[9:]
        # get the snp information from the vcf file
        snp_information = self.vcf_data.vcf_data[individuals]
        # get the snp positions from the vcf file
        snp_positions = self.vcf_data.vcf_data["POS"]
        # get the reference alleles from the vcf file
        ref_alleles = self.vcf_data.vcf_data["REF"]
        # get the alternative alleles from the vcf file
        alt_alleles = self.vcf_data.vcf_data["ALT"]
        # get the start position of the sequence from the bed file 
        start_position = self.bed_data.bed_data["chromStart"][0]

        # get the fasta sequence
        fasta_sequence = self.fasta_data.sequences[0]

        # for each position in the snp positions, and each reference allele, check that the nucleotide in the fasta sequence matches the reference allele in the vcf file
        for snp_position, ref_allele in zip(snp_positions, ref_alleles):
            # check that the fasta sequence contains the reference allele, if it is not the case, print the sequence centered around the snp position, with the snp position in upper case and the reference allele
            if fasta_sequence[snp_position - start_position].upper() != ref_allele:
                print(f"Position {snp_position - start_position} does not match the reference allele {ref_allele} in the fasta sequence")

                # for clarity, print the fasta sequence with the snp position in upper case
                # copy the fasta sequence to a lower case variable
                fasta_sequence_lower = fasta_sequence.lower()
                # make the snp position upper case
                fasta_sequence_lower = fasta_sequence_lower[:snp_position - start_position] + fasta_sequence_lower[snp_position - start_position].upper() + fasta_sequence_lower[snp_position - start_position + 1:]
                print(fasta_sequence_lower)

        # for each individual in the vcf file, build two new sequences (one for each allele) from the reference sequence and the snps
        # the snp information is written as is : x|y, where x corresponds to the first allele and y the second allele
        # if x = 1, we need to introduce the snp in the first allele
        # if y = 1, we need to introduce the snp in the second allele
        # if x = 0 or . we do not need to introduce the snp in the first allele
        # if y = 0 or . we do not need to introduce the snp in the second allele
        # for GTEx vcf, the column containing the individuals start with the GTEX prefix

        # for each individual, build two new sequences (one for each allele) from the reference sequence and the snps
        for individual in individuals:

            snp_info = snp_information[individual]
            
            # get the name of the sequence
            name = self.fasta_data.sequence_names[0]

            # copy the sequence into the first allele sequence
            allele_one = copy.deepcopy(self.fasta_data.sequences[0]).lower()

            # copy the sequence into the second allele sequence
            allele_two = copy.deepcopy(self.fasta_data.sequences[0]).lower()


            # for each snp position, check if the snp information is 1, if it is the case, introduce the snp in the sequence
            for snp, snp_position, ref_allele, alt_allele in zip(snp_info, snp_positions, ref_alleles, alt_alleles):
                # get x and y from the snp information (x|y)
                x, y = snp.split('|')

                if x == "1":
                    # introduce the snp in the first allele
                    allele_one = allele_one[:snp_position - start_position] + alt_allele.upper() + allele_one[snp_position - start_position + 1:]

                if y == "1":
                    # introduce the snp in the second allele
                    allele_two = allele_two[:snp_position - start_position] + alt_allele.upper() + allele_two[snp_position - start_position + 1:]



            # add the allele_one sequence to the list of sequences with snps
            self.sequences_with_snps.append(allele_one)
            # add the name to the list of names with snps, name should reflect the allele as well as the individual
            self.names_with_snps.append(f"{name}:allele_one:{individual}")

            # add the allele_two sequence to the list of sequences with snps
            self.sequences_with_snps.append(allele_two)
            # add the name to the list of names with snps, name should reflect the allele as well as the individual
            self.names_with_snps.append(f"{name}:allele_two:{individual}")


        # duplicate the tags for the sequences with snps
        self.tags_with_snps = self.fasta_data.tags * 2 * len(individuals)

if __name__ == "__main__":

    fasta_file = "/Users/mgrapotte/LabWork/tSTR/plot_model_prediction_rna_gtex/results/filtered_fasta.fa"
    vcf_file = "/Users/mgrapotte/LabWork/tSTR/plot_model_prediction_rna_gtex/results/filtered_vcf.vcf"
    bed_file = "/Users/mgrapotte/LabWork/tSTR/plot_model_prediction_rna_gtex/results/filtered_bed.bed"
    fasta_snp = FastaSnpIntroduction(fasta_file, vcf_file, bed_file)
    fasta_snp.introduce_snps() 

