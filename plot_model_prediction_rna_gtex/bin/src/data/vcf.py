"""
This is a class to handle vcf files
"""

import pandas as pd

class VcfGtex:

    def __init__(self, vcf_file):
        self.vcf_file = vcf_file
        self.vcf_data = self.read_vcf(vcf_file)

    def read_vcf(self, vcf_file):

        with open(vcf_file, 'r') as f:
            for line in f:
                # check if the line is a part of the header
                if line[0] == "#":
                    if line[1] == "#":
                        # double hash means the line is meta information (in Gtex vcfs)
                        continue
                    else:
                        print("hola")
                        # single hash means the line is a header
                        # this is the pandas dataframe column vector
                        columns = line.strip().split("\t")

                        # use it to initialize the dataframe -> code might break if there are multiple lines starting with '#'
                        df = pd.DataFrame(columns=columns)
                else:
                    # the line is not a header
                    # it is a data line

                    line = line.strip().split("\t")

                    # check if df exists
                    if "df" not in locals():
                        raise ValueError("The first line of the vcf file is not a header")

                    # add the line to the dataframe
                    df = df._append(pd.Series(line, index=columns), ignore_index=True)

        # convert POS to int
        df["POS"] = df["POS"].astype(int)

        # remove 1 to all POS since vcf is 1-based and bed is 0-based
        df["POS"] = df["POS"] - 1
        return df



if __name__ == "__main__":
    # minimal code for testing the class
    vcf_file = "/Users/mgrapotte/LabWork/tSTR/plot_model_prediction_rna_gtex/results/filtered_vcf.vcf"
    vcf = VcfGtex(vcf_file)
    print(vcf.read_vcf("/Users/mgrapotte/LabWork/tSTR/plot_model_prediction_rna_gtex/results/filtered_vcf.vcf"))