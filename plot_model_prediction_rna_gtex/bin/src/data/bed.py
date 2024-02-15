"""This is a class for handling bed files"""

import pandas as pd

class Bed:
    """
    This is a class to handle bed files
    """

    def __init__(self, bed_file):
        self.bed_file = bed_file
        self.bed_data = self.read_bed(bed_file)

    def read_bed(self, bed_file):
        columns = ["chrom", "chromStart", "chromEnd", "name", "score", "strand"]
        df = pd.DataFrame(columns=columns)
        with open(bed_file, 'r') as f:
            for line in f:
                # check if the line is a part of the header
                if line[0] == "#":
                    continue
                else:
                    # the line is not a header
                    # it is a data line

                    line = line.strip().split("\t")

                    # add the line to the dataframe
                    df = df._append(pd.Series(line, index=columns), ignore_index=True)

        # convert chromStart and chromEnd to int
        df["chromStart"] = df["chromStart"].astype(int)
        df["chromEnd"] = df["chromEnd"].astype(int)

        # convert score to float
        df["score"] = df["score"].astype(float)
        
        return df
    
if __name__ == "__main__":
    # minimal code for testing the class

    bed_file = "/Users/mgrapotte/LabWork/tSTR/plot_model_prediction_rna_gtex/results/filtered_bed.bed"
    bed = Bed(bed_file)
    print(bed.read_bed("/Users/mgrapotte/LabWork/tSTR/plot_model_prediction_rna_gtex/results/filtered_bed.bed"))