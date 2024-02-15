#!/usr/bin/env python3

"""
The goal of this script is to merge all datasets in a single tsv file

The input files should be : 
1- A prediction file (with a header as column names) with the following columns  : 
    - individual
    - allele (can be either allele_one or allele_two)
    - prediction

2- A metadata file (with a header as column names) with the following columns :
    - SampleID (the sample experiment, each individual has multiple samples from individual tissues)
    - indId (the individual id)
    - group (the tissue group)
    - and other irrelevant columns

3- A gct file (with a header as column names) with the following columns :
    - gene_id
    - description
    - each SampleID as a column name (containing the expression value for each gene in each sample)

The output file should be a tsv file with the following columns :
    - sequence name
    - gene_id
    - IndId
    - allele 1 prediction
    - allele 2 prediction
    - tissue group
    - expression value 

this file should be obtained by first modifying the prediction tsv to have the following columns :
    - individual 
    - allele_one predictions
    - allele_two predictions

Then the new prediction tsv file is merged to the metadata file using the individual column as a key resulting in an temp file. 

We then build the output file by getting the right sampleID associated transcription from the gct file and adding it to a new column in the temp file.

"""

import pandas as pd
import argparse
from os import path

def get_args():
    parser = argparse.ArgumentParser(description="Launch merge datasets")
    parser.add_argument("-p", "--prediction_file", help="Prediction file")
    parser.add_argument("-m", "--metadata_file", help="Metadata file")
    parser.add_argument("-g", "--gct_file", help="Gct file")
    parser.add_argument("-o", "--output", help="Output file", default="output.tsv")
    return parser.parse_args()



def main(prediction_file, metadata_file, gct_file, sequence_name, output):
    # Read the prediction file
    prediction = pd.read_csv(prediction_file, sep="\t")

    # modify the prediction file to have the following columns : 
    # individual, allele_one predictions, allele_two predictions
    prediction = prediction.pivot(index="individual", columns="allele", values="prediction").reset_index()
    prediction.columns = ["individual", "allele_one", "allele_two"]

    # Read the metadata file
    metadata = pd.read_csv(metadata_file, sep="\t")

    # Keep only the relevant columns
    metadata = metadata[["sampleId", "indId", "group"]]

    # Merge the prediction file to the metadata file using the individual column as a key
    temp = pd.merge(metadata, prediction, left_on="indId", right_on="individual")

    # Read the gct file
    gct = pd.read_csv(gct_file, sep="\t")

    temp_col = []
    for sample in temp["sampleId"]:
        # if the sample is found append to the colunm
        if sample in gct.columns:
            temp_col.append(gct[sample].values[0])
        else:
            temp_col.append("NA")
        
    temp["expression_value"] = temp_col

    # drop all "NA" values
    temp = temp[temp["expression_value"] != "NA"]

    # Build the output file
    temp.to_csv(output, sep="\t", index=False)


if __name__ == "__main__":
    args = get_args()
    prediction_file = args.prediction_file
    metadata_file = args.metadata_file
    gct_file = args.gct_file
    output = args.output
    main(prediction_file, metadata_file, gct_file, output)




