#!/usr/bin/env python3

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data.pytorch_loaders import fastaDataset
from src.data.fasta import Fasta
from torch.utils.data import DataLoader

class BlockMax(nn.Module):
    def __init__(self, filter_size):
        super(BlockMax, self).__init__()
        self.conv = nn.Conv2d(1, 1, (filter_size, 4), bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = x.reshape(x.shape[0], x.shape[2])
        output = torch.max(x, dim=1)[0]
        return output


class BlockNet(nn.Module):
    def __init__(self, filter_size, size=101):
        super(BlockNet, self).__init__()
        self.conv = nn.Conv2d(1, 1, (filter_size, 4), bias=False)
        self.dense = nn.Linear(size - (filter_size - 1), 1)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = x.reshape(x.shape[0], x.shape[2])
        output = self.dense(x)
        return output

class Net(nn.Module):
    def __init__(self, filter_size, block_type):
        super(Net, self).__init__()
        if block_type == "max":
            self.last_block = BlockMax(filter_size)
        else:
            self.last_block = BlockNet(filter_size)
        self.blocks = nn.ModuleList()
        self.len = 0
        self.linear = nn.Linear(self.len + 1, 1)

    def add_block(self, filter_size, block_type):
        self.last_block.require_grad = False
        self.blocks.append(self.last_block)
        if block_type == "max":
            self.last_block = BlockMax(filter_size)
        else:
            self.last_block = BlockNet(filter_size)
        self.len = len(self.blocks)
        self.linear = nn.Linear(self.len + 1, 1)

    def forward(self, x):
        hidden = self.last_block(x).view(x.shape[0], 1)
        for i, l in enumerate(self.blocks):
            hidden = torch.cat((hidden, self.blocks[i](x).view(x.shape[0], 1)), dim=1)
        output = self.linear(hidden)
        return output
    
def build_modular(params):
    netmodel = Net(int(params[0][0]), 'net')
    for i in range(1, len(params)):
        netmodel.add_block(int(params[i][0]), 'net')
    return netmodel


def load_model(params_path, keys_path):
    params = np.load(params_path)
    model = build_modular(params)
    model.load_state_dict(torch.load(keys_path, map_location=torch.device('cpu')))
    return model


def get_args():
    parser = argparse.ArgumentParser(description="Launch make predictions")
    parser.add_argument("-f", "--fasta_file", help="Fasta file")
    parser.add_argument("-mw", "--model_weights", help="Model weights")
    parser.add_argument("-ma", "--model_archi", help="Model architecture")
    parser.add_argument("-o", "--output", help="Output file", default="output.fasta")
    return parser.parse_args()

def get_reverse_complement(sequence):
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(complement[base] for base in reversed(sequence))

def get_model_path(models, name):
    sequence_name, sequence_class, strand = name.split(";")
    if strand == "+":
        path = os.path.join(models, "hg38"+sequence_class)
        return path
    else:
        path = os.path.join(models, "hg38"+get_reverse_complement(sequence_class.upper()))
        return path



def main(fasta_file, model_weights, model_architecture, output):

    model = load_model(model_architecture, model_weights)

    rawFasta = Fasta()
    rawFasta.load_fasta(fasta_file )
    fastaData = fastaDataset(fasta_file)
    fastaLoader = DataLoader(fastaData, batch_size=1, shuffle=False)

    # set the model to eval mode
    model.eval()

    # get the outputs on the predicted values in the fastaLoader
    outputs = []
    allele = []
    individual = []
    for i, data in enumerate(fastaLoader):
        name = data[2][0]
        allele.append(name.split(":")[1])
        individual.append(name.split(":")[2])
        sequence = data[0].float()
        sequence = sequence.reshape(1, 1, sequence.shape[2], sequence.shape[1])
        outputs.append(model(sequence).flatten().detach().numpy()[0])

    # save a tsv file containing individual, allele and prediction (in order) with a header
    with open(output, "w") as file:
        file.write("individual\tallele\tprediction\n")
        for i in range(len(outputs)):
            file.write(f"{individual[i]}\t{allele[i]}\t{outputs[i]}\n")

if __name__ == "__main__":
    args = get_args()
    main(args.fasta_file, args.model_weights, args.model_archi, args.output)

    
    

