# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:21:41 2023

@author: 10979
"""
import argparse
import sys
from Bio import SeqIO
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

location = os.getcwd()

from Function.Protein_Embedding import *


def main():
    parser = argparse.ArgumentParser(usage='%(prog)s [options]', description='Chitinase Classifier based on deep learning')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0 - 2023')

    parser.add_argument('-o', '--out', action='store', dest='out_prefix', 
                        help='(Required) The output prefix of results')

    ## lncRNA identification options
    parser.add_argument('-f', '--fasta', action='store', dest='fasta_file', 
                        help='Protein Sequences (Msut be fasta)')
    args = parser.parse_args()
    
    print("Loading Protein fasta")
    fasta_train = SeqIO.parse(args.fasta_file, "fasta")
    print("Total Nb of Elements : ", len(list(fasta_train)))
    
    ids_list = []
    embed_vects_list = []

    for item in tqdm(SeqIO.parse(args.fasta_file, "fasta")):
        ids_list.append(item.id)
        embed_vects_list.append(get_bert_embedding(sequence = item.seq, len_seq_limit = 1200))

    df_res = pd.DataFrame(data={"id" : ids_list, "embed_vect" : embed_vects_list})
    np.save('ID.npy',np.array(ids_list))
    np.save('Embedding.npy',np.array(embed_vects_list))
    print('Embedding is finished')
    
    model_dir = location+'/Chitin_08_06.h5'
    model = tf.keras.models.load_model(model_dir, compile=False)
    embed_vects_list = np.array(embed_vects_list)
    embed_vects_list = embed_vects_list.reshape(embed_vects_list.shape[0],1,1024)
    
    print('Model prediction is finished')
    pre = pd.DataFrame(model.predict(embed_vects_list))
    pre.rename(columns={0: "Probability of nonChitinase", 1: "Probability of Chitinase"})
    pre.to_csv(args.out_prefix)
    print('Complete')

if __name__ == '__main__':
    main()
