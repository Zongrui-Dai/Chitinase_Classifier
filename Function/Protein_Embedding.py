# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:29:55 2023

@author: 10979
"""
#!pip install transformers
#!pip install Bio

from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from Bio import SeqIO
import os
from Bio import Seq
import regex as re
import pandas as pd
import numpy as np

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert").to('cuda')

def get_bert_embedding(
    sequence : str,
    len_seq_limit : int
):
    sequence_w_spaces = ' '.join(list(sequence))
    encoded_input = tokenizer(
        sequence_w_spaces,
        truncation=True,
        max_length=len_seq_limit,
        padding='max_length',
        return_tensors='pt').to('cuda')
    output = model(**encoded_input)
    output_hidden = output['last_hidden_state'][:,0][0].detach().cpu().numpy()
    assert len(output_hidden)==1024
    return output_hidden
