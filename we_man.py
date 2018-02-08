#!/usr/bin/env python3
import os
import sys
import numpy as np
from gensim.models import Word2Vec as we
import csv

# tokenization
f = open("manchester.csv","r")
reader = csv.reader(f,delimiter=',')

utterances = [sent[3].split() for sent in reader if (len(sent[3])>1 and sent[2]!="CHI")]

# model
model = we(utterances, window = 5, min_count=3)
model.save("manchester_model")

