#!/usr/bin/env python3
import os
import sys
import numpy as np
from gensim.models import Word2Vec as we
# $model.wv.most_similar('kill',topn=30)

def w2v(inp, method, window, saved):
    # tokenization
    f = open(inp,"r")
    utterances = [sent.split() for sent in f if len(sent)>1]

    # model
    model = we(utterances,sg=method, window=window, min_count=3)
    model.save(saved)

if __name__ == "__main__":
	# Parse arguments
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", default="bnc_lemma_pos.txt", type=str, help="input file")
	parser.add_argument("--sg", default=0, type=int, help="method")
	parser.add_argument("--w", default=3, type=int, help="window size")
	parser.add_argument("--saved", default="model_tmp", type=str, help="saved filename")

	args = parser.parse_args()
	w2v(args.input,args.sg,args.w,args.saved)