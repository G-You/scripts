import numpy as np
import sys
import os
import string

f = open("bnc_pos.txt","r")
utterances = [sent.split() for sent in f if len(sent)>1]

voc = {}

for s in utterances:
	counts = 0
	target = "" 
	for w in s:
		if (w.isupper()!=1):
			if (target!=""):
				voc[target].append(counts)
			target = w
			if target not in voc:
			    voc[target]=[]
			counts = 0
		elif (target!=""):
			if w in ["SUBST","PRON"]:
				counts += 1
			elif w in ["CONJ","PREP"]:
				voc[target].append(counts)
				target = ""
				counts = 0
	if (target!=""):
		voc[target].append(counts) 

write_f = open("bnc_nomials.txt","w+")

for v in voc:
	if (len(voc[v])>=10):
		voc[v] = sum(voc[v])/float(len(voc[v]))
	    write_f.write(v+"\t"+str(voc[v])+'\n')

