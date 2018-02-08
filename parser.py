import xml.etree.ElementTree as ET
import string
import sys
import os

file = open("utterances_bnc_lemma.txt","w+")

def trees(xml_file):
    tree = ET.ElementTree(file=xml_file)
    root = tree.getroot()
    uttr = []

    for child in root[1].iter():	#each utterance
        if (child.tag=="s"):
    	    if (uttr!=[]):
    		    out = " ".join(uttr)
    		    file.write(out+"\n")
    	    uttr = []
#        if (child.tag=="w" or child.tag=="c"):
        if (child.tag=="w"):
        	if (child.text != None):
                    uttr.append(child.attrib['hw'])
#uttr.append(child.text)
root = "./2554/2554/download/Texts"
for r,d,files in os.walk(root):
	for f in files:
		if f.endswith(".xml"):
		    full_add = os.path.join(r, f)
		    trees(full_add)