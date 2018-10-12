# Extract the labels of documents, and store them into an array.

indir = "../data_sample/raw/"
outdir = "../data_sample/processed/"
import numpy as np
import pandas as pd
import pickle

indata = pd.read_table(indir + "variant-pubmed-ev.tsv", sep='\t', header=0)
pmids = indata["pubmed_id"]
categories = indata["categories"]

uniq_pmids = list(set(pmids))
uniq_pmids.sort()
uniq_categories = []

for i in range(len(categories)):
    tmp = str(categories[i]).split('|')
    uniq_categories = list(set(uniq_categories + tmp))
uniq_categories.sort()

labels = pd.DataFrame(False, index=uniq_pmids, columns=uniq_categories)

for i in range(len(categories)):
    if i % 1000 == 0:
        print(str(i) + " rows processed ...")
    if str(categories[i]) == "nan":
    	continue
    pmid = pmids[i]
    tmp = str(categories[i]).split('|')
    for label in tmp:
        labels.loc[pmid, label] = True

with open(outdir + "doc_labels.pkl", 'wb') as fout:
    pickle.dump(labels, fout)
