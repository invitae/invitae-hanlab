# Extract the labels of documents, and store them into an array.

import pickle
import numpy as np
import pandas as pd
from modules import *

if __name__ == "__main__":
    INDIR = "../data_sample/raw/"
    OUTDIR = "../data_sample/processed/"
    logger = get_console_logger("doc_labels")
    logger.info("Collecting doc labels ...")
    indata = pd.read_table(INDIR + "variant-pubmed-ev.tsv", sep='\t', header=0)
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
            logger.info(str(i) + " rows processed ...")
        if str(categories[i]) == "nan":
        	continue
        pmid = pmids[i]
        tmp = str(categories[i]).split('|')
        for label in tmp:
            labels.loc[pmid, label] = True
    
    outfile = OUTDIR + "doc_labels.pkl"
    logger.info("Writing to %s ..." % outfile)
    with open(outfile, 'wb') as fout:
        pickle.dump(labels, fout)
