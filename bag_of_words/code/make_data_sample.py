import os
from shutil import copyfile
import pickle
import pandas as pd

'''
Authors:
    Jinfeng Xiao (jxiao13@illinois.edu)
    Matthew Davis (matthew.davis@invitae.com)
    
# This script makes a small sample dataset for code review.

'''





if __name__ == "__main__":
    INDIR = "../data/"
    OUTDIR = "../data_sample/"
    with open(INDIR + "processed/doc_labels.pkl", 'rb') as fin:
        labels = pickle.load(fin)  # doc labels
    
    # Compute a list of PMIDs to include in the sample dataset,
    # so that it contains ~100 docs for 4 labels and 100 negative docs.
    CLASS_SIZE = 100
    # Add negative samples
    pmids = labels.index[(labels.sum(axis=1) == 0)][:CLASS_SIZE].tolist()
    # Add positive samples
    for label in ["individual_observations", "functional_experiments",
                  "family_studies", "sequence_observations"]:
        pmids += labels.index[labels[label]][:CLASS_SIZE].tolist()
    # Remove redundance
    pmids = set(pmids)

    # Write sample raw labels to file
    labels_raw = pd.read_table(INDIR + "raw/variant-pubmed-ev.tsv", sep='\t', header=0)
    labels_raw_sample = labels_raw.loc[labels_raw["pubmed_id"].isin(pmids), :]
    labels_raw_sample.to_csv(OUTDIR + "raw/variant-pubmed-ev.tsv", sep='\t', header=True, index=False)
    
    # Copy the sample corpus
    for pmid in pmids:
        if not os.path.exists(OUTDIR + "raw/pubmed-txts-mf/" + str(pmid)):
            os.makedirs(OUTDIR + "raw/pubmed-txts-mf/" + str(pmid))
        files = os.listdir(INDIR + "raw/pubmed-txts-mf/" + str(pmid) + "/")
        for file in files:
            copyfile(INDIR + "raw/pubmed-txts-mf/" + str(pmid) + "/" + file, OUTDIR + "raw/pubmed-txts-mf/" + str(pmid) + "/" + file)
