import os
from shutil import copyfile
import pickle
import pandas as pd
import argparse
import sys

'''
Authors:
    Jinfeng Xiao (jxiao13@illinois.edu)
    Matthew Davis (matthew.davis@invitae.com)
    
    
Summary:
    Creates small sample data set for local testing.
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Generate sample data for Bag of Words.")
    parser.add_argument("--input-dir",
                        type=str,
                        dest='input_dir',
                        default="./data_sample/",
                        help="name of processed pickle file of labeled documents")
    parser.add_argument("--output-dir",
                        dest='output_dir',
                        type=str,
                        default="../data_sample/",
                        help="name of output files")
    parser.add_argument("--class-size",
                        type=int,
                        dest='class_size',
                        default=100,
                        help="Number of PMID documents to include in a balanced set of positive and negative data.")

    return parser.parse_args(sys.argv[1:])



if __name__ == "__main__":


    prog_args = parse_args()

    input_dir = prog_args.input_dir
    output_dir = prog_args.output_dir

    input_fn = "/".join([input_dir, 'processed', 'doc_labels.pkl'])
    labels = pickle.load(open(input_fn))

    # Compute a list of PMIDs to include in the sample dataset,
    # so that it contains ~100 docs for 4 labels and 100 negative docs.
    class_size = prog_args.class_size

    # Add negative samples
    pmids = labels.index[(labels.sum(axis=1) == 0)][:class_size].tolist()

    # Add positive samples
    for label in ["individual_observations",
                  "functional_experiments",
                  "family_studies",
                  "sequence_observations"]:
        pmids += labels.index[labels[label]][:class_size].tolist()
    # Remove redundancy
    pmids = set(pmids)

    # Write sample raw labels to file
    labels_raw = pd.read_table(input_dir + "raw/variant-pubmed-ev.tsv", sep='\t', header=0)
    labels_raw_sample = labels_raw.loc[labels_raw["pubmed_id"].isin(pmids), :]
    labels_raw_sample.to_csv(output_dir + "raw/variant-pubmed-ev.tsv", sep='\t', header=True, index=False)
    
    # Copy the sample corpus
    for pmid in pmids:
        if not os.path.exists(output_dir + "raw/pubmed-txts-mf/" + str(pmid)):
            os.makedirs(output_dir + "raw/pubmed-txts-mf/" + str(pmid))
        files = os.listdir(input_dir + "raw/pubmed-txts-mf/" + str(pmid) + "/")
        for file in files:
            copyfile(input_dir + "raw/pubmed-txts-mf/" + str(pmid) + "/" + file, output_dir + "raw/pubmed-txts-mf/" + str(pmid) + "/" + file)
