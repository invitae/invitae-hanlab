# Read papers and convert them to bag-of-words representations.

import os
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from modules import *

def read_corpus(indir):
    # Read papers into a list
    logger = get_console_logger("read_corpus")
    corpus = {x:'' for x in os.listdir(indir)}
    logger.info("Reading corpus ...")
    for pmid in corpus:
        for field in os.listdir(indir + pmid + '/'):
            with open(indir + pmid + '/' + field, 'r') as fin:
                text = fin.read().replace('\n', ' ')
                corpus[pmid] += text
    return corpus

def tokenize_corpus(corpus, outdir, tokenizer="tfidf", output="pkl", max_df=0.5, min_df=0.01):
    # Tokenize a corpus into a doc-by-token sparse matrix
    logger = get_console_logger("tokenize_corpus")
    logger.info("Building bag-of-words representation ...")
    if tokenizer == "tf":
        vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)
    elif tokenizer == "tfidf":
        vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
    else:
        logger.error("Invalid tokenizer.")
    #bow = vectorizer.fit_transform(list(corpus.values()))
    bow = vectorizer.fit_transform([corpus[x] for x in sorted(corpus)])
    if output == "pkl":
        outfile = outdir + "bag_of_words_" + tokenizer + ".pkl"
        logger.info("Writing to %s ..." % format(outfile))
        with open(outfile, 'wb') as fout:
            pickle.dump(bow, fout)
            #bow_rownames = list(corpus.keys())
            bow_rownames = sorted(corpus)
            pickle.dump(bow_rownames, fout)
            bow_colnames = vectorizer.get_feature_names()
            pickle.dump(bow_colnames, fout)
    #elif output == "csv":
    #    
    else:
        logger.error("Invalid tokenization output format.")
    logger.info("Done!")

if __name__ == "__main__":
    INDIR = "../data/raw/pubmed-txts-mf/"
    OUTDIR = "../data/processed/"
    corpus = read_corpus(indir=INDIR)
    tokenize_corpus(corpus, OUTDIR, tokenizer="tfidf", output="pkl", max_df=0.5, min_df=0.01)

