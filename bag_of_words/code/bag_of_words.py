# Read papers and convert them to bag-of-words representations.

indir = "../data/raw/pubmed-txts-mf/"
outdir = "../data/processed/"
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
import os
import numpy as np
import pickle

# Read papers into a list
pmids = os.listdir(indir)
pmids.sort()
# pmids = pmids[:5]  # For debugging
corpus = [''] * len(pmids)
print("Reading corpus ...")
for i in range(len(pmids)):
    if i % 10000 == 0:
        print(str(i) + " docs processed ...")
    for field in os.listdir(indir + pmids[i] + '/'):
        with open(indir + pmids[i] + '/' + field, 'r') as fin:
            text = fin.read().replace('\n', ' ')
            corpus[i] += text
print('')

# Convert papers to bags of words, and save objects to file
print("Building bag-of-words representation ...")
bow = vectorizer.fit_transform(corpus)
with open(outdir + "bag_of_words.pkl", 'wb') as fout:
    pickle.dump(bow, fout)
    bow_rownames = pmids
    pickle.dump(bow_rownames, fout)
    bow_colnames = vectorizer.get_feature_names()
    pickle.dump(bow_colnames, fout)
print("Done!")
