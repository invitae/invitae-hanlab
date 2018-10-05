import pickle
from itertools import compress
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

indir = "../data/processed/"

with open(indir + "bag_of_words.pkl", 'rb') as fin:
    bow = pickle.load(fin)  # doc-by-word count sparse matrix
    bow_rownames = pickle.load(fin)  # list of PMIDs
    bow_colnames = pickle.load(fin)  # list of distinct words

with open(indir + "doc_labels.pkl", 'rb') as fin:
    labels = pickle.load(fin)  # doc labels

# Remove docs that have no labels
labels_pos_ind = (labels.sum(axis=1) > 0)
labels_pos_ind.sum() * 1.0 / len(labels_pos_ind)
bow = bow[labels_pos_ind.values,:]
bow_rownames = list(compress(bow_rownames, labels_pos_ind.values))
labels = labels[labels_pos_ind.values]

# Take the most frequent ~1000 words
bow_colsums = (bow > 0).sum(axis=0) / len(bow_rownames)
(bow_colsums > 0.2).sum()  # The most frequent 1155 words appear in >20% of all papers
bow_freq = bow[:,(bow_colsums > 0.2).A1]

# Proportion of False labels (i.e. accuracy of an all-False classifier)
false_prop = (labels == False).sum(axis=0) / labels.shape[0]
#false_prop.sort_values()


# Train and test classifiers for a specific label
label = "family_studies"

clf = LinearSVC(random_state=0, tol=1e-5)
cv_results = cross_validate(clf, bow_freq, labels[label], cv=3)
sum(cv_results['test_score']) / len(cv_results['test_score'])

clf = DecisionTreeClassifier(random_state=0, max_depth=10)
cv_results = cross_validate(clf, bow_freq, labels[label], cv=3)
sum(cv_results['test_score']) / len(cv_results['test_score'])
