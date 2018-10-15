# Train and test bag-of-words classifiers for doc label prediction

import pickle
import heapq
from itertools import compress
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from modules import *

INDIR = "../data_sample/processed/"

with open(INDIR + "bag_of_words.pkl", 'rb') as fin:
    bow = pickle.load(fin)  # doc-by-word count sparse matrix
    bow_rownames = pickle.load(fin)  # list of PMIDs
    bow_colnames = pickle.load(fin)  # list of distinct words

with open(INDIR + "doc_labels.pkl", 'rb') as fin:
    labels = pickle.load(fin)  # doc labels

# Remove docs that have no labels
labels_pos_ind = (labels.sum(axis=1) > 0)
labels_pos_ind.sum() * 1.0 / len(labels_pos_ind)
bow = bow[labels_pos_ind.values,:]
bow_rownames = list(compress(bow_rownames, labels_pos_ind.values))
labels = labels[labels_pos_ind.values]

# Ignore words whose DF does not fall within a given range
max_df = 0.5
min_df = 0.01
df = (bow > 0).sum(axis=0) / len(bow_rownames)
index = np.logical_and((df <= max_df), (df >= min_df)) # 13,255 words remain
bow = bow[:, index.A1]
bow_freq = bow

# Take the topk tokens
#topk = index.sum()
#df = (bow > 0).sum(axis=0)
#tf = bow.sum(axis=0)
#idf = np.log( (len(bow_rownames) + 1) / df )
#tfidf = np.multiply(tf, idf)
# Select tokens based on TF-IDF
#cutoff = heapq.nlargest(topk, tfidf.A1.tolist())[topk-1]
#bow_freq = bow[:,(tfidf >= cutoff).A1]
# Select tokens based on DF
#cutoff = heapq.nlargest(topk, df.A1.tolist())[topk-1]
#bow_freq = bow[:,(df >= cutoff).A1]

# Proportion of False labels (i.e. accuracy of an all-False classifier)
#false_prop = (labels == False).sum(axis=0) / labels.shape[0]
#false_prop.sort_values()


# Train and test classifiers for a specific label
cv_fold = 3
for label in ["individual_observations", "functional_experiments", "family_studies", "sequence_observations"]:
    print("Class: " + label)
    scoring = ["precision", "recall", "f1", "accuracy"]
    
    print("LinearSVC with L2:")
    clf = LinearSVC(random_state=0, tol=1e-5)
    cv_results = cross_validate(clf, bow_freq, labels[label], cv=cv_fold, n_jobs=-1, scoring=scoring, return_train_score=False)
    print("  Accuracy: " + str(sum(cv_results['test_accuracy']) / cv_fold))
    print("  Precision: " + str(sum(cv_results['test_precision']) / cv_fold))
    print("  Recall: " + str(sum(cv_results['test_recall']) / cv_fold))
    print("  F1: " + str(sum(cv_results['test_f1']) / cv_fold))
    print("")
    
    print("LinearSVC with L1:")
    clf = LinearSVC(dual=False, random_state=0, tol=1e-5, penalty='l1')
    cv_results = cross_validate(clf, bow_freq, labels[label], cv=cv_fold, n_jobs=-1, scoring=scoring, return_train_score=False)
    print("  Accuracy: " + str(sum(cv_results['test_accuracy']) / cv_fold))
    print("  Precision: " + str(sum(cv_results['test_precision']) / cv_fold))
    print("  Recall: " + str(sum(cv_results['test_recall']) / cv_fold))
    print("  F1: " + str(sum(cv_results['test_f1']) / cv_fold))
    print("")
    
    print("Decision Tree:")
    clf = DecisionTreeClassifier(random_state=0, max_depth=10)
    cv_results = cross_validate(clf, bow_freq, labels[label], cv=cv_fold, n_jobs=-1, scoring=scoring, return_train_score=False)
    print("  Accuracy: " + str(sum(cv_results['test_accuracy']) / cv_fold))
    print("  Precision: " + str(sum(cv_results['test_precision']) / cv_fold))
    print("  Recall: " + str(sum(cv_results['test_recall']) / cv_fold))
    print("  F1: " + str(sum(cv_results['test_f1']) / cv_fold))
    print("")
    
    print("Random Forest:")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=10)
    cv_results = cross_validate(clf, bow_freq, labels[label], cv=cv_fold, n_jobs=-1, scoring=scoring, return_train_score=False)
    print("  Accuracy: " + str(sum(cv_results['test_accuracy']) / cv_fold))
    print("  Precision: " + str(sum(cv_results['test_precision']) / cv_fold))
    print("  Recall: " + str(sum(cv_results['test_recall']) / cv_fold))
    print("  F1: " + str(sum(cv_results['test_f1']) / cv_fold))
    print("")
    print("")
    print("")
