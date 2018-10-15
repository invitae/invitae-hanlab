# Train and test bag-of-words (BoW) classifiers for doc label prediction

import sys
import pickle
import heapq
from itertools import compress
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from modules import *

def read_bag_of_words(indir, tokenizer="tfidf"):
    with open(indir + "bag_of_words_" + tokenizer + ".pkl", 'rb') as fin:
        bow = pickle.load(fin)  # doc-by-word count sparse matrix
        bow_rownames = pickle.load(fin)  # list of PMIDs
        bow_colnames = pickle.load(fin)  # list of distinct words
    return bow, bow_rownames, bow_colnames

def read_doc_labels(indir):
    with open(indir + "doc_labels.pkl", 'rb') as fin:
        labels = pickle.load(fin)  # doc labels
    return labels

def rm_doc_without_label(bow, bow_rownames, labels):
    # Remove docs that have no labels from the BoW matrix
    labels_pos_ind = (labels.sum(axis=1) > 0)
    labels_pos_ind.sum() * 1.0 / len(labels_pos_ind)
    # Return subsets of bow, bow_rownames, and labels
    return bow[labels_pos_ind.values,:], list(compress(bow_rownames, labels_pos_ind.values)), labels[labels_pos_ind.values]

def model_initiator(classifier, n_estimators=1):
    clf = None
    logger = get_console_logger(name="model_initiator")
    if classifier == "svc-linear-l1":
        clf = LinearSVC(dual=False, random_state=0, tol=1e-5, penalty='l1')
    elif classifier == "svc-linear-l2":
        clf = LinearSVC(random_state=0, tol=1e-5)
    elif classifier == "svc-radial":
        clf = SVC(kernel='rbf', gamma='auto')
    elif classifier == "decision-tree":
        clf = DecisionTreeClassifier(random_state=0, max_depth=10)
    elif classifier == "random-forest":
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=10)
    else:
        logger.error("Classifier is not supported.")
    if n_estimators > 1:
        clf_bagging = BalancedBaggingClassifier(base_estimator=clf, n_estimators=n_estimators, max_samples=1.0/n_estimators)
        return clf_bagging
    else:
        return clf


def cross_validation(classifier, X, y, cv_fold=3, n_estimators=1, n_jobs=-1, par=[]):
    scoring = ["precision", "recall", "f1", "accuracy"]
    logger = get_console_logger(name=classifier)
    logger.info("Training & testing {0} with {1}-fold CV ...".format(classifier, cv_fold))
    clf = model_initiator(classifier, n_estimators)
    cv_results = cross_validate(clf, X, y, cv=cv_fold, n_jobs=n_jobs, scoring=scoring, return_train_score=False)
    logger.info("  Accuracy: {0}".format(sum(cv_results['test_accuracy']) / cv_fold))
    logger.info("  Precision: " + str(sum(cv_results['test_precision']) / cv_fold))
    logger.info("  Recall: " + str(sum(cv_results['test_recall']) / cv_fold))
    logger.info("  F1: " + str(sum(cv_results['test_f1']) / cv_fold))
    logger.info("")
    return cv_results


if __name__ == "__main__":
    INDIR = "../data/processed/"
    CV_FOLD = 3
    CLASSIFIER = sys.argv[1]
    N_ESTIMATORS = int(sys.argv[2])
    logger = get_console_logger(name="classifiers")
    bow, bow_rownames, bow_colnames = read_bag_of_words(INDIR, tokenizer='tfidf')
    labels = read_doc_labels(INDIR)
    bow, bow_rownames, labels = rm_doc_without_label(bow, bow_rownames, labels)
    print(len(bow_colnames))
    for label in ["individual_observations", "functional_experiments", "family_studies", "sequence_observations"]:
        logger.info("Class: " + label)
        cv_results = cross_validation(classifier=CLASSIFIER, X=bow, y=labels[label], cv_fold=CV_FOLD, n_estimators=N_ESTIMATORS, n_jobs=-1, par=[])
    logger.info("N_ESTIMATORS: " + str(N_ESTIMATORS))
    logger.info("N_TOKENS: " + str(len(bow_colnames)))


# Ignore words whose DF does not fall within a given range
#max_df = 0.5
#min_df = 0.01
#df = (bow > 0).sum(axis=0) / len(bow_rownames)
#index = np.logical_and((df <= max_df), (df >= min_df)) # 13,255 words remain
#bow = bow[:, index.A1]
#bow_freq = bow

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
