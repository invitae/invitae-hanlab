import pickle, heapq
from itertools import compress
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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

# Take the topk tokens
topk = 1000
df = (bow > 0).sum(axis=0)
bow = bow[:,(df > 0).A1]  # Remove words with zero occurance in the sub-corpus
df = (bow > 0).sum(axis=0)
tf = bow.sum(axis=0)
idf = np.log( (len(bow_rownames) + 1) / df )
tfidf = np.multiply(tf, idf)
# Select tokens based on TF-IDF
cutoff = heapq.nlargest(topk, tfidf.A1.tolist())[topk-1]
bow_freq = bow[:,(tfidf >= cutoff).A1]
# Select tokens based on DF
#cutoff = heapq.nlargest(topk, df.A1.tolist())[topk-1]
#bow_freq = bow[:,(df >= cutoff).A1]

# Proportion of False labels (i.e. accuracy of an all-False classifier)
false_prop = (labels == False).sum(axis=0) / labels.shape[0]
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

