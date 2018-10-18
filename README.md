# invitae-hanlab

## ElasticSearch

Instruction on running a local Elasticsearch cluster on Kubernetes can be found [here](elasticsearch).

## Bag of Words (BoW)

Note: Codes and a sample dataset are included. The original data files are not. The sample dataset was generated from 
the original dataset with the script `make_data_sample.py`.

Steps:

1. Run `bag_of_words.py` to convert the corpus to a doc-by-word BoW matrix.
2. Run `doc_labels.py` to build a doc-by-label matrix.
3. Run `classifiers.py` to train and test various classifiers.

