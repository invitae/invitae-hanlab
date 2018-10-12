# invitae-hanlab

## ElasticSearch

ElasticSearch docker-compose config was adapted from [Elastic's Docker documentation](https://www.elastic.co/guide/en/elasticsearch/reference/6.3/docker.html).
A custom Dockerfile was created to bootstrap the installation with the [ElasticSearch Learning to Rank Plugin](https://elasticsearch-learning-to-rank.readthedocs.io/en/latest/).
We are using the default container number recommended by ElasticSearch of 2 for local development, though in Invitae's production
infrastructure we can scale this to more containers.

To start the ElasticSearch cluster locally, navigate to `./elasticsearch` and run:

```
docker-compose up
```

To load test documents into the index, start up the cluster and run:

```
$ test_index.sh
```

## Bag of Words (BoW)

Note: Codes and a sample dataset are included. The original data files are not. The sample dataset was generated from the original dataset with the script `make_data_sample.py`.

Steps:

1. Run `bag_of_words.py` to convert the corpus to a doc-by-word BoW matrix.
2. Run `doc_labels.py` to build a doc-by-label matrix.
3. Run `classifiers.py` to train and test various classifiers.

