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

## How to Run WeSTClass

In `bag_of_words/data_westclass` there is a sample dataset for testing the WeSTClass weakly-supervised text classification tool. You may follow those steps:

1. Clone the WeSTClass repo at `https://github.com/invitae/WeSTClass.git`.
2. Replace the files in the folder `WestClass/yelp` with the files in `bag_of_words/data_westclass`.
3. Modify the file `WeSTClass/main.py` and set the variable `max_sequence_length` to 10000 or a larger number.
4. In WeSTClass folder, execute `python main.py --dataset yelp --sup_source docs --model cnn --with_evaluation False`. Make sure you are using Python 3.6. You may change the parameters as needed.