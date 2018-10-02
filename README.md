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

