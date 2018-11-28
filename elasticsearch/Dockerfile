FROM docker.elastic.co/elasticsearch/elasticsearch:6.5.0

# install ElasticSearch Learning to Rank plugin
RUN ./bin/elasticsearch-plugin install --batch http://es-learn-to-rank.labs.o19s.com/ltr-1.1.0-es6.5.0.zip
RUN ./bin/elasticsearch-plugin install --batch ingest-attachment

