#FROM gradle:jdk10 as ltr-builder
#
#RUN git clone https://github.com/o19s/elasticsearch-learning-to-rank.git
#WORKDIR /home/gradle/elasticsearch-learning-to-rank
#RUN git checkout 6344408d06461acd270d64a75d12a6c1c118e796
#RUN sed -i 's/6\.4\.1/6\.4\.2/g' build.gradle
#RUN cat build.gradle
#RUN ./gradlew clean check

# install the ltr-plugin
FROM docker.elastic.co/elasticsearch/elasticsearch:6.4.2

# install ElasticSearch Learning to Rank plugin
#COPY --from=ltr-builder /home/gradle/elasticsearch-learning-to-rank/build/distributions/ltr-1.1.0-es6.4.2.zip /tmp/
#RUN elasticsearch-plugin install --batch file:///tmp/ltr-1.1.0-es6.4.2.zip

RUN elasticsearch-plugin install --batch https://github.com/invitae/invitae-hanlab/releases/download/v0.0.0b/ltr-1.1.0-es6.4.2.zip
