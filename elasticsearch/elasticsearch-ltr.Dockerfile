FROM docker.elastic.co/elasticsearch/elasticsearch:6.4.2

# Pulling the elasticsearch-ltr plugin from our repo beacuse the upstream hasn't made this version available
RUN elasticsearch-plugin install --batch https://github.com/invitae/invitae-hanlab/releases/download/v0.0.0b/ltr-1.1.0-es6.4.2.zip
