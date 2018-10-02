#!/usr/bin/env bash

for file in $(ls data/test-json)
do 
  echo indexing $file
  curl -s -H "Content-Type: application/json" -XPUT localhost:9200/test/_doc/$file --data-binary @data/test-json/$file
done

