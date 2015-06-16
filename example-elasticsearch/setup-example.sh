#!/bin/bash

# get the test data, see http://www.cs.cornell.edu/People/pabo/movie-review-data/
mkdir data
cd data
wget -nc  http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz 
tar xf review_polarity.tar.gz

cd ..
cd example-elasticsearch

# Install elasticsearch 1.5.2
wget -nc https://download.elastic.co/elasticsearch/elasticsearch/elasticsearch-1.5.2.tar.gz
tar xvf elasticsearch-1.5.2.tar.gz
# build plugin  

rm -rf token-plugin/
mkdir token-plugin
cd token-plugin
git clone https://github.com/brwe/es-token-plugin .

mvn clean -DskipTests $2 package

currentpath=$(pwd)
../elasticsearch-1.5.2/bin/plugin -r token_plugin
../elasticsearch-1.5.2/bin/plugin -i token_plugin -u "file://localhost$currentpath/target/releases/es-token-plugin-0.0.0-SNAPSHOT.zip"

# install marvel because sense is so convinient
../elasticsearch-1.5.2/bin/plugin -i elasticsearch/marvel/latest

# enable dynamic scripting
echo 'script.disable_dynamic: false' >> ../elasticsearch-1.5.2/config/elasticsearch.yml

# set reloading of file scripts lower
echo 'watcher.interval: "10s"' >> ../elasticsearch-1.5.2/config/elasticsearch.yml

# start elasticsearch
cd ..
elasticsearch-1.5.2/bin/elasticsearch 

