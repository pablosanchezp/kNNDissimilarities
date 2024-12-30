#!/bin/bash


java_command=java
JAR=kNNDissimilarities.jar
jvm_memory=-Xmx100G


mkdir -p OriginalFiles



# Download GoodReads
echo "Downloading Goodreads"
urlgoodreads=https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_reviews_spoiler.json.gz
curl -s $urlgoodreads | gunzip -c > OriginalFiles/goodreads_reviews_spoiler.json

echo "Downloading Movielesn20M"
# Download Movielens 20M
urlMoviels20M=https://files.grouplens.org/datasets/movielens/ml-20m.zip
curl -O $urlMoviels20M && unzip ml-20m.zip && rm ml-20m.zip
tail -n +2 ml-20m/ratings.csv | tr ',' '\t' > OriginalFiles/ratingsNoFirstLineTabs.csv

# Download Lasftm dataset
urllastfm=https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
curl -O $urllastfm && unzip hetrec2011-lastfm-2k.zip

$java_command $jvm_memory -jar $JAR -o ImplicitToExplicit -trf user_artists.dat OriginalFiles/Original_LastfmHetrec.txt "\t"
rm user_artists.dat user_friends.dat user_taggedartists-timestamps.dat user_taggedartists.dat hetrec2011-lastfm-2k.zip tags.dat artists.dat

# Download Amazon Discs and Vinyls
echo "Downloading AmazonVinyls"
urlamazondiscs=https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_CDs_and_Vinyl.csv
curl -O $urlamazondiscs
mv ratings_CDs_and_Vinyl.csv OriginalFiles/
