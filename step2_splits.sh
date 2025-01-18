#!/bin/bash


java_command=java
JAR=kNNDissimilarities.jar
jvm_memory=-Xmx100G


datasetMovielens="Movielens20M"
datasetGoodReads="GoodReadsSpoiler"
datasetAmazonDiscVinyls="AmazonDiscVinyls"
datasetLastfmHetRec="LastfmHetrec"
kCore=5


#First, new Ids for goodreads
$java_command $jvm_memory -jar $JAR -o processJSONGoodReads -trf OriginalFiles/goodreads_reviews_spoiler.json -orf OriginalFiles/goodreads_reviews_spoiler_JSON_Parsed.txt
$java_command $jvm_memory -jar $JAR -o DatasetTransform -trf OriginalFiles/goodreads_reviews_spoiler_JSON_Parsed.txt OriginalFiles/goodreads_reviews_spoiler_JSON_ParsedTransformed.txt "\t" true


#Processing the Amazon Review dataset (discs)
$java_command $jvm_memory -jar $JAR -o DatasetTransform -trf OriginalFiles/ratings_CDs_and_Vinyl.csv OriginalFiles/ratings_CDs_and_VinylTransformed.txt "," true
$java_command $jvm_memory -jar $JAR -o DatasetReduction -trf OriginalFiles/ratings_CDs_and_VinylTransformed.txt -mru $kCore -mri $kCore -orf OriginalFiles/ratings_CDs_and_VinylTransformed"$kCore"Core.txt
rm OriginalFiles/ratings_CDs_and_VinylTransformed.txt


#Random Global
train_test=TrainTestRandomGlobal8020
mkdir -p $train_test


#For Movielens 20M
$java_command $jvm_memory -jar $JAR -o simpleRandomSplit -trf "OriginalFiles/ratingsNoFirstLineTabs.csv" $train_test/"$datasetMovielens"RandomGlobal8020Train.txt $train_test/"$datasetMovielens"RandomGlobal8020Test.txt 0.8 2242
$java_command $jvm_memory -jar $JAR -o Statistics -trf "OriginalFiles/ratingsNoFirstLineTabs.csv" Movielens20MStats.txt

#For lastFM
$java_command $jvm_memory -jar $JAR -o simpleRandomSplit -trf OriginalFiles/Original_LastfmHetrec.txt $train_test/"$datasetLastfmHetRec"RandomGlobal8020Train.txt $train_test/"$datasetLastfmHetRec"RandomGlobal8020Test.txt  0.8 2242
$java_command $jvm_memory -jar $JAR -o Statistics -trf "OriginalFiles/Original_LastfmHetrec.txt" LastfmStats.txt

#For AmazonDiscs
$java_command $jvm_memory -jar $JAR -o simpleRandomSplit -trf "OriginalFiles/ratings_CDs_and_VinylTransformed"$kCore"Core.txt" $train_test/"$datasetAmazonDiscVinyls""$kCore"RandomGlobal8020Train.txt $train_test/"$datasetAmazonDiscVinyls""$kCore"RandomGlobal8020Test.txt 0.8 2242
$java_command $jvm_memory -jar $JAR -o Statistics -trf "OriginalFiles/ratings_CDs_and_VinylTransformed"$kCore"Core.txt" VynilsStats.txt

#For GoodReads
$java_command $jvm_memory -jar $JAR -o simpleRandomSplit -trf "OriginalFiles/goodreads_reviews_spoiler_JSON_ParsedTransformed.txt" $train_test/"$datasetGoodReads"RandomGlobal8020Train.txt $train_test/"$datasetGoodReads"RandomGlobal8020Test.txt 0.8 2242
$java_command $jvm_memory -jar $JAR -o Statistics -trf "OriginalFiles/goodreads_reviews_spoiler_JSON_ParsedTransformed.txt" GoodReadsStats.txt

chmod +777 ./MyMediaLite-3.11/bin/item_recommendation
