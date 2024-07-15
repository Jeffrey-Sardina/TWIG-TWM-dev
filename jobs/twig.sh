#!/bin/bash

cd ../src

# global constants
n_first_epochs=5
n_second_epochs=10
kfold=0
test_mode="hyp"


# run all datasets
tag="main-run"
datasets="CoDExSmall-DBpedia50-Kinships-OpenEA-UMLS"
./twig-pipeline.sh \
    $datasets \
    $n_first_epochs \
    $n_second_epochs \
    $tag \
    $kfold \
    $test_mode 


# run with one holdout
tag="pretrain-4"
datasets="CoDExSmall-DBpedia50-Kinships-OpenEA"
./twig-pipeline.sh \
    $datasets \
    $n_first_epochs \
    $n_second_epochs \
    $tag \
    $kfold \
    $test_mode 

datasets="CoDExSmall-DBpedia50-Kinships-UMLS"
./twig-pipeline.sh \
    $datasets \
    $n_first_epochs \
    $n_second_epochs \
    $tag \
    $kfold \
    $test_mode 

datasets="CoDExSmall-DBpedia50-OpenEA-UMLS"
./twig-pipeline.sh \
    $datasets \
    $n_first_epochs \
    $n_second_epochs \
    $tag \
    $kfold \
    $test_mode 

datasets="CoDExSmall-Kinships-OpenEA-UMLS"
./twig-pipeline.sh \
    $datasets \
    $n_first_epochs \
    $n_second_epochs \
    $tag \
    $kfold \
    $test_mode 
    
datasets="DBpedia50-Kinships-OpenEA-UMLS"
./twig-pipeline.sh \
    $datasets \
    $n_first_epochs \
    $n_second_epochs \
    $tag \
    $kfold \
    $test_mode 
