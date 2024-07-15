#!/bin/bash

cd ../src

export TWIG_CHANNEL=2

# ./TWIG_pipeline.sh CoDExSmall-UMLS-DBpedia50-OpenEA-Kinships 5 10 TWM_run_hyp 0 hyp

datasets=$1 #ex "CoDExSmall_UMLS_DBpedia50_OpenEA_Kinships" # delimit by "_"
n_first_epochs=$2
n_second_epochs=$3
tag=$4
kfold=$5
test_mode=$6 #exp or hyp
test_ratio=$7
versions="3"
normalisation="zscore"
rescale_y="1"

for version in $versions
do
    out_file="rec_v${version}_${datasets}_norm-${normalisation}_e${n_first_epochs}-e${n_second_epochs}_tag-${tag}.log"

    echo "args" 1>> $out_file
    echo "datasets: $datasets" 1>> $out_file
    echo "n_first_epochs: $n_first_epochs" 1>> $out_file
    echo "n_second_epochs: $n_second_epochs" 1>> $out_file
    echo "tag: $tag" 1>> $out_file
    echo "kfold: $kfold" 1>> $out_file
    echo "test_mode: $test_mode" 1>> $out_file
    echo "test_ratio: $test_ratio" 1>> $out_file
    echo "versions: $versions" 1>> $out_file
    echo "normalisation: $normalisation" 1>> $out_file
    echo "rescale_y: $rescale_y" 1>> $out_file
    echo "" 1>> $out_file

    exp_start=`date +%s`
    python -u run_exp.py \
        $version \
        "$datasets" \
        $n_first_epochs \
        $n_second_epochs \
        $normalisation \
        $rescale_y \
        $test_mode \
        $kfold \
        $test_ratio &> $out_file
    exp_end=`date +%s`
    exp_runtime=$((end-start))
    echo "Experiments took $exp_runtime seconds" 1>> $out_file
done

