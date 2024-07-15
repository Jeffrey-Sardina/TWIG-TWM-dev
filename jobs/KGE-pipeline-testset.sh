#!/bin/bash

cd ../src

hyp_id=$1
model=$2
dataset=$3

out_dir="TWIG-I/baseline_exps/${model}_${dataset}_hypID-${hyp_id}/"
echo "running ${out_dir}"
mkdir $out_dir

python KGE-pipeline-testset.py \
    $hyp_id \
    $model \
    $dataset \
    $out_dir \
    1>> $out_dir/out.res \
    2>> $out_dir/out.log