#!/bin/bash

cd ../src

run_nums=$1
datasets=$2 # ex "CoDExSMall DBpedia50 OpenEA" #$4 #delimit by spaces
models=$3 # ex"DistMult TransE" #delimit by spaces
exp_name="TWM"
num_processes=1

start=`date +%s`
seed=None

echo "pipeline configured with $num_processes processes"
echo "if memory issues occur, please restart with fewer processes"

for dataset in $datasets
do
    for model in $models
    do
        for run_num in $run_nums
        do
            run_name="$dataset-$model-$exp_name-run2.$run_num"
            echo "running $run_name"

            mkdir output/$run_name &> /dev/null
            python KGE_pipeline.py \
                output/$run_name/$run_name.grid \
                output/$run_name/ \
                $num_processes \
                $dataset \
                $model \
                $seed \
                1>> output/$run_name/$run_name.res \
                2>> output/$run_name/$run_name.log

            end=`date +%s`
            runtime=$((end-start))
            echo "Experiments took $runtime seconds" 1>> output/$run_name/$run_name.log
        done
    done
done
