#!/bin/bash

cd ../src

exp_name=$1
run_nums=$2
num_processes=$3
datasets=$4 #delimit by spaces

start=`date +%s`
seed=None

echo "re-eval pipeline configured with $num_processes processes"
echo "if memory issues occur, please restart with fewer processes"

for dataset in $datasets
do
    for run_num in $run_nums
    do
        run_name="$dataset-$exp_name-run2.$run_num"
        echo "running eval for $run_name"

        python KGE-pipeline-reeval.py \
            output/$dataset/$run_name/$run_name.grid \
            output/$dataset/$run_name/ \
            $num_processes \
            $dataset \
            $seed \
            1>> output/$dataset/$run_name/$run_name.reval.res \
            2>> output/$dataset/$run_name/$run_name.reval.log

        end=`date +%s`
        runtime=$((end-start))
        echo "Experiments took $runtime seconds" 1>> output/$dataset/$run_name/$run_name.log
    done
done
