#!/bin/bash

cd ../

datasets="Kinships"
n_first_epochs=5
n_second_epochs=10
kfold=0
test_mode="hyp"

for test_ratio in 0.25 0.5 0.75 0.9 0.95 0.99
do
    for repliate_num in 1
    do
        tag="test-ratio-${test_ratio}_rep-${repliate_num}"
        ./twig-pipeline.sh \
            $datasets \
            $n_first_epochs \
            $n_second_epochs \
            $tag \
            $kfold \
            $test_mode \
            $test_ratio
    done
done
