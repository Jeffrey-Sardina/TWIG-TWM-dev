#!/bin/bash

cd ../

# Note: You will need to run this from your own checkpoint_name and epoch_state
# this is an illustrative example, but will not run
checkpoint_name="chkpt-ID_430396720502062_v3_CoDExSmall-DBpedia50-Kinships-OpenEA"
epoch_state="e5-e10"
override_file="override-example.json"
tag="TWIG_Model"
./TWIG-from-checkpoint.sh \
    $checkpoint_name \
    $epoch_state \
    $override_file \
    $tag
