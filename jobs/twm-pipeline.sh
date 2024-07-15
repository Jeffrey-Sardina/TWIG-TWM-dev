#!/bin/bash

cd ../src

user_cmd=$1

if [[ $user_cmd = "kill" ]]
then
    echo "Killing running serves processes only; no servers will be started"
    ps -ef | grep "TWM_app_base.py" | grep -v "grep" | awk '{print $2}' | xargs kill -9 &> /dev/null
else
    echo "If you have previously started these serves, those processes will be automatically killed"
    echo "Your new servers will then be created"
    ps -ef | grep "TWM_app_base.py" | grep -v "grep" | awk '{print $2}' | xargs kill -9 &> /dev/null

    TWM_LOG_PATH=$(realpath "../jobs/twm.log")

    # run flash app and Gephi lite in subshells so they stay in the correct
    # dirs regardless of future any possible cd's in this file (main shell)
    (cd TWM/ && python TWM_app_base.py &> "$TWM_LOG_PATH") &

    echo "Note: this may take a minute to start..."
    echo "TWIG: http://127.0.0.1:5000/"
    echo "External Visualiser: https://gephi.org/gephi-lite/"

    sleep 10
    echo "If you want to kill the background job, here are the process IDs:"
    echo $(ps -ef | grep "TWM_app_base.py" | grep -v "grep" | awk '{print $2}') 
fi
