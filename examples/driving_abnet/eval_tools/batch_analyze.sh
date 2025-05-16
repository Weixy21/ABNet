#!/bin/bash

ROOT_DIR=$1

i=0
for FILE in $(ls -d $ROOT_DIR/*/results.pkl); do
    echo $i $FILE
    i=$((i+1))
    python eval_tools/analyze_vista_results.py --results-path $FILE
done