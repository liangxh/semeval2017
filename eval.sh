#! /bin/bash

key=$1
perl eval/score-semeval2016-task4-subtask${key}.pl ../data/result/gold_result${key}.txt ../data/result/pred_result${key}.txt
