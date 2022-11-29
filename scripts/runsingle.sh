#!/bin/bash

echo "writing experiments to ${5}"
mkdir -p ${5}

echo "running ${1} - ${2} - ${3} - setting ${4}"
python ~/MultiInstancePU/runExperiment.py ${5}/${3}_${2}_${1}_setting${4} \
--problem_name ${3} --clf ${1} --problem_setting ${4} --minClassCount 1 > \
${5}/${3}_${2}_${1}_setting${4}.out 2>&1
tail -n 10 ${5}/${3}_${2}_${1}_setting${4}.out
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"