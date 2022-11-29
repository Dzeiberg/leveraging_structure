#!/bin/bash

echo "writing experiments to ${1}"
mkdir -p ${1}

for clf in rf
do
    for rep in {0..4}
    do

        for dim in 1 4 16 64
        do
            for nClusters in 1 2 4 8
            do
                for settingNum in {1..2}
                do
                    echo "running ${clf} - ${rep} - dim ${dim} - ${nClusters} clusters - setting ${settingNum}"
                    python ~/MultiInstancePU/runExperiment.py ${1}/synthetic_dim${dim}_nClusters${nClusters}_setting${settingNum}_${clf}_${rep} \
                    --problem_name synthetic --datasetType synthetic --syntheticDim ${dim} --syntheticNClusters ${nClusters} --clf ${clf} --problem_setting ${settingNum} > \
                    ${1}/synthetic_dim${dim}_nClusters${nClusters}_setting${settingNum}_${clf}_${rep}.out 2>&1
                    tail -n 10 ${1}/synthetic_dim${dim}_nClusters${nClusters}_setting${settingNum}_${clf}_${rep}.out
                    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                done
            done
        done

        for problem in income employment income_poverty_ratio
        do
            for settingNum in {1..2}
            do
                echo "running ${clf} - ${rep} - ${problem} - setting ${settingNum}"
                python ~/MultiInstancePU/runExperiment.py ${1}/${problem}_${rep}_${clf}_setting${settingNum} \
                --problem_name ${problem} --clf ${clf} --problem_setting ${settingNum} > \
                ${1}/${problem}_${rep}_${clf}_setting${settingNum}.out 2>&1
                tail -n 10 ${1}/${problem}_${rep}_${clf}_setting${settingNum}.out
                echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            done
        done

        
    done
done