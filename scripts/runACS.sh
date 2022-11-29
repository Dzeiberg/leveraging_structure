for i in {0..49}
do
	echo "Running Income ${i}"
	python runExperiment.py --experimentPath /data/dzeiberg/leveragingStructureResponseExperiments/experiments/income_setting_2_$i --datasetType acs --problem_name income >>output/income_$i.out 2>&1

	echo "Running Employment ${i}"
	python runExperiment.py --experimentPath /data/dzeiberg/leveragingStructureResponseExperiments/experiments/employment_setting_2_$i --datasetType acs --problem_name employment >> output/employment_$i 2>&1

	echo "Running IPR ${i}"
	python runExperiment.py --experimentPath /data/dzeiberg/leveragingStructureResponseExperiments/experiments/ipr_setting_2_$i --datasetType acs --problem_name income_poverty_ratio >> output/ipr_$i.out 2>&1
	
	# echo "Running Amazon ${i}"
	# python runExperiment.py --experimentPath /data/dzeiberg/leveragingStructureResponseExperiments/experiments/amazon_reviews_pca_setting_2_$i --datasetType huggingface --problem_name amazon_reviews_multi >>output/amazon_$i.out 2>&1
done
