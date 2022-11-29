for i in {0..9}
do
	echo "Running Amazon ${i}"
	python runExperiment.py --experimentPath /data/dzeiberg/leveragingStructureResponseExperiments/experiments/amazon_reviews_pca_setting_2_$i --datasetType huggingface --problem_name amazon_reviews_multi --pca True
done
