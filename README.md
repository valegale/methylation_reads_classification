# A collection of scripts to run a read classification based on methylation profile

## Conda environment

Create a new environment with:

`conda env create -f environment.yml`

It includes pytorch plus other required packages to run the python scripts. 

## Feature-based analysis (Logistic regression, Random forest, SVM)
Enter the `feature_based_classifier subfolder` and tun the classifier script with:
`python -m training.classifiers` which will train the three classifiers:

`Logistic_Regression_model_cpg_True_mean_False_cutoff_0.8.joblib`
`Random_Forest_model_cpg_True_mean_False_cutoff_0.8.joblib`
`SVM_Forest_model_cpg_True_mean_False_cutoff_0.8.joblib`

and save them inside the `models` directory. 

Finally run 
`python -m test.test_results`

which will perform statistical tests and save a plot accuracy.png. 

## CNN


First create the input tensors based on the bam and bed files. After entering the `cnn` subfodler, run: 

`python -m training.create_input_batches 5 0.8`

The parameter 5 indicates a window of 5 nt around the methylated base that are not masked in the tensor. The parameter 0.8 is a cutoff for methylation (only bases with a methylation probability > 0.8 are considered). 
The tensors will be saved in the folder: `training_data_window5_cutoff0.8`.

Train the cnn with:
`python -m training.train_cnn 5 0.8`

The model will be saved as `training_data_window5_cutoff0.8.pth`.

Similarly, to test the cnn, first create the batches with:
`python -m test.create_input_batches 5 0.8`

and then run:

`python -m test.test_cnn 5 0.8`

The results will be saved in `test_results_window5_cutoff0.8.txt`.