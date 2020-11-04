# Code for replication of "Treatment Policy Learning in Multiobjective Settings with Fully Observed Outcomes"

This code is meant to be used in conjunction with the [AMR-UTI dataset](http://www.clinicalml.org/data/amr-dataset) for replication of the main analyses in the paper. 

## Preface: Important Note on Replication

The uncomplicated specimens included in this dataset release (in both the train and test sets) are identical to those used for our analyses.

Nonetheless, there are minor differences that will arise when replicating our analyses with the released dataset.  There are two broad reasons for this:
1. Slight differences in features, primarily due to de-identification efforts.
2. The absence of patient-level identifiers, which were used in our original analysis to construct all train/validate splits.

Regarding (1), the differences are as follows:
* Any binary feature with fewer than 20 positive observations was dropped from the dataset.
* All colonization pressure features were rounded to the nearest 0.01
* Age was censored, so that any patient with an age > 89 has their age set to 90.
* Basic laboratory values (WBC, lymphocytes, and neutrophil counts) are excluded from the dataset release, due to inconsistencies in reporting of laboratory values. These features did not have a noticeable impact on our results.

Regarding (2), our analysis used patient identifiers to ensure that there were no patients with specimens in both the train and validate sets.  Because there are no patient identifiers included in this dataset release, the splits performed by our replication code are done without knowledge of patient identifiers.  As a result, they are necessarily different from the ones we used in our work.

For this reason, we provide utilities to run our scripts both in an end-to-end fashion (replicating the approach taken in the paper, but with different train/validate splits and therefore different selection of hyperparameters), as well as directly using the hyperparameters and thresholds chosen by our original analysis.

# Replication Instructions

## Setup 

First, you need to load in the data from Physionet.  See the [project website](http://www.clinicalml.org/data/amr-dataset) for more information on how to access this data. Place the Physionet data in a folder **outside** this repository to avoid accidentally uploading any files to Github.

Second, you need to **edit the paths in `setup/paths.sh`** 
* `DATA_PATH` in `setup/paths.sh` should reflect the absolute path to the files from Physionet. The files should be all be accessible at `${DATA_PATH}/<file_name>.csv`.
* `REPO_PATH` in `setup/paths.sh` should reflect the absolute path to this directory.  That is, this file should be located at `${REPO_PATH}/README.md`.

Third, you need to set up your `python` and bash environment.
* Run `bash setup/setup_env.sh` to create a python environment `amr` that will contain the necessary packages.
* Run `conda activate amr` to activate the environment
* Run `source setup/paths.sh` (note: use `source`, not `bash` here) to populate the necessary bash variables that define paths
* *Going forward (e.g., in subsequent terminal sessions), you will need to run `conda activate amr` and `source setup/paths.sh` before running the experiment scripts*

Finally, you need to run `python setup/load_data.py` to split the data release into train/test splits that the remaining code expects.  This will create additional `.csv` files in `${DATA_PATH}`

## Running the experiment scripts

We present two options for replication:
1. Using our original hyperparameters and thresholds: Run `bash run_all_rep.sh`
2. Running the analysis end-to-end: Run `bash run_all.sh`

Note that Option 1 is much faster for the thresholding and expected reward maximization, as it skips all hyperparameter tuning, while Option 2 will take approximately 2-3 hours longer to run, driven primarily by threshold selection.  The longest-running item in both of these scripts is the direct learning script.

After either of these scripts have been run, see the "Replicating Figures and Tables" section below for instructions on running the analysis notebook + plotting code.  If you have run with Option 1, then you will want to ensure that the `USE_REP_HP` flag in the notebook is set to `True`, and vice versa if you are using Option 2.

## (Optional) Manually Running the Scripts

Alternatively, you can manually run the relevant experiment scripts that are called by `run_all.sh` and `run_all_rep.sh` respectively.  These details are given below.

Before manually running the experiments in this section, run `cd ${REPO_PATH}/experiments` to go to the right directory.  All the below assumes you are in that directory, and there will be errors if you try to run from a different directory.

When you run these scripts for the first time, they will create (if it does not already exist) a folder `${REPO_PATH}/experiments/experiment_results/<exp_type>/<exp_name>/` that contains `results` to store the artifacts, and `logs` where you can watch progress by examining the log files.  In this context, `<exp_type>` might be `train_outcome_models` and `<exp_name>` might be `train_outcome_models_validation`.  

**NOTE**: The scripts (and paths in `setup/paths.sh`) assume the experiment names that are already given in these scripts, so **do not change them**

### Option 1: Running the analysis with the original hyperparameters

This will skip the validation scripts, because those are used to choose hyperparameters, which in this analysis is not necessary.

To run these analyses, you can either of the following equivalent options:
* Run `bash run_all_rep.sh` from the base directory `${REPO_PATH}`
* Move into the `${REPO_PATH}/experiments` directory and run the following scripts
```
bash scripts/eval_test_scripts/train_outcome_models_eval_test_rep.sh
bash scripts/eval_test_scripts/thresholding_test_rep.sh
bash scripts/eval_test_scripts/expected_reward_test_rep.sh
```
In sequence, these commands will (a) train models with the original hyperparameters on the entire training set, and then evaluate on the test set, (b) choose thresholds based on the observed validation performance on the original data, and evaluate on the test set, and (c) evaluate the expected reward maximization on the test set.

### Option 2: Running our analysis end-to-end

This will re-run the entire analysis, including the automatic selection of thresholds and hyperparameters using train/validate splits.  As noted above, this will result in slightly different choices than in our published analysis, in part because the splits will differ.

To run these analyses, you can either of the following equivalent options:
* Run `bash run_all.sh` from the base directory `${REPO_PATH}`
* Move into the `${REPO_PATH}/experiments` directory and run the following scripts (in order)

There are three types of models that we compare in the paper, and to generate results for each one, you need to run the following:
* Training outcome models (prerequisite for thresholding / expected reward experiments)
  + Run `bash scripts/validation_scripts/train_outcome_models_validation.sh` to train outcome models using the train/validation set, using the hyperparameter grid defined in `../models/hyperparameter_grids.py`. This also generates train/validation set predictions using the chosen hyperparameters.  **NOTE**: In our experiments we observed that logistic regression and random forests performed comparably, and by default the script will only investigate hyperparameters for logistic regression models.  To change this, you will need to change the code in `experiment_train_outcome_model.py:113, 123` where we have commented out random forests.
  + Run `bash scripts/eval_test_scripts/train_outcome_models_eval_test.sh` to use the best hyperparameters from the previous step to train a model on all of the train/validate data, and evaluate on the test set.

* Thresholding Experiments: Run
```
bash scripts/validation_scripts/thresholding_validation.sh
bash scripts/eval_test_scripts/thresholding_test.sh
```
  to choose thresholds based on the validation set, using the validation-set predictions generated by `train_outcome_models_validation.sh`, and then to generate results on the test set.  Note that the search over thresholds takes a significant amount of time to run, which occurs in `thresholding_validation.sh`. 

* Expected Reward Experiments: Run 
```
bash scripts/eval_test_scripts/expected_reward_test.sh
```
  to generate results on the test set, using the test set predictions generated by `train_outcome_models_test.sh`.  Note that there is no hyperparameter tuning (other than what was already done for the outcome models).  If you wish to play around with new models on the validation set, you can run the corresponding validation script, but it is not necessary for reproducing our results.

* Direct Learning Experiments: Note that these experiments take significantly longer to run, as we build new models for every value of `omega` that is tested. Run
```
bash scripts/eval_test_scripts/direct_learning_test.sh
```
to generate results on the test set.  Similar to the expected reward experiments, there is no need to run the corresponding validation script to replicate our results (note that this script internally does training on the training/validation set, with no hyperparameter tuning, and then evaluates on the test set), but it is available if you wish to play around with the train/validate data.

In addition, to generate the data for Figure 2 in the main paper, you will need to run
```
bash scripts/eval_test_scripts/direct_learning_defer_interventions_test.sh
```
**Note**: This experiment takes a very long time (24hr+) to run, because we run a large number of trials for each setting.  As such, it is not included in the main replication script.

## Evaluating Baselines

The script `scripts/eval_test_scripts/test_yelin_baseline.sh` will run the baselines we discuss in the paper, and will print their values to the screen.  These are then hard-coded into the notebooks (see below) for producing the figures.  Note that this requires running `train_outcome_models_eval_test.sh` as a prerequisite.

## Replicating tables and figures

Within this repository, `notebooks/` contains a jupyter notebook that can be used to replicate figures from the paper and examine results.

To replicate figures, you will need to do the following:
* First, navigate to the folder `notebooks/` and open the Jupyter notebook `figures_and_tables.ipynb`;  This will replicate selected figures and tables.
* Set the flag `USE_REP_HP` in this notebook based on whether or not you wish to compute results using the original hyperparameters, or the results of the end-to-end analysis applied to the dataset release.  Note that either of these options assumes you have already run the relevant code above.
* For the synthetic experiment, run `experiment_synthetic_notebook.ipynb`

