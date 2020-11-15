# Toxicity Prediction with python
Data cannote be represented, since not published yet. Howerver a preprocess module is provided

util.args.py contains the models' hyperparameter candidates

# How to run (if data exists)
0. `python resplit.py` for curation and split the data into train and test
1. `python data_loader.py --mode train` ; `python data_loader.py --mode test` for preprocess
2. `python model.py` for train
