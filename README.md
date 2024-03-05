# Kaggle-ASCII-Sign-Language-Classification


## Requirements
First install the requirements by running the following command:
```
pip install -r requirements.txt
```

## Data folder
Please put all the following files within a folder, **All at the zero depth**. For instance:
```
...
ascii-sign-language/
    Letters.xlsx
    sign_mnist_train.csv
    test.csv
```
Please rename the ```sign_mnist_test.csv``` into ```test.csv``` file.

## How to run
### Convolutional Neural Network

To run all the experiments and configurations performed for the convolutional neural network, run the code in the "Convolutional neural network.ipynb" jupyter notebook. The code here depends on the main libraries of keras and tensor flow and uses the "make_predictions_test" function from the python code "Predictions_test_set.py".

### Random forest
To run the random forest, use the following command
```
python train_random_forest.py [data folder]
```

### AdaBoost
To run the AdaBoost, use the following command:
```
python train_adaboost.py [data folder]
```
