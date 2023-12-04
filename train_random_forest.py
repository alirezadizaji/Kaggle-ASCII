import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Predictions_test_set import make_predictions_test

from stdout_stderr_setter import stdout_stderr_setter
from random_forest.random_forest import RandomForestClassifier

@stdout_stderr_setter("./Consoles_random_forest")
def main():
     #Import training and test sets
    train=pd.read_csv(os.path.join(sys.argv[1], "sign_mnist_train.csv"))
    test = pd.read_csv(os.path.join(sys.argv[1], "test.csv"))

    #Create dictionary to map the predicted numbers (0-25) (and no cases for 9=J or 25=Z because of gesture motions).
    map_letters=pd.read_excel(os.path.join(sys.argv[1], "Letters.xlsx"))
    map_letters["label"] = np.where(map_letters["label"] >= 9, map_letters["label"] - 1, map_letters["label"])
    dictionary_letters = dict(map_letters.values)
    
    num_cls = train["label"].unique().size
    x_train, x_val,y_train, y_val = train_test_split(train.iloc[:,1:] , train["label"],
                                                  stratify=train["label"], random_state=0)

    # Reshape the data to be suitable for CNN
    x_train=np.array(x_train).astype(np.float32)
    x_val=np.array(x_val).astype(np.float32)

    #LETS ADJUST THE NUMBER OF CLASSES SINCE 9 IS NOT INCLUDED
    y_train=np.array(y_train)
    y_train = np.where(y_train >= 9, y_train - 1, y_train)
    y_val=np.array(y_val)
    y_val = np.where(y_val >= 9, y_val - 1, y_val)

    # Normalize pixel values to be between 0 and 1
    x_train /= 255.0
    x_val /= 255.0

    hyperparameters = [
    #    {"n_estimators": 2, "p_bootstraping": 0.8, "max_depth": 5, "p_featuring": 1.0, 'num_cls': num_cls},
    #    {"n_estimators": 2, "p_bootstraping": 0.8, "max_depth": 10, "p_featuring": 1.0, 'num_cls': num_cls},
    #    {"n_estimators": 2, "p_bootstraping": 0.8, "max_depth": 15, "p_featuring": 1.0, 'num_cls': num_cls},
    #    {"n_estimators": 3, "p_bootstraping": 0.8, "max_depth": 10, "p_featuring": 1.0, 'num_cls': num_cls},
      # {"n_estimators": 5, "p_bootstraping": 0.8, "max_depth": 20, "p_featuring": 1.0, 'num_cls': num_cls},
      # {"n_estimators": 10, "p_bootstraping": 0.8, "max_depth": 20, "p_featuring": 1.0, 'num_cls': num_cls},
     #  {"n_estimators": 15, "p_bootstraping": 0.8, "max_depth": 20, "p_featuring": 1.0, 'num_cls': num_cls},
      # {"n_estimators": 15, "p_bootstraping": 0.8, "max_depth": 25, "p_featuring": 1.0, 'num_cls': num_cls},

       {"n_estimators": 50, "p_bootstraping": 0.3, "max_depth": 15, "p_featuring": 0.6, 'num_cls': num_cls},
       {"n_estimators": 50, "p_bootstraping": 0.3, "max_depth": 15, "p_featuring": 0.6, 'num_cls': num_cls},
       
       {"n_estimators": 100, "p_bootstraping": 0.2, "max_depth": 12, "p_featuring": 0.3, 'num_cls': num_cls},

       {"n_estimators": 400, "p_bootstraping": 0.2, "max_depth": 15, "p_featuring": 0.2, 'num_cls': num_cls}
    ]

    best_h: dict = None
    best_val_acc: float = -np.inf

    for h in hyperparameters:
        rf = RandomForestClassifier(**h)
        rf.fit(x_train, y_train)
        pred_val = rf.predict(x_val).argmax(1)
        val_acc = (pred_val == y_val).sum() / y_val.size
        print(f"@@ Random Forest Classifer: val acc {val_acc}, params {h}", flush=True)
        if val_acc > best_val_acc:
            print(f"@@@ BEST VAL ACCURACY HAS JUST CHANGED @@@", flush=True)
            best_h = h
            best_val_acc = val_acc
    
    model = RandomForestClassifier(**best_h)
    X = np.vstack((x_train, x_val))
    y = np.concatenate((y_train, y_val))
    model.fit(X, y)
    final_pred=make_predictions_test(test,model,dictionary_letters, to_img_form=False)
    final_pred.to_csv("rfc_first_trial.csv",index=False)


if __name__ == "__main__":
   main()