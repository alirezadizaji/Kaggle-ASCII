import pandas as pd
import numpy as np

def make_predictions_test(df, apply_model, dictionary_letters, to_img_form: bool = True):
    
    first = np.array(df.iloc[:,1:784+1])
    second = np.array(df.iloc[:,784+1:])
    
    if to_img_form:
        first = first.reshape(first.shape[0], 28, 28, 1)
        second = second.reshape(second.shape[0], 28, 28, 1)
    
    first = first.astype(np.float32)
    second = second.astype(np.float32)
    
    first /= 255
    second /= 255
    
    #Predictions for image A
    y_pred_1 = apply_model.predict(first)
    if y_pred_1.ndim == 2:
        y_pred_1 = [np.argmax(i) for i in y_pred_1]
    y_pred_1 = pd.Series(y_pred_1).map(dictionary_letters)
    y_pred_1_ascii = np.array([ord(i) for i in y_pred_1])
    
    #Predictions for image B
    y_pred_2 = apply_model.predict(second)
    if y_pred_2.ndim == 2:
        y_pred_2 = [np.argmax(i) for i in y_pred_2]
    y_pred_2 = pd.Series(y_pred_2).map(dictionary_letters)
    y_pred_2_ascii = np.array([ord(i) for i in y_pred_2])
    
    #Sum of both ASCII
    summa = y_pred_1_ascii + y_pred_2_ascii

    for i in range(len(summa)):
        while summa[i] >= 122:
            summa[i] -= 65 
        else:
            summa[i] = summa[i]

    y_pred = np.array([chr(i) for i in summa])
    
    
    df=df[["id"]]
    df.loc[:,"label"] = y_pred

    
    return df

