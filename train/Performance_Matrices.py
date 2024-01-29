import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def confussion_maxtrix(y_ture, y_pred, class_name):
    y_ture= np.array(y_ture); y_pred= np.array(y_pred); class_name= np.array(class_name)
    ture_numbers_of_each_class= np.array(pd.DataFrame(y_ture).value_counts().sort_index())
    ture_numbers_of_each_class = np.append(ture_numbers_of_each_class, 0)
    classification_correction_table= pd.DataFrame(np.zeros(shape= (len(class_name), len(class_name))), columns= class_name, index= class_name)
    # classification_correction_table.loc["Nonetype"] = 0  # 添加一行，所有元素初始化为0
    classification_correction_table["Nonetype"] = 0  # 添加一列，所有元素初始化为0
    for i in range(0, len(y_ture)):
        if y_ture[i] == y_pred[i]:
            classification_correction_table.loc[str(y_ture[i]), str(y_ture[i])]= classification_correction_table.loc[str(y_ture[i]), str(y_ture[i])]+ 1
        elif y_pred[i] not in class_name:
            # print('find nonetype')
            classification_correction_table.loc[str(y_ture[i]), 'Nonetype'] += 1 
        else:
            classification_correction_table.loc[str(y_ture[i]), str(y_pred[i])]= classification_correction_table.loc[str(y_ture[i]), str(y_pred[i])] + 1 ##左側為target 上面為pred 此為FN
    # for i in range(0, 2):
    #     classification_correction_table.iloc[i, :]= round(classification_correction_table.iloc[i, :]/ ture_numbers_of_each_class[i], 4)
    confussion_maxtrix_result = classification_correction_table
    return confussion_maxtrix_result   
    
def measurements(confussion_maxtrix, class_name):
    class_name_list= class_name
    precision_list= []; recall_list= []; f1_list= []; accuracy_list= []
    
    for i in range(0, len(class_name)):
        
        accuracy= round(confussion_maxtrix[class_name[i]][i]/np.sum(confussion_maxtrix.iloc[i, :]), 4)
        # accuracy= round(confussion_maxtrix[class_name[i]][i], 4)
        if  np.sum(confussion_maxtrix[class_name[i]]) == 0:
            precision = 0.0000
        else:
            precision= round(confussion_maxtrix[class_name[i]][i]/ np.sum(confussion_maxtrix[class_name[i]]), 4)
        recall= round(confussion_maxtrix.iloc[i, :][i]/ np.sum(confussion_maxtrix.iloc[i, :]), 4)
        if (recall+ precision) == 0 :
            f1 = 0.0000
        else:
            f1= round((2* recall* precision)/ (recall+ precision), 4)
        precision_list.append(precision); recall_list.append(recall); f1_list.append(f1)
        accuracy_list.append(accuracy)
   
   #append average
    precision_list.append(round(np.mean(precision_list[:]), 4))
    recall_list.append(round(np.mean(recall_list[:]), 4))
    f1_list.append(round(np.mean(f1_list[:]), 4))
    accuracy_list.append(round(np.mean(accuracy_list[:]), 4))
        
    precision_list= np.array(precision_list); recall_list= np.array(recall_list); f1_list= np.array(f1_list)
    accuracy_list= np.array(accuracy_list)
    
    class_name_list.append("Average")
    measurement= pd.DataFrame(index= class_name_list)
    
    measurement["accuracy"]= accuracy_list
    measurement["precision"]= precision_list
    measurement["recall"]= recall_list
    measurement["F1 value"]= f1_list
    
    measurement_table= measurement
    return measurement_table