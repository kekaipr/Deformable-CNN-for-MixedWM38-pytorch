import numpy as np 
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import copy
import pandas as pd

source_data = np.load('C:/Users/MA201-Ultima/Desktop/thesis/datasets/Mixed_WM38.npz')
#for mix wm38單類別#
defect_dict= {"Normal": [0, 0, 0, 0, 0, 0, 0, 0], 
            "Center": [1, 0, 0, 0, 0, 0, 0, 0],
            "Donut":  [0, 1, 0, 0, 0, 0, 0, 0],
            "Edge-Loc": [0, 0, 1, 0, 0, 0, 0, 0],
            "Edge-Ring": [0, 0, 0, 1, 0, 0, 0, 0],
            "Loc":[0, 0, 0, 0, 1, 0, 0, 0],
            "Near-full":[0, 0, 0, 0, 0, 1, 0, 0],
            "Scratch":[0, 0, 0, 0, 0, 0, 1, 0],
            "Random":[0, 0, 0, 0, 0, 0, 0, 1]}

defect_dict_inverse= {}
for i in range(0, len(defect_dict)):
    key_name= list(defect_dict.keys())[i]
    defect_dict_inverse[str(np.array(defect_dict[key_name]))]= key_name

y_class= []
y_encode_class= []
Source_index = []
for i in range(0, len(source_data["arr_1"])):
    if  str(source_data["arr_1"][i]) in defect_dict_inverse.keys():
        Source_index.append(i)
        y_encode_class.append(str(source_data["arr_1"][i]))
y_class = pd.DataFrame(y_encode_class).replace(defect_dict_inverse)

source_image = source_data['arr_0'][Source_index]

a = np.zeros(shape =(8015,26,26))
source_image = source_image.astype("uint8")
a = a.astype("uint8")
for i,j in enumerate(source_image):
    a[i] = cv2.resize(source_image[i],(26,26))

data_dict = {
    'arr_0': np.array(a),
    'arr_1': np.array(y_class[0])
}
# np.savez('source_domain.npz', **data_dict, allow_pickle=True)

del data_dict


##### 811K #####
WM811Kdataset = pd.read_pickle('C:/Users/MA201-Ultima/Desktop/thesis/datasets/WM811K.pkl')
ImageData, ImgeLabel= WM811Kdataset["waferMap"], WM811Kdataset["failureType"]        
ImgeLabelNew= []; label_count= {}
for index in range(0, len(ImgeLabel)):
    label= ImgeLabel.iloc[index]
    if len(label)== 0:
        ImgeLabelNew.append("Non-label")
        if "Non-label" not in list(label_count.keys()):
            label_count["Non-label"]= 1
        else:
            label_count["Non-label"]+= 1
    else:
        label= label[0][0]
        if label== "none":
            label= "Normal"
        ImgeLabelNew.append(label)
        if  label not in list(label_count.keys()):
            label_count[ label]= 1
        else:
            label_count[ label]+= 1
ImgeLabel= pd.DataFrame({"failureType": ImgeLabelNew})
target_index = []
test_index = []
a = copy.copy(ImageData)
b = copy.copy(ImageData)
for i,j in enumerate(ImgeLabel['failureType']):
    if (ImageData[i].shape[0]==26) and (ImageData[i].shape[1] ==26):
        a[i] = ImageData[i]
        if j == "Non-label":
            target_index.append(i)
    else:
        a[i] = ImageData[i]

for i,j in enumerate(ImgeLabel['failureType']):
    if (abs(ImageData[i].shape[0]-26)<5) and (abs(ImageData[i].shape[1]-26)<5):
        b[i] = ImageData[i]
        if j != "Non-label":
            test_index.append(i)
    else:
        b[i] = ImageData[i]

target_image = a[target_index]
target_label = ImgeLabel['failureType'][target_index]

test_image = b[test_index]
test_label = ImgeLabel['failureType'][test_index]

final_test_label_index = []
for i,j in enumerate(test_label.unique()):
    label_data = test_label[test_label==j]
    label_data = label_data[:200].index.values.tolist()
    final_test_label_index += label_data

test_image = b[final_test_label_index]
test_label = ImgeLabel['failureType'][final_test_label_index]

data_dict = {
    'arr_0': np.array(target_image),
    'arr_1': np.array(target_label)
}
# np.savez('target_domain_26.npz', **data_dict, allow_pickle=True)

del data_dict
data_dict = {
    'arr_0': np.array(test_image),
    'arr_1': np.array(test_label)
}
# np.savez('test_data_26_5.npz', **data_dict, allow_pickle=True)