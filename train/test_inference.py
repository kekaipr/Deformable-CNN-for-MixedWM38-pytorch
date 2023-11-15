import train_multi_label_torch as train
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from train_multi_label_torch import Train_mix
import Performance_Matrices as PM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.model_selection import train_test_split
# defect_dict= {"normal": [0, 0, 0, 0, 0, 0, 0, 0], 
#               "center": [1, 0, 0, 0, 0, 0, 0, 0],
#               "dount":  [0, 1, 0, 0, 0, 0, 0, 0],
#               "edge_loc": [0, 0, 1, 0, 0, 0, 0, 0],
#               "edge_ring": [0, 0, 0, 1, 0, 0, 0, 0],
#               "loc":[0, 0, 0, 0, 1, 0, 0, 0],
#               "near_full":[0, 0, 0, 0, 0, 1, 0, 0],
#               "scratch":[0, 0, 0, 0, 0, 0, 1, 0],
#               "random":[0, 0, 0, 0, 0, 0, 0, 1],
#               "C+EL":[1, 0, 1, 0, 0, 0, 0, 0],
#               "C+ER":[1, 0, 0, 1, 0, 0, 0, 0],
#               "C+L":[1, 0, 0, 0, 1, 0, 0, 0],
#               "C+S":[1, 0, 0, 0, 0, 0, 1, 0],
#               "D+EL":[0, 1, 1, 0, 0, 0, 0, 0],
#               "D+ER":[0, 1, 0, 1, 0, 0, 0, 0],
#               "D+L":[0, 1, 0, 0, 1, 0, 0, 0],
#               "D+S":[0, 1, 0, 0, 0, 0, 1, 0],
#               "EL+L":[0, 0, 1, 0, 1, 0, 0, 0],
#               "EL+S":[0, 0, 1, 0, 0, 0, 1, 0],
#               "ER+L":[0, 0, 0, 1, 1, 0, 0, 0],
#               "ER+S":[0, 0, 0, 1, 0, 0, 1, 0],
#               "L+S":[0, 0, 0, 0, 1, 0, 1, 0],
#               "C+EL+L":[1, 0, 1, 0, 1, 0, 0, 0],
#               "C+EL+S":[1, 0, 1, 0, 0, 0, 1, 0],
#               "C+ER+L":[1, 0, 0, 1, 1, 0, 0, 0],
#               "C+ER+S":[1, 0, 0, 1, 0, 0, 1, 0],
#               "C+L+S":[1, 0, 0, 0, 1, 0, 1, 0],
#               "D+EL+L":[0, 1, 1, 0, 1, 0, 0, 0],
#               "D+EL+S":[0, 1, 1, 0, 0, 0, 1, 0],
#               "D+ER+L":[0, 1, 0, 1, 1, 0, 0, 0],
#               "D+ER+S":[0, 1, 0, 1, 0, 0, 1, 0],
#               "D+L+S":[0, 1, 0, 0, 1, 0, 1, 0],
#               "EL+L+S":[0, 0, 1, 0, 1, 0, 1, 0],
#               "ER+L+S":[0, 0, 0, 1, 1, 0, 1, 0],
#               "C+L+EL+S":[1, 0, 1, 0, 1, 0, 1, 0],
#               "C+L+ER+S":[1, 0, 0, 1, 1, 0, 1, 0],
#               "D+L+EL+S":[0, 1, 1, 0, 1, 0, 1, 0],
#               "D+L+ER+S":[0, 1, 0, 1, 1, 0, 1, 0],
#               "Nonetype":0}
# defect_dict = {"C+EL":[1, 0, 1, 0, 0, 0, 0, 0],
#                "C+EL+L":[1, 0, 1, 0, 1, 0, 0, 0],
#                "C+EL+S":[1, 0, 1, 0, 0, 0, 1, 0],
#                "C+ER":[1, 0, 0, 1, 0, 0, 0, 0],
#                "C+ER+L":[1, 0, 0, 1, 1, 0, 0, 0],
#                "C+ER+S":[1, 0, 0, 1, 0, 0, 1, 0],
#                "C+L":[1, 0, 0, 0, 1, 0, 0, 0],
#                "C+L+EL+S":[1, 0, 1, 0, 1, 0, 1, 0],
#                "C+L+ER+S":[1, 0, 0, 1, 1, 0, 1, 0],
#                "C+L+S":[1, 0, 0, 0, 1, 0, 1, 0],
#                "C+S":[1, 0, 0, 0, 0, 0, 1, 0],
#                "D+EL":[0, 1, 1, 0, 0, 0, 0, 0],
#                "D+EL+L":[0, 1, 1, 0, 1, 0, 0, 0],
#                "D+EL+S":[0, 1, 1, 0, 0, 0, 1, 0],
#                "D+ER":[0, 1, 0, 1, 0, 0, 0, 0],
#                "D+ER+L":[0, 1, 0, 1, 1, 0, 0, 0],
#                "D+ER+S":[0, 1, 0, 1, 0, 0, 1, 0],
#                "D+L":[0, 1, 0, 0, 1, 0, 0, 0],
#                "D+L+EL+S":[0, 1, 1, 0, 1, 0, 1, 0],
#                "D+L+ER+S":[0, 1, 0, 1, 1, 0, 1, 0],
#                "D+L+S":[0, 1, 0, 0, 1, 0, 1, 0],
#                "D+S":[0, 1, 0, 0, 0, 0, 1, 0],
#                "EL+L":[0, 0, 1, 0, 1, 0, 0, 0],
#                "EL+L+S":[0, 0, 1, 0, 1, 0, 1, 0],
#                "EL+S":[0, 0, 1, 0, 0, 0, 1, 0],
#                "ER+L":[0, 0, 0, 1, 1, 0, 0, 0],
#                "ER+L+S":[0, 0, 0, 1, 1, 0, 1, 0],
#                "ER+S":[0, 0, 0, 1, 0, 0, 1, 0],
#                "L+S":[0, 0, 0, 0, 1, 0, 1, 0],
#                "center": [1, 0, 0, 0, 0, 0, 0, 0],
#                "dount":  [0, 1, 0, 0, 0, 0, 0, 0],
#                "edge_loc": [0, 0, 1, 0, 0, 0, 0, 0],
#                "edge_ring": [0, 0, 0, 1, 0, 0, 0, 0],
#                "loc":[0, 0, 0, 0, 1, 0, 0, 0],
#                "near_full":[0, 0, 0, 0, 0, 1, 0, 0],
#                "normal": [0, 0, 0, 0, 0, 0, 0, 0], 
#                "random":[0, 0, 0, 0, 0, 0, 0, 1],
#                "scratch":[0, 0, 0, 0, 0, 0, 1, 0], 
#                }
defect_dict = {"Center": [1, 0, 0, 0, 0, 0, 0, 0],
               "Donut":  [0, 1, 0, 0, 0, 0, 0, 0],
               "Edge-Loc": [0, 0, 1, 0, 0, 0, 0, 0],
               "Edge-Ring": [0, 0, 0, 1, 0, 0, 0, 0],
               "Loc":[0, 0, 0, 0, 1, 0, 0, 0],
               "Near-full":[0, 0, 0, 0, 0, 1, 0, 0],
               "none": [0, 0, 0, 0, 0, 0, 0, 0],
               "Random":[0, 0, 0, 0, 0, 0, 0, 1],
               "Scratch":[0, 0, 0, 0, 0, 0, 1, 0]}

defect_dict_inverse = {}
for i in range(0, len(defect_dict)):
    key_name= list(defect_dict.keys())[i]
    defect_dict_inverse[str(np.array(defect_dict[key_name]))]= key_name

# b = np.load('C:/Users/MA201-Ultima/Desktop/thesis/Wafer_Map_Datasets.npz')
b = np.load('C:/Users/MA201-Ultima/Desktop/thesis/datasets/811k_clean.npz')

model = train.Mymodel(inputs_shape=(1,52,52))
# model = train.Xception(num_classes=8)
# model.load_state_dict(torch.load('C:/Users/MA201-Ultima/Desktop/X_D_1_b2_1e3_epoch_100_epoch300/X_D_1_b4_1e3_epoch_100_epoch300.pt'))
model.load_state_dict(torch.load('C:/Users/MA201-Ultima/Desktop/De_Adam/wm38_D_262.pt'))

model.to(device)
model.eval()


testx = b['arr_0']
textx = np.expand_dims(testx, axis=-1)
# testx = testx.reshape(38015, 1, 52, 52)
# testx = testx.reshape(615521, 1, 52, 52)
# testx = testx.reshape(182786,1,52,52)
testx = testx.reshape(164777,1,52,52)
testy = b['arr_1']

# x_train, x_test, y_train, y_test = train_test_split(testx, testy, test_size=0.2, random_state=10)


test_dataset = TensorDataset(torch.tensor(testx, dtype=torch.float32),
                             torch.tensor(testy, dtype=torch.float32))
# test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                            #  torch.tensor(y_test, dtype=torch.float32))
test_loader = DataLoader(test_dataset)


target = []
pred = []

class_name = [defect_dict_inverse[str(sublist).replace(",","")] for sublist in list(defect_dict.values())]

with torch.no_grad():
    for i, j in test_loader:
        # print(j)
        i,j = i.to(device),j.to(device)
        output = model(i)
        # print(output)
        result = torch.round(output)
        
        j_list ,result_list =j[0].int().tolist(), result[0].int().tolist()
        j_string ,result_string =str(j_list).replace(",","") ,str(result_list).replace(",","")
        
        if result_string in defect_dict_inverse.keys():
            pred.append(defect_dict_inverse[result_string])
        else : 
            pred.append('Nonetype')
            
        target.append(defect_dict_inverse[j_string])

        

    # print(len(class_name))
    result2 = PM.confussion_maxtrix(target,pred,class_name)
    print(result2)
    result3 = PM.measurements(result2,class_name)
    print(result3)


