import numpy as np
from model_0108 import Mymodel
import torch, os
import torch.nn as nn
import torch.utils.data as data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

def MMD(x, y, kernel, device):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz
    XX, YY, XY = (torch.zeros(xx.shape).to(device), torch.zeros(xx.shape).to(device), torch.zeros(xx.shape).to(device))
    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
    return torch.mean(XX + YY - 2. * XY)

if __name__== "__main__":
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_training_data= np.load("source_domain_26.npz", allow_pickle= True)
    targets_training_data= np.load("target_domain_26.npz", allow_pickle= True)

    targets_test_data= np.load("test_data_26_5.npz", allow_pickle= True)

    source_training_imag= source_training_data["arr_0"]
    source_training_imag= source_training_imag.reshape(len(source_training_imag), 1, 26, 26)
    source_training_label= source_training_data["arr_1"]

    target_training_imag_= targets_training_data["arr_0"]
    target_training_imag= np.zeros(shape= (len(target_training_imag_), 1, 26, 26))
    for index in range(0, len(target_training_imag)):
        target_training_imag[index]= target_training_imag_[index].reshape(1, 1, 26, 26)
    del target_training_imag_

    target_test_imag_= targets_test_data["arr_0"]
    target_test_imag= np.zeros(shape= (len(target_test_imag_), 1, 26, 26))
    for index in range(0, len(target_test_imag)):
        target_test_imag[index]= target_test_imag_[index].reshape(1, 1, 26, 26)
    del target_test_imag_

    target_test_label= targets_test_data["arr_1"]

    label_encoding= LabelEncoder()
    source_training_label= label_encoding.fit_transform(source_training_label)
    target_test_label= label_encoding.transform(target_test_label)

    target_training_imag= target_training_imag[: len(source_training_imag), :, :]

    source_x_train, source_y_train=  torch.FloatTensor(source_training_imag), torch.LongTensor(source_training_label)
    target_training_imag, target_test_imag, target_test_label= torch.FloatTensor(target_training_imag), torch.FloatTensor(target_test_imag), torch.LongTensor(target_test_label)

    model= Mymodel(in_channels= source_x_train.shape[1] , out_class_dim= len(label_encoding.classes_)).to(device= device)

    batch_size= 5000
    epoch= 1000
    learning_rate= 0.000005
    optimizer= torch.optim.Adam(model.parameters(), lr= learning_rate)
    loss= nn.CrossEntropyLoss()

    source_train_dataloader= data.TensorDataset(source_x_train, source_y_train, target_training_imag)
    source_train_dataloader= data.DataLoader(source_train_dataloader, batch_size= batch_size, shuffle= True)

    target_test_dataloader= data.TensorDataset(target_test_imag, target_test_label)
    target_test_dataloader= data.DataLoader(target_test_dataloader, batch_size= batch_size, shuffle= True)

    early_stop= {"max_iter": 100, "epoch_rec": 0.0, "domain_loss": 100000.0, "best_epoch": 0.0,"total_loss": 100000.0 }
    epcoh_cls_loss= []; epoch_cls_acc= []; epoch_dom_loss= []
    epcoh_cls_target_loss= []; epoch_cls_target_acc= []

    
    trainLossLs, trainAccLs, valLossLs, valAccLs, domainLossLs = [], [], [], [], []

    for e in range(1, epoch+ 1):
        model.train()
        batch_cls_loss= 0.0; batch_accuracy= 0.0; batch_domain_loss= 0.0
        for source_x, source_y, target_x in source_train_dataloader:
            source_x, source_y, target_x= source_x.to(device), source_y.to(device), target_x.to(device)
            optimizer.zero_grad()                                                                            # 
            source_outputs, source_latent_space_1, source_latent_space_2, source_latent_space_3, source_latent_space_4, source_latent_space_5 , source_latent_space_6, source_latent_space_7, source_latent_space_8= model(source_x)
            _, target_latent_space_1, target_latent_space_2, target_latent_space_3, target_latent_space_4, target_latent_space_5, target_latent_space_6, target_latent_space_7, target_latent_space_8= model(target_x)

            source_latent_space_6= source_latent_space_6.reshape(source_latent_space_6.shape[0],source_latent_space_6.shape[1])
            target_latent_space_6= target_latent_space_6.reshape(target_latent_space_6.shape[0],target_latent_space_6.shape[1])

            mmd_loss_6= MMD(x= source_latent_space_6, y= target_latent_space_6, kernel= "rbf", device= device)
            mmd_loss_7= MMD(x= source_latent_space_7, y= target_latent_space_7, kernel= "rbf", device= device)
            mmd_loss_8= MMD(x= source_latent_space_8, y= target_latent_space_8, kernel= "rbf", device= device)
            
            mmd_loss=  mmd_loss_6+ mmd_loss_7+ mmd_loss_8
            mmd_loss= torch.div(mmd_loss, 3)
            class_loss= loss(source_outputs, source_y)
            
            total_loss= class_loss*0.1+ mmd_loss
            total_loss.backward()
            optimizer.step()

            batch_cls_loss+= class_loss.item()
            batch_accuracy+= accuracy_score(source_y.data.cpu().numpy(), np.argmax(source_outputs.data.cpu().numpy(), axis= 1))
            batch_domain_loss+= mmd_loss.item()
        
        batch_cls_loss= batch_cls_loss/ len(source_train_dataloader)
        batch_accuracy= batch_accuracy / len(source_train_dataloader)
        batch_domain_loss= batch_domain_loss/ len(source_train_dataloader)
        epcoh_cls_loss.append(batch_cls_loss); epoch_cls_acc.append(batch_accuracy); epoch_dom_loss.append(batch_domain_loss)

        if e% 1== 0:
            model.eval()
            with torch.no_grad():
                batch_cls_val_loss= 0.0; batch_val_accuracy= 0.0
                for target_val_x, target_val_y in target_test_dataloader:
                    target_val_x, target_val_y= target_val_x.to(device), target_val_y.to(device)
                    target_outputs, _, _, _, _, _, _, _, _= model(target_val_x)
                    class_loss= loss(target_outputs, target_val_y)
                    batch_cls_val_loss+= class_loss.item()
                    batch_val_accuracy+= accuracy_score(target_val_y.data.cpu().numpy(), np.argmax(target_outputs.data.cpu().numpy(), axis= 1))
                batch_cls_val_loss= batch_cls_val_loss/ len(target_test_dataloader)
                batch_val_accuracy= batch_val_accuracy / len(target_test_dataloader)
                epcoh_cls_target_loss.append(batch_cls_val_loss); epoch_cls_target_acc.append(batch_val_accuracy)
            if e>= 10 and early_stop["total_loss"]<= batch_domain_loss+ batch_cls_loss:
                early_stop["epoch_rec"]+= 1.0
                if early_stop["epoch_rec"]== early_stop["max_iter"]:
                    break
            else:
                early_stop["epoch_rec"]= 0.0
                early_stop["total_loss"]= batch_domain_loss+ batch_cls_loss
                early_stop["best_epoch"]= e
                torch.save(model, "best.pth")
                print("Model updating : ", os.path.join(os.getcwd(), "best.pth"))
            print("Epoch: %4d|| train loss: %2.6f|| train accuracy: %2.6f|| domain loss: %2.6f|| val loss: %2.6f|| val accuracy: %2.6f " %(e, batch_cls_loss, batch_accuracy, batch_domain_loss, batch_cls_val_loss, batch_val_accuracy))
            print("-----------------------------------------------------------------------------------------------------------------------------------------------------------")


            trainLossLs.append(batch_cls_loss)
            trainAccLs.append(batch_accuracy)
            valLossLs.append(batch_cls_val_loss)
            valAccLs.append(batch_val_accuracy)
            domainLossLs.append(batch_domain_loss)
    plt.figure(num= 1, figsize= (8, 4))
    ### training classification accuracy
    plt.plot(np.arange(0, len(epoch_cls_acc)), epoch_cls_acc, ls= "--", color= "darkred", label= "train - cls acc")
    plt.scatter(np.arange(0, len(epoch_cls_acc)), epoch_cls_acc, marker= "o", edgecolors= "black", color= "darkred")
    ### validation classification accuracy
    plt.plot(np.arange(0, len(epoch_cls_target_acc)), epoch_cls_target_acc, ls= "--", color= "green", label= "val - cls acc")
    plt.scatter(np.arange(0, len(epoch_cls_target_acc)), epoch_cls_target_acc, marker= "v", edgecolors= "black", color= "green")
    plt.grid(True)
    plt.legend(loc= "best", fontsize= 15)
    plt.xticks(fontsize= 18)
    plt.yticks(fontsize= 18)
    plt.show()

    plt.figure(num= 2, figsize= (8, 4))
    ### training classification loss
    plt.plot(np.arange(0, len(epcoh_cls_loss)), epcoh_cls_loss, ls= "--", color= "darkred", label= "train - cls loss")
    plt.scatter(np.arange(0, len(epcoh_cls_loss)), epcoh_cls_loss, marker= "o", edgecolors= "black", color= "darkred")
    ### validation classification loss
    plt.plot(np.arange(0, len(epcoh_cls_target_loss)), epcoh_cls_target_loss, ls= "--", color= "green", label= "val - cls acc")
    plt.scatter(np.arange(0, len(epcoh_cls_target_loss)), epcoh_cls_target_loss, marker= "v", edgecolors= "black", color= "green")
    plt.grid(True)
    plt.legend(loc= "best", fontsize= 15)
    plt.xticks(fontsize= 18)
    plt.yticks(fontsize= 18)
    plt.show()

    plt.figure(num= 3, figsize= (8, 4))
    ### domain loss
    plt.plot(np.arange(0, len(epoch_dom_loss)), epoch_dom_loss, ls= "--", color= "yellow", label= "train - domain loss")
    plt.scatter(np.arange(0, len(epoch_dom_loss)), epoch_dom_loss, marker= "o", edgecolors= "black", color= "yellow")
    plt.grid(True)
    plt.legend(loc= "best", fontsize= 15)
    plt.xticks(fontsize= 18)
    plt.yticks(fontsize= 18)
    plt.show()

    model= torch.load("best.pth")
    model.eval(); pred_label= []
    with torch.no_grad():
        for target_val_x, target_val_y in target_test_dataloader:
            target_val_x, target_val_y= target_val_x.to(device), target_val_y.to(device)
            target_outputs, _, _, _, _, _, _, _, _= model(target_val_x)
            target_outputs= np.argmax(target_outputs.data.cpu().numpy(), axis= 1)
            pred_label+= list(target_outputs)
    pred_label= np.array(pred_label); target_test_label= target_test_label.data.cpu().numpy()
    cm= confusion_matrix(y_pred= pred_label, y_true= target_test_label, normalize= "true")#
    plt.figure(figsize=(10, 6))
    labels=np.arange(len(label_encoding.classes_))
    plt.ylabel("Actual label", fontsize= 12)
    plt.xlabel("Predicted label", fontsize= 12)
    sns.heatmap(cm, xticklabels= label_encoding.inverse_transform(labels), yticklabels= label_encoding.inverse_transform(labels), annot= True, linewidths= 0.1, fmt= 'f', cmap= 'YlGnBu')
    plt.show()

    
    
    
    
    
    plt.plot(range(epoch), trainAccLs, marker='.',  label='train_accuracy')
    plt.plot(range(epoch), valAccLs, marker='.',  label='val_accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('C:/Users/MA201-Ultima/Desktop/accuracy.jpg',bbox_inches = 'tight')
    
    plt.clf()

    plt.plot(range(epoch), trainLossLs, marker='.', label='train_loss')
    plt.plot(range(epoch), valLossLs, marker='.', label='val_loss')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('cls_loss')
    plt.savefig('C:/Users/MA201-Ultima/Desktop/cls_loss.jpg',bbox_inches = 'tight')

    plt.clf()

    plt.plot(range(epoch), domainLossLs, marker='.', label='train_loss')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('domain_loss')
    plt.savefig('C:/Users/MA201-Ultima/Desktop/domain_loss.jpg',bbox_inches = 'tight')