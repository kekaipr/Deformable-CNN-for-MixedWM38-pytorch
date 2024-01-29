import os
import numpy as np
import time
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from layers_train_pytorch import ConvOffset2D
import torch.nn.functional as F
import Performance_Matrices as PM
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

class Mymodel(nn.Module):
    # Deformable init
    def __init__(self, inputs_shape, classes=1, trainable=True):
        super().__init__()
        bn_axis = 1 # PyTorch batch normalization axis
        self.conv_num = 32  # Assuming this value

        self.offset1 = ConvOffset2D(1)
        self.conv1 = nn.Conv2d(1, self.conv_num, (3,3), stride=(2,2), padding=1)
        self.batch_norm1 = nn.BatchNorm2d(self.conv_num)
        
        self.offset2 = ConvOffset2D(32)
        self.conv2 = nn.Conv2d(32, self.conv_num*2, (3,3), stride=(2,2), padding=1)
        self.batch_norm2 = nn.BatchNorm2d(self.conv_num*2)

        self.offset3 = ConvOffset2D(64)
        self.conv3 = nn.Conv2d(64, self.conv_num*4, (3,3), stride=(2,2), padding=1)
        self.batch_norm3 = nn.BatchNorm2d(self.conv_num*4)

        self.offset4 = ConvOffset2D(128)
        self.conv4 = nn.Conv2d(128, self.conv_num*8, (3,3), padding=1)
        self.batch_norm4 = nn.BatchNorm2d(self.conv_num*8)
        
        self.offset5 = ConvOffset2D(256)
        self.conv5 = nn.Conv2d(256, self.conv_num*4, (3,3), stride=(2,2), padding=1)
        self.batch_norm5 = nn.BatchNorm2d(self.conv_num*4)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.conv_num*4, 9)
        self.sigmoid = nn.Sigmoid()
    
#     # Deformable forward
    def forward(self, x):      
        x = self.offset1(x)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        
        x = self.offset2(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        
        x = self.offset3(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        
        x = self.offset4(x)
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        
        x = self.offset5(x)
        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = F.relu(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class Train_mix():
    def __init__(self,datapath='merged.npz',lr=0.00001,fullnet_num=128,conv_num=32,deconv_size=(3,3)):
        self.datapath=datapath
        self.lr = lr
        self.fullnet_num = fullnet_num
        self.conv_num = conv_num
        self.deconv_size = deconv_size
 
    def start_train(self):
        epochs = 1000
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = np.load(os.path.join(self.datapath), allow_pickle= True)

        trainx = data["arr_0"]
        trainy = data["arr_1"]
        # print('$$$$$$$$$$$$trainx',trainx.shape)
        label_encoding= LabelEncoder()
        source_training_label= label_encoding.fit_transform(trainy)

        trainx = np.expand_dims(trainx, axis=-1)
        ##以下用於須解出來，以上用於不須解
        # trainx_ = np.zeros(shape= (len(trainx), 1, 40, 40))
        # for index in range(0, len(trainx_)):
        #     trainx_[index]= trainx[index].reshape(1, 1, 40, 40)
        # del trainx
        # trainx = trainx_

        trainx = trainx.reshape(trainx.shape[0], 1, 40, 40)
        data_shape = trainx.shape[1:]
        num_classes = 9
        model = Mymodel(data_shape, num_classes)

        model.to(device)
        print(model)
        loss_fn= nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
                              
        x_train, x_test, y_train, y_test = train_test_split(trainx, source_training_label, test_size=0.01, random_state=10 )#,stratify = trainy
        train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                    torch.LongTensor(y_train))
        
        test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                    torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
        test_loader = DataLoader(test_dataset)
        
        accuracy_list = []
        loss_list = []
        val_accuracy_list = []
        val_loss_list = []

        count = 0
        early_stop = {'epoch':0,'best_epoch':0,'loss':10000}
        for epoch in range(epochs+1):
            model.train()
            # model.load_state_dict(torch.load('C:/Users/MA201-Ultima/Desktop/40sourceWeight/00epoch100_38.pt'))
            total_loss = 0.0
            total_accuracy = 0.0
            accuracy = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                accuracy= accuracy_score(targets.data.cpu().numpy(), np.argmax(outputs.data.cpu().numpy(), axis= 1))* inputs.size(0)
                total_accuracy += accuracy
            average_loss = total_loss / len(train_loader.dataset)
            average_accuracy = total_accuracy / len(train_loader.dataset)  
            accuracy_list.append(average_accuracy.item())
            loss_list.append(average_loss)
            print("Epoch %4d , train_Loss: %2.6f, train_Accuracy: %2.6f" %(epoch,average_loss,average_accuracy))
            
            model.eval()
            total_loss_val = 0
            total_accuracy_val = 0
            accuracy_val = 0.0
            torch.save(model.state_dict(), f'C:/Users/MA201-Ultima/Desktop/mixtrain/epoch{epoch}.pt')
        
            with torch.no_grad():
                for input_val, target_val in test_loader:
                    input_val, target_val = input_val.to(device), target_val.to(device)
                    output_val = model(input_val)
                    
                    loss_val = loss_fn(output_val,target_val).item()
                    # accuracy_val = self.acc_myself(target_val, output_val)
                    accuracy_val= accuracy_score(target_val.data.cpu().numpy(), np.argmax(output_val.data.cpu().numpy(), axis= 1))* input_val.size(0)

                    total_loss_val += loss_val * input_val.size(0)
                    total_accuracy_val += accuracy_val 
                        
                    average_loss_val = total_loss_val / len(test_loader.dataset)
                    average_accuracy_val = total_accuracy_val / len(test_loader.dataset)
                val_accuracy_list.append(average_accuracy_val.item())
                val_loss_list.append(average_loss_val)
                print("test_Loss: %2.6f, test_Accuracy: %2.6f" %(average_loss_val,average_accuracy_val))

            if average_loss_val < early_stop['loss']:
                early_stop['loss'] = average_loss_val
                early_stop['best_epoch'] = epoch
                early_stop['epoch'] = epoch
                count = 0
            else:
                count += 1
                early_stop['epoch'] = epoch
            if count == 10:
                break
        print("the lowest val_loss is epoch: %2.0f, the val_loss is: %2.6f" %(early_stop["best_epoch"],early_stop["loss"]))      

        # model.load_state_dict(torch.load('C:/Users/MA201-Ultima/Desktop/epoch2500.pt'))
        plt.plot(range(early_stop["epoch"]+1), accuracy_list, marker='.',  label='train_accuracy')
        plt.plot(range(early_stop["epoch"]+1), val_accuracy_list, marker='.',  label='val_accuracy')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.savefig('C:/Users/MA201-Ultima/Desktop/accuracy.jpg',bbox_inches = 'tight')
        
        plt.clf()

        plt.plot(range(early_stop["epoch"]+1), loss_list, marker='.', label='train_loss')
        plt.plot(range(early_stop["epoch"]+1), val_loss_list, marker='.', label='val_loss')
        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.savefig('C:/Users/MA201-Ultima/Desktop/loss.jpg',bbox_inches = 'tight')

if __name__ == '__main__':
    history = Train_mix()
    history.start_train()
    