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



class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.offset = ConvOffset2D(in_filters)
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(ConvOffset2D(in_filters)) 
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(ConvOffset2D(filters)) 
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(ConvOffset2D(in_filters)) 
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x

# class Xception(nn.Module):
 
#     def __init__(self, num_classes=8):
#         super().__init__()
#         self.offset1 = ConvOffset2D(3)
#         self.conv1 = nn.Conv2d(3, 32, (3,3),stride=(2,2), padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu = nn.ReLU(inplace=True)

#         self.offset2 = ConvOffset2D(32)
#         self.conv2 = nn.Conv2d(32,64,3,bias=False)
#         self.bn2 = nn.BatchNorm2d(64)
#         #do relu here
        
#         self.block1=Block(64,128,2,2,start_with_relu=False, grow_first=True)
#         self.block2=Block(128,256,2,2,start_with_relu=True, grow_first=True)
#         self.block3=Block(256,728,2,2,start_with_relu=True, grow_first=True)
        
#         self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
#         self.block5=Block(728,1024,3,1,start_with_relu=True,grow_first=False)
#         # self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
#         # self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)  

#         # self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
#         # self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
#         # self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
#         # self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

#         # self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

#         self.conv3 = SeparableConv2d(1024,1536,3,1,1)
#         self.bn3 = nn.BatchNorm2d(1536)

#         #do relu here
#         self.conv4 = SeparableConv2d(1536,2048,3,1,1)
#         self.bn4 = nn.BatchNorm2d(2048)

#         self.fc = nn.Linear(2048, num_classes)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.offset1(x)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         x = self.offset2(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
        
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)

#         x = self.block4(x)
#         x = self.block5(x)
#         # x = self.block6(x)
#         # x = self.block7(x)

#         # x = self.block8(x)
#         # x = self.block9(x)
#         # x = self.block10(x)
#         # x = self.block11(x)
#         # x = self.block12(x)

#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)

#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.relu(x)
#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         x = x.view(x.size(0), -1)
        
#         x = self.fc(x)
#         x = self.sigmoid(x)
#         return x

class Mymodel(nn.Module):
    # Deformable init
    def __init__(self, inputs_shape, classes=8, trainable=True):
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
        self.fc = nn.Linear(self.conv_num*4, classes)
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
    def __init__(self,datapath='C:/Users/MA201-Ultima/Desktop/thesis/datasets/Mixed_WM38_OH.npz',lr=0.0001,fullnet_num=128,conv_num=32,deconv_size=(3,3)):
        self.datapath=datapath
        self.lr = lr
        self.fullnet_num = fullnet_num
        self.conv_num = conv_num
        self.deconv_size = deconv_size

    def auc1(self,inputs,pre):
        inputs_T=inputs.T
        pre_T=pre.T
        acc=np.mean(np.equal(inputs_T,pre_T).astype(np.float64),axis=1)
        return acc
    
        
    def acc_myself(self, y_true, y_pre):
        y_pre = torch.round(y_pre.data.cpu())
        r = torch.eq(y_true.data.cpu(), y_pre)  # torch.eq 與 tf.equal 功能相似
        r = r.to(torch.float32)
        r = torch.sum(r, dim=1)
        d = torch.zeros_like(r, dtype=torch.float32) + 8
        c = torch.eq(r, d)
        c = c.to(torch.float32)
        
        return torch.div(torch.sum(c), torch.tensor(c.size(), dtype=torch.float32))
 

    def start_train(self):
        epochs = 200
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = np.load(os.path.join(self.datapath))

        trainx = data["arr_0"]
        trainy = data["arr_1"]
        
        trainx = np.expand_dims(trainx, axis=-1)
        trainx = trainx.reshape(38015, 3, 52, 52)
        data_shape = trainx.shape[1:]
        num_classes = trainy.shape[-1]
        model = Mymodel(data_shape, num_classes)
        # model = Xception(num_classes)

        model.to(device)
        print(model)
        loss_fn = nn.BCELoss()  
        # optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-6)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
                              
        x_train, x_test, y_train, y_test = train_test_split(trainx, trainy, test_size=0.2, random_state=10,stratify = trainy)
        train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                    torch.tensor(y_train, dtype=torch.float32))
        
        test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                    torch.tensor(y_test, dtype=torch.float32))
        
        train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)
        test_loader = DataLoader(test_dataset)
        
        accuracy_list = []
        loss_list = []
        val_accuracy_list = []
        val_loss_list = []

        # not_frozen = ['offset3.weight','conv3.weight','conv3.bias','batch_norm3.weight','batch_norm3.bias','offset4.weight','conv4.weight','conv4.bias','batch_norm4.weight','batch_norm4.bias','offset5.weight','conv5.weight','conv5.bias','batch_norm5.weight','batch_norm5.bias','fc.weight','fc.bias']
        # not_frozen = ['conv1.weight','conv1.bias','conv2.weight','conv2.bias','conv3.weight','conv3.bias','conv4.weight','conv4.bias','conv5.weight','conv5.bias']
        # for name, param in model.named_parameters():
        #     if name in not_frozen:  
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        count = 0
        early_stop = {'epoch':0,'best_epoch':0,'loss':10000}
        for epoch in range(epochs+1):
            model.train()
            total_loss = 0.0
            total_accuracy = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()     
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                accuracy = self.acc_myself(targets, outputs)
                total_accuracy += accuracy * inputs.size(0)
            average_loss = total_loss / len(train_loader.dataset)
            average_accuracy = total_accuracy / len(train_loader.dataset)  
            accuracy_list.append(average_accuracy.item())
            loss_list.append(average_loss)
            print("Epoch %4d , train_Loss: %2.6f, train_Accuracy: %2.6f" %(epoch,average_loss,average_accuracy))
            
            model.eval()
            total_loss_val = 0
            total_accuracy_val = 0
            torch.save(model.state_dict(), f'C:/Users/MA201-Ultima/Desktop/exp_1108/epoch{epoch}.pt')
        
            with torch.no_grad():
                for input_val, target_val in test_loader:
                    input_val, target_val = input_val.to(device), target_val.to(device)
                    output_val = model(input_val)
                    
                    loss_val = loss_fn(output_val,target_val).item()
                    accuracy_val = self.acc_myself(target_val, output_val)

                    total_loss_val += loss_val * input_val.size(0)
                    total_accuracy_val += accuracy_val * input_val.size(0)
                        
                    average_loss_val = total_loss_val / len(test_loader.dataset)
                    average_accuracy_val = total_accuracy_val / len(test_loader.dataset)
                val_accuracy_list.append(average_accuracy_val.item())
                val_loss_list.append(average_loss_val)
                print("test_Loss: %2.6f, test_Accuracy: %2.6f" %(average_loss_val,average_accuracy_val))

            if average_loss_val < early_stop['loss']:
                early_stop['loss'] = average_loss_val
                early_stop['best_epoch'] = epoch
                early_stop['epoch'] = epoch
            else:
                count += 1
                early_stop['epoch'] = epoch
            if count == 200:
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
    