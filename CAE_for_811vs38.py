import os
import numpy as np
import time
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F

class CAE(nn.Module):
    def __init__(self):
        super().__init__()
        # self.maxpooling= nn.MaxPool2d((2, 2))
        self.conv1 = nn.Conv2d(3, 32, (2,2), 1,padding=1)
        self.conv2 = nn.Conv2d(32, 64, (2,2), 1,padding=1)
        self.conv3 = nn.Conv2d(64, 128, (2,2), 1,padding=1)
        self.conv4 = nn.Conv2d(128, 256, (2,2), 1,padding=1)
        self.conv5 = nn.Conv2d(256, 256, (2,2), 2,padding=1)
        self.conv6 = nn.Conv2d(256, 256, (2,2), 2,padding=1)
        self.conv7 = nn.Conv2d(256, 256, (2,2), 2,padding=1)

        self.convtrans1 = nn.ConvTranspose2d(256, 256, (3,3), 2,padding=1)
        self.convtrans2 = nn.ConvTranspose2d(256, 256, (2,2), 2,padding=1)
        self.convtrans4 = nn.ConvTranspose2d(256, 128, (2,2), 2,padding=1)
        self.convtrans5 = nn.ConvTranspose2d(128, 64, (2,2), 1,padding=1)
        self.convtrans6 = nn.ConvTranspose2d(64, 32, (2,2), 1,padding=1)
        self.convtrans7 = nn.ConvTranspose2d(32, 3, (3,3), 1,padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)

        x = self.convtrans1(x)
        x = F.relu(x)
        x = self.convtrans2(x)
        x = F.relu(x)
        # x = self.convtrans3(x)
        # x = F.relu(x)
        x = self.convtrans4(x)
        x = F.relu(x)
        x = self.convtrans5(x)
        x = F.relu(x)
        x = self.convtrans6(x)
        x = F.relu(x)
        x = self.convtrans7(x)

        x = F.sigmoid(x)
        return x 

if __name__ == "__main__":
    defect = 'nearfull'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data=np.load(f'C:/Users/MA201-Ultima/Desktop/thesis/datasets/for_CAE/38_{defect}_OH.npz')
    
    trainx = data["arr_0"]
    trainy = data["arr_1"]
    # trainx = np.expand_dims(trainx, axis=-1)
    trainx = trainx.reshape(149, 3, 52, 52)
    data_shape = trainx.shape[1:]
    model = CAE()
    model.to(device)
    print(model)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 3000
    x_train, x_test, y_train, y_test = train_test_split(trainx, trainy, test_size=0.2, random_state=10,stratify = trainy)
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                    torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                    torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=400, shuffle=True)
    val_loader = DataLoader(val_dataset)

    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []
    imgW, imgH, imgC= trainx.shape[2], trainx.shape[3], trainx.shape[1]
    

    # Train
    early_stop, count = {'epoch':0,'best_epoch':0,'loss':10000}, 0
    for epoch in range(epochs+1):
        model.train()
        Tbatch_loss= []; Vbatch_loss= []
        for image, label in train_loader:
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(image)
            # print(output.shape)
            loss = loss_fn(output,image)
            Tbatch_loss.append(loss.data.cpu().numpy())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            for image_val, label_val in val_loader:
                image_val, label = image_val.to(device), label_val.to(device)
                output_val = model(image_val)
                loss_val = loss_fn(output_val, image_val)
                Vbatch_loss.append(loss_val.data.cpu().numpy())
        train_loss.append(np.mean(Tbatch_loss))
        val_loss.append(np.mean(Vbatch_loss))
        
        #print per 100 epoch 
        if epoch%50 ==0:
            print("[NOTE] Iteration: %5d// Train loss: %2.6f// Valid loss %2.6f"%(epoch, np.mean(Tbatch_loss), np.mean(Vbatch_loss)))
            plt.figure(figsize= (8, 8))
            plt.subplot(221)
            plt.title("[Train] actual image")
            plt.imshow(image[0, :, :, :].data.cpu().numpy().reshape(imgH, imgW, imgC))
            plt.subplot(222)
            plt.title("[Train] reconstruct image")
            plt.imshow(output[0, :, :, :].data.cpu().numpy().reshape(imgH, imgW, imgC))
            plt.subplot(223)
            plt.title("[Valid] actual image")
            plt.imshow(image_val[0, :, :, :].data.cpu().numpy().reshape(imgH, imgW, imgC))
            plt.subplot(224)
            plt.title("[Valid] reconstruct image")
            plt.imshow(output_val[0, :, :, :].data.cpu().numpy().reshape(imgH, imgW, imgC))
            plt.tight_layout()
            plt.show()
            torch.save(model.state_dict(), f'C:/Users/MA201-Ultima/Desktop/CAE_weight/{defect}/CAE_{defect}_epoch{epoch}.pt')

        #early stop
        # if np.mean(Vbatch_loss) < early_stop['loss']:
        #         early_stop['loss'] = np.mean(Vbatch_loss)
        #         early_stop['best_epoch'] = epoch
        #         early_stop['epoch'] = epoch
        # else:
        #         count += 1
        #         early_stop['epoch'] = epoch
        # if count == 100:
        #         break

    plt.figure(num= 1, figsize= (8, 3))
    plt.plot(np.arange(1, len(train_loss)+ 1), train_loss, label= "train_loss", color= "darkblue")
    plt.plot(np.arange(1, len(val_loss)+ 1), val_loss, label= "val_loss", color= "darkred")
    plt.legend(loc= "best", fontsize= 12)
    plt.xlabel("Epoch", fontsize= 12)
    plt.ylabel("MSE_LOSS", fontsize= 12)
    plt.grid(True)
    plt.show()


    ##test