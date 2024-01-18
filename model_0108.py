import torch.nn as nn
import torch
from layers_train_pytorch import ConvOffset2D
import torch.nn.functional as F

class Mymodel(nn.Module):
    def __init__(self, in_channels, out_class_dim, trainable=True):
        super().__init__()
        self.conv_num = 32  # Assuming this value

        self.conv_block_1= self.ConvBolck(in_channels, self.conv_num)
        self.conv_block_2= self.ConvBolck(self.conv_num, self.conv_num* 2)
        self.conv_block_3= self.ConvBolck(self.conv_num* 2, self.conv_num* 4)
        self.conv_block_4= self.ConvBolck(self.conv_num* 4, self.conv_num* 8)
        self.conv_block_5= self.ConvBolck(self.conv_num* 8, self.conv_num* 4)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.conv_num*4, out_class_dim)
        self.sigmoid = nn.Sigmoid()
    
    def ConvBolck(self, in_channels, out_channels):
        conv_block= nn.Sequential(ConvOffset2D(in_channels),
        nn.Conv2d(in_channels, out_channels, (3,3), stride=(2, 2), padding= 1),
                                                     nn.BatchNorm2d(out_channels),
                                                     nn.ReLU())
        return conv_block
#     # Deformable forward
    def forward(self, x):      
        x_1= self.conv_block_1(x)
        x_2= self.conv_block_2(x_1)
        x_3= self.conv_block_3(x_2)
        x_4= self.conv_block_4(x_3)
        x_5= self.conv_block_5(x_4)
        x_6 = self.global_avg_pool(x_5)
        x_7 = torch.flatten(x_6, 1)
        x_8 = self.fc(x_7)
        output = self.sigmoid(x_8)
        return output, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8
    #x_4