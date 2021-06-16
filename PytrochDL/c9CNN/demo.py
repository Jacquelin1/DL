
import torch
import numpy as np
import  torch.nn as nn
import  torch.nn.functional as F


def print_conv(name):

    # Conv2d
    x=torch.rand(1,1,28,28)  # 1 photo
    # layer=nn.Conv2d(1,3,kernel_size=3,stride=1,padding=0)  # kernel size:[3,1,3,3]
    # out=layer.forward(x)
    # print(out.shape)  # torch.Size([1, 3, 26, 26])
    #
    layer2=nn.Conv2d(1,3,kernel_size=3,stride=1,padding=0)
    out=layer2.forward(x)
    print(out.shape)  # torch.Size([1, 3, 28, 28])
    #
    # layer3 = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=0)
    # out = layer3.forward(x)
    # print(out.shape)  # torch.Size([1, 3, 13, 13])
    #
    # out=layer3(x)  # call function
    # print(out.shape)  # torch.Size([1, 3, 13, 13])
    # print(layer3.weight)  # [3,1,3,3] [kernel numbers,input channel,kernel w,kernel l]
    # print(layer3.weight.shape)  # torch.Size([3, 1, 3, 3])
    # print(layer3.bias)  # tensor([ 0.2156,  0.0640, -0.3235], requires_grad=True)  kernel channel=3
    #
    # x2=torch.rand(1,3,28,28)
    # w=torch.rand(16,3,5,5)  # 16个kernel or filter，3为input的channel，5*5代表kernel的尺寸
    # b=torch.rand(16)
    # out=F.conv2d(x2,w,b,stride=1,padding=1)
    # print(out.shape)  # torch.Size([1, 16, 26, 26]) 每个kernel都会对这张图过一遍

    # BatchNorm
    # x=torch.rand(1,16,784)  # 一维数据，28*28=784
    # layer=nn.BatchNorm1d(16)  # 有几个channel，BatchNorm中的参数就需要设几，在每个channel上生成均值和方差
    # out=layer(x)
    # print(out.shape)
    # print(layer.running_mean)
    # print(layer.running_var)
    # print(layer.weight)
    # print(layer.bias)
    # print(x[0][0])
    # print(out[0][0])

    # x2=torch.rand(1,16,7,7)
    # layer2=nn.BatchNorm2d(16)
    # out=layer2(x2)
    # print(out.shape)  # torch.Size([1, 16, 7, 7])
    # print(layer2.running_mean)
    # # print(layer2.running_mean[0])
    # print(layer2.running_var)
    # # print(layer2.running_var[0])
    # print("x",x2[0][0])
    # # print(x2[0][0][0][0])
    # print("out",out[0][0])
    # print(layer2.weight)
    # print(layer2.bias)
    # # print(out[0][0][0][0])
    # print("(num-mean)/var",(x2[0][0][0][0]-layer2.running_mean[0])/layer2.running_var[0])
    # print(vars(layer2))








if __name__ == '__main__':
    print_conv('tensor foundation')