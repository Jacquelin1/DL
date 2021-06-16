import torch
import numpy as np

print(torch.__version__)


def print_hi(name):

    # view
    # a=torch.rand(4,1,28,28)
    # print(a.shape)
    # print(4*28*28)
    # print(a.numel())  # number elements 获取tensor中一共包含多少个元素
    # print(a.view(4,28*28).shape)  # torch.Size([4, 784])

    # squeeze维度增加/unsqueeze新插入一个维度
    # b=torch.rand(4,3,28,28)
    # print(b.shape)
    # print(b.unsqueeze(0).shape)  # torch.Size([1, 4, 3, 28, 28])  unsqueeze的参数0表示在哪个维度
    # print(b.unsqueeze(4).shape)  # torch.Size([4, 3, 28, 28, 1])
    # print(b.unsqueeze(2).shape)  # torch.Size([4, 3, 1, 28, 28])
    # 在channel上加bias
    # c=torch.rand(32)
    # print(c)
    # d=torch.rand(4,32,14,14)
    # print(c.unsqueeze(1).shape)  # torch.Size([32, 1])
    # print(c.unsqueeze(1).unsqueeze(2).unsqueeze(0).shape)  # torch.Size([1, 32, 1, 1])
    # e=c.unsqueeze(1).unsqueeze(2).unsqueeze(0)
    # print(e)
    # print(e.squeeze().shape)  # torch.Size([32])
    # print(e.squeeze(0).shape)  # torch.Size([32, 1, 1])
    # print(e.squeeze(-4).shape)  # torch.Size([32, 1, 1])
    # print(e.squeeze(1).shape)  # torch.Size([1, 32, 1, 1]) dim=1的维度上不为0，所以挤压不掉

    # expand/repeat
    # c = torch.rand(32)
    # e = c.unsqueeze(1).unsqueeze(2).unsqueeze(0)
    # print(e.shape)  # torch.Size([1, 32, 1, 1])
    # print(e.expand(4,32,14,14).shape)  # torch.Size([4, 32, 14, 14]) dim=1的维度必须与原来的32一致；必须由1扩展到m
    # print(e.expand(-1,32,-1,-1).shape)  # torch.Size([1, 32, 1, 1])
    # print(e.repeat(4,1,14,14).shape)  # torch.Size([4, 32, 14, 14])

    # t  !! t() expects a tensor with <= 2 dimensions, but self is 3D
    # f=torch.rand(3,4)
    # print(f)
    # print(f.t())

    # transpose
    g=torch.rand(2,3)
    print(g)
    g1=g.transpose(0,1)
    print(g1)
    h=torch.rand(4,3,14,14)
    print(h.transpose(1,3).shape)  # torch.Size([4, 14, 14, 3])
    print(h.transpose(1, 3).contiguous().view(4,3*14*14).shape)  # torch.Size([4, 588])
    print(h.transpose(1, 3).contiguous().view(4, 3 * 14 * 14).view(4, 3 , 14 ,14).shape)  # torch.Size([4, 3, 14, 14])
    h1=h.transpose(1, 3).contiguous().view(4, 3 * 14 * 14).view(4, 3 , 14 ,14)
    print(h1.shape)  # torch.Size([4, 3, 14, 14])
    h2=h.transpose(1, 3).contiguous().view(4, 3 * 14 * 14).view(4, 14 , 14 ,3).transpose(1,3)
    print(h2.shape)  # torch.Size([4, 3, 14, 14])
    print(torch.all(torch.eq(h,h1)))  # tensor(False)  证明view会污染数据，必须人为跟踪，不能强行转换，不然顺序会打乱
    print(torch.all(torch.eq(h,h2)))  # tensor(True)

    # permute
    i=torch.rand(4,3,24,32)
    print(i.permute(0,2,3,1).shape)  # torch.Size([4, 24, 32, 3])  将channel维度换到最后，原来的图像还是原来图像


if __name__ == '__main__':
        print_hi('tensor foundation')