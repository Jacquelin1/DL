import torch
import numpy as np

print(torch.__version__)


def print_hi(name):
    # idx
    a = torch.rand(4, 3, 28, 28)  # 4张图片，3个channel，28*28的像素
    # print("a[0]",a[0].shape)  # torch.Size([3, 28, 28])  第0张图片
    # print(a[0,0].shape)  # torch.Size([28, 28])  第0张图片第0个通道
    # print(a[0,0,2,4])  # tensor(0.5066)  第0张图片第0个通道(2,4)点

    # select first/lastN
    # print("a[:2].shape",a[:2].shape)  # 第0,1两张图片 torch.Size([2, 3, 28, 28])
    # print("a[:2,:1,:,:].shape",a[:2,:1,:,:].shape)  # 第0,1两张图片的第0个通道 torch.Size([2, 1, 28, 28])
    # print("a[:2,1:].shape",a[:2,1:].shape)  # 前两张图片的后两个通道 torch.Size([2, 2, 28, 28])
    # print("a[:2,-1:,:,:].shape",a[:2,-1:,:,:].shape)  # 前两张图片的最后一个通道 torch.Size([2, 1, 28, 28])

    # select by steps
    # print("a[:,:,0:28:2,0:28:2]", a[:, :, 0:28:2, 0:28:2].shape)  # 像素隔一行隔一列torch.Size([4, 3, 14, 14])
    # print("a[:,:,0:28:3,0:28:2]", a[:, :, 0:28:3, 0:28:2].shape)  # torch.Size([4, 3, 10, 14])
    # print("a[:,:,0:28:3,0:28:2]", a[:, :, ::2, ::2].shape)  # torch.Size([4, 3, 14, 14])

    # select by specific idx
    # print(a.index_select(0,torch.tensor([0,2])).shape)  # 对第一个维度select，选取前两张图 torch.Size([2, 3, 28, 28])
    # print(a.index_select(2,torch.arange(8)).shape)  # torch.Size([4, 3, 8, 28])
    # print(torch.arange(8))  # tensor([0, 1, 2, 3, 4, 5, 6, 7])

    # ...
    # print(a[..., :2].shape)

    # select by mask
    # b=torch.randn(3,4)
    # print(b)
    # # tensor([[-0.5925, -0.5023, 1.0493, -0.4419],
    # #         [0.2347, 1.4010, 0.0417, -2.1166],
    # #         [0.1378, 0.9809, -1.7706, -1.0642]])
    # mask=b.ge(0.5)  # 如果两个张量有相同的形状和元素值，则返回True ，否则 False。 第二个参数可以为一个数或与第一个参数相同形状和类型的张量
    # print(mask)
    # # tensor([[False, False, True, False],
    # #         [False, True, False, False],
    # #         [False, True, False, False]])
    # c=torch.masked_select(b,mask)  # 取出b中所有大于0.5的元素
    # print(c)  # tensor([1.0493, 1.4010, 0.9809])

    # select by flatten index
    d=torch.randn(2,3)
    print(d)
    # tensor([[-0.7545, -1.0704, 0.6819],
    #         [-2.1728, 0.8400, -1.7062]])
    print(torch.take(d,torch.tensor([0,2,5])))
    # tensor([-0.7545, 0.6819, -1.7062])


if __name__ == '__main__':
    print_hi('tensor foundation')
