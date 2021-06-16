import torch
import numpy as np

print(torch.__version__)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    # a = torch.randn(2, 3)
    # print(a)
    # print(type(a))
    # print(isinstance(a, torch.FloatTensor))
    # print(a.type())
    # b = torch.tensor(0.99)
    # print(b)
    # print(b.size())  # torch.Size([])
    #
    # # dim=1
    # c = torch.tensor([1, 3])
    # print("torch.tensor([1,3])", c)
    #
    # d = torch.FloatTensor(1)
    # print("torch.FloatTensor(1)", d)
    #
    # e = torch.FloatTensor(2)
    # print("torch.FloatTensor(2)", e)
    #
    # f = np.ones(2)
    # g = torch.from_numpy(f)
    # print("torch.from_numpy(f)", g)
    # print("torch.from_numpy(f) size", g.size())
    #
    # h = torch.ones(3)
    # print("torch.ones(3)", h.shape)
    # print("torch.ones(3)", h.shape)
    # print(h)  # tensor([1., 1., 1.])
    #
    # # dim = 2  如[4,784]，有四张照片，每张照片有784个维度
    # i = torch.randn(2, 3)
    # print(i)
    # print(i.shape)  # torch.Size([2, 3])
    # print(i.size())  # torch.Size([2, 3])
    # print(i.size(0))  # 2
    # print(i.size(1))  # 3
    # print(i.shape[1])  # 3
    #
    # # dim = 3  典型的RNN输入：一句话，有10个单词，用onehot编码100维。有20句话，[10,20,100]
    # j=torch.rand(1,2,3)
    # print("torch.rand(1,2,3)",j)  # torch.rand(1,1,3) tensor([[[0.8063, 0.3698, 0.7915],[0.0573, 0.0628, 0.6368]]])
    # print(j.shape)
    # print(j[0])  # 第一个维度里的第一个元素
    # print(j[0][1])  # tensor([0.2175, 0.7022, 0.5565])
    # print(list(j.shape))  # [1, 2, 3]
    #
    # # dim = 4  CNN 输入，三张图片，三个RGB通道，图片像素长，图片像素宽 [3,3,28,28]
    # k=torch.rand(2,3,2,2)
    # print(k)
    # print(k.shape)
    # print("k.numel",k.numel())   # 24
    # print("k.dim()",k.dim())  # 4
    # print(torch.tensor(90).dim())  # 0
    #
    # l1=torch.FloatTensor(2,3)  # input shape
    # print("torch.FloatTensor(2,3)",l1)
    # l2 = torch.FloatTensor([2, 3])  # input data
    # print("torch.FloatTensor([2, 3])", l2)
    #
    # m=torch.tensor([2.3,4])  # tensor([2.3000, 4.0000])
    # n = torch.FloatTensor(2, 4)   # tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
    #                                 # [3.9710e+33, 3.0634e-41, 0.0000e+00, 0.0000e+00]])
    # print(m)
    # print(n)
    # print(m.type())

    # # rand
    # o1=torch.rand(2,3)
    # o2=torch.randint(1,10,[3,3])
    # print(o1)
    # print(o2)

    # # randn
    # p1=torch.randn(3,3)
    # print(p1)
    # p2=torch.normal(mean=torch.full([10],0.0),std=torch.arange(1,0,-0.1))
    # print(p2)
    # p3=p2.resize(2,5)
    # print(p3)

    # # full
    # q1=torch.full([2,3],9)
    # print(q1)

    # # range/arange
    # r1=torch.range(0,10)
    # print(r1)
    # r2=torch.arange(0,10)
    # r3=torch.arange(0,10,2)
    # print(r2)
    # print(r3)

    # # linspace/logspace
    # s1=torch.linspace(0,10,steps=5)  # tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])
    # s2=torch.linspace(0,10,steps=11)  # tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
    # print(s1)
    # print(s2)
    # s3=torch.logspace(0,-1,steps=10)
    # print(s3)
    # s4=torch.logspace(0,8,steps=3)  # tensor([1.0000e+00, 1.0000e+04, 1.0000e+08])
    # print(s4)

    # # ones/zeros/eye
    # t1=torch.ones(3,3)
    # print(t1)
    # # tensor([[1., 1., 1.],
    # #     [1., 1., 1.],
    # #     [1., 1., 1.]])
    # t2= torch.zeros(3, 3)
    # print(t2)
    # # tensor([[0., 0., 0.],
    # #         [0., 0., 0.],
    # #         [0., 0., 0.]])
    # t3 = torch.eye(3, 3)
    # print(t3)
    # # tensor([[1., 0., 0.],
    # #         [0., 1., 0.],
    # #         [0., 0., 1.]])

    # randperm  : shuffle
    u1=torch.rand(2,3)
    u2=torch.rand(2,2)
    print(u1)
    print(u2)
    u3=torch.randperm(2)
    print(u3)
    u4=u1[u3]
    u5=u2[u3]
    print(u4)
    print(u5)


if __name__ == '__main__':
    print_hi('tensor foundation')
