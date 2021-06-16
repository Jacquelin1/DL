import torch
import numpy as np


def print_hi(name):
    a1 = torch.rand(4, 3,16,32)
    a2 = torch.rand(4, 3,16,32)

    # cat / stack
    b=torch.cat([a1,a2],dim=2)
    print(b.shape)  # torch.Size([4, 3, 32, 32])
    c=torch.stack([a1,a2],dim=2)
    print(c.shape)  # torch.Size([4, 3, 2, 16, 32]) 多一个维度

    # split / chunk
    aa,bb=b.split([1,2],dim=1)
    print(aa.shape,bb.shape)  # torch.Size([4, 1, 32, 32]) torch.Size([4, 2, 32, 32])
    m=torch.rand(3,2,32,8)
    cc,dd=m.chunk(2,0)
    print(cc.shape,dd.shape)
    # data = torch.from_numpy(np.random.rand(3, 5))
    # for i, data_i in enumerate(data.chunk(5, 1)):  # 沿1轴分为5块
    #     print(str(data_i))

if __name__ == '__main__':
    print_hi('tensor foundation')