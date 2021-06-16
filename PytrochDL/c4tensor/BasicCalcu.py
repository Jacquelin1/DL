import torch
import numpy as np

def print_hi(name):

    # +/-
    a=torch.rand(3,4)
    b=torch.rand(4)
    c=a+b
    print("c",c)
    print("a",a)
    print("b",b)
    d=torch.add(a,b)
    print(d)
    print(d.max(1)[1])
    print(torch.all(torch.eq(a-b,torch.sub(a,b))))
    print(torch.all(torch.eq(a/b,torch.div(a,b))))
    print(torch.all(torch.eq(a*b,torch.mul(a,b))))

    # c
    # tensor([[0.9538, 1.0292, 1.3806, 1.1365],
    #         [1.3058, 1.7152, 1.5823, 1.2357],
    #         [1.6568, 1.5475, 1.5506, 1.3027]])
    # a
    # tensor([[0.1013, 0.1491, 0.7412, 0.3872],
    #         [0.4533, 0.8351, 0.9429, 0.4864],
    #         [0.8043, 0.6674, 0.9112, 0.5533]])
    # b
    # tensor([0.8525, 0.8801, 0.6394, 0.7493])
    # tensor([[0.9538, 1.0292, 1.3806, 1.1365],
    #         [1.3058, 1.7152, 1.5823, 1.2357],
    #         [1.6568, 1.5475, 1.5506, 1.3027]])
    # tensor(True)
    # tensor(True)
    # tensor(True)

if __name__ == '__main__':
    print_hi('tensor foundation')