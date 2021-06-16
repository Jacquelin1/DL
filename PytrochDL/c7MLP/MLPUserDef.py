import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms
from torchvision.datasets import MNIST
import torchvision
from visdom import Visdom



if __name__ == '__main__':

    # 创建一个自己的类（网络结构），封装性更强，继承自nn.Module
    class MLP(nn.Module):

        def __init__(self):
            super(MLP, self).__init__()

            self.model = nn.Sequential(
                nn.Linear(784, 200),
                nn.LeakyReLU(inplace=True),  # 激活函数Relu/LeakyReLU
                nn.Linear(200, 200),  # 全连接层
                nn.LeakyReLU(inplace=True),
                nn.Linear(200, 10),
                nn.LeakyReLU(inplace=True),
            )

        def forward(self, x):
            x = self.model(x)

            return x

    # 一次训练所选取的样本数,Batch Size的大小影响模型的优化程度和速度。
    # 同时其直接影响到GPU内存的使用情况，假如GPU内存不大，该数值最好设置小一点
    # 把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大
    batch_size = 200
    # 学习率
    learning_rate = 0.01
    epochs = 2
    # 1.download the data,then use torch.utils.data.DataLoader to import
    # train_loader = MNIST(root='./', train=True, download=True, transform=torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(
    #         (0.1307,), (0.3081,))
    # ]))
    #
    # test_loader = MNIST(root='./', train=False, download=True, transform=torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(
    #         (0.1307,), (0.3081,))
    # ]))

    # 60000
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    # 10000
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    # create MLP
    net = MLP()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # params(iterable) – 待优化参数的iterable或者是定义了参数组的dict
    # lr(float) – 学习率
    # momentum(float, 可选) – 动量因子（默认：0）
    # weight_decay(float, 可选) – 权重衰减（L2惩罚）（默认：0）
    # dampening(float, 可选) – 动量的抑制因子（默认：0）
    # nesterov(bool, 可选) – 使用Nesterov动量（默认：False）

    # 交叉熵误差 -> loss function
    criteon = nn.CrossEntropyLoss()

    viz = Visdom()

    viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
    viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.',
                                                       legend=['loss', 'acc.']))
    global_step = 0
    for epoch in range(epochs):

        for batch_idx, (data, target) in enumerate(train_loader):
            # print("batch_idx",batch_idx)
            # print("data.shape",data.shape)  # data.shape torch.Size([200, 1, 28, 28])
            data = data.view(-1, 28*28)  # data.shape torch.Size([200, 784])
            # print("data.shape", data.shape)
            # print(data[0])

            # 前向传播求出预测的值
            logits = net(data)
            # print("logits",logits.shape)  # logits torch.Size([200, 10])
            # print("logits",logits[0])  # logits tensor([0.0026, 0.0226, 0.0660, 0.0000, 0.0000, 0.0145, 0.0000, 0.0508, 0.0435,0.0000], grad_fn=<SelectBackward>)
            # print(target[0])  # tensor(3)
            # 求loss
            loss = criteon(logits, target)
            # print("loss",loss)  # loss tensor(2.2879, grad_fn=<NllLossBackward>)

            # 模型中参数的梯度设为0
            optimizer.zero_grad()
            # 反向传播求梯度得到每个参数的梯度值
            loss.backward()
            # print(w1.grad.norm(), w2.grad.norm())
            # 通过梯度下降执行一步参数更新
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            # batch_idx 100
            # loss tensor(1.8665, grad_fn= < NllLossBackward >)
            # Train Epoch: 0[20000 / 60000(33 %)] Loss: 1.866482

        # evaluate
        test_loss = 0
        correct = 0
        # print("len(test_loader)",len(test_loader))  # 50
        for data, target in test_loader:
            # print(data.shape)  # torch.Size([200, 1, 28, 28])
            data = data.view(-1, 28 * 28)
            # print(data.shape)  # torch.Size([200, 784])
            logits = net(data)
            test_loss += criteon(logits, target).item()

            # print("logits",logits.shape)  # torch.Size([200, 10])
            # print("logits",logits)
            # print("logits.data.max()",logits.data.max())
            # print("logits.data.max(1)",logits.data.max(1))  # max(dim)
            # print("logits.data.max(1)[]",logits.data.max(1)[1])  # idx
            pred = logits.data.max(1)[1]  # 200*10 200张图片，生成了10个类别的预测 每张图片从10个类别里取出最大可能的类别
            correct += pred.eq(target.data).sum()
            # print(pred.eq(target.data).sum())
        viz.line([[test_loss, correct / len(test_loader.dataset)]],
                 [global_step], win='test', update='append')
        viz.images(data.view(-1, 1, 28, 28), win='x')
        # viz.text(str(pred.detach().cpu().numpy()), win='pred',
        #          opts=dict(title='pred'))
        viz.text(str(pred.numpy()), win='pred',
                 opts=dict(title='pred'))

        test_loss /= len(test_loader.dataset)
        print('\nTest set: \tAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

  # Test set: Average loss: 0.0032, Accuracy: 8500/10000 (85%)
  # Test set: Average loss: 0.0019, Accuracy: 8936/10000 (89%)