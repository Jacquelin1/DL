import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim

"""
Resnet18要经过一个卷积层、Pooling层，然后是四个“小方块”，一个方块由两个A Build Block组成，
一个A Build Block又由两个卷积层组成，四个“小方块”即16层，最后是average pool、全连接层。
由于Pooling层不需要参数学习，故去除Pooling层，整个resnet18网络由18层组成
https://blog.csdn.net/weixin_39867066/article/details/112275617

model des:  ResNet18(
  (conv1): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(3, 3))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (blk1): ResBlk(
    (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (extra): Sequential(
      (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2))
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (blk2): ResBlk(
    (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (extra): Sequential(
      (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2))
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (blk3): ResBlk(
    (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (extra): Sequential(
      (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (blk4): ResBlk(
    (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (extra): Sequential()
  )
  (outLayer): Linear(in_features=512, out_features=10, bias=True)
)
"""


class ResBlk(nn.Module):

    def __init__(self, channel_in, channel_out, stride=1):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_out)
        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_out)

        # 短接层
        self.extra = nn.Sequential()
        # 保证X与res层相加时维度是对应的，这步只是进行转化成相同维度
        if channel_out != channel_in:  # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(channel_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )

        self.blk1 = ResBlk(64, 128, stride=2)  # 减少了feature像素，可以增加通道
        self.blk2 = ResBlk(128, 256, stride=2)
        self.blk3 = ResBlk(256, 512, stride=2)
        self.blk4 = ResBlk(512, 512, stride=2)

        self.outLayer = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = F.adaptive_avg_pool2d(x, [1, 1])

        x = x.view(x.size(0), -1)
        x = self.outLayer(x)

        return x

class RunResNet18:
    def run(self):
        batchSize = 128
        epochs = 10
        # download data
        cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]), download=True)
        cifar_train = DataLoader(cifar_train, batch_size=batchSize, shuffle=True)

        cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]), download=True)
        cifar_test = DataLoader(cifar_test, batch_size=batchSize, shuffle=True)

        # x,label=iter(cifar_train).next()
        x, label = iter(cifar_test).next()
        print("x: ", x.shape, "label: ", label.shape)  # x:  torch.Size([128, 3, 32, 32]) label:  torch.Size([128])

        # model
        model = ResNet18()
        # loss
        criteon = nn.CrossEntropyLoss()
        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        print("model des: ", model)

        for epoch in range(epochs):

            model.train()
            for batchIdx, (x, label) in enumerate(cifar_train):
                # x:[b,3,32,32] label:[b]
                # forward
                logits = model(x)  # logits:[b,10]
                loss = criteon(logits, label)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batchIdx % 300 == 0:
                    print(epoch, batchIdx,"loss: ", loss.item())

            # test
            model.eval()
            with torch.no_grad():
                total_correct = 0
                total_num=0
                for x,label in cifar_test:
                    logits=model(x)  # logits:[b,10]
                    pred=logits.argmax(dim=1)  # pred:[b]
                    # print("logits[0]",logits[0])
                    correct=torch.eq(pred,label).float().sum().item()
                    total_correct+=correct
                    total_num+=x.size(0)

                accuracy=total_correct/total_num
                print(epoch,"acc:",accuracy)


if __name__ == '__main__':
    # blk = ResBlk(64,128,4)
    # tmp = torch.rand(2, 64, 32, 32)
    # out = blk(tmp)
    # print(out.shape)  # torch.Size([2, 128, 8, 8])

    # x = torch.rand(2, 3, 32, 32)
    # model = ResNet18()
    # out = model(x)
    # print(out.shape)  # torch.Size([2, 10])

    cl=RunResNet18()
    cl.run()