
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim


"""
model des:  Lenet5(
  (conv_unit): Sequential(
    (0): Conv2d(3, 16, kernel_size=(5, 5), stride=(1, 1))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1))
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
   flatten: x = x.view(batchSize, 32 * 5 * 5) 
  (fc_unit): Sequential(
    (0): Linear(in_features=800, out_features=32, bias=True)
    (1): ReLU()
    (2): Linear(in_features=32, out_features=10, bias=True)
  )
)"""
class Lenet5(nn.Module):

    def __init__(self):
        super(Lenet5, self).__init__()

        # conv
        self.conv_unit = nn.Sequential(
            # [b,3,32,32] ->conv-> [b,16,28,28] ->maxPool-> [b,16,14,14]
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
            # [in_channels,out_channels, , , ] cifar10数据是rgb3通道,，使用16个kernel/filter
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # [b,16,14,14] ->conv-> [b,32,10,10] ->maxPool-> [b,32,5,5]
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # flat
        self.fc_unit = nn.Sequential(
            nn.Linear(32 * 5 * 5, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        # x [b,3,32,32]
        batchSize = x.size(0)
        # conv
        x = self.conv_unit(x)
        # flat
        x = x.view(batchSize, 32 * 5 * 5)  # flatten
        logits = self.fc_unit(x)

        return logits


class RunLenet5:

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
        model = Lenet5()
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
    cl=RunLenet5()
    cl.run()

    # lenet5Model = Lenet5()
    # tmp = torch.rand(2, 3, 32, 32)
    # out = lenet5Model.conv_unit(tmp)
    # print(out.shape)  # torch.Size([2, 32, 5, 5])
