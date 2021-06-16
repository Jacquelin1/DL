#
# from visdom import Visdom
# import  torch
# import  torch.nn as nn
# import  torch.nn.functional as F
# import  torch.optim as optim
# from    torchvision import datasets, transforms
# from torchvision.datasets import MNIST
# import torchvision
# viz=Visdom()
#
#
# if __name__ == '__main__':
#     batch_size = 10000
#     # 学习率
#     learning_rate = 0.01
#     epochs = 2
#     # # 60000
#     # train_loader = torch.utils.data.DataLoader(
#     #     datasets.MNIST('./', train=True, download=True,
#     #                    transform=transforms.Compose([
#     #                        transforms.ToTensor(),
#     #                        transforms.Normalize((0.1307,), (0.3081,))
#     #                    ])),
#     #     batch_size=batch_size, shuffle=True)
#     # 10000
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('./', train=False, transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])),
#         batch_size=batch_size, shuffle=True)
#
#     (data, target)=test_loader
#
#     data = data.view(-1, 1,28, 28)
#     data=data[0]
#     viz.images(data.view(-1, 1, 28, 28), win='x')
#     viz.text(data[0].numpy(), win='pred',
#              opts=dict(title='pred'))