# 分类为10
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import argparse
import os

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from thop import profile
from tensorboardX import SummaryWriter

import timm

ViT0 = timm.create_model('vit_small_resnet26d_224', pretrained=False)
ViT1 = timm.create_model('vit_small_resnet26d_224', pretrained=True)

SumWriter = SummaryWriter(log_dir="transformer_10_1")
dummyinput = torch.rand(12,3,32,32)
SumWriter.add_graph(ViT0 , dummyinput ,verbose = True)


#flops
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear1 = nn.Linear(512 * block.expansion, num_classes)
        #self.linear2 = nn.Linear(128 * block.expansion, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = self.pool(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = nn.Dropout(p=0.5)(out)
        out = self.linear1(out)
        # out = self.linear2(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Res_Net(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(Res_Net, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=4)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = self.pool(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = nn.Dropout(p=0.5)(out)
        out = self.linear(out)
        return out

def ResNet152():
    return  Res_Net(Bottleneck, [3, 8, 36, 3])

# 超参数设置
EPOCH = 150  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128  # 批处理尺寸(batch_size)
LR = 0.01  # 学习率

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf1', default='./model1_10', help='folder to output images and model checkpoints')
parser.add_argument('--outf2', default='./model2_10', help='folder to output images and model checkpoints')
parser.add_argument('--outf3', default='./model3_10', help='folder to output images and model checkpoints')
parser.add_argument('--outf4', default='./model4_10', help='folder to output images and model checkpoints')# 输出结果保存路径
args = parser.parse_args()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)  # 训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ViT0
net1 = ViT0.to(device)
net2 = ViT1.to(device)
net3 = ResNet34().to(device)
net4 = ResNet152().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
# 每个网络对应各自的优化器
optimizer1 = optim.SGD(net1.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)


print_step = 100  # 100次迭代后输出损失

# 训练
#resnet
if __name__ == "__main__":
    if not os.path.exists(args.outf1):
        os.makedirs(args.outf1)
    best_acc1 = 50  #2 初始化best test accuracy
    print("Start Training, ViT0!")  # 定义遍历数据集的次数
    with open("acc_ViT0.txt", "w") as f_1:
        with open("log_ViT0.txt", "w")as f2_1:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net1.train()
                sum_loss1 = 0.0
                correct1 = 0.0
                total1 = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer1.zero_grad()

                    # forward + backward
                    outputs1 = net1(inputs)
                    loss1 = criterion(outputs1, labels)
                    loss1.backward()
                    optimizer1.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss1 += loss1.item()

                    #test_loss_all.append()

                    _, predicted1 = torch.max(outputs1.data, 1)
                    total1 += labels.size(0)
                    correct1 += predicted1.eq(labels.data).cpu().sum()
                    niter1 = i + 1 + epoch * length
                    train_acc1 = 100. * correct1 / total1
                    train_loss1 = sum_loss1 / (i+1)
                    print('[epoch:%d, iter:%d] Loss: %.03f_1 | Acc: %.3f_1%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss1 / (i + 1), 100. * correct1 / total1))
                    f2_1.write('%03d  %05d |Loss: %.03f_1 | Acc: %.3f_1%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss1 / (i + 1), 100. * correct1 / total1))
                    f2_1.write('\n')
                    f2_1.flush()
                    if niter1 % print_step == 0:
                        # 为日志添加损失函数
                        SumWriter.add_scalar("train_loss_ViT0", train_loss1 , niter1)
                        testinputs1_im = vutils.make_grid(inputs, nrow=12)  # (数据预处理？)
                        SumWriter.add_image("input image sample_ViT0", testinputs1_im, niter1)
                        testout1_im = vutils.make_grid(outputs1, nrow=128)
                        SumWriter.add_image("output image sample_ViT0", testout1_im, niter1)

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct1 = 0
                    total1 = 0
                    for data in testloader:
                        net1.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs1 = net1(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted1 = torch.max(outputs1.data, 1)
                        total1 += labels.size(0)
                        correct1 += (predicted1 == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct1 / total1))
                    acc1 = 100. * correct1 / total1
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net1.state_dict(), '%s/net_%03d.pth' % (args.outf1, epoch + 1))
                    f_1.write("EPOCH=%03d  |Accuracy= %.3f_1%%" % (epoch + 1,acc1))
                    f_1.write('\n')
                    f_1.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc1 > best_acc1:
                        f3_1 = open("best_acc_resnet.txt", "w")
                        f3_1.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc1))
                        f3_1.close()
                        best_acc1 = acc1
                SumWriter.add_scalars("train/test_acc_epoch_ViT0", {'train_acc': train_acc1,'test_acc':acc1}, epoch)
                SumWriter.add_scalars("train_acc_epoch",{'ViT0':train_acc1},epoch)
                SumWriter.add_scalars("test_acc_epoch", {'ViT0': acc1}, epoch)
                SumWriter.add_scalars("train_loss_epoch", {'ViT0':train_loss1}, epoch)
                testout1_im = vutils.make_grid(outputs1, nrow=12)
                SumWriter.add_image("output image sample epoch_ViT0", testout1_im,global_step=epoch, dataformats='CHW')
            print("Training Finished, TotalEPOCH=%d" % EPOCH)

#net0

optimizer2 = optim.SGD(net2.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

if __name__ == "__main__":
    if not os.path.exists(args.outf2):
        os.makedirs(args.outf2)
    best_acc2 = 70  #2 初始化best test accuracy
    print("Start Training, ViT1!")  # 定义遍历数据集的次数
    with open("acc_ViT1.txt", "w") as f:
        with open("log_ViT1.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net2.train()
                sum_loss2 = 0.0
                correct2 = 0.0
                total2 = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer2.zero_grad()

                    # forward + backward
                    outputs2 = net2(inputs)
                    loss2 = criterion(outputs2, labels)
                    loss2.backward()
                    optimizer2.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss2 += loss2.item()

                    #test_loss_all.append()

                    _, predicted2 = torch.max(outputs2.data, 1)
                    total2 += labels.size(0)
                    correct2 += predicted2.eq(labels.data).cpu().sum()
                    niter2 = i + 1 + epoch * length
                    train_acc2 = 100. * correct2 / total2
                    train_loss2 = sum_loss2 / (i+1)
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss2 / (i + 1), 100. * correct2 / total2))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss2 / (i + 1), 100. * correct2 / total2))
                    f2.write('\n')
                    f2.flush()
                    if niter2 % print_step == 0:
                        # 为日志添加损失函数
                        SumWriter.add_scalar("train_loss_ViT1", train_loss2 , niter2)
                        testinputs2_im = vutils.make_grid(inputs, nrow=12)  # (数据预处理？)
                        SumWriter.add_image("input image sample_ViT1", testinputs2_im, niter2)
                        testout2_im = vutils.make_grid(outputs2, nrow=128)
                        SumWriter.add_image("output image sample_ViT1", testout2_im, niter2)

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct2 = 0
                    total2 = 0
                    for data in testloader:
                        net2.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs2 = net2(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted2 = torch.max(outputs2.data, 1)
                        total2 += labels.size(0)
                        correct2 += (predicted2 == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct2 / total2))
                    acc2 = 100. * correct2 / total2
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net2.state_dict(), '%s/net_%03d.pth' % (args.outf2, epoch + 1))
                    f.write("EPOCH=%03d  |Accuracy= %.3f%%" % (epoch + 1,acc2))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc2 > best_acc2:
                        f3 = open("best_acc0_1.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc2))
                        f3.close()
                        best_acc2 = acc2
                SumWriter.add_scalars("train/test_acc_epoch_ViT1",{'train_acc':train_acc2,'test_acc':acc2},epoch)
                SumWriter.add_scalars("train_acc_epoch", {'ViT1': train_acc2}, epoch)
                SumWriter.add_scalars("test_acc_epoch", {'ViT1': acc2}, epoch)
                SumWriter.add_scalars("train_loss_epoch", {'ViT1':train_loss2}, epoch)
                testout2_im = vutils.make_grid(outputs2, nrow=12)
                SumWriter.add_image("output image sample epoch_ViT1", testout2_im,global_step=epoch, dataformats='CHW')
            print("Training Finished, TotalEPOCH=%d" % EPOCH)

#net1
optimizer3 = optim.SGD(net3.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
if __name__ == "__main__":
    if not os.path.exists(args.outf3):
        os.makedirs(args.outf3)
    best_acc3 = 70  #2 初始化best test accuracy
    print("Start Training, CNN_net1!")  # 定义遍历数据集的次数
    with open("acc1_1.txt", "w") as f_3:
        with open("log1_1.txt", "w")as f2_3:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net3.train()
                sum_loss3 = 0.0
                correct3 = 0.0
                total3 = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer3.zero_grad()

                    # forward + backward
                    outputs3 = net3(inputs)
                    loss3 = criterion(outputs3, labels)
                    loss3.backward()
                    optimizer3.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss3 += loss3.item()

                    #test_loss_all.append()

                    _, predicted3 = torch.max(outputs3.data, 1)
                    total3 += labels.size(0)
                    correct3 += predicted3.eq(labels.data).cpu().sum()
                    niter3 = i + 1 + epoch * length
                    train_acc3 = 100. * correct3 / total3
                    train_loss3 = sum_loss3 / (i+1)
                    print('[epoch:%d, iter:%d] Loss: %.03f_3 | Acc: %.3f_3%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss3 / (i + 1), 100. * correct3 / total3))
                    f2_3.write('%03d  %05d |Loss: %.03f_3 | Acc: %.3f_3%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss3 / (i + 1), 100. * correct3 / total3))
                    f2_3.write('\n')
                    f2_3.flush()
                    if niter3 % print_step == 0:
                        # 为日志添加损失函数
                        SumWriter.add_scalar("train_loss_net1", train_loss3 , niter3)
                        testinputs3_im = vutils.make_grid(inputs, nrow=12)  # (数据预处理？)
                        SumWriter.add_image("input image sample_net1", testinputs3_im, niter3)
                        testout3_im = vutils.make_grid(outputs3, nrow=128)
                        SumWriter.add_image("output image sample_net1", testout3_im, niter3)

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct3 = 0
                    total3 = 0
                    for data in testloader:
                        net3.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs3 = net3(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted3 = torch.max(outputs3.data, 1)
                        total3 += labels.size(0)
                        correct3 += (predicted3 == labels).sum()
                    print('测试分类准确率为：%.3f_3%%' % (100 * correct3 / total3))
                    acc3 = 100. * correct3 / total3
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net3.state_dict(), '%s/net_%03d.pth' % (args.outf3, epoch + 1))
                    f_3.write("EPOCH=%03d  |Accuracy= %.3f%%" % (epoch + 1,acc3))
                    f_3.write('\n')
                    f_3.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc3 > best_acc3:
                        f3_3 = open("best_acc1_1.txt", "w")
                        f3_3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc3))
                        f3_3.close()
                        best_acc3 = acc3
                SumWriter.add_scalars("train/test_acc_epoch_net1",{'train_acc':train_acc3,'test_acc':acc3},epoch)
                SumWriter.add_scalars("train_acc_epoch", {'net_1': train_acc3}, epoch)
                SumWriter.add_scalars("test_acc_epoch", {'net_1': acc3}, epoch)
                SumWriter.add_scalars("train_loss_epoch", {'net_1':train_loss3}, epoch)
                testout3_im = vutils.make_grid(outputs3, nrow=12)
                SumWriter.add_image("output image sample epoch_net1", testout3_im,global_step=epoch, dataformats='CHW')
            print("Training Finished, TotalEPOCH=%d" % EPOCH)

#第四个网络
optimizer4 = optim.SGD(net4.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
if __name__ == "__main__":
    if not os.path.exists(args.outf4): #3
        os.makedirs(args.outf4)
    best_acc4 = 70  #2 初始化best test accuracy
    print("Start Training, CNN_net2!")  # 定义遍历数据集的次数
    with open("acc1_2.txt", "w") as f_4:
        with open("log1_2.txt", "w")as f2_4:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net4.train()
                sum_loss4 = 0.0
                correct4 = 0.0
                total4 = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer4.zero_grad()

                    # forward + backward
                    outputs4 = net4(inputs)
                    loss4 = criterion(outputs4, labels)
                    loss4.backward()
                    optimizer4.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss4 += loss4.item()

                    #test_loss_all.append()

                    _, predicted4 = torch.max(outputs4.data, 1)
                    total4 += labels.size(0)
                    correct4 += predicted4.eq(labels.data).cpu().sum()
                    niter4 = i + 1 + epoch * length
                    train_acc4 = 100. * correct4 / total4
                    train_loss4 = sum_loss4 / (i+1)
                    print('[epoch:%d, iter:%d] Loss: %.03f_3 | Acc: %.3f_3%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss4 / (i + 1), 100. * correct4 / total4))
                    f2_4.write('%03d  %05d |Loss: %.03f_3 | Acc: %.3f_3%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss4 / (i + 1), 100. * correct4 / total4))
                    f2_4.write('\n')
                    f2_4.flush()
                    if niter4 % print_step == 0:
                        # 为日志添加损失函数
                        SumWriter.add_scalar("train_loss_net2", train_loss4 , niter4)
                        testinputs4_im = vutils.make_grid(inputs, nrow=12)  # (数据预处理？)
                        SumWriter.add_image("input image sample_net2", testinputs4_im, niter4)
                        testout4_im = vutils.make_grid(outputs4, nrow=128)
                        SumWriter.add_image("output image sample_net2", testout4_im, niter4)

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct4 = 0
                    total4 = 0
                    for data in testloader:
                        net4.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs4 = net4(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted4 = torch.max(outputs4.data, 1)
                        total4 += labels.size(0)
                        correct4 += (predicted4 == labels).sum()
                    print('测试分类准确率为：%.3f_3%%' % (100 * correct4 / total4))
                    acc4 = 100. * correct4 / total4
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net4.state_dict(), '%s/net_%03d.pth' % (args.outf4, epoch + 1))
                    f_4.write("EPOCH=%03d  |Accuracy= %.3f%%" % (epoch + 1,acc4))
                    f_4.write('\n')
                    f_4.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc4 > best_acc4:
                        f3_4 = open("best_acc1_2.txt", "w")
                        f3_4.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc3))
                        f3_4.close()
                        best_acc4 = acc4
                SumWriter.add_scalars("train/test_acc_epoch_net2",{'train_acc':train_acc4,'test_acc':acc4},epoch)
                SumWriter.add_scalars("train_acc_epoch", {'net_2': train_acc4}, epoch)
                SumWriter.add_scalars("test_acc_epoch", {'net_2': acc4}, epoch)
                SumWriter.add_scalars("train_loss_epoch", {'net_2':train_loss4}, epoch)
                testout4_im = vutils.make_grid(outputs4, nrow=12)
                SumWriter.add_image("output image sample epoch_net2", testout4_im,global_step=epoch, dataformats='CHW')
            print("Training Finished, TotalEPOCH=%d" % EPOCH)