# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data_process import dataAugmentation
from data_process import data_set_load
from torch.optim.lr_scheduler import StepLR
import pickle
import torch.utils.data as data
import numpy as np
import math


class FerSet(data.Dataset):
    def __init__(self, data, labels):
        super(FerSet, self).__init__()
        print "Data set is creating..."
        self.data = data.astype(np.float32)
        self.label = labels
        self.data /= 255.
        print "Done!"

    def __getitem__(self, item):
        return self.data[item, :, :, :], self.label[item]

    def __len__(self):
        return self.data.shape[0]


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=7):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(6, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.softmax(x, dim=1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels.data).sum()
    return correct


def validate(net, loader, use_cuda=False):
    correct_count = 0.
    count = 0.
    if use_cuda:
        net = net.cuda()
    for i, (b_x, b_y) in enumerate(loader, 0):
        size = b_x.shape[0]
        b_x = Variable(b_x)
        b_y = Variable(b_y)
        if use_cuda:
            b_x = b_x.cuda()
            b_y = b_y.cuda()
        outputs = net(b_x)
        c = accuracy(outputs, b_y)
        correct_count += c
        count += size
    acc = correct_count.item() / float(count)
    return acc


def train():
    lr = 0.01
    batch_size = 128
    use_cuda = True
    epochs = 50
    if torch.cuda.is_available() is False:
        use_cuda = False
    # 这里改成任意的网络
    net = ResNet(Bottleneck, [3, 3, 3, 3])
    if use_cuda:
        net = net.cuda()
    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.00001)
    # 定义loss函数
    loss_function = nn.CrossEntropyLoss()
    # 训练集扩增
    # sampled：扩展两圈0，随机采样；rotate：水平和垂直镜像；scale：尺度变换；noise：椒盐噪声
    train_data, train_labels = dataAugmentation('fer2013_test_a.p', sampled=False, rotate=True, scale=True, noise=False)
    train_set = FerSet(train_data, train_labels)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # 学习速率调节器
    lr_scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
    # 测试数据集
    test_data, test_labels, validation_data, validation_labels = data_set_load('fer2013_test_a.p', is_validate=True)
    validating_set = FerSet(validation_data, validation_labels)
    validation_loader = DataLoader(validating_set, batch_size=batch_size)
    # 开始训练
    loss_save = []
    tacc_save = []
    vacc_save = []
    for epoch in range(epochs):
        lr_scheduler.step()
        running_loss = 0.0
        correct_count = 0.
        count = 0
        for i, (b_x, b_y) in enumerate(train_loader):
            size = b_x.shape[0]
            b_x = Variable(b_x)
            b_y = Variable(b_y)
            if use_cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            # 前向
            outputs = net(b_x)
            # 计算误差
            loss = loss_function(outputs, b_y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 参数更新
            optimizer.step()
            # 计算loss
            running_loss += loss.item()
            count += size
            correct_count += accuracy(outputs, b_y).item()
            if (i + 1) % 10 == 0:
                acc = validate(net, validation_loader, True)
                print('[ %d-%d ] loss: %.9f, \n'
                      'training accuracy: %.6f, \n'
                      'validating accuracy: %.6f' % (
                      epoch + 1, i + 1, running_loss / count, correct_count / count, acc))
                tacc_save.append(correct_count / count)
                loss_save.append(running_loss / count)
                vacc_save.append(acc)
        if (epoch + 1) % 5 == 0:
            print "save"
            torch.save(net.state_dict(), '../model/expression_net{}.p'.format(epoch + 1))
    dic = {}
    dic['loss'] = loss_save
    dic['training_accuracy'] = tacc_save
    dic['validating_accuracy'] = vacc_save
    with open('../model/record.p', 'wb') as f:
        pickle.dump(dic, f)

if __name__ == "__main__":
    train()