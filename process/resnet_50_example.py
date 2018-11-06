# -*-coding:utf-8-*-
import torch
import sys
sys.path.append('./../')
# from data.data_loader import *
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch.autograd import Variable
import os
import cv2
import numpy as np
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

class FerSet(data.Dataset):
    def __init__(self, path=None, is_align=True):
        super(FerSet, self).__init__()
        assert path is not None
        self.path = path
        self.setList = os.listdir(path)
        self.setLength = len(self.setList)
        # predictor = '../model/shape_predictor_68_face_landmarks.dat'
        # detector = '../model/haarcascade_frontalface_default.xml'
        # self.faceAlign = process.FaceAlign(predictor, detector)
        # self.is_align = is_align

    def __getitem__(self, item):
        label = self.setList[item].split('_')[-1][0] # the laset word: 0.bmp, the first word:0
        imgPath = os.path.join(self.path, self.setList[item])
        # data = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        data = np.loadtxt(imgPath, dtype=np.float32)
        # alignment
        # if self.is_align:
        #     bb = dlib.rectangle(0, 0, 48, 48)
        #     landmarks = self.faceAlign.predict(data, bb)
        #     data = self.faceAlign.align(data, landmarks)
        return torch.Tensor([data.astype(np.float32)]), eval(label)

    def __len__(self):
        return self.setLength


class NaiveNet(nn.Module):
    def __init__(self):
        super(NaiveNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=48,
            kernel_size=3,
            stride=1,
            padding=1)
        self.f1 = nn.Linear(
            in_features=6*6*48,
            out_features=512
        )
        self.f2 = nn.Linear(
            in_features=512,
            out_features=64
        )
        self.f3 = nn.Linear(
            in_features=64,
            out_features=7
        )

    def forward(self, x):
        #
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        out = self.f3(self.f2(self.f1(x)))
        return F.softmax(out, dim=1)

# begining
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

        return x	


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
		
		
# input : 48*48
# output : 7
def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

# ending
expressionNet = resnet50()
use_cuda = True
if torch.cuda.is_available() is False:
    use_cuda = False
if use_cuda:
    expressionNet = expressionNet.cuda()

def train():
    expressionNet.load_state_dict(torch.load('./../model/example_model/example_model_115_.pkl'))
    expressionNet.train()
    batch_size = 128
    lr_1 = 0.01
    epochs_1 = 101
    lr_2 = 0.001
    epochs_2 = 21
    # 定义优化器
    optimizer = torch.optim.SGD(expressionNet.parameters(), lr=lr_1, momentum=0.9)
    # 定义loss函数
    loss_function = nn.CrossEntropyLoss()
    # 训练集
    trainPath = '../data/fer2013/train_aligned/'
    train_set = FerSet(path=trainPath, is_align=True)
    train_loader = DataLoader(train_set, batch_size=batch_size)

    # 开始训练
    for epoch in range(epochs_1):
        loss_total = 0
        for i, (b_x, b_y) in enumerate(train_loader):
            b_x = Variable(b_x)
            b_y = Variable(b_y)
            if use_cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            # 前向
            out = expressionNet(b_x)
            # 计算误差
            loss = loss_function(out, b_y)
            loss_total = loss_total + float(loss)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 参数更新
            optimizer.step()
        loss_total /= len(train_loader.dataset)
        loss_total *= 128
        print('current epoch: {0}  loss: {1}'.format(epoch, loss_total))
        if epoch % 10 == 0:
            print('first train: the {0} epoch loss is {1}'.format(epoch, loss_total))
            torch.save(expressionNet.state_dict(), './../model/example_model/example_model_'+str(epoch+115)+'_.pkl')

    # 定义优化器
    optimizer = torch.optim.SGD(expressionNet.parameters(), lr=lr_2, momentum=0.9)
    # 开始训练
    for epoch in range(epochs_2):
        loss_total = 0
        for i, (b_x, b_y) in enumerate(train_loader):
            b_x = Variable(b_x)
            b_y = Variable(b_y)
            if use_cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            # 前向
            out = expressionNet(b_x)
            # 计算误差
            loss = loss_function(out, b_y)
            loss_total += loss
            # print b_y
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 参数更新
            optimizer.step()
        loss_total /= len(train_loader.dataset)
        loss_total *= 128
        print('current epoch: {0}  loss: {1}'.format(epoch, loss_total))
        if epoch % 5 == 0:
            print('finetune train: the {0} epoch loss is {1}'.format(epoch, loss_total))
            # save model
            torch.save(expressionNet.state_dict(), './../model/example_model/example_model_'+str(epoch+epochs_1+115)+'_.pkl')

def test():
    # 测试数据集
    test_loss = 0
    correct = 0
    # load model
    # expressionNet = resnet50()
    # use_cuda = True
    # if torch.cuda.is_available() is False:
    #     use_cuda = False
    #     print(use_cuda)
    # if use_cuda:
    #     expressionNet = expressionNet.cuda()
    # print(use_cuda)
    expressionNet.load_state_dict(torch.load('./../model/example_model/example_model_235_.pkl'))
    expressionNet.eval()
    testPath = '../data/fer2013/test_aligned/'
    test_set = FerSet(path=testPath, is_align=True)
    test_loader = DataLoader(test_set, batch_size=1)
    loss_function = nn.CrossEntropyLoss()
    for i, (data, target) in enumerate(test_loader):
        data, target = Variable(data), Variable(target)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # print(expressionNet(data))
        output = expressionNet(data)
        test_loss += float(loss_function(output, target))
        # print(output)
        # print(target)
        pred1, pred2 = output.data.max(1, keepdim=True)
        # print(pred2)
        # print(target)
        if pred2 == target:
            correct = correct + 1
        # if i == 1000:
        #     break

        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /=len(test_loader.dataset)
    print('\n test_set: Average_loss:{:.4f}, Accurracy: {}/{}({:.0f})%)\n').format(
        test_loss, correct, len(test_loader.dataset), 100.*correct/i)


if __name__ == '__main__':
    train()
    test()