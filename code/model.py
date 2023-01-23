from collections import OrderedDict
from typing import Dict

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_class=2, num_sq=8, dim=32):
        super(Net, self).__init__()
        self.num_class = num_class
        self.num_sq = num_sq
        self.dim = dim

        self.block1 = ConvNet(num_classes=-1)
        self.block2 = ConvNet(num_classes=-1)
        self.block3 = ConvNet(num_classes=-1)
        self.block4 = ConvNet(num_classes=-1)
        self.block5 = ConvNet(num_classes=-1)
        self.block6 = ConvNet(num_classes=-1)
        self.block7 = ConvNet(num_classes=-1)
        self.block8 = ConvNet(num_classes=-1)

        self.block1.load_state_dict(torch.load('./trained_models/CBF_cnv5_at_c_32_224.pth',
                                               map_location='cuda:0'), strict=False)
        for k, v in self.block1.named_parameters():
            v.requires_grad = False

        self.block2.load_state_dict(torch.load('./trained_models/CBV_cnv5_at_c_32_224.pth',
                                               map_location='cuda:0'), strict=False)
        for k, v in self.block2.named_parameters():
            v.requires_grad = False

        self.block3.load_state_dict(torch.load('./trained_models/T1_cnv5_at_c_32_224.pth',
                                               map_location='cuda:0'), strict=False)
        for k, v in self.block3.named_parameters():
            v.requires_grad = False

        self.block4.load_state_dict(torch.load('./trained_models/T1c_cnv5_at_c_32_224.pth',
                                               map_location='cuda:0'), strict=False)
        for k, v in self.block4.named_parameters():
            v.requires_grad = False

        self.block5.load_state_dict(torch.load('./trained_models/Flair_cnv5_at_c_32_224.pth',
                                               map_location='cuda:0'), strict=False)
        for k, v in self.block5.named_parameters():
            v.requires_grad = False

        self.block6.load_state_dict(torch.load('./trained_models/T2_cnv5_at_c_32_224.pth',
                                               map_location='cuda:0'), strict=False)
        for k, v in self.block6.named_parameters():
            v.requires_grad = False

        self.block7.load_state_dict(torch.load('./trained_models/rMD_cnv5_at_c_32_224.pth',
                                               map_location='cuda:0'), strict=False)
        for k, v in self.block7.named_parameters():
            v.requires_grad = False

        self.block8.load_state_dict(torch.load('./trained_models/rFA_cnv5_at_c_32_224.pth',
                                               map_location='cuda:0'), strict=False)
        for k, v in self.block8.named_parameters():
            v.requires_grad = False
        self.attention = self_Attention2(dim=self.dim)
        self.layer_norm = nn.LayerNorm(self.dim * self.num_sq)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.dim * self.num_sq, self.dim * self.num_sq)
        self.fc2 = nn.Linear(self.dim * self.num_sq, self.num_class)

    def forward(self, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8):
        size = x_1.shape[0]
        # print(size)
        x_1, _ = self.block1(x_1)
        x_2, _ = self.block2(x_2)
        x_3, _ = self.block3(x_3)
        x_4, _ = self.block4(x_4)
        x_5, _ = self.block5(x_5)
        x_6, _ = self.block6(x_6)
        x_7, _ = self.block7(x_7)
        x_8, _ = self.block8(x_8)

        x = torch.cat((x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8), 1)
        # x = x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8
        x = self.layer_norm(x)

        x = x.reshape(size, self.num_sq, self.dim)
        #
        x = self.attention(x)
        x = x.view(x.size(0), -1)

        # x = self.fc1(x)
        # x = self.relu(x)
        out = self.fc1(x)
        out = self.dropout(out)
        # out = self.sigmod(out)
        feat = out
        out = self.fc2(out)

        # out = self.softmax(x)
        return feat, out


class self_Attention(nn.Module):
    def __init__(self, dim):
        super(self_Attention, self).__init__()
        self.dim = dim
        # self.fc_Q = nn.Linear(self.dim, self.dim, bias=False)
        self.fc_K = nn.Linear(self.dim, self.dim, bias=False)
        self.fc_V = nn.Linear(self.dim, self.dim, bias=False)
        self.fc = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(p=0.5)
        self.layer_norm = nn.LayerNorm(self.dim)

    def forward(self, x1,x2):
        size = x1.shape[0]
        Q = x1
        K = self.fc_K(x2)
        V = self.fc_V(x2)
        scale = K.size(-1) ** -0.5  # 缩放因子
        attention = torch.matmul(Q, K.permute(0, 2, 1)) * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        context = context.view(size, -1, self.dim)
        context = self.fc(context)
        out = self.dropout(context)
        out = out + x1
        out = self.layer_norm(out)
        return out

class self_Attention2(nn.Module):
    def __init__(self, dim):
        super(self_Attention2, self).__init__()
        self.dim = dim
        self.fc_Q = nn.Linear(self.dim, self.dim, bias=False)
        self.fc_K = nn.Linear(self.dim, self.dim, bias=False)
        self.fc_V = nn.Linear(self.dim, self.dim, bias=False)
        self.fc = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(p=0.5)
        self.layer_norm = nn.LayerNorm(self.dim)

    def forward(self, x):
        size = x.shape[0]
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        scale = K.size(-1) ** -0.5  # 缩放因子
        attention = torch.matmul(Q, K.permute(0, 2, 1)) * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        context = context.view(size, -1, self.dim)
        context = self.fc(context)
        out = self.dropout(context)
        out = out + x
        out = self.layer_norm(out)
        return out


class featureMap_Attention(nn.Module):
    def __init__(self):
        super(featureMap_Attention, self).__init__()
        self.mapCov1 = nn.Conv2d(1536, 512, kernel_size=1)
        # self.mapCov2 = nn.Conv2d(512, 1000, kernel_size=1)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), 1)
        x = self.mapCov1(x)
        # x = self.mapCov2(x)
        out = self.avgPool(x)

        return out


class resnet_block(nn.Module):
    def __init__(self):
        super(resnet_block, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        # self.resnet.load_state_dict(torch.load('D:/pyProject/Radiomics-Features-Extractor-main/'
        #                                        'Deep-learning-based-radiomics/8-Resnet50/'
        #                                        'pretrained_model/resnet18-f37072fd.pth'))
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)

        # for k, v in self.resnet.named_parameters():
        #     if k != 'fc.weight' and k != 'fc.bias' and k != 'conv1':
        #         v.requires_grad = False

    def forward(self, x):
        feat, out = self.resnet(x)
        return out


class ConvNet2(nn.Module):
    def __init__(self, num_classes=-1):
        super(ConvNet2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            # nn.BatchNorm2d(32, momentum=1, affine=True, track_running_stats=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.avgpool3 = nn.AdaptiveAvgPool2d(1)
        self.avgpool4 = nn.AdaptiveAvgPool2d(1)
        self.avgpool5 = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.5)

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.classifier = nn.Linear(64, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, is_feat=True):
        out = self.layer1(x)
        f0 = self.avgpool1(out)
        # print(f0.shape)
        out = self.layer2(out)
        f1 = self.avgpool2(out)
        # print(f1.shape)
        out = self.layer3(out)
        f2 = self.avgpool3(out)
        # print(f2.shape)
        out = self.layer4(out)
        f3 = self.avgpool4(out)
        # print(f3.shape)
        out = self.avgpool5(out)
        # print(out.shape)
        out = torch.cat([out, f0, f1, f2, f2], 1)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.drop(out)
        feat = out

        if self.num_classes > 0:
            out = self.classifier(out)

        if is_feat:
            return feat, out
        else:
            return out


class Conv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1,
                 padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias, groups=groups)
        )

    def forward(self, x):
        return self.net(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x