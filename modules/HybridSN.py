import torch
import torch.nn as nn
import torch.nn.functional as F

windowSize = 25
K = 30  # 参考Hybrid-Spectral-Net
rate = 16


class SSSE_test(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes):
        super(SSSE_test, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='zeros',
                               bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='zeros',
                               bias=True)
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 128, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(128)
        self.activation4 = nn.ReLU()
        if patch_size == 7:
            self.SpectralSE = SpectralSE(128, 128, self.sz)
        elif patch_size == 3:
            self.SpectralSE = SpectralSE_three(128, 128, self.sz)
        elif patch_size == 5:
            self.SpectralSE = SpectralSE_five(128, 128, self.sz)
        elif patch_size == 9:
            self.SpectralSE = SpectralSE_nine(128, 128, self.sz)
        elif patch_size == 11:
            self.SpectralSE = SpectralSE_eleven(128, 128, self.sz)
        else:
            raise Exception('this patch not define')
        # self.SpectralSE = SpectralSE_R(128, 128, self.sz)
        # self.SpectralSE = SpectralSE_S(128, 128, self.sz)

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='replicate',
                               bias=True)
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='replicate',
                               bias=True)
        self.bn7 = nn.BatchNorm3d(24)
        self.activation7 = nn.ReLU()
        self.SpatialSE = SpatialSE(24, 1)
        self.conv8 = nn.Conv3d(24, 24, kernel_size=1)
        # Finish

        # Combination shape
        self.inter_size = 128 + 24

        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               padding_mode='replicate', bias=True)
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                padding_mode='replicate', bias=True)
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))
        # self.avgpool = nn.Conv2d(in_channels=self.inter_size,out_channels=self.inter_size,kernel_size=self.sz)

        # Fully connected Layer
        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, bounds=None):
        # Convolution layer 1
        x = x.unsqueeze(1)
        # spatial
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))
        e1 = self.SpectralSE(x1)
        x1 = torch.mul(e1, x1)

        # spatial
        x2 = self.conv5(x)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)
        x2 = self.conv6(x2)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))
        e2 = self.SpatialSE(x2)
        x2 = torch.mul(e2, x2)

        # concat spatial and spectral information
        x = torch.cat((x1, x2), 1)

        x = self.avgpool(x)
        x = x.reshape((x.size(0), -1))

        # Fully connected layer
        # x = self.fc1(x)

        return x


class SSSE_Houston(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes):
        super(SSSE_Houston, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='zeros',
                               bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='zeros',
                               bias=True)
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 128, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(128)
        self.activation4 = nn.ReLU()
        if patch_size == 7:
            self.SpectralSE = SpectralSE(128, 128, self.sz)
        elif patch_size == 3:
            self.SpectralSE = SpectralSE_three(128, 128, self.sz)
        elif patch_size == 5:
            self.SpectralSE = SpectralSE_five(128, 128, self.sz)
        elif patch_size == 9:
            self.SpectralSE = SpectralSE_nine(128, 128, self.sz)
        elif patch_size == 11:
            self.SpectralSE = SpectralSE_eleven(128, 128, self.sz)
        elif patch_size == 13:
            self.SpectralSE = SpectralSE_thirteen(128, 128, self.sz)
        else:
            raise Exception('this patch not define')
        # self.SpectralSE = SpectralSE_R(128, 128, self.sz)
        # self.SpectralSE = SpectralSE_S(128, 128, self.sz)

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='zeros',
                               bias=True)
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='zeros',
                               bias=True)
        self.bn7 = nn.BatchNorm3d(24)
        self.activation7 = nn.ReLU()
        self.SpatialSE = SpatialSE(24, 1)
        self.conv8 = nn.Conv3d(24, 24, kernel_size=1)
        # Finish

        # Combination shape
        self.inter_size = 128 + 24

        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               padding_mode='zeros', bias=True)
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                padding_mode='zeros', bias=True)
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))
        # self.avgpool = nn.Conv2d(in_channels=self.inter_size,out_channels=self.inter_size,kernel_size=self.sz)

        # Fully connected Layer
        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, bounds=None):
        # Convolution layer 1
        x = x.unsqueeze(1)
        # spatial
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))
        e1 = self.SpectralSE(x1)
        x1 = torch.mul(e1, x1)

        # spatial
        x2 = self.conv5(x)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)
        x2 = self.conv6(x2)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))
        e2 = self.SpatialSE(x2)
        x2 = torch.mul(e2, x2)

        # concat spatial and spectral information
        x = torch.cat((x1, x2), 1)

        x = self.avgpool(x)
        x = x.reshape((x.size(0), -1))

        # Fully connected layer
        # x = self.fc1(x)

        return x


class SSSE_Contrastive(nn.Module):
    def __init__(self, input_channels, patch_size, n_classes, is_houston=False, hide_dim=512, projection_dim=128):
        super(SSSE_Contrastive, self).__init__()
        # 原有 SSSE_Houston 结构
        if is_houston:
            self.ssse = SSSE_Houston(input_channels, patch_size, n_classes)
        else:
            self.ssse = SSSE_test(input_channels, patch_size, n_classes)
        # self.ssse_288 = SSSE_288(input_channels, patch_size, n_classes)
        # self.ssse_csr = SSSE_with_CSR(input_channels, patch_size, n_classes)

        # 添加投影头
        self.projection_head = nn.Sequential(
            nn.Linear(self.ssse.inter_size, hide_dim),
            nn.BatchNorm1d(hide_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hide_dim, projection_dim)
        )

    def forward(self, x):
        features = self.ssse(x)
        # features = self.ssse_288(x)
        projections = self.projection_head(features)
        projections = F.normalize(projections, dim=1)
        return features, projections


class SpatialSE(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, out_channel):
        super(SpatialSE, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        C = x.shape[1]
        x = self.conv(x)
        x = torch.sigmoid(x)
        out = x.repeat(1, C, 1, 1)
        return out


# patch size 7
class SpectralSE(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C // 4, kernel_size=3, stride=1, padding=3)
        self.conv2 = nn.Conv2d(C // 4, C, kernel_size=3, stride=1, padding=2)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out


# patch_size = 5
class SpectralSE_five(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE_five, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C // 4, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(C // 4, C, kernel_size=3, stride=1, padding=2)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out


# patch_size = 9
class SpectralSE_nine(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE_nine, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C // 4, kernel_size=3, stride=1, padding=3)
        self.conv2 = nn.Conv2d(C // 4, C, kernel_size=3, stride=1, padding=3)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out


# patch_size = 11
class SpectralSE_eleven(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE_eleven, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C // 4, kernel_size=3, stride=1, padding=3)
        self.conv2 = nn.Conv2d(C // 4, C, kernel_size=3, stride=1, padding=4)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out


class SpectralSE_thirteen(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE_thirteen, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C // 4, kernel_size=3, stride=1, padding=4)
        self.conv2 = nn.Conv2d(C // 4, C, kernel_size=3, stride=1, padding=4)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out


# patch_size = 3
class SpectralSE_three(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE_three, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C // 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(C // 4, C, kernel_size=3, stride=1, padding=2)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out
