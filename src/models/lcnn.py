import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AngleLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        
        self.iter = 0
        self.lambda_min = 5.0
        self.lambda_max = 1500.0

    def forward(self, x, target=None):
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        
        cos_theta = F.linear(x, weight_norm) / x_norm
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if target is None:
            return x_norm * cos_theta

        cos_4_theta = 8 * (cos_theta ** 4) - 8 * (cos_theta ** 2) + 1
        theta = cos_theta.acos()
        k = (4 * theta / math.pi).floor()
        phi_theta = ((-1.0) ** k) * cos_4_theta - 2 * k

        self.iter += 1
        lamb = max(self.lambda_min, self.lambda_max / (1 + 0.1 * self.iter))
        
        output = x_norm * cos_theta
        batch_size = x.size(0)
        target_cos = cos_theta[torch.arange(batch_size), target]
        target_phi = phi_theta[torch.arange(batch_size), target]
        
        output[torch.arange(batch_size), target] = x_norm.squeeze(1) * (
            target_phi + lamb * target_cos
        ) / (1 + lamb)
        
        return output


class MFM(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, is_conv=True):
        super(MFM, self).__init__()
        self.is_conv = is_conv
        if is_conv:
            self.filter = nn.Conv2d(
                in_channels, out_channels * 2, kernel_size, stride, padding
            )
        else:
            self.filter = nn.Linear(in_channels, out_channels * 2)

    def forward(self, x):
        x = self.filter(x)
        out1, out2 = torch.chunk(x, 2, dim=1 if self.is_conv else -1)
        return torch.max(out1, out2)


class LCNN(nn.Module):

    def __init__(self, num_classes=2):
        super(LCNN, self).__init__()
        self.layer1 = nn.Sequential(
            MFM(1, 32, kernel_size=5, stride=1, padding=2, is_conv=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            MFM(32, 32, kernel_size=1, stride=1, padding=0, is_conv=True),
            nn.BatchNorm2d(32),
            MFM(32, 48, kernel_size=3, stride=1, padding=1, is_conv=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48)
        )
        self.layer3 = nn.Sequential(
            MFM(48, 48, kernel_size=1, stride=1, padding=0, is_conv=True),
            nn.BatchNorm2d(48),
            MFM(48, 64, kernel_size=3, stride=1, padding=1, is_conv=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            MFM(64, 64, kernel_size=1, stride=1, padding=0, is_conv=True),
            nn.BatchNorm2d(64),
            MFM(64, 32, kernel_size=3, stride=1, padding=1, is_conv=True),
            nn.BatchNorm2d(32)
        )
        self.layer5 = nn.Sequential(
            MFM(32, 32, kernel_size=1, stride=1, padding=0, is_conv=True),
            nn.BatchNorm2d(32),
            MFM(32, 32, kernel_size=3, stride=1, padding=1, is_conv=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc1 = MFM(32 * 53 * 37, 80, is_conv=False)
        self.dropout = nn.Dropout(p=0.75)
        self.bn_fc = nn.BatchNorm1d(80)
        self.classifier = AngleLinear(80, num_classes)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, target=None, return_embeddings=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        embeddings = self.bn_fc(x)
        
        logits = self.classifier(embeddings, target)
        
        if return_embeddings:
            return logits, embeddings
        return logits