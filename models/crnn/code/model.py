import torch
import torch.nn as nn
import torchvision.models as models

class CRNN(nn.Module):
    def __init__(self, num_classes=11, rnn_hidden_size=256):
        super(CRNN, self).__init__()
        
        # 使用ResNet18作为骨干网络，提取特征
        resnet = models.resnet18(pretrained=True)
        self.conv_layers = nn.Sequential(*list(resnet.children())[:-2])  # 移除最后两层
        
        # 自适应池化层，将特征图高度固定为1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # BiLSTM层
        self.rnn = nn.LSTM(
            input_size=512,          # ResNet18的最后一个卷积层输出通道数
            hidden_size=rnn_hidden_size,
            bidirectional=True,
            num_layers=2,
            dropout=0.5,
            batch_first=True
        )
        
        # 分类层
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)  # 双向LSTM的输出通道数为hidden_size*2
    
    def forward(self, x):
        # 卷积层提取特征
        features = self.conv_layers(x)  # 输出形状: (batch_size, 512, H, W)
        
        # 自适应池化，将H变为1
        pooled_features = self.adaptive_pool(features)  # 输出形状: (batch_size, 512, 1, W)
        pooled_features = pooled_features.squeeze(2)  # 输出形状: (batch_size, 512, W)
        
        # 调整形状以适应RNN输入
        pooled_features = pooled_features.permute(2, 0, 1)  # 输出形状: (W, batch_size, 512)
        
        # LSTM层
        lstm_out, _ = self.rnn(pooled_features)  # 输出形状: (W, batch_size, rnn_hidden_size*2)
        
        # 分类层
        logits = self.fc(lstm_out)  # 输出形状: (W, batch_size, num_classes)
        
        return logits





