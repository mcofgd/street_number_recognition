import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def print_tensor_stats(name, tensor):
    """打印张量的统计信息"""
    if not torch.isfinite(tensor).all():
        print(f"[ERROR] {name} contains NaN or Inf")

    if torch.is_floating_point(tensor):
        print(f"{name} stats: min={tensor.min().item():.4f}, "
              f"max={tensor.max().item():.4f}, "
              f"mean={tensor.mean().item():.4f}")
    else:
        print(f"{name} is not a floating-point tensor, skipping mean calculation.")


class SimpleCharClassifier(nn.Module):
    def __init__(self, num_classes=10):  # 注意这里去掉了blank类
        super(SimpleCharClassifier, self).__init__()

        # 使用 ResNet18 作为骨干网络
        resnet = torchvision.models.resnet18(pretrained=True)

        # 替换最后两层（去掉 avgpool 和 fc）
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # 自定义全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

        # 初始化分类层
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        :param x: shape [B, C, H, W]
        :return: logits [B, num_classes]
        """
        features = self.backbone(x)  # [B, 512, H', W']
        pooled = self.global_pool(features)  # [B, 512, 1, 1]
        logits = self.classifier(pooled)  # [B, num_classes]

        # 检查点
        print_tensor_stats("CNN特征", features)
        print_tensor_stats("分类logits", logits)

        probs = F.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

        return logits, confidence, predicted
    
