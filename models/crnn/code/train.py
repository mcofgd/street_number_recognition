import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import SimpleCharClassifier
from data_loader import get_dataloader, CharImageDataset  # 确保数据集类支持单字符分类任务
from utils import train_epoch, evaluate  # 确保这些辅助函数也适应单字符分类任务


def main():
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.0005
    early_stop_patience = 15

    # 数据集路径
    base_dir = '../tcdata'
    train_data_dir = os.path.join(base_dir, 'mchar_train')
    train_json = os.path.join(base_dir, 'train.json')
    val_data_dir = os.path.join(base_dir, 'mchar_val')
    val_json = os.path.join(base_dir, 'mchar_val.json')

    # 模型保存路径
    model_save_dir = '../user_data/model_data'
    os.makedirs(model_save_dir, exist_ok=True)

    # 获取数据加载器
    train_loader = get_dataloader(train_data_dir, train_json, batch_size, mode='train')
    val_loader = get_dataloader(val_data_dir, val_json, batch_size, mode='val')

    # 初始化模型
    model = SimpleCharClassifier(num_classes=10)  # 数字0~9共10类
    model = model.to(device)

    # 使用交叉熵损失函数
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 更改为交叉熵损失函数
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-5,  # 更小的初始学习率
        weight_decay=1e-3,  # 更强的L2正则化
        eps=1e-8  # 提高数值稳定性
    )

    # 梯度裁剪回调
    def safe_backward(loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True  # 显式报错
        )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 训练循环
    best_val_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # 训练
        train_loss, train_acc, train_conf = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Conf: {train_conf:.4f}")

        # 验证
        val_loss, val_acc, val_conf = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Conf: {val_conf:.4f}")

        # 学习率调整
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(model_save_dir, 'best_model.pth'))
            print("保存最佳模型！")
        else:
            no_improve_epochs += 1

        # 保存最后模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, os.path.join(model_save_dir, 'last_model.pth'))

        # 早停检查
        if no_improve_epochs >= early_stop_patience:
            print(f"\n早停：验证损失连续{early_stop_patience}轮没有改善")
            break

    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最后模型保存在: {os.path.join(model_save_dir, 'last_model.pth')}")
    print(f"最佳模型保存在: {os.path.join(model_save_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()