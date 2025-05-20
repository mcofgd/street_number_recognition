import os
import torch
import numpy as np
from tqdm import tqdm


def print_tensor_stats(name, tensor):
    """打印张量的统计信息"""
    if not torch.isfinite(tensor).all():
        print(f"[ERROR] {name} contains NaN or Inf")

    # 只有浮点类型才能计算 min/max/mean
    if torch.is_floating_point(tensor):
        print(f"{name} stats: min={tensor.min().item():.4f}, "
              f"max={tensor.max().item():.4f}, "
              f"mean={tensor.mean().item():.4f}")
    else:
        print(f"{name} is not a floating-point tensor, skipping mean calculation.")


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    total_confidence = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, targets in pbar:  # 移除了target_lengths
        batch_size = images.size(0)

        images = images.to(device)
        targets = targets.to(device)

        # 前向传播
        logits, confidence, preds = model(images)  # 现在logits形状应为[B, num_classes]

        # 计算损失
        loss = criterion(logits, targets)

        # 检查是否有NaN或Inf值
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("[WARNING] Loss contains NaN or Inf values.")
        else:
            print_tensor_stats("Loss", loss)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_confidence += confidence.mean().item()

        # 计算准确率
        _, preds = logits.max(1)  # 对于单标签分类问题，dim=1
        correct += (preds == targets).sum().item()
        total_samples += batch_size

        pbar.set_postfix({
            "loss": loss.item(),
            "accuracy": correct / total_samples,
            "confidence": total_confidence / (pbar.n + 1)
        })

    return total_loss / len(dataloader), correct / total_samples, total_confidence / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    total_confidence = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            # 前向传播
            logits, confidence, preds = model(images)

            # 计算损失
            loss = criterion(logits, targets)

            total_loss += loss.item()
            total_confidence += confidence.mean().item()

            # 计算准确率
            _, preds = logits.max(1)
            correct += (preds == targets).sum().item()
            total_samples += images.size(0)

            pbar.set_postfix({
                "val_loss": loss.item(),
                "val_accuracy": correct / total_samples,
                "val_confidence": total_confidence / (pbar.n + 1)
            })

    return total_loss / len(dataloader), correct / total_samples, total_confidence / len(dataloader)

import torch.nn.functional as F  # 添加这一行
def decode_predictions(logits, confidence):
    probs = F.softmax(logits, dim=1)
    _, preds = probs.max(1)

    decoded_preds = []
    decoded_confs = []

    for i in range(logits.size(0)):
        pred = preds[i].cpu().numpy()
        conf = probs[i][pred].cpu().numpy()
        decoded_preds.append(pred.tolist())
        decoded_confs.append(conf.tolist())

    return decoded_preds, decoded_confs
import json

def predict(model, dataloader, device, output_dir):
    model.eval()
    predictions = {}

    with torch.no_grad():
        for images, img_names in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)

            # 前向传播
            logits, confidence, preds = model(images)

            # 解码预测
            decoded_preds, decoded_confs = decode_predictions(logits, confidence)

            # 将预测结果保存
            for i, img_name in enumerate(img_names):
                predictions[img_name] = {
                    "label": int(decoded_preds[i][0]),  # 转换为整数
                    "confidence": float(decoded_confs[i][0])
                }

    # 保存预测结果
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "predictions.json"), "w") as f:
        json.dump(predictions, f)

    return predictions