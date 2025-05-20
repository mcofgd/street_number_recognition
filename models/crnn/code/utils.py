import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import json
from PIL import Image
from torchvision import transforms

class strLabelConverter(object):
    """
    在字符串和标签之间进行转换
    """
    def __init__(self, alphabet):
        """
        初始化转换器
        
        Args:
            alphabet: 字符集字符串
        """
        self.alphabet = alphabet
        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1  # 0保留给空白标签
        
    def encode(self, text):
        """
        将文本转换为标签序列
        
        Args:
            text: 文本字符串或字符串列表
            
        Returns:
            torch.LongTensor: 标签序列
            torch.IntTensor: 每个序列的长度
        """
        if isinstance(text, str):
            text = [text]
            
        length = []
        result = []
        for item in text:
            length.append(len(item))
            for char in item:
                if char in self.dict:
                    result.append(self.dict[char])
                else:
                    result.append(0)  # 未知字符用0表示
                    
        return torch.LongTensor(result), torch.IntTensor(length)
    
    def decode(self, t, length, raw=False):
        """
        将标签序列转换为文本
        
        Args:
            t: 标签序列
            length: 序列长度
            raw: 是否返回原始解码结果
            
        Returns:
            解码后的文本字符串
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # 批处理模式
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode(t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

class resizeNormalize(object):
    """
    调整图像大小并标准化
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        
    def __call__(self, img):
        # 调整大小
        img = img.resize(self.size, self.interpolation)
        # 转换为张量
        img = self.toTensor(img)
        # 标准化
        img.sub_(0.5).div_(0.5)
        return img

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images, targets, target_lengths, bbox_targets = batch
        images = images.to(device)
        targets = targets.to(device)
        bbox_targets = bbox_targets.to(device)
        
        # 前向传播
        logits, bbox_pred = model(images)
        
        # 计算序列损失
        batch_size, seq_len, num_classes = logits.size()
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        
        # CTC损失
        log_probs = F.log_softmax(logits, dim=2).transpose(0, 1)  # (seq_len, batch, num_classes)
        ctc_loss = criterion(log_probs, targets, input_lengths, target_lengths)
        
        # 定位损失 (使用MSE)
        bbox_loss = F.mse_loss(bbox_pred, bbox_targets)
        
        # 总损失
        loss = ctc_loss + bbox_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算准确率
        _, preds = logits.max(2)
        for i in range(batch_size):
            pred_str = ''.join([str(p.item()) for p in preds[i] if p != 0])
            target_str = ''.join([str(targets[i][j].item()) for j in range(target_lengths[i].item())])
            if pred_str == target_str:
                correct += 1
        total_samples += batch_size
        
        pbar.set_postfix({"loss": loss.item(), "accuracy": correct / total_samples})
    
    return total_loss / len(dataloader), correct / total_samples

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            images, targets, target_lengths, bbox_targets = batch
            images = images.to(device)
            targets = targets.to(device)
            bbox_targets = bbox_targets.to(device)
            
            # 前向传播
            logits, bbox_pred = model(images)
            
            # 计算序列损失
            batch_size, seq_len, num_classes = logits.size()
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
            
            # CTC损失
            log_probs = F.log_softmax(logits, dim=2).transpose(0, 1)
            ctc_loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            # 定位损失
            bbox_loss = F.mse_loss(bbox_pred, bbox_targets)
            
            # 总损失
            loss = ctc_loss + bbox_loss
            
            total_loss += loss.item()
            
            # 计算准确率
            _, preds = logits.max(2)
            for i in range(batch_size):
                pred_str = ''.join([str(p.item()) for p in preds[i] if p != 0])
                target_str = ''.join([str(targets[i][j].item()) for j in range(target_lengths[i].item())])
                if pred_str == target_str:
                    correct += 1
            total_samples += batch_size
            
            pbar.set_postfix({"val_loss": loss.item(), "val_accuracy": correct / total_samples})
    
    return total_loss / len(dataloader), correct / total_samples

def decode_predictions(logits, image_size):
    """从模型输出解码得到字符和位置"""
    batch_size, seq_len, num_classes = logits.size()
    
    # 解码序列
    _, preds = logits.max(2)
    
    # 移除重复字符和空白符（CTC解码）
    decoded_preds = []
    for i in range(batch_size):
        pred = preds[i].cpu().numpy()
        decoded = []
        prev_char = -1
        for p in pred:
            if p != 0 and p != prev_char:  # 0通常是空白符
                decoded.append(p)
            prev_char = p
        decoded_preds.append(decoded)
    
    return decoded_preds

def predict(model, dataloader, device, output_dir):
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            images, img_names = batch
            images = images.to(device)
            
            # 前向传播
            logits, bbox_pred = model(images)
            
            # 解码预测
            decoded = decode_predictions(logits, images.shape[-2:])
            
            # 将预测结果保存
            for i, img_name in enumerate(img_names):
                h, w = 128, 384  # 模型输入的图像大小
                
                # 获取原始图像大小
                orig_img = cv2.imread(os.path.join(dataloader.dataset.data_dir, img_name))
                orig_h, orig_w = orig_img.shape[:2]
                
                # 缩放边界框到原始图像大小
                x, y, width, height = bbox_pred[i].cpu().numpy()
                x = int(x * orig_w)
                y = int(y * orig_h)
                width = int(width * orig_w)
                height = int(height * orig_h)
                
                # 保存预测结果
                label = [int(c) for c in decoded[i]]
                
                if img_name not in predictions:
                    predictions[img_name] = {
                        "label": label,
                        "top": [y],
                        "left": [x],
                        "height": [height],
                        "width": [width]
                    }
                else:
                    predictions[img_name]["label"].extend(label)
                    predictions[img_name]["top"].extend([y])
                    predictions[img_name]["left"].extend([x])
                    predictions[img_name]["height"].extend([height])
                    predictions[img_name]["width"].extend([width])
    
    # 保存预测结果
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "predictions.json"), "w") as f:
        json.dump(predictions, f)
    
    return predictions