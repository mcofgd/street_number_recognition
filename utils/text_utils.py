import torch

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