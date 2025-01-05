import torch
import torch.nn as nn

class Attention(nn.Module):
    """注意力机制模块
    
    实现了基于加法注意力的机制,用于对序列中不同时间步赋予不同的重要性权重。
    通过学习权重来确定哪些时间步的信息更重要。
    
    Args:
        hidden_size (int): 隐藏层的维度,因为是双向LSTM,所以实际使用的是hidden_size*2
        
    Returns:
        context: 注意力加权后的上下文向量
        attention_weights: 各时间步的注意力权重,用于可视化分析
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # 注意力层,将hidden_size*2维的输入映射到1维,用于计算权重
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        # 计算注意力权重并通过softmax归一化
        # lstm_output的形状为(batch, seq_len, hidden_size*2)
        attention_weights = torch.softmax(
            self.attention(lstm_output),  # 计算每个时间步的权重分数
            dim=1  # 在序列长度维度上进行softmax
        )
        
        # 使用注意力权重对LSTM输出进行加权
        weighted_output = lstm_output * attention_weights
        # 在序列长度维度上求和,得到上下文向量
        context = weighted_output.sum(dim=1)
        return context, attention_weights


class BiLSTMAttentionClassifier(nn.Module):
    """带注意力机制的双向LSTM分类器模型
    
    该模型结合了双向LSTM和注意力机制的优点:
    1. 双向LSTM捕获序列的双向依赖关系
    2. 注意力机制关注重要的时间步
    3. 全连接层进行最终分类
    
    Args:
        input_size (int): 输入特征的维度,即每个时间步的特征数量
        hidden_size (int): LSTM隐藏层的维度,决定模型的容量
        num_classes (int): 分类类别数(二分类为1)
        
    Returns:
        outputs: 模型的预测概率(0-1之间)
        attention_weights: 注意力权重分布,用于分析模型关注点
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(BiLSTMAttentionClassifier, self).__init__()
        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )
        # 注意力层
        self.attention = Attention(hidden_size)
        # 全连接层,输入维度为hidden_size*2(双向LSTM的输出)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # 通过双向LSTM层得到所有时间步的输出
        lstm_out, _ = self.lstm(x)
        # 应用注意力机制
        context, attention_weights = self.attention(lstm_out)
        # 通过全连接层进行分类
        out = self.fc(context)
        return out, attention_weights
