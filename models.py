import torch
import torch.nn as nn
from torchvision.models import resnet50

# 定义注意力机制模块
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)  # 线性层，将隐藏状态映射到注意力权重

    def forward(self, lstm_output):
        # 计算每个时间步的注意力权重
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # 对LSTM输出加权得到上下文向量
        weighted_output = lstm_output * attention_weights
        context = weighted_output.sum(dim=1)  # 对加权后的输出按时间步求和得到上下文向量
        return context, attention_weights

# 定义BiLSTM和注意力模块
class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTMAttentionClassifier, self).__init__()
        # 双向LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        # 注意力机制模块
        self.attention = Attention(hidden_size)
        # 输出全连接层
        self.fc = nn.Linear(hidden_size * 2, 128)

    def forward(self, x):
        # 获取LSTM的输出
        lstm_out, _ = self.lstm(x)
        # 计算注意力和上下文向量
        context, attention_weights = self.attention(lstm_out)
        # 输出分类结果
        out = self.fc(context)
        return out, attention_weights

# 定义多模态模型
class MultimodalModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MultimodalModel, self).__init__()
        # ResNet50 模块
        self.resnet50 = resnet50(pretrained=True)
        self.resnet50.fc = nn.Sequential(
            nn.Linear(self.resnet50.fc.in_features, 128),  # 增加一层全连接层，输出大小为128
            nn.ReLU()
        )

        # BiLSTM-Attention 模块
        self.bilstm_attention = BiLSTMAttentionClassifier(input_size, hidden_size)

        # 融合层
        self.fusion_layer = nn.Linear(128 * 2, 1)  # 将两种模态的128维特征融合，输出最终分类结果

    def forward(self, image, sequence):
        # 图像输入通过 ResNet50
        image_features = self.resnet50(image)

        # 时序数据输入通过 BiLSTM-Attention
        sequence_features, _ = self.bilstm_attention(sequence)

        # 融合两个模态的特征
        combined_features = torch.cat((image_features, sequence_features), dim=1)
        output = self.fusion_layer(combined_features)

        return output
