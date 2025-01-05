import torch
import torch.nn as nn
from torchvision.models import resnet50

# 定义BiLSTM-Attention模块
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        weighted_output = lstm_output * attention_weights
        context = weighted_output.sum(dim=1)
        return context, attention_weights

class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTMAttentionClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, 128)  # 输出大小为128

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attention_weights = self.attention(lstm_out)
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

# 测试多模态模型
if __name__ == "__main__":
    # 定义输入参数
    input_size = 8  # 时序特征维度
    hidden_size = 64  # LSTM隐藏层维度

    # 创建模型
    model = MultimodalModel(input_size, hidden_size)
    print(model)

    # 测试输入数据
    image_input = torch.randn(4, 3, 256, 256)  # 图像输入 (batch_size, channels, height, width)
    sequence_input = torch.randn(4, 100, input_size)  # 时序数据输入 (batch_size, seq_len, input_size)

    # 模型前向传播
    output = model(image_input, sequence_input)
    print("输出形状:", output.shape)  # 输出形状 (batch_size, 1)
