import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Adopting convolution to match input channel
        self.adopting_conv = nn.Conv2d(1, 3, kernel_size=(3, 3), padding="same")

        # Load pretrained ResNet50 with ImageNet weights
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Fully connected layer to get the desired output size
        self.fc = nn.Linear(self.resnet.fc.out_features, 1024)

    def forward(self, x):
        x = self.adopting_conv(x)
        x = self.resnet(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super(Decoder, self).__init__()

        # If no hidden_dim is provided, use half of input_dim
        hidden_dim = hidden_dim or (input_dim // 2)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        # Weight initialization
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads=8, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        # Ensure embed_dim is divisible by heads
        assert self.head_dim * heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        # Linear projections for query, key, and value
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Input shape: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.size()

        # Residual connection
        residual = x

        # Project inputs
        query = self.query_proj(x).view(batch_size, seq_len, self.heads, self.head_dim)
        key = self.key_proj(x).view(batch_size, seq_len, self.heads, self.head_dim)
        value = self.value_proj(x).view(batch_size, seq_len, self.heads, self.head_dim)

        # Transpose for multi-head attention
        query = query.transpose(1, 2)  # [batch_size, heads, seq_len, head_dim]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)

        # Softmax attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        # Compute attended values
        context = torch.matmul(attention_weights, value)

        # Reshape and combine heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Output projection
        output = self.out_proj(context)

        # Residual connection and layer normalization
        output = self.layer_norm(output + residual)

        return output


class EncoderDecoderWithAttention(nn.Module):
    def __init__(self, output_dim, embed_dim=1024, heads=8):
        super(EncoderDecoderWithAttention, self).__init__()
        self.encoder = Encoder()
        self.attention = SelfAttention(embed_dim=embed_dim, heads=heads)
        self.decoder = Decoder(input_dim=embed_dim, output_dim=output_dim)

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        attended_features = self.attention(encoder_outputs.unsqueeze(1))
        output = self.decoder(attended_features.squeeze(1))

        return output


class Custom2DDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)  # Apply transform

        return sample, target


# main
if __name__ == "__main__":
    model = EncoderDecoderWithAttention(output_dim=283)

    x = torch.randn(16, 1, 224, 224)

    outputs = model(x)

    print(outputs.shape)
