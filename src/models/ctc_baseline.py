import torch
import torch.nn as nn
import torch.nn.functional as F

import src.config as config
from src.models.resnet_encoder import ResNetEncoder

class CTCBaseline(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 256, num_layers: int = 2):
        """
        Mô hình Baseline: CNN (ResNet18) + BiLSTM + CTC.
        - vocab_size: kích thước từ vựng (bao gồm cả pad/blank ở index 0).
        - hidden_dim: kích thước ẩn của LSTM.
        - num_layers: số lớp LSTM xếp chồng.
        """
        super().__init__()
        
        # Sử dụng ResNetEncoder dừng ở layer3 để trích xuất đặc trưng
        self.encoder = ResNetEncoder(pretrained=True, freeze=True, out_layer="layer3")
        
        # Tính toán kích thước đầu vào của LSTM
        enc_channels, enc_h, enc_w = self.encoder.get_output_size(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        encoder_dim = enc_channels * enc_h  # 256 * 4 = 1024
        
        # Mạng hồi quy 2 chiều BiLSTM
        self.bilstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        
        # Lớp chiếu tuyến tính sang số lượng từ vựng
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, 64, 256)
        Trả về:
        - log_probs: (T, B, vocab_size) — Log probabilities dùng cho CTCLoss
        """
        # 1. Trích xuất đặc trưng: (B, C, H', W') -> (B, 256, 4, 16)
        enc_features = self.encoder(images)
        
        # 2. Reshape ghép chiều cao vào channels: (B, W', C * H') -> (B, 16, 1024)
        B, C, H_prime, W_prime = enc_features.shape
        enc_seq = enc_features.permute(0, 3, 1, 2).contiguous().view(B, W_prime, C * H_prime)
        
        # 3. Đi qua BiLSTM: lstm_out shape: (B, W', hidden_dim * 2)
        lstm_out, _ = self.bilstm(enc_seq)
        
        # 4. Chiếu tuyến tính sang logits: (B, W', vocab_size)
        logits = self.fc(lstm_out)
        
        # 5. Tính log softmax trên chiều vocab_size: (B, W', vocab_size)
        # Permute thành (W', B, vocab_size) tương ứng (T, B, C) theo yêu cầu của nn.CTCLoss
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
        
        return log_probs

    def freeze_encoder(self):
        """Đóng băng encoder để chỉ train BiLSTM và Classifier."""
        self.encoder.freeze_all()

    def unfreeze_encoder(self, from_layer: str = "layer3"):
        """Mở đóng băng encoder để fine-tune."""
        self.encoder.unfreeze_from(from_layer)
