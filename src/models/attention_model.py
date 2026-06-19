import torch
import torch.nn as nn
from typing import Tuple, Optional

import src.config as config
from src.models.resnet_encoder import ResNetEncoder
from src.models.gru_decoder import GRUDecoder

class AttentionHTR(nn.Module):
    def __init__(self, vocab_size: int, pretrained: bool = True, freeze: bool = True,
                 out_layer: str = "layer3", embed_dim: int = 128, 
                 decoder_hidden_dim: int = 256, attention_dim: int = 256, 
                 dropout: float = 0.3):
        """
        Mô hình nhận dạng chữ viết tay Attention-HTR tích hợp Encoder và Decoder.
        """
        super().__init__()
        
        # 1. Khởi tạo CNN Encoder
        self.encoder = ResNetEncoder(pretrained=pretrained, freeze=freeze, out_layer=out_layer)
        
        # Lấy kích thước đầu ra thực tế của Encoder dựa trên cấu hình ảnh
        enc_channels, enc_h, enc_w = self.encoder.get_output_size(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        
        # Gộp chiều cao H' vào số channels C làm đặc trưng cho mỗi cột sequence chiều rộng W'
        raw_encoder_dim = enc_channels * enc_h
        
        # Khối BiLSTM ngữ cảnh (Context LSTM)
        self.context_lstm = nn.LSTM(
            input_size=config.ATTENTION_LSTM_INPUT_DIM,
            hidden_size=config.ATTENTION_LSTM_HIDDEN_DIM,
            num_layers=config.ATTENTION_LSTM_NUM_LAYERS,
            bidirectional=config.ATTENTION_LSTM_BIDIRECTIONAL,
            dropout=config.ATTENTION_LSTM_DROPOUT if config.ATTENTION_LSTM_NUM_LAYERS > 1 else 0.0,
            batch_first=True
        )
        
        # 2. Khởi tạo GRU Decoder với encoder_dim mới (đã đi qua BiLSTM)
        self.decoder = GRUDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            encoder_dim=config.ATTENTION_ENCODER_DIM,
            decoder_hidden_dim=decoder_hidden_dim,
            attention_dim=attention_dim,
            dropout=dropout
        )

    def extract_features(self, images: torch.Tensor, return_debug: bool = False):
        """
        Trích xuất đặc trưng hình ảnh qua ResNet, biến đổi sang chuỗi, và đi qua BiLSTM.
        Trả về chuỗi ngữ cảnh có kích thước (B, W', 512).
        """
        enc_features = self.encoder(images)
        B, C, H_prime, W_prime = enc_features.shape
        enc_seq_raw = enc_features.permute(0, 3, 1, 2).contiguous().view(B, W_prime, C * H_prime)
        enc_seq, _ = self.context_lstm(enc_seq_raw)
        
        if return_debug:
            return enc_seq, enc_features, enc_seq_raw
        return enc_seq

    def forward(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5, max_len: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        images: (B, 3, 64, 512)
        targets: (B, target_len)
        
        Trả về:
        - logits: (B, T, vocab_size)
        - attn_weights: (B, T, seq_len)
        """
        # 1 & 2. Trích xuất đặc trưng và đi qua BiLSTM ngữ cảnh
        enc_seq = self.extract_features(images)
        
        # 3. Đi qua Attention Decoder để sinh chuỗi logits
        logits, attn_weights = self.decoder(
            encoder_outputs=enc_seq,
            targets=targets,
            teacher_forcing_ratio=teacher_forcing_ratio,
            max_len=max_len
        )
        
        return logits, attn_weights

    def freeze_encoder(self):
        """Đóng băng toàn bộ tham số của encoder."""
        self.encoder.freeze_all()

    def unfreeze_encoder(self, from_layer: str = "layer3"):
        """Mở đóng băng các tham số encoder từ layer chỉ định."""
        self.encoder.unfreeze_from(from_layer)