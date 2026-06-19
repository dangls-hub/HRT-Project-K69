import random
import torch
import torch.nn as nn
from typing import Tuple, Optional
from src.models.attention import BahdanauAttention

class GRUDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder_dim: int,
                 decoder_hidden_dim: int, attention_dim: int, dropout: float):
        """
        GRU Decoder với Bahdanau Attention để sinh chuỗi ký tự.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.decoder_hidden_dim = decoder_hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(encoder_dim, decoder_hidden_dim, attention_dim)
        
        # Đầu vào của GRUCell là sự kết hợp của embedding ký tự và context vector
        self.gru = nn.GRUCell(embed_dim + encoder_dim, decoder_hidden_dim)
        
        # Đầu ra kết hợp hidden state, context vector và embedding
        self.fc_out = nn.Linear(decoder_hidden_dim + encoder_dim + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Khởi tạo hidden state cho GRUCell từ trung bình đầu ra của Encoder
        self.init_h = nn.Linear(encoder_dim, decoder_hidden_dim)

    def init_hidden(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Khởi tạo hidden state ban đầu cho Decoder bằng cách tính trung bình 
        các đặc trưng encoder theo chiều sequence (seq_len) và đi qua một lớp Linear + Tanh.
        encoder_outputs: (B, seq_len, encoder_dim)
        Trả về: (B, decoder_hidden_dim)
        """
        mean_enc = encoder_outputs.mean(dim=1)  # (B, encoder_dim)
        h0 = torch.tanh(self.init_h(mean_enc))  # (B, decoder_hidden_dim)
        return h0

    def forward_step(self, input_char: torch.Tensor, hidden: torch.Tensor, 
                     encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Giải mã một bước đơn lẻ (Single-step decoding).
        - input_char: (B,) — Chỉ số của ký tự đầu vào hiện tại.
        - hidden: (B, decoder_hidden_dim) — Hidden state hiện tại.
        - encoder_outputs: (B, seq_len, encoder_dim) — Chuỗi đặc trưng encoder.
        
        Trả về:
        - logits: (B, vocab_size) — Phân phối điểm số trên từ vựng.
        - hidden_new: (B, decoder_hidden_dim) — Hidden state mới.
        - attn_w: (B, seq_len) — Trọng số attention cho bước này.
        """
        # 1. Nhúng ký tự đầu vào: (B, embed_dim)
        emb = self.dropout(self.embedding(input_char))
        
        # 2. Tính toán Attention: context (B, encoder_dim), attn_w (B, seq_len)
        context, attn_w = self.attention(hidden, encoder_outputs)
        
        # 3. Concat embedding và context vector làm đầu vào GRUCell
        gru_input = torch.cat([emb, context], dim=1)  # (B, embed_dim + encoder_dim)
        
        # 4. Cập nhật hidden state của GRU
        hidden_new = self.gru(gru_input, hidden)  # (B, decoder_hidden_dim)
        
        # 5. Chiếu đầu ra kết hợp để dự đoán ký tự tiếp theo
        fc_input = torch.cat([hidden_new, context, emb], dim=1)  # (B, hidden + enc + emb)
        logits = self.fc_out(self.dropout(fc_input))  # (B, vocab_size)
        
        return logits, hidden_new, attn_w

    def forward(self, encoder_outputs: torch.Tensor, targets: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5, max_len: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Giải mã toàn bộ chuỗi (Full forward pass).
        - encoder_outputs: (B, seq_len, encoder_dim)
        - targets: (B, target_len) — Nhãn thực tế (đã loại bỏ sos, kết thúc bằng eos).
        - teacher_forcing_ratio: tỉ lệ sử dụng nhãn thật thay vì ký tự dự đoán ở bước trước.
        - max_len: chiều dài giải mã tối đa khi không có targets (ở chế độ inference).
        
        Trả về:
        - all_logits: (B, T, vocab_size)
        - all_attn_weights: (B, T, seq_len)
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Khởi tạo hidden state ban đầu
        hidden = self.init_hidden(encoder_outputs)
        
        # Ký tự bắt đầu giải mã là <sos> (sos_idx = 1)
        input_char = torch.full((batch_size,), fill_value=1, dtype=torch.long, device=device)
        
        all_logits = []
        all_attn_weights = []
        
        # Xác định số bước giải mã
        decode_len = targets.size(1) if targets is not None else max_len
        
        for t in range(decode_len):
            logits, hidden, attn_w = self.forward_step(input_char, hidden, encoder_outputs)
            all_logits.append(logits)
            all_attn_weights.append(attn_w)
            
            # Lựa chọn ký tự tiếp theo làm đầu vào cho bước tiếp theo
            if targets is not None and t < decode_len - 1:
                is_teacher = random.random() < teacher_forcing_ratio
                if is_teacher:
                    # Sử dụng nhãn thực tế
                    input_char = targets[:, t]
                else:
                    # Sử dụng ký tự dự đoán có điểm số cao nhất
                    input_char = logits.argmax(dim=-1)
            else:
                input_char = logits.argmax(dim=-1)
                
        # Stack kết quả qua chiều thời gian (dim=1)
        all_logits_tensor = torch.stack(all_logits, dim=1)  # (B, T, vocab_size)
        all_attn_weights_tensor = torch.stack(all_attn_weights, dim=1)  # (B, T, seq_len)
        
        return all_logits_tensor, all_attn_weights_tensor