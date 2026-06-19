import torch
import numpy as np
from typing import List, Tuple
from src.data.vocab import Vocabulary

def greedy_decode(model: torch.nn.Module, images: torch.Tensor, vocab: Vocabulary, 
                  device: torch.device, max_len: int = 32) -> Tuple[List[str], List[np.ndarray]]:
    """
    Giải mã tham lam (Greedy decoding) cho Attention HTR Model.
    - model: AttentionHTR model.
    - images: Batch ảnh tensor (B, 3, 64, 256).
    - vocab: Đối tượng Vocabulary.
    - device: Thiết bị chạy (cuda hoặc cpu).
    - max_len: Chiều dài giải mã tối đa.
    
    Trả về:
    - decoded_words: List chứa chuỗi ký tự dự đoán cho từng ảnh trong batch.
    - final_attn_weights: List chứa các attention weights tương ứng dạng numpy array (T, seq_len) cho mỗi ảnh.
    """
    model.eval()
    batch_size = images.size(0)
    
    with torch.no_grad():
        # 1. Trích xuất đặc trưng qua mô hình (bao gồm BiLSTM ngữ cảnh)
        enc_seq = model.extract_features(images)
        W_prime = enc_seq.size(1)
        
        # 2. Khởi tạo hidden state cho Decoder
        hidden = model.decoder.init_hidden(enc_seq)
        
        # 3. Bắt đầu bằng token <sos>
        input_char = torch.full((batch_size,), fill_value=vocab.sos_idx, dtype=torch.long, device=device)
        
        # Lưu các chỉ số ký tự giải mã được và lịch sử attention
        decoded_indices = [[] for _ in range(batch_size)]
        attn_history = [[] for _ in range(batch_size)]
        
        # Mảng theo dõi các mẫu đã hoàn thành gặp <eos>
        finished = [False] * batch_size
        
        for t in range(max_len):
            logits, hidden, attn_w = model.decoder.forward_step(input_char, hidden, enc_seq)
            
            # Dự đoán nhãn có điểm số cao nhất (Greedy)
            preds = logits.argmax(dim=-1)  # (B,)
            
            # Ghi nhận ký tự cho từng mẫu trong batch
            for i in range(batch_size):
                if not finished[i]:
                    idx = preds[i].item()
                    if idx == vocab.eos_idx:
                        finished[i] = True
                    else:
                        decoded_indices[i].append(idx)
                        attn_history[i].append(attn_w[i].cpu().numpy())
            
            # Dừng sớm nếu tất cả các mẫu trong batch đều đã sinh ra <eos>
            if all(finished):
                break
                
            # Cập nhật input cho bước tiếp theo
            input_char = preds
            
        # Dịch chuỗi chỉ số thành văn bản
        decoded_words = []
        final_attn_weights = []
        for i in range(batch_size):
            word = vocab.decode(decoded_indices[i], stop_at_eos=True)
            decoded_words.append(word)
            
            # Tổng hợp trọng số attention dạng numpy array (T, seq_len)
            if len(attn_history[i]) > 0:
                final_attn_weights.append(np.stack(attn_history[i], axis=0))
            else:
                final_attn_weights.append(np.zeros((0, W_prime)))
                
        return decoded_words, final_attn_weights