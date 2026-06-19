import torch
from typing import List
from src.data.vocab import Vocabulary

def ctc_decode(log_probs: torch.Tensor, vocab: Vocabulary) -> List[str]:
    """
    Giải mã tham lam CTC (CTC Greedy decoding).
    - log_probs: (T, B, vocab_size) — Mảng xác suất log từ CTC model.
    - vocab: Đối tượng Vocabulary.
    
    Trả về:
    - decoded_words: Danh sách các chuỗi ký tự được giải mã cho từng mẫu trong batch.
    """
    # Lấy nhãn dự đoán có điểm số cao nhất tại mỗi thời điểm: (T, B)
    preds = log_probs.argmax(dim=-1)
    
    # Chuyển về dạng (B, T) để duyệt theo từng mẫu
    preds_batch = preds.permute(1, 0).cpu().tolist()
    
    decoded_words = []
    for seq in preds_batch:
        # 1. Loại bỏ các chỉ số lặp liên tiếp
        collapsed = []
        prev_idx = -1
        for idx in seq:
            if idx != prev_idx:
                collapsed.append(idx)
                prev_idx = idx
                
        # 2. Loại bỏ blank token (index 0 - tương ứng với <pad>)
        cleaned = [idx for idx in collapsed if idx != 0]
        
        # Dịch chuỗi chỉ số sang văn bản (vocab.decode tự động dừng lại ở <eos> nếu có)
        word = vocab.decode(cleaned, stop_at_eos=True)
        decoded_words.append(word)
        
    return decoded_words