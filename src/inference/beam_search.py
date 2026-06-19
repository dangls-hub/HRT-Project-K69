import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from src.data.vocab import Vocabulary

def beam_search_decode(model: torch.nn.Module, image: torch.Tensor, vocab: Vocabulary,
                       device: torch.device, beam_width: int = 5, max_len: int = 32) -> Tuple[str, np.ndarray]:
    """
    Thuật toán giải mã Beam Search (Beam Search decoding) cho một ảnh viết tay đơn lẻ.
    - model: Mô hình Attention HTR.
    - image: Tensor ảnh đầu vào (3, 64, 256) hoặc (1, 3, 64, 256).
    - vocab: Đối tượng Vocabulary.
    - device: Thiết bị chạy (cuda hoặc cpu).
    - beam_width: Độ rộng của chùm (số nhánh tối đa được giữ lại ở mỗi bước).
    - max_len: Chiều dài giải mã tối đa.
    
    Trả về:
    - predicted_word: Từ dự đoán tốt nhất.
    - final_attn: Trọng số attention tương ứng dạng numpy array (T, seq_len).
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
        
    model.eval()
    with torch.no_grad():
        # 1. Trích xuất đặc trưng qua mô hình (bao gồm BiLSTM ngữ cảnh)
        enc_seq = model.extract_features(image.to(device))
        W_prime = enc_seq.size(1)
        
        # 2. Khởi tạo hidden state cho Decoder
        h0 = model.decoder.init_hidden(enc_seq)
        
        # Một phần tử chùm (beam item) gồm: (score, sequence_indices, hidden_state, attn_history)
        # Bắt đầu với token <sos>
        beams = [(0.0, [vocab.sos_idx], h0, [])]
        completed_beams = []
        
        for t in range(max_len):
            candidates = []
            
            for score, seq, hidden, attn_history in beams:
                # Ký tự đầu vào là ký tự cuối cùng của chuỗi hiện tại
                input_char = torch.tensor([seq[-1]], dtype=torch.long, device=device)
                
                # Đi qua 1 bước decode
                logits, hidden_new, attn_w = model.decoder.forward_step(input_char, hidden, enc_seq)
                
                # Tính Log probabilities
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (vocab_size,)
                
                # Lấy top k ứng viên tốt nhất để mở rộng nhánh (tránh duyệt hết toàn bộ vocab)
                top_indices = np.argsort(log_probs)[-beam_width:]
                
                for idx in top_indices:
                    idx = int(idx)
                    new_score = score + log_probs[idx]
                    new_seq = seq + [idx]
                    new_attn_history = attn_history + [attn_w[0].cpu().numpy()]
                    
                    if idx == vocab.eos_idx:
                        # Chuỗi hoàn thành: chuẩn hóa điểm theo độ dài chuỗi (chiều dài không tính <sos>)
                        seq_len_norm = len(new_seq) - 1
                        norm_score = new_score / (seq_len_norm ** 0.7) if seq_len_norm > 0 else new_score
                        completed_beams.append((norm_score, new_score, new_seq, new_attn_history))
                    else:
                        candidates.append((new_score, new_seq, hidden_new, new_attn_history))
            
            # Sắp xếp các ứng viên chưa hoàn thành và giữ lại top beam_width
            new_beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
            
            # Nếu không còn nhánh nào hoạt động (tất cả đã gặp <eos>), dừng vòng lặp sớm
            if not new_beams:
                break
                
            beams = new_beams
            
        # Nếu không thu được chuỗi hoàn thành nào (vượt quá max_len), chuyển các chùm còn lại vào completed_beams
        if not completed_beams:
            for score, seq, hidden, attn_history in beams:
                seq_len_norm = len(seq) - 1
                norm_score = score / (seq_len_norm ** 0.7) if seq_len_norm > 0 else score
                completed_beams.append((norm_score, score, seq, attn_history))
                
        # Lấy chùm có điểm chuẩn hóa cao nhất
        completed_beams = sorted(completed_beams, key=lambda x: x[0], reverse=True)
        best_norm_score, best_score, best_seq, best_attn_history = completed_beams[0]
        
        # Dịch chuỗi chỉ số thành từ (loại bỏ sos/eos)
        predicted_word = vocab.decode(best_seq, stop_at_eos=True)
        
        # Tạo numpy array cho attention weights
        final_attn = np.stack(best_attn_history, axis=0) if best_attn_history else np.zeros((0, W_prime))
        
        # Cắt bớt phần attention dư thừa (như phần dự đoán <eos>) để độ dài khớp hoàn toàn với số chữ cái dự đoán
        final_attn = final_attn[:len(predicted_word)]
        
        return predicted_word, final_attn