import os
import sys
import io
import argparse
import cv2
import torch
import numpy as np

# Ép kiểu stdout sử dụng UTF-8 trên Windows để tránh crash khi in tiếng Việt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import src.config as config
from src.data.vocab import Vocabulary
from src.data.transforms import resize_and_pad, to_tensor_and_normalize
from src.models.attention_model import AttentionHTR
from src.inference.greedy_decode import greedy_decode
from src.utils.image_utils import save_attention_map
from src.inference.beam_search import beam_search_decode
from src.utils.metrics import compute_all_metrics
from src.utils.checkpoint import load_checkpoint

def predict_path(path: str, checkpoint_path: str, save_attention: bool = False, use_beam: bool = False):
    """
    Dự đoán nhãn cho một ảnh hoặc một thư mục ảnh sử dụng mô hình Attention.
    Nếu là thư mục, tên file (không gồm đuôi mở rộng) sẽ được dùng làm nhãn thực tế để tính toán WER/CER.
    """
    device = torch.device(config.DEVICE)
    print(f"Chạy inference Attention trên thiết bị: {device}")
    
    # 1. Khởi tạo vocab và tải mô hình
    vocab = Vocabulary(config.VOCAB_CHARS)
    
    # Khởi tạo mô hình ở chế độ freeze để an toàn
    model = AttentionHTR(
        vocab_size=vocab.size,
        pretrained=False,  # Không tải trọng số ImageNet vì ta sẽ nạp checkpoint
        freeze=True,
        out_layer="layer3"
    ).to(device)
    
    # Nạp trọng số từ checkpoint
    load_checkpoint(model, checkpoint_path, device)
    model.eval()
    
    # 2. Kiểm tra đường dẫn tồn tại
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy đường dẫn tại: {path}")
        
    if os.path.isdir(path):
        # --- CHẾ ĐỘ DỰ ĐOÁN CẢ THƯ MỤC ---
        files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            print(f"Không tìm thấy file ảnh nào trong thư mục: {path}")
            return
            
        print(f"Đang chạy dự đoán cho {len(files)} ảnh trong thư mục: {path}...")
        print(f"Chế độ giải mã: {'Beam Search (width=' + str(config.BEAM_WIDTH) + ')' if use_beam else 'Greedy Decode'}")
        print("-" * 80)
        
        all_targets = []
        all_predictions = []
        
        for f in files:
            # Lấy nhãn chuẩn từ tên file (chuyển sang chữ thường)
            label = os.path.splitext(f)[0].lower()
            all_targets.append(label)
            
            img_path = os.path.join(path, f)
            image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image_gray is None:
                print(f"Cảnh báo: Không thể giải mã ảnh {f}, bỏ qua.")
                continue
                
            image_processed = resize_and_pad(image_gray, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
            image_tensor = to_tensor_and_normalize(image_processed).unsqueeze(0).to(device)
            
            with torch.no_grad():
                if use_beam:
                    pred_word, _ = beam_search_decode(
                        model=model,
                        image=image_tensor.squeeze(0),
                        vocab=vocab,
                        device=device,
                        beam_width=config.BEAM_WIDTH,
                        max_len=config.MAX_LABEL_LENGTH
                    )
                else:
                    decoded_words, _ = greedy_decode(
                        model=model,
                        images=image_tensor,
                        vocab=vocab,
                        device=device,
                        max_len=config.MAX_LABEL_LENGTH
                    )
                    pred_word = decoded_words[0]
            
            all_predictions.append(pred_word)
            print(f"Ảnh: {f:25s} | Thực tế: {label:15s} | Dự đoán: {pred_word:15s} | Kết quả: {'ĐÚNG' if pred_word == label else 'SAI'}")
            
        # Tính toán metrics cho toàn bộ thư mục
        metrics = compute_all_metrics(all_predictions, all_targets)
        print("\n" + "=" * 60)
        print("          KẾT QUẢ ĐÁNH GIÁ THƯ MỤC - ATTENTION")
        print("=" * 60)
        print(f"Số lượng ảnh:                     {len(all_targets)}")
        print(f"Word Accuracy (WA):              {metrics['word_accuracy']:.2f}%")
        print(f"Word Error Rate (WER):           {100.0 - metrics['word_accuracy']:.2f}%")
        print(f"Character Error Rate:            {metrics['cer']:.2f}%")
        print(f"Normalized Edit Distance:        {metrics['ned']:.2f}%")
        print("=" * 60 + "\n")
        
    else:
        # --- CHẾ ĐỘ DỰ ĐOÁN ẢNH ĐƠN LẺ ---
        image_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            raise ValueError(f"Không thể giải mã ảnh: {path}")
            
        # Resize giữ aspect ratio và pad về 64x512
        image_processed = resize_and_pad(image_gray, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        
        # Chuyển thành tensor và thêm chiều batch
        image_tensor = to_tensor_and_normalize(image_processed).unsqueeze(0).to(device)
        
        print("Đang giải mã...")
        with torch.no_grad():
            if use_beam:
                predicted_word, attn_weights = beam_search_decode(
                    model=model,
                    image=image_tensor.squeeze(0),
                    vocab=vocab,
                    device=device,
                    beam_width=config.BEAM_WIDTH,
                    max_len=config.MAX_LABEL_LENGTH
                )
            else:
                decoded_words, attn_weights_list = greedy_decode(
                    model=model,
                    images=image_tensor,
                    vocab=vocab,
                    device=device,
                    max_len=config.MAX_LABEL_LENGTH
                )
                predicted_word = decoded_words[0]
                attn_weights = attn_weights_list[0]  # (T, seq_len)
            
        print("-" * 40)
        print(f"Ảnh: {path}")
        print(f"Từ dự đoán: '{predicted_word}'")
        print("-" * 40)
        
        # Trực quan hóa bản đồ attention (chỉ hỗ trợ khi dùng Greedy Decode đơn lẻ)
        if save_attention:
            if use_beam:
                print("Lưu ý: Không hỗ trợ lưu bản đồ attention hàng loạt khi chạy với Beam Search.")
                return
            if len(predicted_word) == 0:
                print("Không thể lưu bản đồ attention do không dự đoán được ký tự nào (từ rỗng).")
                return
                
            base_name = os.path.splitext(os.path.basename(path))[0]
            save_name = f"predict_{base_name}_attn.png"
            save_path = os.path.join(config.ATTENTION_MAP_DIR, save_name)
            
            # Chuyển đổi ký tự sang danh sách đơn lẻ
            predicted_chars = list(predicted_word)
            
            print(f"Đang tạo và lưu bản đồ attention tại: {save_path}...")
            save_attention_map(
                image=image_processed,
                attention_weights=attn_weights,
                predicted_chars=predicted_chars,
                save_path=save_path
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict single image or folder of images using Attention-based Model.")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn đến file ảnh hoặc thư mục ảnh cần nhận dạng.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/attention_bilstm/best_attention_model.pth", 
                        help="Đường dẫn đến file checkpoint (.pth).")
    parser.add_argument("--save_attention", action="store_true", help="Lưu bản đồ attention trực quan hóa (chỉ cho Greedy Decode ảnh đơn lẻ).")
    parser.add_argument("--beam", action="store_true", help="Sử dụng Beam Search thay cho Greedy Decode.")
    
    args = parser.parse_args()
    
    # Giải quyết đường dẫn tuyệt đối hoặc tương đối
    img_path = os.path.abspath(args.image)
    ckpt_path = os.path.abspath(args.checkpoint)
    
    predict_path(img_path, ckpt_path, args.save_attention, args.beam)