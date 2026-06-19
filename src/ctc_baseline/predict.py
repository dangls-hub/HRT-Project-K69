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
from src.models.ctc_baseline import CTCBaseline
from src.inference.ctc_decode import ctc_decode
from src.utils.metrics import compute_all_metrics
from src.utils.checkpoint import load_checkpoint

def predict_path(path: str, checkpoint_path: str):
    """
    Dự đoán nhãn cho một ảnh hoặc một thư mục ảnh sử dụng mô hình CTC Baseline.
    Nếu là thư mục, tên file (không gồm đuôi mở rộng) sẽ được dùng làm nhãn thực tế để tính toán WER/CER.
    """
    device = torch.device(config.DEVICE)
    print(f"Chạy inference CTC trên thiết bị: {device}")
    
    # 1. Khởi tạo vocab và tải mô hình
    vocab = Vocabulary(config.VOCAB_CHARS)
    
    model = CTCBaseline(
        vocab_size=vocab.size,
        hidden_dim=config.CTC_HIDDEN_DIM,
        num_layers=config.CTC_NUM_LAYERS
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
                log_probs = model(image_tensor)
                decoded_ctc = ctc_decode(log_probs, vocab)[0]
            
            all_predictions.append(decoded_ctc)
            print(f"Ảnh: {f:25s} | Thực tế: {label:15s} | Dự đoán: {decoded_ctc:15s} | Kết quả: {'ĐÚNG' if decoded_ctc == label else 'SAI'}")
            
        # Tính toán metrics cho toàn bộ thư mục
        metrics = compute_all_metrics(all_predictions, all_targets)
        print("\n" + "=" * 60)
        print("          KẾT QUẢ ĐÁNH GIÁ THƯ MỤC - CTC")
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
            
        # Resize giữ aspect ratio và pad về kích thước trong config
        image_processed = resize_and_pad(image_gray, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        
        # Chuyển thành tensor và thêm chiều batch
        image_tensor = to_tensor_and_normalize(image_processed).unsqueeze(0).to(device)
        
        print("Đang giải mã...")
        with torch.no_grad():
            log_probs = model(image_tensor)
            decoded_ctc = ctc_decode(log_probs, vocab)
            
        predicted_word = decoded_ctc[0]
        
        print("-" * 40)
        print(f"Ảnh: {path}")
        print(f"Từ dự đoán (CTC): '{predicted_word}'")
        print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict single image or folder of images using CTC Baseline Model.")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn đến file ảnh hoặc thư mục ảnh cần nhận dạng.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ctc_baseline/best_ctc_baseline.pth", 
                        help="Đường dẫn đến file checkpoint (.pth).")
    
    args = parser.parse_args()
    
    img_path = os.path.abspath(args.image)
    ckpt_path = os.path.abspath(args.checkpoint)
    
    predict_path(img_path, ckpt_path)