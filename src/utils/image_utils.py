import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def load_image(path: str) -> np.ndarray:
    """
    Đọc ảnh từ đường dẫn tuyệt đối ở dạng grayscale.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {path}")
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Không thể giải mã ảnh: {path}")
    return image

def show_image_with_label(image: np.ndarray, label: str):
    """
    Hiển thị ảnh và nhãn tương ứng bằng matplotlib phục vụ mục đích debug.
    """
    plt.figure(figsize=(6, 2))
    if len(image.shape) == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()

def save_attention_map(image: np.ndarray, attention_weights: np.ndarray,
                       predicted_chars: List[str], save_path: str):
    """
    Tạo và lưu lưới hình ảnh trực quan hóa Attention (Attention Heatmap Overlay):
    - Trực quan hóa xem mô hình 'nhìn' vào vị trí nào theo trục ngang khi sinh từng ký tự.
    - Lưới gồm: 1 ảnh gốc trên cùng, bên dưới là các subplots tương ứng với từng bước ký tự.
    
    attention_weights: (T, seq_len) — Trọng số attention cho từng bước giải mã.
    predicted_chars: danh sách T ký tự được dự đoán.
    save_path: đường dẫn lưu ảnh kết quả.
    """
    T, seq_len = attention_weights.shape
    assert len(predicted_chars) == T, f"Số ký tự dự đoán ({len(predicted_chars)}) phải khớp với số bước giải mã ({T})."

    h, w = image.shape[:2]
    
    # Tạo hình ảnh lớn gồm T + 1 hàng (ảnh gốc ở hàng 0, tiếp theo là các ký tự)
    fig, axes = plt.subplots(T + 1, 1, figsize=(6, (T + 1) * 1.5))
    
    # 1. Vẽ ảnh gốc trên cùng
    if len(image.shape) == 2:
        axes[0].imshow(image, cmap="gray")
    else:
        axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # 2. Vẽ heatmap cho từng ký tự
    for t in range(T):
        char = predicted_chars[t]
        weights = attention_weights[t]  # (seq_len,)
        
        # Biến đổi vector weights (seq_len,) thành ma trận (1, seq_len) để resize
        weights_2d = weights.reshape(1, seq_len)
        
        # Nội suy (upsample) trọng số theo chiều ngang từ seq_len sang chiều rộng w của ảnh gốc
        weights_resized = cv2.resize(weights_2d, (w, 1), interpolation=cv2.INTER_CUBIC)
        
        # Lặp lại dòng trọng số này theo chiều dọc h lần để tạo heatmap 2D có hình dạng (h, w)
        heatmap = np.repeat(weights_resized, h, axis=0)
        
        # Chuẩn hóa giá trị về [0, 255]
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        if (heatmap_max - heatmap_min) > 1e-8:
            heatmap_norm = np.uint8(255 * (heatmap - heatmap_min) / (heatmap_max - heatmap_min))
        else:
            heatmap_norm = np.zeros_like(heatmap, dtype=np.uint8)
            
        # Áp dụng Colormap JET (Màu đỏ là attention mạnh nhất, màu xanh là yếu nhất)
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Chuyển ảnh gốc sang RGB để trộn màu
        if len(image.shape) == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = image.copy()
            
        # Trộn ảnh gốc và heatmap với tỉ lệ 60% ảnh gốc + 40% heatmap màu
        blended = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)
        
        axes[t + 1].imshow(blended)
        axes[t + 1].set_title(f"Bước {t+1}: Ký tự '{char}'")
        axes[t + 1].axis("off")
        
    plt.tight_layout()
    
    # Đảm bảo thư mục lưu tồn tại
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()