import os
import random
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, List

import src.config as config
from src.data.vocab import Vocabulary
from src.data.transforms import resize_and_pad, to_tensor_and_normalize

class HTRDataset(Dataset):
    def __init__(self, csv_path: str, vocab: Vocabulary, transform=None, is_train: bool = False):
        """
        Khởi tạo HTRDataset.
        - csv_path: đường dẫn đến file CSV chứa image_path và label.
        - vocab: đối tượng Vocabulary để mã hóa nhãn.
        - transform: Albumentations composition để augmentation.
        - is_train: cờ xác định chế độ huấn luyện (dùng để xử lý logic đặc thù nếu cần).
        """
        self.df = pd.read_csv(csv_path)
        self.vocab = vocab
        self.transform = transform
        self.is_train = is_train
        self.cache = {}  # Bộ nhớ đệm RAM lưu ảnh gốc

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Tải ảnh từ đĩa (hoặc lấy từ cache RAM), áp dụng transform, sinh target tensor và trả về:
        (image_tensor, target_tensor, target_length)
        """
        row = self.df.iloc[idx]
        image_relative_path = row["image_path"]
        label = str(row["label"])

        # Kiểm tra ảnh trong cache RAM trước
        if idx in self.cache:
            image = self.cache[idx]
        else:
            # Tạo đường dẫn tuyệt đối đến file ảnh
            abs_image_path = os.path.join(config.PROJECT_ROOT, image_relative_path)

            # Đọc ảnh ở dạng grayscale
            image = cv2.imread(abs_image_path, cv2.IMREAD_GRAYSCALE)

            # Cơ chế fallback nếu ảnh không load được (file bị corrupt hoặc mất mát trên đĩa sau khi split)
            if image is None:
                print(f"[Cảnh báo] Không thể đọc ảnh: {abs_image_path}. Thử lấy mẫu ngẫu nhiên khác.")
                random_idx = random.randint(0, len(self) - 1)
                return self.__getitem__(random_idx)
            
            # Lưu vào cache RAM
            self.cache[idx] = image

        # Áp dụng Albumentations transform (nếu có) trước khi resize và pad
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Resize giữ tỉ lệ và pad ảnh về kích thước chuẩn (64x256)
        image = resize_and_pad(image, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

        # Chuyển đổi thành tensor và chuẩn hóa ImageNet
        image_tensor = to_tensor_and_normalize(image)

        # Mã hóa nhãn văn bản sang index
        encoded_label = self.vocab.encode(label)
        
        # Luôn thêm token <eos> vào cuối nhãn
        encoded_label.append(self.vocab.eos_idx)
        
        target_tensor = torch.tensor(encoded_label, dtype=torch.long)
        target_length = len(encoded_label)

        return image_tensor, target_tensor, target_length

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]], pad_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hàm ghép các mẫu đơn lẻ thành một batch dữ liệu:
    1. Gộp các tensor ảnh thành một batch tensor (B, C, H, W).
    2. Điền pad_idx vào các target tensor có độ dài ngắn hơn để chúng có kích thước bằng nhau.
    3. Gộp các target lengths thành một tensor (B,).
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    
    # Tìm độ dài nhãn lớn nhất trong batch hiện tại
    max_target_len = max(item[2] for item in batch)
    
    # Tạo tensor chứa nhãn đã pad, mặc định điền pad_idx (0)
    targets = torch.full((len(batch), max_target_len), fill_value=pad_idx, dtype=torch.long)
    for i, item in enumerate(batch):
        target_length = item[2]
        targets[i, :target_length] = item[1]
        
    target_lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)
    
    return images, targets, target_lengths