import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Thiết lập random seed cho tất cả các thư viện để đảm bảo khả năng tái lặp kết quả (reproducibility).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Đảm bảo tính toán trên GPU là deterministic (nhưng có thể làm giảm hiệu năng nhẹ)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Đã thiết lập random seed: {seed}")