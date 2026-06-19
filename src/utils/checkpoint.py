import os
import shutil
import torch
from typing import Dict, Any, Optional

import src.config as config

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    epoch: int, metrics: Dict[str, float], path: str, 
                    scheduler: Optional[Any] = None, is_best: bool = False):
    """
    Lưu checkpoint hiện tại của mô hình.
    Nếu is_best=True, nhân bản file checkpoint này sang định dạng "best_..." tương ứng.
    """
    # Trích xuất state_dict của model (hỗ trợ cả DataParallel nếu có)
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    
    state = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
        "vocab_chars": config.VOCAB_CHARS
    }
    
    # Tạo thư mục cha nếu chưa tồn tại
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Lưu checkpoint hiện tại (last)
    torch.save(state, path)
    
    # Nếu là mô hình tốt nhất, copy sang file best
    if is_best:
        dir_name = os.path.dirname(path)
        base_name = os.path.basename(path)
        if base_name.startswith("last_"):
            best_base = base_name.replace("last_", "best_")
        else:
            best_base = "best_" + base_name
        best_path = os.path.join(dir_name, best_base)
        shutil.copyfile(path, best_path)
        print(f"Đã lưu checkpoint tốt nhất tại: {best_path} (Val CER: {metrics.get('val_cer', 100.0):.2f}%)")

def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None) -> Dict[str, Any]:
    """
    Tải checkpoint từ file và nạp trạng thái vào model, optimizer, scheduler.
    Trả về dictionary thông tin metadata (epoch, metrics).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {path}")
        
    print(f"Đang tải checkpoint từ: {path}...")
    checkpoint = torch.load(path, map_location=device)
    
    # Nạp trọng số vào model
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        
    # Nạp trạng thái optimizer nếu truyền vào
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    # Nạp trạng thái scheduler nếu truyền vào
    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "vocab_chars": checkpoint.get("vocab_chars", config.VOCAB_CHARS)
    }