import os
import time
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Any

import src.config as config
from src.data.vocab import Vocabulary
from src.data.dataset import HTRDataset, collate_fn
from src.data.transforms import get_train_transform, get_val_transform
from src.models.ctc_baseline import CTCBaseline
from src.inference.ctc_decode import ctc_decode
from src.utils.seed import set_seed
from src.utils.metrics import compute_all_metrics
from src.utils.checkpoint import save_checkpoint

def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device,
                    scaler: Optional[torch.cuda.amp.GradScaler] = None) -> float:
    """
    Huấn luyện mô hình CTC Baseline trong một epoch.
    """
    model.train()
    total_loss = 0.0
    
    for images, targets, target_lengths in tqdm(dataloader, desc="Training Batch", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # log_probs shape: (T, B, vocab_size)
                log_probs = model(images)
                T, B, _ = log_probs.shape
                
                # Chiều dài chuỗi đầu ra của CTC model
                input_lengths = torch.full((B,), fill_value=T, dtype=torch.long, device=device)
                
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            log_probs = model(images)
            T, B, _ = log_probs.shape
            input_lengths = torch.full((B,), fill_value=T, dtype=torch.long, device=device)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
            optimizer.step()
            
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate(model: nn.Module, dataloader: DataLoader, vocab: Vocabulary,
             criterion: nn.Module, device: torch.device) -> Tuple[float, float, float]:
    """
    Đánh giá mô hình CTC Baseline trên tập validation.
    """
    model.eval()
    total_loss = 0.0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, target_lengths in tqdm(dataloader, desc="Validating Batch", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            
            log_probs = model(images)
            T, B, _ = log_probs.shape
            input_lengths = torch.full((B,), fill_value=T, dtype=torch.long, device=device)
            
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            
            # Giải mã bằng CTC Greedy Decoding
            decoded_words = ctc_decode(log_probs, vocab)
            all_predictions.extend(decoded_words)
            
            # Giải mã nhãn đích
            for i in range(B):
                target_indices = targets[i].cpu().tolist()
                target_word = vocab.decode(target_indices, stop_at_eos=True)
                all_targets.append(target_word)
                
    metrics = compute_all_metrics(all_predictions, all_targets)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, metrics["word_accuracy"], metrics["cer"]

def main():
    # 1. Thiết lập Seed và thiết bị
    set_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)
    print(f"Huấn luyện CTC Baseline trên thiết bị: {device}")
    
    # 2. Khởi tạo vocab
    vocab = Vocabulary(config.VOCAB_CHARS)
    
    # 3. Tạo Datasets và Dataloaders
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    train_dataset = HTRDataset(
        csv_path=os.path.join(config.DATA_PROCESSED_DIR, "train.csv"),
        vocab=vocab,
        transform=train_transform,
        is_train=True
    )
    val_dataset = HTRDataset(
        csv_path=os.path.join(config.DATA_PROCESSED_DIR, "val.csv"),
        vocab=vocab,
        transform=val_transform,
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.CTC_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=(config.NUM_WORKERS > 0),
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.CTC_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=(config.NUM_WORKERS > 0),
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx)
    )
    
    print(f"Số lượng mẫu huấn luyện: {len(train_dataset):,}")
    print(f"Số lượng mẫu validation: {len(val_dataset):,}")
    
    # 4. Khởi tạo mô hình CTCBaseline
    # Đóng băng Encoder để huấn luyện BiLSTM/Linear trước (giống Attention model)
    model = CTCBaseline(
        vocab_size=vocab.size,
        hidden_dim=config.CTC_HIDDEN_DIM,
        num_layers=config.CTC_NUM_LAYERS
    ).to(device)
    
    # 5. Cấu hình Loss và Optimizer
    # nn.CTCLoss yêu cầu:
    # - blank: index của blank token (index 0 ứng với <pad>)
    # - zero_infinity: True để gán 0 cho giá trị vô cùng (tránh crash)
    criterion = nn.CTCLoss(blank=vocab.pad_idx, zero_infinity=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.CTC_LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP and device.type == "cuda" else None
    
    # Khởi tạo TensorBoard cho ctc baseline
    ctc_log_dir = os.path.join(config.LOG_DIR, "ctc_baseline")
    writer = SummaryWriter(log_dir=ctc_log_dir)
    
    best_cer = float("inf")
    epochs_no_improve = 0
    checkpoint_filename = os.path.join(config.CHECKPOINT_DIR, "ctc_baseline", "last_ctc_baseline.pth")
    
    # 6. Vòng lặp huấn luyện chính
    for epoch in range(config.CTC_EPOCHS):
        start_time = time.time()
        
        # Giải đóng băng encoder tại epoch thứ 5 để fine-tune (nhất quán với Attention)
        if epoch == 5:
            print("\n>>> Giải đóng băng encoder để fine-tune...")
            model.unfreeze_encoder(from_layer="layer3")
            
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        
        val_loss, val_wa, val_cer = validate(model, val_loader, vocab, criterion, device)
        
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        # Ghi log
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Metrics/Word_Accuracy", val_wa, epoch)
        writer.add_scalar("Metrics/CER", val_cer, epoch)
        
        print(f"Epoch {epoch+1:02d}/{config.CTC_EPOCHS:02d} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val WA: {val_wa:.2f}% | Val CER: {val_cer:.2f}% | "
              f"Time: {epoch_time:.1f}s")
              
        metrics_dict = {
            "val_loss": val_loss,
            "val_wa": val_wa,
            "val_cer": val_cer
        }
        
        is_best = val_cer < best_cer
        if is_best:
            best_cer = val_cer
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics_dict,
            path=checkpoint_filename,
            scheduler=scheduler,
            is_best=is_best
        )
        
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n[Early Stopping] Dừng sớm do Val CER không cải thiện sau {config.EARLY_STOPPING_PATIENCE} epochs.")
            break
            
    writer.close()
    print("\nQuá trình huấn luyện CTC Baseline hoàn tất!")

if __name__ == "__main__":
    main()
