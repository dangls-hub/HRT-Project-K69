import os
import time
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Tuple, Optional

import src.config as config
from src.data.vocab import Vocabulary
from src.data.dataset import HTRDataset, collate_fn
from src.data.transforms import get_train_transform, get_val_transform
from src.models.attention_model import AttentionHTR
from src.inference.greedy_decode import greedy_decode
from src.utils.seed import set_seed
from src.utils.metrics import compute_all_metrics
from src.utils.checkpoint import save_checkpoint, load_checkpoint

def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device, teacher_forcing_ratio: float,
                    scaler: Optional[torch.cuda.amp.GradScaler] = None) -> float:
    """
    Huấn luyện mô hình trong một epoch.
    """
    model.train()
    total_loss = 0.0
    
    for images, targets, target_lengths in tqdm(dataloader, desc="Training Batch", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Sử dụng chế độ Mixed Precision (AMP) nếu scaler được cung cấp
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # logits shape: (B, T, vocab_size)
                logits, _ = model(images, targets, teacher_forcing_ratio)
                vocab_size = logits.size(-1)
                
                # Reshape logits: (B*T, vocab_size) và targets: (B*T) để tính CrossEntropyLoss
                loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(images, targets, teacher_forcing_ratio)
            vocab_size = logits.size(-1)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
            optimizer.step()
            
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate(model: nn.Module, dataloader: DataLoader, vocab: Vocabulary,
             criterion: nn.Module, device: torch.device) -> Tuple[float, float, float]:
    """
    Đánh giá mô hình trên tập validation (tính loss và tính metrics qua greedy_decode).
    """
    model.eval()
    total_loss = 0.0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, target_lengths in tqdm(dataloader, desc="Validating Batch", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            
            # 1. Tính loss validation (không dùng teacher forcing)
            logits, _ = model(images, targets, teacher_forcing_ratio=0.0)
            vocab_size = logits.size(-1)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            total_loss += loss.item()
            
            # 2. Sinh chuỗi giải mã tham lam phục vụ tính toán WA và CER
            decoded_words, _ = greedy_decode(model, images, vocab, device, max_len=config.MAX_LABEL_LENGTH)
            all_predictions.extend(decoded_words)
            
            # Giải mã nhãn nhắm đích từ target tensor
            for i in range(targets.size(0)):
                target_indices = targets[i].cpu().tolist()
                # Giải mã nhãn đích (dừng tại eos_idx)
                target_word = vocab.decode(target_indices, stop_at_eos=True)
                all_targets.append(target_word)
                
    # Tính toán các metrics
    metrics = compute_all_metrics(all_predictions, all_targets)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, metrics["word_accuracy"], metrics["cer"]

def main():
    # 1. Thiết lập seed và môi trường
    set_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)
    print(f"Huấn luyện trên thiết bị: {device}")
    
    # 2. Khởi tạo Vocabulary
    vocab = Vocabulary(config.VOCAB_CHARS)
    vocab_size = vocab.size
    
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
    
    # collate_fn pad nhãn với pad_idx=0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=(config.NUM_WORKERS > 0),
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=(config.NUM_WORKERS > 0),
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx)
    )
    
    print(f"Số lượng mẫu huấn luyện: {len(train_dataset):,}")
    print(f"Số lượng mẫu validation: {len(val_dataset):,}")
    
    # 4. Khởi tạo mô hình
    # Mặc định Phase 1: freeze=True (đóng băng ResNet encoder)
    model = AttentionHTR(
        vocab_size=vocab_size,
        pretrained=True,
        freeze=True,
        out_layer="layer3",
        embed_dim=config.EMBED_DIM,
        decoder_hidden_dim=config.DECODER_HIDDEN_DIM,
        attention_dim=config.ATTENTION_DIM,
        dropout=config.DROPOUT
    ).to(device)
    
    # 5. Cấu hình Loss và Optimizer
    # Bỏ qua pad_idx (0) khi tính loss
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    
    # Phân chia tham số an toàn theo yêu cầu của Phase 1
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            decoder_params.append(param)
            
    optimizer = torch.optim.Adam([
        {"params": decoder_params, "lr": config.LEARNING_RATE_DECODER},
        {"params": encoder_params, "lr": config.LEARNING_RATE_ENCODER}
    ], weight_decay=config.WEIGHT_DECAY)
    
    # LR Scheduler điều chỉnh dựa trên val_loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    # Cấu hình AMP nếu sử dụng GPU và được cấu hình
    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP and device.type == "cuda" else None
    if scaler is not None:
        print("Đã kích hoạt chế độ Mixed Precision (AMP) training.")
        
    # Khởi tạo TensorBoard writer
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    
    # Các biến theo dõi huấn luyện
    best_cer = float("inf")
    epochs_no_improve = 0
    checkpoint_filename = os.path.join(config.CHECKPOINT_DIR, "attention_bilstm", "last_attention_model.pth")
    
    # 6. Vòng lặp huấn luyện chính
    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        
        # --- Quản lý các Phase huấn luyện ---
        # Epoch 0 - 4 (5 epochs đầu): Phase 1 - Chỉ huấn luyện Decoder (Encoder đóng băng)
        # Epoch 5 trở đi: Phase 2 - Unfreeze layer3+layer4 để fine-tune toàn bộ
        if epoch == 5:
            print("\n>>> Chuyển sang Phase 2: Mở đóng băng encoder từ layer3 để fine-tune...")
            model.unfreeze_encoder(from_layer="layer3")
            # In thông tin kiểm tra grad của mô hình
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Số lượng tham số huấn luyện mới: {trainable_params:,}")
            
        # Tính toán tỷ lệ Teacher Forcing cho epoch này
        tf_ratio = max(0.01, config.TEACHER_FORCING_RATIO - epoch * config.TEACHER_FORCING_DECAY)
        
        # Huấn luyện
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, tf_ratio, scaler)
        
        # Validation
        val_loss, val_wa, val_cer = validate(model, val_loader, vocab, criterion, device)
        
        # Cập nhật scheduler
        scheduler.step(val_loss)
        
        # Thời gian chạy epoch
        epoch_time = time.time() - start_time
        
        # 7. Ghi logs
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Metrics/Word_Accuracy", val_wa, epoch)
        writer.add_scalar("Metrics/CER", val_cer, epoch)
        writer.add_scalar("Hyperparameters/Teacher_Forcing_Ratio", tf_ratio, epoch)
        
        # Lấy LR hiện tại của decoder để log
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Hyperparameters/Learning_Rate_Decoder", current_lr, epoch)
        
        print(f"Epoch {epoch+1:02d}/{config.NUM_EPOCHS:02d} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val WA: {val_wa:.2f}% | Val CER: {val_cer:.2f}% | "
              f"TF Ratio: {tf_ratio:.2f} | Time: {epoch_time:.1f}s")
        
        # 8. Lưu Checkpoint
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
        
        # 9. Dừng sớm (Early Stopping)
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n[Early Stopping] Đã dừng sớm ở epoch {epoch+1} do Val CER không cải thiện sau {config.EARLY_STOPPING_PATIENCE} epochs.")
            break
            
    writer.close()
    print("\nQuá trình huấn luyện mô hình Attention hoàn tất!")

if __name__ == "__main__":
    main()