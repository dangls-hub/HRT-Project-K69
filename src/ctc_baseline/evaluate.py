import os
import argparse
from functools import partial
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
import io

# Ép kiểu stdout sử dụng UTF-8 trên Windows để tránh crash khi in tiếng Việt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import src.config as config
from src.data.vocab import Vocabulary
from src.data.dataset import HTRDataset, collate_fn
from src.data.transforms import get_val_transform
from src.models.ctc_baseline import CTCBaseline
from src.inference.ctc_decode import ctc_decode
from src.utils.metrics import compute_all_metrics, edit_distance
from src.utils.checkpoint import load_checkpoint

def evaluate_ctc_test_set(checkpoint_path: str):
    """
    Đánh giá mô hình CTC Baseline trên toàn bộ test set.
    """
    device = torch.device(config.DEVICE)
    print(f"Đánh giá CTC Baseline trên thiết bị: {device}")
    
    # 1. Khởi tạo vocab và tải mô hình
    vocab = Vocabulary(config.VOCAB_CHARS)
    model = CTCBaseline(
        vocab_size=vocab.size,
        hidden_dim=config.CTC_HIDDEN_DIM,
        num_layers=config.CTC_NUM_LAYERS
    ).to(device)
    
    # Tải checkpoint
    load_checkpoint(model, checkpoint_path, device)
    model.eval()
    
    # 2. Tạo test dataloader
    val_transform = get_val_transform()
    test_csv_path = os.path.join(config.DATA_PROCESSED_DIR, "test.csv")
    
    test_dataset = HTRDataset(
        csv_path=test_csv_path,
        vocab=vocab,
        transform=val_transform,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx)
    )
    
    print(f"Số lượng mẫu kiểm thử (test set): {len(test_dataset):,}")
    
    # 3. Chạy dự đoán
    all_image_paths = test_dataset.df["image_path"].tolist()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, target_lengths in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            
            # Forward pass -> (T, B, vocab_size)
            log_probs = model(images)
            
            # Giải mã CTC Greedy
            decoded_words = ctc_decode(log_probs, vocab)
            all_predictions.extend(decoded_words)
            
            # Lấy ground truth
            for i in range(targets.size(0)):
                target_indices = targets[i].cpu().tolist()
                target_word = vocab.decode(target_indices, stop_at_eos=True)
                all_targets.append(target_word)
                
    # 4. Tính toán các metrics
    metrics = compute_all_metrics(all_predictions, all_targets)
    
    # 5. Lưu kết quả ra file predictions_ctc.csv
    results = []
    for img_path, gt, pred in zip(all_image_paths, all_targets, all_predictions):
        dist = edit_distance(pred, gt)
        is_correct = (pred == gt)
        results.append({
            "image_path": img_path,
            "ground_truth": gt,
            "prediction": pred,
            "edit_distance": dist,
            "correct": is_correct
        })
        
    predictions_df = pd.DataFrame(results)
    output_csv_path = os.path.join(config.OUTPUT_DIR, "predictions_ctc.csv")
    predictions_df.to_csv(output_csv_path, index=False, encoding="utf-8")
    
    # 6. In ra summary kết quả đánh giá theo đúng format yêu cầu
    print("\n" + "=" * 60)
    print("          EVALUATION RESULTS - CTC BASELINE MODEL")
    print("=" * 60)
    print(f"Test samples:                    {len(test_dataset):,}")
    print(f"Word Accuracy:                   {metrics['word_accuracy']:.2f}%")
    print(f"Character Error Rate:            {metrics['cer']:.2f}%")
    print(f"Normalized Edit Distance:        {metrics['ned']:.2f}%")
    print("-" * 60)
    print(f"Predictions saved to: {output_csv_path}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CTC Baseline Model on Test Set.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ctc_baseline/best_ctc_baseline.pth",
                        help="Đường dẫn đến file checkpoint (.pth).")
    
    args = parser.parse_args()
    
    ckpt_path = os.path.abspath(args.checkpoint)
    evaluate_ctc_test_set(ckpt_path)
