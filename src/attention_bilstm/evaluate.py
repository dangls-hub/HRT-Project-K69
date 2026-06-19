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
from src.models.attention_model import AttentionHTR
from src.inference.greedy_decode import greedy_decode
from src.inference.beam_search import beam_search_decode
from src.utils.metrics import compute_all_metrics, edit_distance
from src.utils.checkpoint import load_checkpoint

def evaluate_test_set(checkpoint_path: str, use_beam: bool = False):
    """
    Đánh giá mô hình Attention trên toàn bộ test set và ghi nhận kết quả.
    - use_beam: Nếu True, sử dụng Beam Search Decode thay vì Greedy Decode.
    """
    device = torch.device(config.DEVICE)
    print(f"Đánh giá mô hình trên thiết bị: {device}")
    print(f"Chế độ giải mã: {'Beam Search' if use_beam else 'Greedy Decode'}")
    if use_beam:
        print(f"Độ rộng Beam (Beam Width): {config.BEAM_WIDTH}")
    
    # 1. Khởi tạo vocab và tải mô hình
    vocab = Vocabulary(config.VOCAB_CHARS)
    model = AttentionHTR(
        vocab_size=vocab.size,
        pretrained=False,
        freeze=True,
        out_layer="layer3"
    ).to(device)
    
    # Tải checkpoint
    load_checkpoint(model, checkpoint_path, device)
    model.eval()
    
    # 2. Tạo test dataset và dataloader
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
    
    # 3. Chạy dự đoán trên toàn bộ test set
    all_image_paths = test_dataset.df["image_path"].tolist()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, target_lengths in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            
            if use_beam:
                # Giải mã bằng Beam Search (duyệt từng ảnh trong batch)
                for i in range(images.size(0)):
                    img = images[i]  # shape: (3, H, W)
                    pred_word, _ = beam_search_decode(
                        model=model,
                        image=img,
                        vocab=vocab,
                        device=device,
                        beam_width=config.BEAM_WIDTH,
                        max_len=config.MAX_LABEL_LENGTH
                    )
                    all_predictions.append(pred_word)
            else:
                # Giải mã bằng Greedy Decode (mặc định)
                decoded_words, _ = greedy_decode(
                    model=model,
                    images=images,
                    vocab=vocab,
                    device=device,
                    max_len=config.MAX_LABEL_LENGTH
                )
                all_predictions.extend(decoded_words)
            
            # Lấy ground truth labels
            for i in range(targets.size(0)):
                target_indices = targets[i].cpu().tolist()
                target_word = vocab.decode(target_indices, stop_at_eos=True)
                all_targets.append(target_word)
                
    # 4. Tính toán các metrics
    metrics = compute_all_metrics(all_predictions, all_targets)
    
    # 5. Lưu báo cáo predictions.csv chi tiết
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
    output_csv_path = os.path.join(config.OUTPUT_DIR, "predictions.csv")
    predictions_df.to_csv(output_csv_path, index=False, encoding="utf-8")
    
    # 6. In ra summary kết quả đánh giá theo đúng format yêu cầu
    print("\n" + "=" * 60)
    print(f"          EVALUATION RESULTS - ATTENTION MODEL")
    print("=" * 60)
    print(f"Test samples:                    {len(test_dataset):,}")
    print(f"Word Accuracy:                   {metrics['word_accuracy']:.2f}%")
    print(f"Character Error Rate:            {metrics['cer']:.2f}%")
    print(f"Normalized Edit Distance:        {metrics['ned']:.2f}%")
    print("-" * 60)
    print(f"Predictions saved to: {output_csv_path}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Attention HTR Model on Test Set.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/attention_bilstm/best_attention_model.pth",
                        help="Đường dẫn đến file checkpoint (.pth).")
    parser.add_argument("--beam", action="store_true", help="Sử dụng giải mã Beam Search thay vì Greedy Decode.")
    
    args = parser.parse_args()
    
    ckpt_path = os.path.abspath(args.checkpoint)
    evaluate_test_set(ckpt_path, use_beam=args.beam)