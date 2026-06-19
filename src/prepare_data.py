import os
import html
import random
import csv
from PIL import Image
from tqdm import tqdm

# Import config và Vocabulary
import src.config as config
from src.data.vocab import Vocabulary

def load_label_file(label_path: str):
    """
    Đọc tệp label.txt (TSV), parse từng dòng thành (relative_image_path, label).
    """
    samples = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                samples.append((parts[0], parts[1]))
            else:
                # Dòng không đúng định dạng TSV
                pass
    return samples

def validate_sample(relative_path: str, label: str, vocab: Vocabulary):
    """
    Kiểm tra tính hợp lệ của sample:
    1. Kiểm tra ảnh tồn tại trên đĩa.
    2. Kiểm tra nhãn có độ dài hợp lệ [MIN_LABEL_LENGTH, MAX_LABEL_LENGTH].
    3. Kiểm tra nhãn chỉ chứa các ký tự trong Vocabulary.
    Trả về (is_valid, reason).
    """
    # 1. Kiểm tra ảnh tồn tại
    full_image_path = os.path.join(config.DATASET_DIR, relative_path)
    if not os.path.exists(full_image_path):
        return False, "image_not_found"

    # 2. Kiểm tra độ dài nhãn
    if not label or len(label) < config.MIN_LABEL_LENGTH or len(label) > config.MAX_LABEL_LENGTH:
        return False, "label_length_invalid"

    # 3. Kiểm tra ký tự nhãn hợp lệ
    if not vocab.is_valid_label(label):
        return False, "invalid_chars"

    return True, "valid"

def prepare_data():
    """
    Hàm xử lý chính: đọc nhãn, làm sạch, lọc mẫu, chia bộ dữ liệu và lưu CSV.
    """
    print("Bắt đầu xử lý dữ liệu...")
    vocab = Vocabulary(config.VOCAB_CHARS)

    # 1. Tải toàn bộ dòng từ label.txt
    raw_samples = load_label_file(config.LABEL_FILE)
    total_lines = len(raw_samples)

    valid_samples = []
    
    # Bộ đếm để in summary
    skipped_not_found = 0
    skipped_corrupt = 0
    skipped_empty = 0
    skipped_length = 0
    skipped_invalid_chars = 0

    # 2. Lặp qua từng mẫu dữ liệu để làm sạch và kiểm tra
    for rel_path, raw_label in tqdm(raw_samples, desc="Validating dataset"):
        # Giải mã các ký tự HTML (ví dụ: &apos; -> ', &quot; -> ")
        clean_label = html.unescape(raw_label).strip()

        # Lowercase nhãn (vì Phase 1 chỉ dùng bộ chữ thường a-z)
        clean_label = clean_label.lower()

        # Xử lý trường hợp nhãn rỗng trước khi kiểm tra sâu hơn
        if not clean_label:
            skipped_empty += 1
            continue

        # Kiểm tra tính hợp lệ
        is_valid, reason = validate_sample(rel_path, clean_label, vocab)

        if is_valid:
            # Lưu đường dẫn tương đối tính từ project root để tiện load
            csv_path = os.path.join("dataset", rel_path).replace("\\", "/")
            valid_samples.append((csv_path, clean_label))
        else:
            if reason == "image_not_found":
                skipped_not_found += 1
            elif reason == "image_corrupt":
                skipped_corrupt += 1
            elif reason == "label_length_invalid":
                skipped_length += 1
            elif reason == "invalid_chars":
                skipped_invalid_chars += 1

    # 3. Shuffle với random seed cố định để đảm bảo lặp lại được kết quả
    random.seed(config.RANDOM_SEED)
    random.shuffle(valid_samples)

    # 4. Chia dữ liệu theo tỷ lệ
    total_valid = len(valid_samples)
    train_end = int(total_valid * config.TRAIN_RATIO)
    val_end = train_end + int(total_valid * config.VAL_RATIO)

    train_samples = valid_samples[:train_end]
    val_samples = valid_samples[train_end:val_end]
    test_samples = valid_samples[val_end:]

    # 5. Ghi dữ liệu ra các file CSV
    splits = {
        "train.csv": train_samples,
        "val.csv": val_samples,
        "test.csv": test_samples
    }

    for filename, samples in splits.items():
        csv_path = os.path.join(config.DATA_PROCESSED_DIR, filename)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label"])
            writer.writerows(samples)

    # 6. In ra tóm tắt (Summary) theo đúng format yêu cầu
    print("\n" + "=" * 60)
    print("               DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Total lines in label.txt:          {total_lines:,}")
    print("-" * 60)
    print(f"Valid samples:                      {total_valid:,}")
    print(f"Skipped - image not found:          {skipped_not_found:,}")
    print(f"Skipped - label empty/whitespace:   {skipped_empty:,}")
    print(f"Skipped - label has invalid chars:  {skipped_invalid_chars:,}")
    print(f"Skipped - label too short/long:     {skipped_length:,}")
    print(f"Skipped - image cannot be opened:   {skipped_corrupt:,}")
    print("-" * 60)
    print(f"Train set:                          {len(train_samples):,}  ({config.TRAIN_RATIO * 100:.1f}%)")
    print(f"Validation set:                      {len(val_samples):,}  ({config.VAL_RATIO * 100:.1f}%)")
    print(f"Test set:                            {len(test_samples):,}  ({config.TEST_RATIO * 100:.1f}%)")
    print("-" * 60)
    print("Files saved:")
    print(f"  → {os.path.join('data_processed', 'train.csv')}")
    print(f"  → {os.path.join('data_processed', 'val.csv')}")
    print(f"  → {os.path.join('data_processed', 'test.csv')}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    prepare_data()