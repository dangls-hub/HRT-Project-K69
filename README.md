# Word-Level Handwritten Text Recognition (HTR)

Hệ thống nhận dạng chữ viết tay mức độ từ đơn lẻ sử dụng PyTorch chạy local. Dự án xây dựng và so sánh **hai kiến trúc** trên bộ dữ liệu IAM (96,835 mẫu):

| Mô hình | Kiến trúc | Word Accuracy | CER |
|---|---|---|---|
| **CTC Baseline** | ResNet18 + BiLSTM + CTC Loss | **80.75%** | **7.27%** |
| **Attention (Proposed)** | ResNet18 + BiLSTM + Bahdanau Attention + GRU Decoder | **83.09%** | **7.06%** |

---

## 📁 Cấu trúc thư mục dự án

```text
HRT-Project/
├── src/                            # Mã nguồn chính của dự án
│   ├── data/                       # Dataloader, transforms và vocab
│   ├── models/                     # Kiến trúc mô hình
│   │   ├── ctc_baseline.py        #   └─ CTC Baseline (ResNet18 + BiLSTM + CTC)
│   │   ├── resnet_encoder.py      #   └─ CNN Encoder dùng chung
│   │   ├── attention.py           #   └─ Bahdanau Attention
│   │   ├── attention_model.py     #   └─ Attention Model tổng hợp
│   │   └── gru_decoder.py         #   └─ GRU Decoder
│   ├── inference/                  # Các giải thuật giải mã
│   │   ├── ctc_decode.py          #   └─ CTC Greedy Decoding
│   │   ├── greedy_decode.py       #   └─ Attention Greedy Decoding
│   │   └── beam_search.py         #   └─ Attention Beam Search
│   ├── utils/                      # Tiện ích (metrics, checkpoint, seed, visualization)
│   ├── ctc_baseline/               # Pipeline mô hình CTC Baseline
│   │   ├── train.py                #   └─ Script huấn luyện
│   │   ├── evaluate.py             #   └─ Đánh giá trên Test Set
│   │   └── predict.py              #   └─ Dự đoán ảnh đơn / thư mục
│   ├── attention_bilstm/           # Pipeline mô hình Attention Model
│   │   ├── train.py                #   └─ Script huấn luyện
│   │   ├── evaluate.py             #   └─ Đánh giá trên Test Set
│   │   └── predict.py              #   └─ Dự đoán ảnh đơn / thư mục
│   └── prepare_data.py            # Tiền xử lý & chia dữ liệu
├── report/                         # Báo cáo kỹ thuật chi tiết phục vụ viết báo cáo/LaTeX
│   └── project_report.md           # [BÁO CÁO TỔNG HỢP DUY NHẤT]
├── powerpoint/                     # Tài liệu phục vụ thiết kế slide thuyết trình
│   └── slides_outline.md           # Dàn ý chi tiết 10 slide thuyết trình & lời thoại gợi ý
├── checkpoints/                    # Thư mục chứa trọng số mô hình (.pth)
│   ├── ctc_baseline/
│   │   └── best_ctc_baseline.pth  #   └─ Checkpoint tốt nhất CTC Baseline
│   └── attention_bilstm/
│       └── best_attention_model.pth #  └─ Checkpoint tốt nhất Attention
├── dataset/                        # Dữ liệu ảnh words/ và nhãn label.txt (Tải riêng)
├── requirements.txt                # Danh sách thư viện phụ thuộc
└── README.md                       # Hướng dẫn nhanh này
```

---

## ⚡ Hướng dẫn chạy nhanh (Quickstart)

### 1. Cài đặt môi trường
Xem chi tiết hướng dẫn cài đặt Python 3.11, PyTorch CUDA tương thích với card đồ họa tại [report/project_report.md (Mục 3)](file:///d:/Downloads/HRT-Project/report/project_report.md).
Sau khi kích hoạt môi trường:
```bash
python -m pip install -r requirements.txt
```

### 2. Tải dữ liệu & checkpoint
Do kích thước lớn, các file nặng được lưu trên Google Drive. Tải về và đặt đúng cấu trúc thư mục:

| File | Mô tả | Link |
|---|---|---|
| `dataset/` (IAM words) | Ảnh + nhãn gốc (~1.1 GB) | [Tải dataset](https://drive.google.com/file/d/1uC2H2NCVvU1pPz-OId8KKfJiyI3nq5lj/view?usp=sharing) |
| `best_attention_model.pth` | Checkpoint Attention (~91 MB) | [Tải checkpoint](https://drive.google.com/file/d/1yuN5fDkaIQ7PCj3Q-Kq5-muftEd0w8_a/view?usp=sharing) |
| `best_ctc_baseline.pth` | Checkpoint CTC Baseline (~75 MB) | *(có sẵn trong repo)* |

Sau khi tải checkpoint, đặt file vào đúng vị trí:
```
checkpoints/attention_bilstm/best_attention_model.pth
checkpoints/ctc_baseline/best_ctc_baseline.pth
```

### 3. Chuẩn bị dữ liệu
Đặt bộ dữ liệu IAM vào thư mục `dataset/` (gồm file `label.txt` và thư mục `words/`). Sau đó chạy lệnh chia tập dữ liệu:
```bash
python -m src.prepare_data
```

### 4. Đánh giá mô hình trên tập kiểm thử (Test Set)
```bash
# Đánh giá CTC Baseline:
python -m src.ctc_baseline.evaluate

# Đánh giá Attention Model:
python -m src.attention_bilstm.evaluate
```

### 5. Chạy dự đoán thực tế
> **Lưu ý:** Tham số `--image` nhận được cả **đường dẫn file ảnh** lẫn **đường dẫn thư mục**.

```bash
# --- CTC Baseline ---
# Nhận dạng một ảnh đơn lẻ:
python -m src.ctc_baseline.predict --image duong_dan_anh.png

# Nhận dạng toàn bộ ảnh trong thư mục (tên file không có đuôi = nhãn chuẩn):
python -m src.ctc_baseline.predict --image duong_dan_thu_muc/

# --- Attention Model ---
# Nhận dạng một ảnh đơn lẻ và trực quan hóa bản đồ chú ý:
python -m src.attention_bilstm.predict --image duong_dan_anh.png --save_attention

# Nhận dạng toàn bộ ảnh trong thư mục:
python -m src.attention_bilstm.predict --image duong_dan_thu_muc/
```

---

## 📖 Hướng dẫn chạy chi tiết từng Tiến trình

<details>
<summary><b>Xem chi tiết các câu lệnh và tham số nâng cao (Click để mở rộng)</b></summary>

### A. Tiền xử lý dữ liệu (`src.prepare_data`)
*   **Chức năng:** Làm sạch nhãn (lowercase, bỏ ký tự đặc biệt), lọc ảnh lỗi, chia dữ liệu thành 3 tập `train.csv`, `val.csv`, `test.csv` lưu trong `data_processed/`.
*   **Lệnh chạy:**
    ```bash
    python -m src.prepare_data
    ```

### B. Huấn luyện mô hình (Training)

#### B1. CTC Baseline
```bash
python -m src.ctc_baseline.train
```
*Huấn luyện CTC Baseline qua 50 epoch, tự động đóng băng Encoder 5 epoch đầu rồi mở khóa fine-tune. Checkpoint tốt nhất lưu tại `checkpoints/ctc_baseline/best_ctc_baseline.pth`.*

#### B2. Attention Model
```bash
python -m src.attention_bilstm.train
```
*Huấn luyện Attention Model qua 80 epoch với Teacher Forcing giảm dần. Checkpoint tốt nhất lưu tại `checkpoints/attention_bilstm/best_attention_model.pth`.*

### C. Đánh giá mô hình trên tập kiểm thử

#### C1. CTC Baseline
```bash
python -m src.ctc_baseline.evaluate --checkpoint checkpoints/ctc_baseline/best_ctc_baseline.pth
```
*(Dự đoán chi tiết từng ảnh được lưu ra file `outputs/predictions_ctc.csv`)*

#### C2. Attention Model
Đánh giá độ chính xác (Word Accuracy, CER, NED) trên 9,684 mẫu test:
```bash
python -m src.attention_bilstm.evaluate --checkpoint checkpoints/attention_bilstm/best_attention_model.pth
```
*(Dự đoán chi tiết từng ảnh được lưu ra file `outputs/predictions.csv`)*

### D. Nhận dạng ảnh thực tế

#### D1. CTC Baseline (`src.ctc_baseline.predict`)
*   **Ảnh đơn lẻ:**
    ```bash
    python -m src.ctc_baseline.predict --image duong_dan_anh.png
    ```
*   **Toàn bộ thư mục ảnh (tên file = nhãn chuẩn):**
    ```bash
    python -m src.ctc_baseline.predict --image duong_dan_thu_muc/
    ```

#### D2. Attention Model (`src.attention_bilstm.predict`)
*   **Ảnh đơn lẻ (Greedy Decode & Lưu Attention Map):**
    ```bash
    python -m src.attention_bilstm.predict --image duong_dan_anh.png --save_attention
    ```
    *(Ảnh trực quan hóa attention heatmap được lưu tại `outputs/attention_maps/`)*
*   **Ảnh đơn lẻ (Beam Search nâng cao):**
    ```bash
    python -m src.attention_bilstm.predict --image duong_dan_anh.png --beam
    ```
*   **Toàn bộ thư mục ảnh (tên file = nhãn chuẩn):**
    ```bash
    python -m src.attention_bilstm.predict --image duong_dan_thu_muc/
    ```

</details>

---

## 👥 Phân công nhiệm vụ thành viên (Task Allocation)

Dưới đây là bảng phân công công việc chi tiết cho các thành viên trong nhóm thực hiện dự án:

| STT | Họ và tên | MSSV | Nhiệm vụ cụ thể | Đóng góp |
| :---: | :--- | :---: | :--- | :---: |
| 1 | **Nguyễn Trí Hiếu** | 202416203 | - Thu thập, chuẩn bị và làm sạch dữ liệu IAM (`prepare_data.py`).<br>- Thiết kế và triển khai mô hình chính **Attention Model** (Bahdanau Attention + GRU Decoder).<br>- Xây dựng pipeline huấn luyện, tinh chỉnh siêu tham số và Teacher Forcing cho Attention Model.<br>- Biên soạn báo cáo kỹ thuật tổng hợp và chuyển đổi sang báo cáo LaTeX (`Report_HTR`). | 100% |
| 2 | **Đỗ Hải Đăng** | 202400035 | - Triển khai cấu trúc Encoder-Decoder dùng chung và trích xuất đặc trưng với ResNet18.<br>- Triển khai mô hình baseline **CTC Baseline** (ResNet18 + BiLSTM + CTC Loss).<br>- Xây dựng pipeline huấn luyện, đánh giá cho CTC Baseline.<br>- Chuẩn bị tài liệu thuyết trình (slides) và dàn ý báo cáo. | 100% |
| 3 | **Đinh Thái Sơn** | 202416746 | - Triển khai các thuật toán giải mã ở thư mục `inference/` (CTC decode, Greedy decode, Beam Search).<br>- Viết mã nguồn tính toán các chỉ số đánh giá (Word Accuracy, CER, NED).<br>- Phát triển công cụ trực quan hóa Attention Map (`save_attention_map`).<br>- Viết script dự đoán ảnh thực tế (`predict.py`) và thực hiện kiểm thử hệ thống. | 100% |

---

## 📄 Tài liệu chi tiết nộp bài
*   **Dành cho viết báo cáo (Word/LaTeX):** [report/project_report.md](file:///d:/Downloads/HRT-Project/report/project_report.md)
*   **Dành cho thiết kế slide thuyết trình:** [powerpoint/slides_outline.md](file:///d:/Downloads/HRT-Project/powerpoint/slides_outline.md)

