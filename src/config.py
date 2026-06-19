import os
import torch

# --- Paths ---
# Tự động xác định project root dựa trên vị trí của config.py
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
LABEL_FILE = os.path.join(DATASET_DIR, "label.txt")
WORDS_DIR = os.path.join(DATASET_DIR, "words")

DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data_processed")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
ATTENTION_MAP_DIR = os.path.join(OUTPUT_DIR, "attention_maps")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Tự động tạo các thư mục output nếu chưa tồn tại
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ATTENTION_MAP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Image ---
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 512
IMAGE_CHANNELS = 3         # 3 vì ResNet18 pretrained cần RGB

# --- Vocabulary ---
MAX_LABEL_LENGTH = 32
MIN_LABEL_LENGTH = 1
# Bộ ký tự chuẩn tiếng Anh (chữ thường) dùng cho Phase 1
VOCAB_CHARS = "abcdefghijklmnopqrstuvwxyz"

# --- Model ---
ENCODER_MODEL = "resnet18"
EMBED_DIM = 128
DECODER_HIDDEN_DIM = 256
ATTENTION_DIM = 256
DECODER_NUM_LAYERS = 1
DROPOUT = 0.3

# --- Attention BiLSTM (Context LSTM) ---
ATTENTION_LSTM_INPUT_DIM = 1024
ATTENTION_LSTM_HIDDEN_DIM = 256
ATTENTION_LSTM_NUM_LAYERS = 2
ATTENTION_LSTM_DROPOUT = 0.3
ATTENTION_LSTM_BIDIRECTIONAL = True
ATTENTION_ENCODER_DIM = 512  # ATTENTION_LSTM_HIDDEN_DIM * 2


# --- Training ---
BATCH_SIZE = 256
NUM_EPOCHS = 80
LEARNING_RATE_DECODER = 1e-3
LEARNING_RATE_ENCODER = 5e-5
WEIGHT_DECAY = 1e-5
TEACHER_FORCING_RATIO = 0.5
TEACHER_FORCING_DECAY = 0.01    # Giảm dần mỗi epoch
GRAD_CLIP = 5.0
EARLY_STOPPING_PATIENCE = 10
USE_AMP = True                  # Mixed precision
 
# --- Data ---
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
NUM_WORKERS = 4                 # Đã tăng từ 0 lên 4 để tối ưu hóa đa luồng trên Windows
PIN_MEMORY = True
RANDOM_SEED = 42

# --- Inference ---
BEAM_WIDTH = 2

# --- CTC Baseline ---
CTC_HIDDEN_DIM = 256
CTC_NUM_LAYERS = 2
CTC_LEARNING_RATE = 1e-3
CTC_BATCH_SIZE = 256
CTC_EPOCHS = 50

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"