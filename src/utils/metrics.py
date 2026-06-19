import editdistance
from typing import List, Dict

def edit_distance(pred: str, target: str) -> int:
    """
    Tính khoảng cách Levenshtein (Edit Distance) giữa 2 chuỗi.
    Sử dụng thư viện C-optimized `editdistance`.
    """
    return editdistance.eval(pred, target)

def word_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Tính tỷ lệ từ dự đoán chính xác hoàn toàn (exact match) theo phần trăm [0, 100].
    """
    if not targets:
        return 0.0
    
    correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    return (correct / len(targets)) * 100.0

def character_error_rate(predictions: List[str], targets: List[str]) -> float:
    """
    Tính Character Error Rate (CER) = tổng edit distance / tổng độ dài nhãn thật theo phần trăm [0, 100].
    """
    if not targets:
        return 0.0
    
    total_dist = 0
    total_len = 0
    
    for pred, target in zip(predictions, targets):
        total_dist += edit_distance(pred, target)
        total_len += len(target)
        
    if total_len == 0:
        return 0.0
        
    return (total_dist / total_len) * 100.0

def normalized_edit_distance(predictions: List[str], targets: List[str]) -> float:
    """
    Tính Normalized Edit Distance (NED) = 1 - mean(edit_dist / max_len) theo phần trăm [0, 100].
    """
    if not targets:
        return 0.0
        
    normalized_dists = []
    
    for pred, target in zip(predictions, targets):
        dist = edit_distance(pred, target)
        max_len = max(len(pred), len(target))
        
        if max_len == 0:
            norm_dist = 0.0
        else:
            norm_dist = dist / max_len
            
        normalized_dists.append(norm_dist)
        
    mean_norm_dist = sum(normalized_dists) / len(normalized_dists)
    return (1.0 - mean_norm_dist) * 100.0

def compute_all_metrics(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """
    Tính toán và trả về một dictionary gồm các metrics: word_accuracy, cer, ned.
    """
    return {
        "word_accuracy": word_accuracy(predictions, targets),
        "cer": character_error_rate(predictions, targets),
        "ned": normalized_edit_distance(predictions, targets)
    }