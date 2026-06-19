from typing import List

class Vocabulary:
    PAD_TOKEN = "<pad>"  # index 0
    SOS_TOKEN = "<sos>"  # index 1
    EOS_TOKEN = "<eos>"  # index 2

    def __init__(self, chars: str):
        """
        Khởi tạo vocab từ chuỗi ký tự cho phép.
        Tự động thêm <pad>, <sos>, <eos> vào các index 0, 1, 2.
        """
        self.chars = chars
        self.special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        
        # Tạo mapping
        self._char_to_idx = {token: i for i, token in enumerate(self.special_tokens)}
        for i, char in enumerate(self.chars):
            self._char_to_idx[char] = len(self.special_tokens) + i
            
        self._idx_to_char = {idx: char for char, idx in self._char_to_idx.items()}

    @property
    def pad_idx(self) -> int:
        return 0

    @property
    def sos_idx(self) -> int:
        return 1

    @property
    def eos_idx(self) -> int:
        return 2

    @property
    def size(self) -> int:
        return len(self._char_to_idx)

    def char_to_idx_fn(self, char: str) -> int:
        """Trả về index của ký tự, nếu không có trả về lỗi Key."""
        return self._char_to_idx[char]

    def idx_to_char_fn(self, idx: int) -> str:
        """Trả về ký tự của index, nếu không có trả về lỗi Key."""
        return self._idx_to_char[idx]

    def encode(self, text: str) -> List[int]:
        """
        Chuyển đổi chuỗi text thành list các indices tương ứng.
        LƯU Ý: Không tự động thêm <sos> hoặc <eos>.
        """
        return [self._char_to_idx[char] for char in text]

    def decode(self, indices: List[int], stop_at_eos: bool = True) -> str:
        """
        Chuyển list of indices thành chuỗi text.
        Bỏ qua <pad>, <sos>. Dừng lại tại <eos> nếu stop_at_eos=True.
        """
        decoded_chars = []
        for idx in indices:
            if idx == self.pad_idx or idx == self.sos_idx:
                continue
            if idx == self.eos_idx and stop_at_eos:
                break
            # Trường hợp index nằm ngoài vocab
            if idx in self._idx_to_char:
                char = self._idx_to_char[idx]
                if char not in self.special_tokens:
                    decoded_chars.append(char)
        return "".join(decoded_chars)

    def is_valid_label(self, text: str) -> bool:
        """
        Kiểm tra chuỗi text chỉ chứa ký tự hợp lệ nằm trong vocab
        (không bao gồm các ký tự đặc biệt).
        """
        if not text:
            return False
        return all(char in self._char_to_idx for char in text)