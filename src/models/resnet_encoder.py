import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from typing import Tuple

class ResNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, freeze: bool = True, out_layer: str = "layer3"):
        """
        CNN Encoder sử dụng kiến trúc ResNet18.
        - pretrained: Load trọng số pretrain trên ImageNet.
        - freeze: Đóng băng trọng số khi khởi tạo (phục vụ Phase 1 training).
        - out_layer: Lớp cuối cùng giữ lại ('layer3' hoặc 'layer4').
        """
        super().__init__()
        self.out_layer = out_layer
        
        # Load mô hình ResNet18
        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            resnet = models.resnet18(weights=None)
            
        # Tách các thành phần của ResNet
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        
        if out_layer == "layer4":
            self.layer4 = resnet.layer4
        else:
            self.layer4 = None

        # Đóng băng toàn bộ tham số nếu có yêu cầu
        if freeze:
            self.freeze_all()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        Nếu out_layer = 'layer3' -> output: (B, 256, H/16, W/16)
        Nếu out_layer = 'layer4' -> output: (B, 512, H/32, W/32)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        if self.out_layer == "layer4" and self.layer4 is not None:
            x = self.layer4(x)
            
        return x

    def freeze_all(self):
        """Đóng băng toàn bộ encoder."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_from(self, layer_name: str):
        """
        Mở đóng băng các tham số từ layer_name trở đi.
        Các layer hợp lệ: 'conv1', 'layer1', 'layer2', 'layer3', 'layer4'.
        """
        layers = ["conv1", "layer1", "layer2", "layer3", "layer4"]
        if layer_name not in layers:
            raise ValueError(f"layer_name phải thuộc {layers}")
            
        start_idx = layers.index(layer_name)
        
        # Đầu tiên đóng băng tất cả
        self.freeze_all()
        
        # Mở đóng băng cho conv1 và bn1
        if layer_name == "conv1":
            for param in self.conv1.parameters():
                param.requires_grad = True
            for param in self.bn1.parameters():
                param.requires_grad = True
                
        # Mở đóng băng cho các block layer tiếp theo
        if start_idx <= 1:
            for param in self.layer1.parameters():
                param.requires_grad = True
        if start_idx <= 2:
            for param in self.layer2.parameters():
                param.requires_grad = True
        if start_idx <= 3:
            for param in self.layer3.parameters():
                param.requires_grad = True
        if start_idx <= 4 and self.out_layer == "layer4" and self.layer4 is not None:
            for param in self.layer4.parameters():
                param.requires_grad = True

    def get_output_size(self, input_h: int, input_w: int) -> Tuple[int, int, int]:
        """
        Tính toán kích thước đầu ra (Channels, Height, Width) tự động dựa trên kích thước ảnh đầu vào.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_h, input_w)
            out = self.forward(dummy)
            shape = out.shape
            
        if was_training:
            self.train()
            
        return shape[1], shape[2], shape[3]
