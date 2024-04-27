# Define LORA adaptation (simplified example)
import torch.nn as nn
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, r=16, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module
        kernel_size = int(conv_module.kernel_size[0])
        in_channels = conv_module.in_channels
        out_channels = conv_module.out_channels
        
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
                ,requires_grad=True)
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            ,requires_grad=True)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.merged = False


    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

class LinearLoRA(nn.Module):
    def __init__(self, adapted_layer, rank):
        super(LinearLoRA, self).__init__()
        self.adapted_layer = adapted_layer
        # Initialize low-rank matrices A and B
        self.A = nn.Parameter(torch.randn(adapted_layer.weight.size(1), rank),requires_grad=True)
        self.B = nn.Parameter(torch.zeros(rank, adapted_layer.weight.size(0)),requires_grad=True)

    def forward(self, x):
        low_rank_matrix = self.A @ self.B
        adapted_weight = self.adapted_layer.weight + low_rank_matrix.t()  # Ensure correct shape
        return nn.functional.linear(x, adapted_weight, self.adapted_layer.bias)
    
def modify_resnet_with_lora(model,rank=16, device='cpu'):
    # Recursively replace all convolutional layers with LoRA-enhanced versions
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # Replace the standard convolutional layer with a LoRAConv2d
            setattr(model, name, ConvLoRA(conv_module=module, r=rank))        
        else:
            # Recurse into submodules if not a convolutional layer
            modify_submodules_with_lora(module, rank, device)
    
    model.fc = LinearLoRA(model.fc, rank=rank) 
    
    return model

def modify_submodules_with_lora(submodule, rank, device):
    for name, module in submodule.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(submodule, name, ConvLoRA(conv_module=module, r=rank))
        else:
            modify_submodules_with_lora(module, rank, device)
