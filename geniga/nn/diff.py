# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange #pip install einops
from typing import List
import math
import numpy as np
import os  # Add this import for directory handling

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        # Register as a non-trainable buffer so it automatically moves with the model
        self.register_buffer("embeddings", embeddings, persistent=False)

    def forward(self, x, t):
        # Ensure the index tensor lives on the same device as the embedding table
        t = t.to(self.embeddings.device)
        embeds = self.embeddings[t]
        return embeds[:, :, None, None]

class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation for scalar/vector conditioning"""
    def __init__(self, input_dim: int, feature_dim: int):
        super().__init__()
        self.scale_transform = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.shift_transform = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, conditioning_input, features):
        """
        Args:
            conditioning_input: (B, input_dim) - the conditioning parameter(s)
            features: (B, C, H, W) - feature maps to modulate
        Returns:
            Modulated features: (B, C, H, W)
        """
        scale = self.scale_transform(conditioning_input).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        shift = self.shift_transform(conditioning_input).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return features * (1 + scale) + shift

# Residual Blocks
class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float, use_film: bool = False, conditioning_dim: int = 1):
        super().__init__()
        self.use_film = use_film
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)
        
        if use_film:
            self.film_conditioner = FiLMConditioner(input_dim=conditioning_dim, feature_dim=C)

    def forward(self, x, embeddings, scalar_conditioning=None):
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        
        # Apply FiLM conditioning if enabled
        if self.use_film and scalar_conditioning is not None:
            r = self.film_conditioner(scalar_conditioning, r)
        
        return r + x
    
    
class Attention(nn.Module):
    def __init__(self, C: int, num_heads:int , dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C*3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q,k,v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q,k,v, is_causal=False, dropout_p=self.dropout_prob)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')
    
class UnetLayer(nn.Module):
    def __init__(self, 
            upscale: bool, 
            attention: bool, 
            num_groups: int, 
            dropout_prob: float,
            num_heads: int,
            C: int,
            use_film: bool = False,
            conditioning_dim: int = 1):
        super().__init__()
        self.ResBlock1 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob, use_film=use_film, conditioning_dim=conditioning_dim)
        self.ResBlock2 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob, use_film=use_film, conditioning_dim=conditioning_dim)
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C//2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(C, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x, embeddings, scalar_conditioning=None):
        x = self.ResBlock1(x, embeddings, scalar_conditioning)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings, scalar_conditioning)
        return self.conv(x), x
    
class UNET(nn.Module):
    def __init__(self,
            Channels: List = [32, 64, 128, 256, 256, 192],
            Attentions: List = [False, True, False, False, False, True],
            Upscales: List = [False, False, False, True, True, True],
            num_groups: int = 4,
            dropout_prob: float = 0.1,
            num_heads: int = 8,
            input_channels: int = 3,  # Updated to 3 input channels
            output_channels: int = 1,
            time_steps: int = 1000,
            use_scalar_conditioning: bool = False,
            conditioning_dim: int = 1):  # New parameter for vector conditioning
        super().__init__()
        self.num_layers = len(Channels)
        self.use_scalar_conditioning = use_scalar_conditioning
        self.conditioning_dim = conditioning_dim
        
        # Conditioning mode: 'film' for FiLM conditioning, 'channel' for channel concatenation
        self.conditioning_mode = 'film'  # Default to FiLM for better vector support
        
        # Adjust input channels if using channel concatenation conditioning
        self.base_input_channels = input_channels
        actual_input_channels = input_channels  # Will be adjusted dynamically
        
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)
        # Separate conv for conditioning channels if using channel concatenation
        if use_scalar_conditioning:
            self.conditioning_conv = nn.Conv2d(conditioning_dim, Channels[0]//4, kernel_size=3, padding=1)
            self.combine_conv = nn.Conv2d(Channels[0] + Channels[0]//4, Channels[0], kernel_size=1)
        
        out_channels = (Channels[-1]//2)+Channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(Channels))
        
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads,
                use_film=(use_scalar_conditioning and self.conditioning_mode == 'film'),
                conditioning_dim=conditioning_dim
            )
            setattr(self, f'Layer{i+1}', layer)

    def set_conditioning_mode(self, conditioning_mode: str = 'film'):
        """
        Set the conditioning mode
        
        Args:
            conditioning_mode: 'film' for FiLM conditioning, 'channel' for channel concatenation
        """
        if not self.use_scalar_conditioning:
            raise ValueError("use_scalar_conditioning must be True to use conditioning modes")
        
        if conditioning_mode not in ['film', 'channel']:
            raise ValueError("conditioning_mode must be 'film' or 'channel'")
            
        self.conditioning_mode = conditioning_mode
        
        # Update layers to use FiLM conditioning or not
        use_film = (conditioning_mode == 'film')
        for i in range(self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            layer.ResBlock1.use_film = use_film
            layer.ResBlock2.use_film = use_film
            
        print(f"Conditioning mode set to: {conditioning_mode.upper()}")

    def _process_conditioning_input(self, conditioning_param, batch_size, device, dtype):
        """
        Process conditioning input to ensure consistent format
        
        Args:
            conditioning_param: scalar, tensor, or None
            batch_size: batch size
            device: target device
            dtype: target dtype
            
        Returns:
            torch.Tensor: (B, conditioning_dim) tensor
        """
        if conditioning_param is None:
            return None
            
        if torch.is_tensor(conditioning_param):
            if conditioning_param.dim() == 0:  # scalar tensor
                return conditioning_param.unsqueeze(0).unsqueeze(0).expand(batch_size, self.conditioning_dim)
            elif conditioning_param.dim() == 1:
                # Prioritize conditioning_dim interpretation over batch_size
                if conditioning_param.shape[0] == self.conditioning_dim:
                    # (conditioning_dim,) -> (B, conditioning_dim) - same conditioning for all batch items
                    return conditioning_param.unsqueeze(0).expand(batch_size, self.conditioning_dim)
                elif conditioning_param.shape[0] == batch_size and self.conditioning_dim == 1:
                    # (B,) -> (B, 1) - different scalar conditioning per batch item
                    return conditioning_param.unsqueeze(1)
                else:
                    raise ValueError(f"conditioning_param shape {conditioning_param.shape} doesn't match conditioning_dim {self.conditioning_dim}. "
                                   f"Expected shape: ({self.conditioning_dim},) for same conditioning across batch, "
                                   f"or ({batch_size}, {self.conditioning_dim}) for different conditioning per batch item.")
            elif conditioning_param.dim() == 2:
                # (B, conditioning_dim) - ideal format
                if conditioning_param.shape == (batch_size, self.conditioning_dim):
                    return conditioning_param
                else:
                    raise ValueError(f"conditioning_param shape {conditioning_param.shape} doesn't match expected (B={batch_size}, conditioning_dim={self.conditioning_dim})")
            else:
                raise ValueError(f"conditioning_param has too many dimensions: {conditioning_param.dim()}")
        else:  # python scalar
            return torch.full((batch_size, self.conditioning_dim), conditioning_param, dtype=dtype, device=device)

    def forward(self, x, t, scalar_param=None):
        B, C, H, W = x.shape
        
        # Handle scalar conditioning
        conditioning_tensor = None
        if self.use_scalar_conditioning:
            if scalar_param is None:
                raise ValueError("scalar_param must be provided when use_scalar_conditioning=True")
                
            conditioning_tensor = self._process_conditioning_input(
                scalar_param, B, x.device, x.dtype
            )
            
            # For channel concatenation, add conditioning as additional channels
            if self.conditioning_mode == 'channel':
                # Create spatial conditioning maps
                conditioning_maps = conditioning_tensor.unsqueeze(-1).unsqueeze(-1).expand(B, self.conditioning_dim, H, W)
                # Process conditioning channels
                conditioning_features = self.conditioning_conv(conditioning_maps)
                # Combine with main features
                x = self.shallow_conv(x)
                x = self.combine_conv(torch.cat([x, conditioning_features], dim=1))
            else:
                # For FiLM conditioning, just process main input
                x = self.shallow_conv(x)
        else:
            x = self.shallow_conv(x)
            
        residuals = []
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(x, t)
            x, r = layer(x, embeddings, conditioning_tensor)
            residuals.append(r)
            
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(x, t)
            x, _ = layer(x, embeddings, conditioning_tensor)
            x = torch.concat((x, residuals[self.num_layers-i-1]), dim=1)
            
        return self.output_conv(self.relu(self.late_conv(x)))
    
class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t, device="cuda"):
        return self.beta.to(device)[t], self.alpha.to(device)[t]