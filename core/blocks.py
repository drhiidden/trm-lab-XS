"""
Bloques básicos para la arquitectura TRM
Incluye implementaciones de MLP, Attention, MLP-Mixer y otros componentes fundamentales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .utils import RMSNorm, SwiGLU, RotaryEmbedding, apply_rotary_pos_emb


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention con soporte para rotary embeddings
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_rotary: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_rotary = use_rotary
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        if use_rotary:
            self.rotary_emb = RotaryEmbedding(self.d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Proyecciones Q, K, V
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Aplicar rotary embeddings si está habilitado
        if self.use_rotary:
            cos, sin = self.rotary_emb(x, seq_len)
            q = apply_rotary_pos_emb(q.transpose(1, 2), cos, sin).transpose(1, 2)
            k = apply_rotary_pos_emb(k.transpose(1, 2), cos, sin).transpose(1, 2)
        
        # Calcular atención
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Aplicar atención a los valores
        context = torch.matmul(attn_weights, v)
        
        # Reorganizar y proyectar
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        return output


class MLPBlock(nn.Module):
    """
    Bloque MLP con SwiGLU y normalización RMS
    """
    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or int(2.67 * d_model)
        
        self.norm = RMSNorm(d_model)
        self.swiglu = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        return x + residual


class AttentionBlock(nn.Module):
    """
    Bloque de atención con normalización y residual connection
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_rotary: bool = True):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, use_rotary)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        residual = x
        x = self.norm(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        return x + residual


class MLPMixerBlock(nn.Module):
    """
    Bloque MLP-Mixer para procesamiento de secuencias
    Alterna entre mezcla de tokens y mezcla de canales
    """
    def __init__(self, seq_len: int, d_model: int, token_mix_dim: Optional[int] = None, 
                 channel_mix_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        token_mix_dim = token_mix_dim or seq_len
        channel_mix_dim = channel_mix_dim or int(2.67 * d_model)
        
        self.norm1 = RMSNorm(d_model)
        self.token_mix = nn.Sequential(
            nn.Linear(seq_len, token_mix_dim),
            nn.GELU(),
            nn.Linear(token_mix_dim, seq_len),
            nn.Dropout(dropout)
        )
        
        self.norm2 = RMSNorm(d_model)
        self.channel_mix = nn.Sequential(
            nn.Linear(d_model, channel_mix_dim),
            nn.GELU(),
            nn.Linear(channel_mix_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        residual = x
        
        # Token mixing
        x = self.norm1(x)
        x = x.transpose(-1, -2)  # (batch_size, d_model, seq_len)
        x = self.token_mix(x)
        x = x.transpose(-1, -2)  # (batch_size, seq_len, d_model)
        x = x + residual
        
        # Channel mixing
        residual = x
        x = self.norm2(x)
        x = self.channel_mix(x)
        x = x + residual
        
        return x


class TransformerBlock(nn.Module):
    """
    Bloque Transformer completo con atención y MLP
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: Optional[int] = None, 
                 dropout: float = 0.1, use_rotary: bool = True):
        super().__init__()
        self.attention = AttentionBlock(d_model, n_heads, dropout, use_rotary)
        self.mlp = MLPBlock(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        x = self.attention(x, mask)
        x = self.mlp(x)
        return x


class ConvBlock(nn.Module):
    """
    Bloque convolucional para procesamiento de imágenes
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, use_bn: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, height, width)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """
    Bloque residual para redes profundas
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        residual = x
        
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x + residual


class StateUpdateBlock(nn.Module):
    """
    Bloque para actualizar estados en TRM
    Combina información del estado actual con la entrada
    """
    def __init__(self, d_model: int, d_state: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Proyección de entrada a estado
        self.input_proj = nn.Linear(d_model, d_state)
        
        # Actualización de estado
        self.state_update = nn.Sequential(
            nn.Linear(d_state * 2, d_state),
            nn.GELU(),
            nn.Linear(d_state, d_state),
            nn.Dropout(dropout)
        )
        
        # Proyección de estado a salida
        self.output_proj = nn.Linear(d_state, d_model)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len, d_model)
        # state: (batch_size, seq_len, d_state)
        
        # Proyectar entrada al espacio de estado
        x_state = self.input_proj(x)
        
        # Combinar estado actual con entrada
        combined = torch.cat([state, x_state], dim=-1)
        
        # Actualizar estado
        state_delta = self.state_update(combined)
        new_state = state + state_delta
        
        # Proyectar estado actualizado a salida
        output = self.output_proj(new_state)
        
        return output, new_state


class GatingMechanism(nn.Module):
    """
    Mecanismo de gating para controlar el flujo de información
    """
    def __init__(self, d_model: int, gate_type: str = 'sigmoid'):
        super().__init__()
        self.gate_type = gate_type
        self.gate_proj = nn.Linear(d_model, d_model)
        
        if gate_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif gate_type == 'tanh':
            self.activation = nn.Tanh()
        elif gate_type == 'softmax':
            self.activation = nn.Softmax(dim=-1)
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        gate = self.gate_proj(x)
        gate = self.activation(gate)
        return x * gate


class AdaptivePooling(nn.Module):
    """
    Pooling adaptativo que se ajusta al tamaño de entrada
    """
    def __init__(self, output_size: int = 1, pool_type: str = 'avg'):
        super().__init__()
        self.output_size = output_size
        self.pool_type = pool_type
        
        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            raise ValueError(f"Tipo de pooling no soportado: {pool_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, height, width)
        return self.pool(x)


class SpatialAttention(nn.Module):
    """
    Módulo de atención espacial para imágenes
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, height, width)
        b, c, h, w = x.size()
        
        # Global average pooling
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out).view(b, c, 1, 1)
        
        # Global max pooling
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out).view(b, c, 1, 1)
        
        # Combinar y aplicar
        attention = avg_out + max_out
        return x * attention


def create_2d_sinusoidal_embeddings(height: int, width: int, d_model: int) -> torch.Tensor:
    """
    Crear embeddings posicionales 2D sinusoidales para imágenes
    """
    pe = torch.zeros(d_model, height, width)
    
    # Frecuencias para altura y ancho
    div_term_h = torch.exp(torch.arange(0, d_model // 2, 2).float() * 
                          (-math.log(10000.0) / (d_model // 2)))
    div_term_w = torch.exp(torch.arange(0, d_model // 2, 2).float() * 
                          (-math.log(10000.0) / (d_model // 2)))
    
    # Posiciones
    pos_h = torch.arange(height).float().unsqueeze(1)
    pos_w = torch.arange(width).float().unsqueeze(0)
    
    # Embeddings para altura
    pe[0:d_model//2:2, :, :] = torch.sin(pos_h * div_term_h).unsqueeze(2).expand(-1, -1, width)
    pe[1:d_model//2:2, :, :] = torch.cos(pos_h * div_term_h).unsqueeze(2).expand(-1, -1, width)
    
    # Embeddings para ancho
    pe[d_model//2::2, :, :] = torch.sin(pos_w * div_term_w).unsqueeze(0).expand(-1, height, -1)
    pe[d_model//2+1::2, :, :] = torch.cos(pos_w * div_term_w).unsqueeze(0).expand(-1, height, -1)
    
    return pe.unsqueeze(0)  # (1, d_model, height, width)
