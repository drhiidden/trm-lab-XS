"""
Utilidades para la arquitectura TRM (Two-state Recursive Model)
Incluye RMSNorm, SwiGLU, rotary embeddings, EMA y otras funciones auxiliares.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Más eficiente que LayerNorm estándar
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, dim) o (batch_size, dim)
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    Activación no lineal con gating para mejor rendimiento
    """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or int(2.67 * dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, dim)
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    Mejora la comprensión de posiciones relativas
    """
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precomputar frecuencias
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len, dim)
        if seq_len is None:
            seq_len = x.shape[1]
        
        # Crear posiciones
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Crear cos y sin embeddings
        cos = emb.cos()[None, :, :]
        sin = emb.sin()[None, :, :]
        
        return cos, sin


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Aplicar rotary position embedding a los tensores
    """
    # x: (batch_size, seq_len, dim)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class EMA(nn.Module):
    """
    Exponential Moving Average para pesos del modelo
    Mejora la estabilidad del entrenamiento
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Registrar todos los parámetros
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Actualizar pesos EMA"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model: nn.Module):
        """Aplicar pesos EMA al modelo"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        """Restaurar pesos originales"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class HaltingHead(nn.Module):
    """
    Cabeza de parada para decidir cuándo terminar el procesamiento recursivo
    """
    def __init__(self, dim: int, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.halt_proj = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, dim)
        halt_scores = self.halt_proj(x)  # (batch_size, seq_len, 1)
        return halt_scores.squeeze(-1)  # (batch_size, seq_len)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Crear máscara causal para atención autoregresiva
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()


def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Función de activación GELU
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def get_device() -> torch.device:
    """
    Obtener el dispositivo disponible (CUDA si está disponible)
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: nn.Module) -> int:
    """
    Contar el número de parámetros entrenables en un modelo
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model: nn.Module, init_type: str = 'xavier_uniform'):
    """
    Inicializar pesos del modelo
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(module.weight)
            elif init_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


class PositionalEncoding(nn.Module):
    """
    Codificación posicional sinusoidal para secuencias
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calcular precisión entre predicciones y objetivos
    """
    if predictions.dim() > 1 and predictions.size(-1) > 1:
        # Clasificación multiclase
        pred_classes = predictions.argmax(dim=-1)
        return (pred_classes == targets).float().mean().item()
    else:
        # Clasificación binaria o regresión
        pred_binary = (predictions > 0.5).float()
        return (pred_binary == targets).float().mean().item()


def compute_loss_with_halting(predictions: torch.Tensor, 
                            targets: torch.Tensor, 
                            halt_scores: torch.Tensor,
                            halt_weight: float = 0.1) -> torch.Tensor:
    """
    Calcular pérdida combinada con penalización de parada
    """
    # Pérdida principal (cross-entropy o MSE)
    if predictions.dim() > 1 and predictions.size(-1) > 1:
        main_loss = F.cross_entropy(predictions, targets)
    else:
        main_loss = F.mse_loss(predictions, targets.float())
    
    # Penalización por parada temprana
    halt_penalty = halt_scores.mean()
    
    return main_loss + halt_weight * halt_penalty
