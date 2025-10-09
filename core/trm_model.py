"""
Implementación de la arquitectura TRM (Two-state Recursive Model)
Basado en el paper arXiv:2510.04871

El modelo TRM utiliza dos estados persistentes (y, z) que se actualizan recursivamente
a través de múltiples ciclos externos (T) e internos (n_inner).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math

from .utils import RMSNorm, EMA, HaltingHead, initialize_weights, get_device
from .blocks import (
    MultiHeadAttention, MLPBlock, MLPMixerBlock, TransformerBlock,
    StateUpdateBlock, GatingMechanism
)


class TRM(nn.Module):
    """
    Two-state Recursive Model (TRM)
    
    Args:
        d_model: Dimensión del modelo
        n_layers: Número de capas
        n_inner: Número de actualizaciones internas de z
        T: Número de ciclos externos
        use_attention: Si usar atención o MLP-Mixer
        n_heads: Número de cabezas de atención (si use_attention=True)
        dropout: Tasa de dropout
        use_ema: Si usar Exponential Moving Average
        ema_decay: Decay rate para EMA
        use_halting: Si usar mecanismo de parada
        halting_threshold: Umbral para parada
        deep_supervision: Número de capas para supervisión profunda
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 2,
        n_inner: int = 6,
        T: int = 3,
        use_attention: bool = False,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        use_halting: bool = True,
        halting_threshold: float = 0.5,
        deep_supervision: int = 16,
        vocab_size: Optional[int] = None,
        max_seq_len: int = 2048,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_inner = n_inner
        self.T = T
        self.use_attention = use_attention
        self.n_heads = n_heads
        self.dropout = dropout
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_halting = use_halting
        self.halting_threshold = halting_threshold
        self.deep_supervision = deep_supervision
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Embedding de entrada (si se especifica vocab_size)
        if vocab_size is not None:
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
        else:
            self.input_projection = nn.Linear(1, d_model)  # Para entradas continuas
        
        # Normalización de entrada
        self.input_norm = RMSNorm(d_model)
        
        # Estados persistentes y y z
        self.y_projection = nn.Linear(d_model, d_model)
        self.z_projection = nn.Linear(d_model, d_model)
        
        # Bloques de procesamiento
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            if use_attention:
                layer = TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    use_rotary=True
                )
            else:
                layer = MLPMixerBlock(
                    seq_len=max_seq_len,
                    d_model=d_model,
                    dropout=dropout
                )
            self.layers.append(layer)
        
        # Bloques de actualización de estado
        self.state_update_blocks = nn.ModuleList([
            StateUpdateBlock(d_model, d_model, dropout)
            for _ in range(n_layers)
        ])
        
        # Mecanismo de gating para estados
        self.y_gate = GatingMechanism(d_model, 'sigmoid')
        self.z_gate = GatingMechanism(d_model, 'sigmoid')
        
        # Cabeza de parada
        if use_halting:
            self.halting_head = HaltingHead(d_model, halting_threshold)
        
        # Salida
        self.output_norm = RMSNorm(d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Supervisión profunda
        if deep_supervision > 0:
            self.deep_supervision_layers = nn.ModuleList([
                nn.Linear(d_model, d_model)
                for _ in range(min(deep_supervision, n_layers))
            ])
        
        # EMA para pesos
        if use_ema:
            self.ema = None  # Se inicializará después
        
        # Inicializar pesos
        self.apply(lambda m: initialize_weights(m, 'xavier_uniform'))
        
        # Inicializar EMA después de la inicialización de pesos
        if use_ema:
            self.ema = EMA(self, ema_decay)

    def initialize_states(self, batch_size: int, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inicializar estados y y z
        """
        y = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        z = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        return y, z

    def embed_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embedding de entrada
        """
        if hasattr(self, 'token_embedding'):
            # Entrada discreta (tokens)
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            token_emb = self.token_embedding(x)
            pos_emb = self.position_embedding(positions)
            return token_emb + pos_emb
        else:
            # Entrada continua
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            return self.input_projection(x)

    def forward(
        self, 
        x: torch.Tensor, 
        y0: Optional[torch.Tensor] = None, 
        z0: Optional[torch.Tensor] = None,
        return_states: bool = False,
        return_halt_scores: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass del modelo TRM
        
        Args:
            x: Entrada (batch_size, seq_len) o (batch_size, seq_len, features)
            y0: Estado inicial y (opcional)
            z0: Estado inicial z (opcional)
            return_states: Si devolver estados intermedios
            return_halt_scores: Si devolver scores de parada
            
        Returns:
            Dict con 'output', 'states' (opcional), 'halt_scores' (opcional)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        device = x.device
        
        # Embedding de entrada
        x_emb = self.embed_input(x)
        x_emb = self.input_norm(x_emb)
        
        # Inicializar estados si no se proporcionan
        if y0 is None:
            y, z = self.initialize_states(batch_size, seq_len, device)
        else:
            y, z = y0, z0
        
        # Almacenar estados para supervisión profunda
        states_history = []
        halt_scores_history = []
        
        # Ciclos externos (T)
        for t in range(self.T):
            # Proyectar estados
            y_proj = self.y_projection(y)
            z_proj = self.z_projection(z)
            
            # Combinar entrada con estados
            combined_input = x_emb + y_proj + z_proj
            
            # Procesar a través de las capas
            layer_output = combined_input
            for i, (layer, state_update) in enumerate(zip(self.layers, self.state_update_blocks)):
                # Procesamiento de la capa
                layer_output = layer(layer_output)
                
                # Actualización de estado z (ciclos internos)
                for _ in range(self.n_inner):
                    layer_output, z = state_update(layer_output, z)
                
                # Aplicar gating a los estados
                y = self.y_gate(y + layer_output)
                z = self.z_gate(z)
                
                # Supervisión profunda
                if i < len(self.deep_supervision_layers):
                    deep_out = self.deep_supervision_layers[i](layer_output)
                    states_history.append(deep_out)
            
            # Calcular scores de parada
            if self.use_halting:
                halt_scores = self.halting_head(layer_output)
                halt_scores_history.append(halt_scores)
                
                # Parada temprana si se alcanza el umbral
                if halt_scores.mean() > self.halting_threshold:
                    break
        
        # Salida final
        output = self.output_norm(layer_output)
        output = self.output_projection(output)
        
        # Actualizar EMA si está habilitado
        if self.use_ema and self.training:
            self.ema.update(self)
        
        # Preparar resultado
        result = {'output': output}
        
        if return_states:
            result['states'] = states_history
        
        if return_halt_scores:
            result['halt_scores'] = halt_scores_history
        
        return result

    def generate(self, x: torch.Tensor, max_length: int = 100, temperature: float = 1.0) -> torch.Tensor:
        """
        Generación autoregresiva
        """
        self.eval()
        with torch.no_grad():
            generated = x.clone()
            
            for _ in range(max_length):
                # Forward pass
                result = self.forward(generated)
                logits = result['output']
                
                # Tomar el último token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Muestrear siguiente token
                if self.vocab_size is not None:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = next_token_logits.unsqueeze(1)
                
                # Añadir a la secuencia
                generated = torch.cat([generated, next_token], dim=1)
            
            return generated

    def get_ema_weights(self) -> Dict[str, torch.Tensor]:
        """
        Obtener pesos EMA
        """
        if self.use_ema and self.ema is not None:
            return self.ema.shadow.copy()
        return {}

    def apply_ema_weights(self):
        """
        Aplicar pesos EMA al modelo
        """
        if self.use_ema and self.ema is not None:
            self.ema.apply_shadow(self)

    def restore_original_weights(self):
        """
        Restaurar pesos originales
        """
        if self.use_ema and self.ema is not None:
            self.ema.restore(self)

    def count_parameters(self) -> int:
        """
        Contar parámetros entrenables
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtener información del modelo
        """
        return {
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_inner': self.n_inner,
            'T': self.T,
            'use_attention': self.use_attention,
            'n_heads': self.n_heads if self.use_attention else None,
            'use_ema': self.use_ema,
            'use_halting': self.use_halting,
            'deep_supervision': self.deep_supervision,
            'total_parameters': self.count_parameters(),
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len
        }


class TRMClassifier(TRM):
    """
    TRM para clasificación
    """
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.classifier_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, num_classes)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        result = super().forward(x, **kwargs)
        logits = self.classifier_head(result['output'])
        result['logits'] = logits
        return result


class TRMRegressor(TRM):
    """
    TRM para regresión
    """
    def __init__(self, output_dim: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.regressor_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, output_dim)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        result = super().forward(x, **kwargs)
        predictions = self.regressor_head(result['output'])
        result['predictions'] = predictions
        return result


class TRMSequenceModel(TRM):
    """
    TRM para modelado de secuencias
    """
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(vocab_size=vocab_size, **kwargs)
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(self.d_model, vocab_size)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        result = super().forward(x, **kwargs)
        logits = self.lm_head(result['output'])
        result['logits'] = logits
        return result

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcular pérdida de lenguaje
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def create_trm_model(config: Dict[str, Any]) -> TRM:
    """
    Factory function para crear modelos TRM
    """
    model_type = config.get('model_type', 'base')
    
    if model_type == 'classifier':
        return TRMClassifier(**config)
    elif model_type == 'regressor':
        return TRMRegressor(**config)
    elif model_type == 'sequence':
        return TRMSequenceModel(**config)
    else:
        return TRM(**config)
