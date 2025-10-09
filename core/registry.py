"""
Sistema de registro de modelos para el monorrepo TRM
Permite registrar y crear fácilmente diferentes arquitecturas y variantes.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Type, Optional, Callable, List
from abc import ABC, abstractmethod
import inspect
import yaml
import json
from pathlib import Path

from .trm_model import TRM, TRMClassifier, TRMRegressor, TRMSequenceModel


class ModelRegistry:
    """
    Registro centralizado de modelos y arquitecturas
    """
    
    def __init__(self):
        self._models: Dict[str, Type[nn.Module]] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._factories: Dict[str, Callable] = {}
        self._aliases: Dict[str, str] = {}
        
        # Registrar modelos por defecto
        self._register_default_models()
    
    def register_model(
        self, 
        name: str, 
        model_class: Type[nn.Module], 
        config: Optional[Dict[str, Any]] = None,
        factory: Optional[Callable] = None,
        aliases: Optional[List[str]] = None
    ):
        """
        Registrar un modelo en el registro
        
        Args:
            name: Nombre único del modelo
            model_class: Clase del modelo
            config: Configuración por defecto
            factory: Función factory personalizada
            aliases: Alias alternativos para el modelo
        """
        self._models[name] = model_class
        
        if config is not None:
            self._configs[name] = config
        
        if factory is not None:
            self._factories[name] = factory
        else:
            # Crear factory por defecto
            self._factories[name] = lambda cfg: model_class(**cfg)
        
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name
    
    def create_model(self, name: str, config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """
        Crear una instancia del modelo
        
        Args:
            name: Nombre del modelo (o alias)
            config: Configuración del modelo
            
        Returns:
            Instancia del modelo
        """
        # Resolver alias
        actual_name = self._aliases.get(name, name)
        
        if actual_name not in self._models:
            raise ValueError(f"Modelo '{name}' no está registrado. Modelos disponibles: {list(self._models.keys())}")
        
        # Combinar configuración por defecto con la proporcionada
        default_config = self._configs.get(actual_name, {})
        if config is None:
            config = default_config.copy()
        else:
            merged_config = default_config.copy()
            merged_config.update(config)
            config = merged_config
        
        # Crear modelo usando factory
        factory = self._factories[actual_name]
        return factory(config)
    
    def list_models(self) -> List[str]:
        """
        Listar todos los modelos registrados
        """
        return list(self._models.keys())
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """
        Obtener información sobre un modelo
        """
        actual_name = self._aliases.get(name, name)
        
        if actual_name not in self._models:
            raise ValueError(f"Modelo '{name}' no está registrado")
        
        model_class = self._models[actual_name]
        config = self._configs.get(actual_name, {})
        
        return {
            'name': actual_name,
            'class': model_class.__name__,
            'module': model_class.__module__,
            'config': config,
            'aliases': [alias for alias, model_name in self._aliases.items() if model_name == actual_name]
        }
    
    def save_config(self, name: str, path: str):
        """
        Guardar configuración de un modelo
        """
        config = self._configs.get(name, {})
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def load_config(self, path: str) -> Dict[str, Any]:
        """
        Cargar configuración desde archivo
        """
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _register_default_models(self):
        """
        Registrar modelos por defecto
        """
        # TRM básico
        self.register_model(
            'trm_base',
            TRM,
            {
                'd_model': 512,
                'n_layers': 2,
                'n_inner': 6,
                'T': 3,
                'use_attention': False,
                'dropout': 0.1,
                'use_ema': True,
                'use_halting': True
            },
            aliases=['trm', 'trm_mlp']
        )
        
        # TRM con atención
        self.register_model(
            'trm_attention',
            TRM,
            {
                'd_model': 512,
                'n_layers': 2,
                'n_inner': 6,
                'T': 3,
                'use_attention': True,
                'n_heads': 8,
                'dropout': 0.1,
                'use_ema': True,
                'use_halting': True
            },
            aliases=['trm_attn', 'trm_transformer']
        )
        
        # TRM para clasificación
        self.register_model(
            'trm_classifier',
            TRMClassifier,
            {
                'd_model': 512,
                'n_layers': 2,
                'n_inner': 6,
                'T': 3,
                'use_attention': False,
                'dropout': 0.1,
                'use_ema': True,
                'use_halting': True,
                'num_classes': 10
            }
        )
        
        # TRM para regresión
        self.register_model(
            'trm_regressor',
            TRMRegressor,
            {
                'd_model': 512,
                'n_layers': 2,
                'n_inner': 6,
                'T': 3,
                'use_attention': False,
                'dropout': 0.1,
                'use_ema': True,
                'use_halting': True,
                'output_dim': 1
            }
        )
        
        # TRM para secuencias
        self.register_model(
            'trm_sequence',
            TRMSequenceModel,
            {
                'd_model': 512,
                'n_layers': 2,
                'n_inner': 6,
                'T': 3,
                'use_attention': True,
                'n_heads': 8,
                'dropout': 0.1,
                'use_ema': True,
                'use_halting': True,
                'vocab_size': 10000
            }
        )
        
        # Variantes de tamaño
        self.register_model(
            'trm_small',
            TRM,
            {
                'd_model': 256,
                'n_layers': 1,
                'n_inner': 3,
                'T': 2,
                'use_attention': False,
                'dropout': 0.1,
                'use_ema': True,
                'use_halting': True
            }
        )
        
        self.register_model(
            'trm_large',
            TRM,
            {
                'd_model': 1024,
                'n_layers': 4,
                'n_inner': 12,
                'T': 5,
                'use_attention': True,
                'n_heads': 16,
                'dropout': 0.1,
                'use_ema': True,
                'use_halting': True
            }
        )


# Instancia global del registro
registry = ModelRegistry()


# Decorador para registrar modelos automáticamente
def register_model(name: str, config: Optional[Dict[str, Any]] = None, aliases: Optional[List[str]] = None):
    """
    Decorador para registrar modelos automáticamente
    """
    def decorator(model_class):
        registry.register_model(name, model_class, config, aliases=aliases)
        return model_class
    return decorator


# Funciones de conveniencia
def create_model(name: str, config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Crear modelo usando el registro global
    """
    return registry.create_model(name, config)


def list_models() -> List[str]:
    """
    Listar modelos disponibles
    """
    return registry.list_models()


def get_model_info(name: str) -> Dict[str, Any]:
    """
    Obtener información del modelo
    """
    return registry.get_model_info(name)


# Clases base para modelos personalizados
class BaseModel(nn.Module, ABC):
    """
    Clase base para modelos personalizados
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineModel(BaseModel):
    """
    Clase base para modelos baseline (CNN, LSTM, Transformer, etc.)
    """
    
    def __init__(self, model_type: str, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type


# Modelos baseline para comparación
@register_model('transformer_baseline', {
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 6,
    'd_ff': 2048,
    'dropout': 0.1,
    'max_seq_len': 2048
})
class TransformerBaseline(BaselineModel):
    """
    Transformer estándar para comparación
    """
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, n_layers: int = 6, 
                 d_ff: int = 2048, dropout: float = 0.1, max_seq_len: int = 2048, **kwargs):
        super().__init__('transformer', **kwargs)
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(10000, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Capas Transformer
        from .blocks import TransformerBlock
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Normalización final
        from .utils import RMSNorm
        self.norm = RMSNorm(d_model)
        
        # Salida
        self.output_projection = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embeddings
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        
        # Procesar a través de las capas
        for layer in self.layers:
            x = layer(x)
        
        # Salida
        x = self.norm(x)
        x = self.output_projection(x)
        
        return x


@register_model('lstm_baseline', {
    'input_size': 512,
    'hidden_size': 512,
    'num_layers': 2,
    'dropout': 0.1,
    'bidirectional': False
})
class LSTMBaseline(BaselineModel):
    """
    LSTM estándar para comparación
    """
    
    def __init__(self, input_size: int = 512, hidden_size: int = 512, 
                 num_layers: int = 2, dropout: float = 0.1, 
                 bidirectional: bool = False, **kwargs):
        super().__init__('lstm', **kwargs)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        output_size = hidden_size * (2 if bidirectional else 1)
        self.output_projection = nn.Linear(output_size, input_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        output = self.output_projection(lstm_out)
        return output


@register_model('cnn_baseline', {
    'input_channels': 1,
    'num_classes': 10,
    'dropout': 0.1
})
class CNNBaseline(BaselineModel):
    """
    CNN estándar para comparación
    """
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10, 
                 dropout: float = 0.1, **kwargs):
        super().__init__('cnn', **kwargs)
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Capas convolucionales
        from .blocks import ConvBlock
        self.conv_layers = nn.Sequential(
            ConvBlock(input_channels, 32, 3, 1, 1),
            ConvBlock(32, 64, 3, 2, 1),
            ConvBlock(64, 128, 3, 2, 1),
            ConvBlock(128, 256, 3, 2, 1),
        )
        
        # Pooling global
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, height, width)
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Funciones de utilidad para el registro
def create_model_from_config(config_path: str) -> nn.Module:
    """
    Crear modelo desde archivo de configuración
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_name = config.pop('model_name')
    return create_model(model_name, config)


def save_model_config(model: nn.Module, config_path: str):
    """
    Guardar configuración del modelo
    """
    if hasattr(model, 'get_config'):
        config = model.get_config()
    else:
        config = {}
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def compare_models(model_names: List[str], configs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Comparar información de múltiples modelos
    """
    if configs is None:
        configs = [{}] * len(model_names)
    
    comparison = {}
    for name, config in zip(model_names, configs):
        try:
            model = create_model(name, config)
            comparison[name] = {
                'parameters': model.count_parameters(),
                'config': config,
                'info': get_model_info(name)
            }
        except Exception as e:
            comparison[name] = {'error': str(e)}
    
    return comparison
