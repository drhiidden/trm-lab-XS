"""
Core module for TRM Lab
"""

from .trm_model import TRM, TRMClassifier, TRMRegressor, TRMSequenceModel
from .blocks import (
    MultiHeadAttention, MLPBlock, AttentionBlock, MLPMixerBlock,
    TransformerBlock, ConvBlock, ResidualBlock, StateUpdateBlock,
    GatingMechanism, AdaptivePooling, SpatialAttention
)
from .utils import (
    RMSNorm, SwiGLU, RotaryEmbedding, EMA, HaltingHead,
    get_device, count_parameters, initialize_weights,
    compute_accuracy, compute_loss_with_halting
)
from .registry import (
    ModelRegistry, registry, register_model, create_model,
    list_models, get_model_info, BaseModel, BaselineModel
)

__all__ = [
    # Models
    'TRM', 'TRMClassifier', 'TRMRegressor', 'TRMSequenceModel',
    
    # Blocks
    'MultiHeadAttention', 'MLPBlock', 'AttentionBlock', 'MLPMixerBlock',
    'TransformerBlock', 'ConvBlock', 'ResidualBlock', 'StateUpdateBlock',
    'GatingMechanism', 'AdaptivePooling', 'SpatialAttention',
    
    # Utils
    'RMSNorm', 'SwiGLU', 'RotaryEmbedding', 'EMA', 'HaltingHead',
    'get_device', 'count_parameters', 'initialize_weights',
    'compute_accuracy', 'compute_loss_with_halting',
    
    # Registry
    'ModelRegistry', 'registry', 'register_model', 'create_model',
    'list_models', 'get_model_info', 'BaseModel', 'BaselineModel'
]
