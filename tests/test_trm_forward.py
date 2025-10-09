"""
Tests unitarios para el forward pass del modelo TRM
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from core.trm_model import TRM, TRMClassifier, TRMRegressor, TRMSequenceModel
from core.utils import get_device


class TestTRMForward:
    """Tests para el forward pass del modelo TRM"""
    
    @pytest.fixture
    def device(self):
        """Dispositivo para tests"""
        return get_device()
    
    @pytest.fixture
    def trm_config(self):
        """Configuración básica para TRM"""
        return {
            'd_model': 128,
            'n_layers': 2,
            'n_inner': 3,
            'T': 2,
            'use_attention': False,
            'dropout': 0.1,
            'use_ema': False,  # Deshabilitar EMA para tests
            'use_halting': True,
            'max_seq_len': 100
        }
    
    def test_trm_forward_basic(self, device, trm_config):
        """Test forward pass básico"""
        model = TRM(**trm_config).to(device)
        
        # Input de prueba
        batch_size = 4
        seq_len = 50
        x = torch.randn(batch_size, seq_len, device=device)
        
        # Forward pass
        output = model(x)
        
        # Verificar output
        assert isinstance(output, dict)
        assert 'output' in output
        assert output['output'].shape == (batch_size, seq_len, trm_config['d_model'])
    
    def test_trm_forward_with_states(self, device, trm_config):
        """Test forward pass con estados"""
        model = TRM(**trm_config).to(device)
        
        batch_size = 2
        seq_len = 30
        x = torch.randn(batch_size, seq_len, device=device)
        
        # Forward pass con estados
        output = model(x, return_states=True, return_halt_scores=True)
        
        # Verificar output
        assert 'output' in output
        assert 'states' in output
        assert 'halt_scores' in output
        
        # Verificar estados
        states = output['states']
        assert isinstance(states, list)
        assert len(states) > 0
        
        # Verificar halt scores
        halt_scores = output['halt_scores']
        assert isinstance(halt_scores, list)
        assert len(halt_scores) > 0
    
    def test_trm_forward_with_attention(self, device):
        """Test forward pass con atención"""
        config = {
            'd_model': 128,
            'n_layers': 2,
            'n_inner': 3,
            'T': 2,
            'use_attention': True,
            'n_heads': 4,
            'dropout': 0.1,
            'use_ema': False,
            'use_halting': True,
            'max_seq_len': 100
        }
        
        model = TRM(**config).to(device)
        
        batch_size = 2
        seq_len = 40
        x = torch.randn(batch_size, seq_len, device=device)
        
        output = model(x)
        
        assert 'output' in output
        assert output['output'].shape == (batch_size, seq_len, config['d_model'])
    
    def test_trm_classifier(self, device, trm_config):
        """Test TRM para clasificación"""
        num_classes = 10
        model = TRMClassifier(num_classes=num_classes, **trm_config).to(device)
        
        batch_size = 4
        seq_len = 50
        x = torch.randn(batch_size, seq_len, device=device)
        
        output = model(x)
        
        assert 'output' in output
        assert 'logits' in output
        assert output['logits'].shape == (batch_size, seq_len, num_classes)
    
    def test_trm_regressor(self, device, trm_config):
        """Test TRM para regresión"""
        output_dim = 1
        model = TRMRegressor(output_dim=output_dim, **trm_config).to(device)
        
        batch_size = 4
        seq_len = 50
        x = torch.randn(batch_size, seq_len, device=device)
        
        output = model(x)
        
        assert 'output' in output
        assert 'predictions' in output
        assert output['predictions'].shape == (batch_size, seq_len, output_dim)
    
    def test_trm_sequence_model(self, device, trm_config):
        """Test TRM para modelado de secuencias"""
        vocab_size = 1000
        config = trm_config.copy()
        config['vocab_size'] = vocab_size
        
        model = TRMSequenceModel(vocab_size=vocab_size, **config).to(device)
        
        batch_size = 4
        seq_len = 50
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        output = model(x)
        
        assert 'output' in output
        assert 'logits' in output
        assert output['logits'].shape == (batch_size, seq_len, vocab_size)
    
    def test_trm_different_sequence_lengths(self, device, trm_config):
        """Test TRM con diferentes longitudes de secuencia"""
        model = TRM(**trm_config).to(device)
        
        # Diferentes longitudes
        seq_lengths = [10, 25, 50, 75]
        batch_size = 2
        
        for seq_len in seq_lengths:
            x = torch.randn(batch_size, seq_len, device=device)
            output = model(x)
            
            assert output['output'].shape == (batch_size, seq_len, trm_config['d_model'])
    
    def test_trm_gradient_flow(self, device, trm_config):
        """Test que los gradientes fluyen correctamente"""
        model = TRM(**trm_config).to(device)
        
        batch_size = 2
        seq_len = 30
        x = torch.randn(batch_size, seq_len, device=device, requires_grad=True)
        
        output = model(x)
        loss = output['output'].sum()
        loss.backward()
        
        # Verificar que x tiene gradientes
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Verificar que los parámetros del modelo tienen gradientes
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
    
    def test_trm_deterministic(self, device, trm_config):
        """Test que el modelo es determinístico con la misma semilla"""
        torch.manual_seed(42)
        model1 = TRM(**trm_config).to(device)
        
        torch.manual_seed(42)
        model2 = TRM(**trm_config).to(device)
        
        # Copiar pesos
        model2.load_state_dict(model1.state_dict())
        
        batch_size = 2
        seq_len = 30
        x = torch.randn(batch_size, seq_len, device=device)
        
        # Forward pass
        output1 = model1(x)
        output2 = model2(x)
        
        # Verificar que los outputs son idénticos
        assert torch.allclose(output1['output'], output2['output'], atol=1e-6)
    
    def test_trm_parameter_count(self, device, trm_config):
        """Test conteo de parámetros"""
        model = TRM(**trm_config).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert total_params == trainable_params  # Todos los parámetros deben ser entrenables
    
    def test_trm_model_info(self, device, trm_config):
        """Test información del modelo"""
        model = TRM(**trm_config).to(device)
        
        info = model.get_model_info()
        
        assert isinstance(info, dict)
        assert 'd_model' in info
        assert 'n_layers' in info
        assert 'T' in info
        assert 'total_parameters' in info
        
        assert info['d_model'] == trm_config['d_model']
        assert info['n_layers'] == trm_config['n_layers']
        assert info['T'] == trm_config['T']


class TestTRMEdgeCases:
    """Tests para casos edge del modelo TRM"""
    
    @pytest.fixture
    def device(self):
        return get_device()
    
    def test_trm_empty_sequence(self, device):
        """Test con secuencia vacía"""
        config = {
            'd_model': 64,
            'n_layers': 1,
            'n_inner': 2,
            'T': 1,
            'use_attention': False,
            'dropout': 0.0,
            'use_ema': False,
            'use_halting': False,
            'max_seq_len': 10
        }
        
        model = TRM(**config).to(device)
        
        # Secuencia vacía
        x = torch.randn(1, 0, device=device)
        
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            output = model(x)
    
    def test_trm_single_element_sequence(self, device):
        """Test con secuencia de un solo elemento"""
        config = {
            'd_model': 64,
            'n_layers': 1,
            'n_inner': 2,
            'T': 1,
            'use_attention': False,
            'dropout': 0.0,
            'use_ema': False,
            'use_halting': False,
            'max_seq_len': 10
        }
        
        model = TRM(**config).to(device)
        
        # Secuencia de un elemento
        x = torch.randn(1, 1, device=device)
        output = model(x)
        
        assert output['output'].shape == (1, 1, config['d_model'])
    
    def test_trm_large_sequence(self, device):
        """Test con secuencia larga"""
        config = {
            'd_model': 32,
            'n_layers': 1,
            'n_inner': 1,
            'T': 1,
            'use_attention': False,
            'dropout': 0.0,
            'use_ema': False,
            'use_halting': False,
            'max_seq_len': 1000
        }
        
        model = TRM(**config).to(device)
        
        # Secuencia larga
        seq_len = 500
        x = torch.randn(1, seq_len, device=device)
        output = model(x)
        
        assert output['output'].shape == (1, seq_len, config['d_model'])
    
    def test_trm_zero_cycles(self, device):
        """Test con T=0 (sin ciclos)"""
        config = {
            'd_model': 64,
            'n_layers': 1,
            'n_inner': 2,
            'T': 0,  # Sin ciclos
            'use_attention': False,
            'dropout': 0.0,
            'use_ema': False,
            'use_halting': False,
            'max_seq_len': 10
        }
        
        model = TRM(**config).to(device)
        
        x = torch.randn(1, 5, device=device)
        output = model(x)
        
        assert output['output'].shape == (1, 5, config['d_model'])
    
    def test_trm_zero_inner_updates(self, device):
        """Test con n_inner=0 (sin actualizaciones internas)"""
        config = {
            'd_model': 64,
            'n_layers': 1,
            'n_inner': 0,  # Sin actualizaciones internas
            'T': 2,
            'use_attention': False,
            'dropout': 0.0,
            'use_ema': False,
            'use_halting': False,
            'max_seq_len': 10
        }
        
        model = TRM(**config).to(device)
        
        x = torch.randn(1, 5, device=device)
        output = model(x)
        
        assert output['output'].shape == (1, 5, config['d_model'])


if __name__ == '__main__':
    pytest.main([__file__])
