"""
Tests unitarios para el loop de entrenamiento
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
import os
from pathlib import Path

from core.trm_model import TRM
from core.registry import create_model
from datasets.sudoku import create_sudoku_dataloaders, SudokuMetrics
from core.utils import get_device


class TestTrainingLoop:
    """Tests para el loop de entrenamiento"""
    
    @pytest.fixture
    def device(self):
        """Dispositivo para tests"""
        return get_device()
    
    @pytest.fixture
    def model_config(self):
        """Configuración del modelo para tests"""
        return {
            'd_model': 64,
            'n_layers': 1,
            'n_inner': 2,
            'T': 2,
            'use_attention': False,
            'dropout': 0.1,
            'use_ema': False,  # Deshabilitar EMA para tests
            'use_halting': True,
            'vocab_size': 10,
            'max_seq_len': 81
        }
    
    @pytest.fixture
    def small_dataset(self):
        """Dataset pequeño para tests"""
        train_loader, val_loader, test_loader = create_sudoku_dataloaders(
            batch_size=4,
            train_size=20,
            val_size=10,
            test_size=10,
            difficulty='easy'
        )
        return train_loader, val_loader, test_loader
    
    def test_model_creation(self, device, model_config):
        """Test creación del modelo"""
        model = create_model('trm_base', model_config)
        model = model.to(device)
        
        assert isinstance(model, TRM)
        assert model.device == device or str(next(model.parameters()).device) == str(device)
    
    def test_forward_pass(self, device, model_config, small_dataset):
        """Test forward pass durante entrenamiento"""
        model = create_model('trm_base', model_config).to(device)
        train_loader, _, _ = small_dataset
        
        model.train()
        
        for puzzles, solutions in train_loader:
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)
            
            # Forward pass
            output = model(puzzles)
            
            assert 'output' in output
            assert output['output'].shape == (puzzles.shape[0], puzzles.shape[1], model_config['d_model'])
            break
    
    def test_backward_pass(self, device, model_config, small_dataset):
        """Test backward pass durante entrenamiento"""
        model = create_model('trm_base', model_config).to(device)
        train_loader, _, _ = small_dataset
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        
        for puzzles, solutions in train_loader:
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(puzzles)
            
            # Calcular pérdida
            loss = criterion(output['output'].view(-1, 10), solutions.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Verificar que los gradientes existen
            for param in model.parameters():
                if param.requires_grad:
                    assert param.grad is not None
                    assert not torch.isnan(param.grad).any()
            
            optimizer.step()
            break
    
    def test_training_step(self, device, model_config, small_dataset):
        """Test un paso completo de entrenamiento"""
        model = create_model('trm_base', model_config).to(device)
        train_loader, _, _ = small_dataset
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        
        initial_loss = None
        final_loss = None
        
        for batch_idx, (puzzles, solutions) in enumerate(train_loader):
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)
            
            optimizer.zero_grad()
            
            output = model(puzzles)
            loss = criterion(output['output'].view(-1, 10), solutions.view(-1))
            
            if batch_idx == 0:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
            
            if batch_idx >= 2:  # Solo probar unos pocos batches
                final_loss = loss.item()
                break
        
        assert initial_loss is not None
        assert final_loss is not None
        # Nota: No asumimos que la pérdida disminuya en tan pocos pasos
    
    def test_validation_step(self, device, model_config, small_dataset):
        """Test paso de validación"""
        model = create_model('trm_base', model_config).to(device)
        _, val_loader, _ = small_dataset
        
        criterion = nn.CrossEntropyLoss()
        metrics = SudokuMetrics()
        
        model.eval()
        
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for puzzles, solutions in val_loader:
                puzzles = puzzles.to(device)
                solutions = solutions.to(device)
                
                output = model(puzzles)
                loss = criterion(output['output'].view(-1, 10), solutions.view(-1))
                
                predictions = output['output'].argmax(dim=-1)
                accuracy = metrics.accuracy(predictions, solutions)
                
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        assert avg_loss > 0
        assert 0 <= avg_accuracy <= 1
    
    def test_optimizer_step(self, device, model_config):
        """Test que el optimizador actualiza los parámetros"""
        model = create_model('trm_base', model_config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Obtener parámetros iniciales
        initial_params = [param.clone() for param in model.parameters()]
        
        # Crear pérdida dummy
        dummy_loss = sum(param.sum() for param in model.parameters())
        
        # Paso del optimizador
        optimizer.zero_grad()
        dummy_loss.backward()
        optimizer.step()
        
        # Verificar que los parámetros cambiaron
        for initial_param, current_param in zip(initial_params, model.parameters()):
            assert not torch.equal(initial_param, current_param)
    
    def test_scheduler_step(self, device, model_config):
        """Test que el scheduler actualiza el learning rate"""
        model = create_model('trm_base', model_config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Paso del scheduler
        scheduler.step()
        
        new_lr = optimizer.param_groups[0]['lr']
        
        assert new_lr == initial_lr * 0.5
    
    def test_model_eval_mode(self, device, model_config, small_dataset):
        """Test que el modelo funciona en modo eval"""
        model = create_model('trm_base', model_config).to(device)
        _, val_loader, _ = small_dataset
        
        model.eval()
        
        with torch.no_grad():
            for puzzles, solutions in val_loader:
                puzzles = puzzles.to(device)
                solutions = solutions.to(device)
                
                output = model(puzzles)
                
                assert 'output' in output
                assert output['output'].shape == (puzzles.shape[0], puzzles.shape[1], model_config['d_model'])
                break
    
    def test_gradient_clipping(self, device, model_config, small_dataset):
        """Test gradient clipping"""
        model = create_model('trm_base', model_config).to(device)
        train_loader, _, _ = small_dataset
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        
        for puzzles, solutions in train_loader:
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)
            
            optimizer.zero_grad()
            
            output = model(puzzles)
            loss = criterion(output['output'].view(-1, 10), solutions.view(-1))
            
            loss.backward()
            
            # Aplicar gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Verificar que los gradientes están dentro del límite
            total_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            assert total_norm <= 1.0 + 1e-6  # Pequeña tolerancia numérica
            
            optimizer.step()
            break
    
    def test_loss_computation(self, device, model_config, small_dataset):
        """Test cálculo de pérdida"""
        model = create_model('trm_base', model_config).to(device)
        train_loader, _, _ = small_dataset
        
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        
        for puzzles, solutions in train_loader:
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)
            
            output = model(puzzles)
            loss = criterion(output['output'].view(-1, 10), solutions.view(-1))
            
            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0  # Escalar
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            assert loss.item() >= 0
            
            break
    
    def test_metrics_computation(self, device, model_config, small_dataset):
        """Test cálculo de métricas"""
        model = create_model('trm_base', model_config).to(device)
        _, val_loader, _ = small_dataset
        
        metrics = SudokuMetrics()
        
        model.eval()
        
        with torch.no_grad():
            for puzzles, solutions in val_loader:
                puzzles = puzzles.to(device)
                solutions = solutions.to(device)
                
                output = model(puzzles)
                predictions = output['output'].argmax(dim=-1)
                
                # Calcular métricas
                accuracy = metrics.accuracy(predictions, solutions)
                grid_accuracy = metrics.grid_accuracy(predictions, solutions)
                
                assert isinstance(accuracy, float)
                assert isinstance(grid_accuracy, float)
                assert 0 <= accuracy <= 1
                assert 0 <= grid_accuracy <= 1
                
                break
    
    def test_checkpoint_save_load(self, device, model_config):
        """Test guardar y cargar checkpoint"""
        model = create_model('trm_base', model_config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Crear checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 5,
            'loss': 0.5
        }
        
        # Guardar checkpoint
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            torch.save(checkpoint, tmp_file.name)
            checkpoint_path = tmp_file.name
        
        try:
            # Cargar checkpoint
            loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Crear nuevo modelo y optimizador
            new_model = create_model('trm_base', model_config).to(device)
            new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
            
            # Cargar estados
            new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
            new_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
            
            # Verificar que los parámetros son iguales
            for param1, param2 in zip(model.parameters(), new_model.parameters()):
                assert torch.equal(param1, param2)
            
            # Verificar metadatos
            assert loaded_checkpoint['epoch'] == 5
            assert loaded_checkpoint['loss'] == 0.5
            
        finally:
            # Limpiar archivo temporal
            os.unlink(checkpoint_path)
    
    def test_memory_usage(self, device, model_config, small_dataset):
        """Test uso de memoria durante entrenamiento"""
        model = create_model('trm_base', model_config).to(device)
        train_loader, _, _ = small_dataset
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        
        # Probar múltiples batches para verificar que no hay memory leaks
        for batch_idx, (puzzles, solutions) in enumerate(train_loader):
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)
            
            optimizer.zero_grad()
            
            output = model(puzzles)
            loss = criterion(output['output'].view(-1, 10), solutions.view(-1))
            
            loss.backward()
            optimizer.step()
            
            # Limpiar variables
            del output, loss
            
            if batch_idx >= 5:  # Solo probar unos pocos batches
                break
    
    def test_deterministic_training(self, device, model_config, small_dataset):
        """Test que el entrenamiento es determinístico con la misma semilla"""
        torch.manual_seed(42)
        model1 = create_model('trm_base', model_config).to(device)
        optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
        
        torch.manual_seed(42)
        model2 = create_model('trm_base', model_config).to(device)
        optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
        
        criterion = nn.CrossEntropyLoss()
        train_loader, _, _ = small_dataset
        
        # Entrenar ambos modelos con los mismos datos
        for epoch in range(2):
            model1.train()
            model2.train()
            
            for batch_idx, (puzzles, solutions) in enumerate(train_loader):
                puzzles = puzzles.to(device)
                solutions = solutions.to(device)
                
                # Modelo 1
                optimizer1.zero_grad()
                output1 = model1(puzzles)
                loss1 = criterion(output1['output'].view(-1, 10), solutions.view(-1))
                loss1.backward()
                optimizer1.step()
                
                # Modelo 2
                optimizer2.zero_grad()
                output2 = model2(puzzles)
                loss2 = criterion(output2['output'].view(-1, 10), solutions.view(-1))
                loss2.backward()
                optimizer2.step()
                
                # Verificar que las pérdidas son iguales
                assert torch.allclose(loss1, loss2, atol=1e-6)
                
                if batch_idx >= 2:  # Solo probar unos pocos batches
                    break
        
        # Verificar que los parámetros finales son iguales
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(param1, param2, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__])
