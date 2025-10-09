"""
Tests unitarios para los datasets
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from datasets.sudoku import SudokuDataset, SudokuMetrics
from datasets.maze import MazeDataset, MazeMetrics
from datasets.mnist import MNISTDataset, MNISTMetrics
from datasets.seqcopy import SequenceCopyDataset, SequenceCopyMetrics
from datasets.cifar10 import CIFAR10Dataset, CIFAR10Metrics
from datasets.arc import ARCDataset, ARCMetrics


class TestSudokuDataset:
    """Tests para el dataset de Sudoku"""
    
    def test_sudoku_dataset_creation(self):
        """Test creación del dataset"""
        dataset = SudokuDataset(num_samples=10, difficulty='easy')
        
        assert len(dataset) == 10
        assert len(dataset.puzzles) == 10
        assert len(dataset.solutions) == 10
    
    def test_sudoku_dataset_getitem(self):
        """Test acceso a elementos del dataset"""
        dataset = SudokuDataset(num_samples=5, difficulty='easy')
        
        puzzle, solution = dataset[0]
        
        assert isinstance(puzzle, torch.Tensor)
        assert isinstance(solution, torch.Tensor)
        assert puzzle.shape == (9, 9)
        assert solution.shape == (9, 9)
        assert puzzle.dtype == torch.long
        assert solution.dtype == torch.long
    
    def test_sudoku_dataset_validity(self):
        """Test que los puzzles generados son válidos"""
        dataset = SudokuDataset(num_samples=5, difficulty='easy')
        
        for i in range(len(dataset)):
            puzzle, solution = dataset[i]
            
            # Verificar que la solución es válida (1-9)
            assert torch.all(solution >= 1)
            assert torch.all(solution <= 9)
            
            # Verificar que el puzzle es un subconjunto de la solución
            mask = puzzle != 0
            assert torch.all(puzzle[mask] == solution[mask])
    
    def test_sudoku_metrics(self):
        """Test métricas de Sudoku"""
        metrics = SudokuMetrics()
        
        # Crear datos de prueba
        predictions = torch.randint(1, 10, (2, 9, 9))
        targets = torch.randint(1, 10, (2, 9, 9))
        
        # Test accuracy
        acc = metrics.accuracy(predictions, targets)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1
        
        # Test grid accuracy
        grid_acc = metrics.grid_accuracy(predictions, targets)
        assert isinstance(grid_acc, float)
        assert 0 <= grid_acc <= 1
        
        # Test row accuracy
        row_acc = metrics.row_accuracy(predictions, targets)
        assert isinstance(row_acc, float)
        assert 0 <= row_acc <= 1
        
        # Test column accuracy
        col_acc = metrics.column_accuracy(predictions, targets)
        assert isinstance(col_acc, float)
        assert 0 <= col_acc <= 1
        
        # Test box accuracy
        box_acc = metrics.box_accuracy(predictions, targets)
        assert isinstance(box_acc, float)
        assert 0 <= box_acc <= 1


class TestMazeDataset:
    """Tests para el dataset de laberintos"""
    
    def test_maze_dataset_creation(self):
        """Test creación del dataset"""
        dataset = MazeDataset(num_samples=5, width=10, height=10)
        
        assert len(dataset) == 5
        assert len(dataset.mazes) == 5
        assert len(dataset.solutions) == 5
    
    def test_maze_dataset_getitem(self):
        """Test acceso a elementos del dataset"""
        dataset = MazeDataset(num_samples=3, width=8, height=8)
        
        maze, solution = dataset[0]
        
        assert isinstance(maze, torch.Tensor)
        assert isinstance(solution, torch.Tensor)
        assert maze.shape == (8, 8)
        assert solution.shape == (8, 8)
        assert maze.dtype == torch.long
        assert solution.dtype == torch.long
    
    def test_maze_dataset_validity(self):
        """Test que los laberintos generados son válidos"""
        dataset = MazeDataset(num_samples=3, width=6, height=6)
        
        for i in range(len(dataset)):
            maze, solution = dataset[i]
            
            # Verificar que el laberinto tiene valores válidos
            assert torch.all(maze >= 0)
            assert torch.all(maze <= 3)  # 0=path, 1=wall, 2=start, 3=end
            
            # Verificar que la solución es binaria
            assert torch.all(solution >= 0)
            assert torch.all(solution <= 1)
    
    def test_maze_metrics(self):
        """Test métricas de laberintos"""
        metrics = MazeMetrics()
        
        # Crear datos de prueba
        predictions = torch.randint(0, 2, (2, 10, 10)).float()
        targets = torch.randint(0, 2, (2, 10, 10)).float()
        
        # Test path accuracy
        path_acc = metrics.path_accuracy(predictions, targets)
        assert isinstance(path_acc, float)
        assert 0 <= path_acc <= 1
        
        # Test path length ratio
        length_ratio = metrics.path_length_ratio(predictions, targets)
        assert isinstance(length_ratio, float)
        assert length_ratio >= 0
        
        # Test connectivity score
        connectivity = metrics.connectivity_score(predictions, targets)
        assert isinstance(connectivity, float)
        assert 0 <= connectivity <= 1


class TestMNISTDataset:
    """Tests para el dataset de MNIST"""
    
    def test_mnist_dataset_creation(self):
        """Test creación del dataset"""
        # Usar dataset pequeño para tests
        dataset = MNISTDataset(root='./test_data', train=True, flatten=True)
        
        assert len(dataset) > 0
    
    def test_mnist_dataset_getitem(self):
        """Test acceso a elementos del dataset"""
        dataset = MNISTDataset(root='./test_data', train=True, flatten=True)
        
        if len(dataset) > 0:
            image, label = dataset[0]
            
            assert isinstance(image, torch.Tensor)
            assert isinstance(label, torch.Tensor)
            assert image.shape == (1, 784)  # Flattened
            assert label.shape == ()
            assert label.dtype == torch.long
    
    def test_mnist_metrics(self):
        """Test métricas de MNIST"""
        metrics = MNISTMetrics()
        
        # Crear datos de prueba
        predictions = torch.randn(2, 10)
        targets = torch.randint(0, 10, (2,))
        
        # Test accuracy
        acc = metrics.accuracy(predictions, targets)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1
        
        # Test top-k accuracy
        top5_acc = metrics.top_k_accuracy(predictions, targets, k=5)
        assert isinstance(top5_acc, float)
        assert 0 <= top5_acc <= 1
        
        # Test per-class accuracy
        per_class_acc = metrics.per_class_accuracy(predictions, targets)
        assert isinstance(per_class_acc, dict)
        assert len(per_class_acc) <= 10  # Máximo 10 clases


class TestSequenceCopyDataset:
    """Tests para el dataset de copia de secuencias"""
    
    def test_seqcopy_dataset_creation(self):
        """Test creación del dataset"""
        dataset = SequenceCopyDataset(num_samples=10, vocab_size=5, max_length=10)
        
        assert len(dataset) == 10
        assert len(dataset.inputs) == 10
        assert len(dataset.outputs) == 10
    
    def test_seqcopy_dataset_getitem(self):
        """Test acceso a elementos del dataset"""
        dataset = SequenceCopyDataset(num_samples=5, vocab_size=5, max_length=8)
        
        input_seq, output_seq = dataset[0]
        
        assert isinstance(input_seq, torch.Tensor)
        assert isinstance(output_seq, torch.Tensor)
        assert input_seq.dtype == torch.long
        assert output_seq.dtype == torch.long
        
        # Verificar que la entrada tiene delimiter
        assert 0 in input_seq  # 0 es el delimiter
    
    def test_seqcopy_metrics(self):
        """Test métricas de copia de secuencias"""
        metrics = SequenceCopyMetrics()
        
        # Crear datos de prueba
        predictions = torch.randn(2, 10, 5)  # (batch, seq, vocab)
        targets = torch.randint(0, 5, (2, 10))
        
        # Test sequence accuracy
        seq_acc = metrics.sequence_accuracy(predictions, targets)
        assert isinstance(seq_acc, float)
        assert 0 <= seq_acc <= 1
        
        # Test token accuracy
        token_acc = metrics.token_accuracy(predictions, targets)
        assert isinstance(token_acc, float)
        assert 0 <= token_acc <= 1
        
        # Test length accuracy
        length_acc = metrics.length_accuracy(predictions, targets)
        assert isinstance(length_acc, float)
        assert 0 <= length_acc <= 1
        
        # Test edit distance
        edit_dist = metrics.edit_distance(predictions, targets)
        assert isinstance(edit_dist, float)
        assert edit_dist >= 0


class TestARCDataset:
    """Tests para el dataset de ARC"""
    
    def test_arc_dataset_creation(self):
        """Test creación del dataset"""
        dataset = ARCDataset(num_samples=5, grid_size=5, max_colors=3)
        
        assert len(dataset) == 5
        assert len(dataset.inputs) == 5
        assert len(dataset.outputs) == 5
        assert len(dataset.task_labels) == 5
    
    def test_arc_dataset_getitem(self):
        """Test acceso a elementos del dataset"""
        dataset = ARCDataset(num_samples=3, grid_size=4, max_colors=3)
        
        input_grid, output_grid, task_label = dataset[0]
        
        assert isinstance(input_grid, torch.Tensor)
        assert isinstance(output_grid, torch.Tensor)
        assert isinstance(task_label, str)
        assert input_grid.shape == (4, 4)
        assert output_grid.shape == (4, 4)
        assert input_grid.dtype == torch.long
        assert output_grid.dtype == torch.long
    
    def test_arc_dataset_validity(self):
        """Test que los grids generados son válidos"""
        dataset = ARCDataset(num_samples=3, grid_size=5, max_colors=3)
        
        for i in range(len(dataset)):
            input_grid, output_grid, task_label = dataset[i]
            
            # Verificar que los grids tienen valores válidos
            assert torch.all(input_grid >= 0)
            assert torch.all(input_grid <= 3)  # 0-3 colores
            assert torch.all(output_grid >= 0)
            assert torch.all(output_grid <= 3)
            
            # Verificar que el task_label es válido
            assert task_label in ['copy', 'color_change', 'mirror', 'rotation', 'fill']
    
    def test_arc_metrics(self):
        """Test métricas de ARC"""
        metrics = ARCMetrics()
        
        # Crear datos de prueba
        predictions = torch.randint(0, 5, (2, 5, 5))
        targets = torch.randint(0, 5, (2, 5, 5))
        task_labels = ['copy', 'mirror']
        
        # Test exact match
        exact_match = metrics.exact_match(predictions, targets)
        assert isinstance(exact_match, float)
        assert 0 <= exact_match <= 1
        
        # Test pixel accuracy
        pixel_acc = metrics.pixel_accuracy(predictions, targets)
        assert isinstance(pixel_acc, float)
        assert 0 <= pixel_acc <= 1
        
        # Test task accuracy
        task_acc = metrics.task_accuracy(predictions, targets, task_labels)
        assert isinstance(task_acc, dict)
        assert 'copy' in task_acc
        assert 'mirror' in task_acc


class TestDataLoaderIntegration:
    """Tests de integración con DataLoader"""
    
    def test_sudoku_dataloader(self):
        """Test DataLoader de Sudoku"""
        dataset = SudokuDataset(num_samples=20, difficulty='easy')
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        for batch_idx, (puzzles, solutions) in enumerate(dataloader):
            assert puzzles.shape[0] <= 4  # batch_size
            assert puzzles.shape[1:] == (9, 9)
            assert solutions.shape == puzzles.shape
            
            if batch_idx >= 2:  # Solo probar primeros batches
                break
    
    def test_maze_dataloader(self):
        """Test DataLoader de laberintos"""
        dataset = MazeDataset(num_samples=15, width=6, height=6)
        dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
        
        for batch_idx, (mazes, solutions) in enumerate(dataloader):
            assert mazes.shape[0] <= 3  # batch_size
            assert mazes.shape[1:] == (6, 6)
            assert solutions.shape == mazes.shape
            
            if batch_idx >= 2:  # Solo probar primeros batches
                break
    
    def test_seqcopy_dataloader(self):
        """Test DataLoader de copia de secuencias"""
        dataset = SequenceCopyDataset(num_samples=20, vocab_size=5, max_length=8)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        for batch_idx, (inputs, outputs) in enumerate(dataloader):
            assert inputs.shape[0] <= 4  # batch_size
            assert outputs.shape[0] <= 4  # batch_size
            assert inputs.dtype == torch.long
            assert outputs.dtype == torch.long
            
            if batch_idx >= 2:  # Solo probar primeros batches
                break
    
    def test_arc_dataloader(self):
        """Test DataLoader de ARC"""
        dataset = ARCDataset(num_samples=15, grid_size=5, max_colors=3)
        dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
        
        for batch_idx, (inputs, outputs, task_labels) in enumerate(dataloader):
            assert inputs.shape[0] <= 3  # batch_size
            assert inputs.shape[1:] == (5, 5)
            assert outputs.shape == inputs.shape
            assert len(task_labels) == inputs.shape[0]
            
            if batch_idx >= 2:  # Solo probar primeros batches
                break


class TestDatasetEdgeCases:
    """Tests para casos edge de los datasets"""
    
    def test_empty_dataset(self):
        """Test dataset vacío"""
        dataset = SudokuDataset(num_samples=0, difficulty='easy')
        assert len(dataset) == 0
    
    def test_single_sample_dataset(self):
        """Test dataset con una sola muestra"""
        dataset = SudokuDataset(num_samples=1, difficulty='easy')
        assert len(dataset) == 1
        
        puzzle, solution = dataset[0]
        assert puzzle.shape == (9, 9)
        assert solution.shape == (9, 9)
    
    def test_large_dataset(self):
        """Test dataset grande"""
        dataset = SudokuDataset(num_samples=100, difficulty='easy')
        assert len(dataset) == 100
        
        # Probar acceso aleatorio
        puzzle, solution = dataset[50]
        assert puzzle.shape == (9, 9)
        assert solution.shape == (9, 9)
    
    def test_different_difficulties(self):
        """Test diferentes dificultades de Sudoku"""
        for difficulty in ['easy', 'medium', 'hard']:
            dataset = SudokuDataset(num_samples=5, difficulty=difficulty)
            assert len(dataset) == 5
            
            # Verificar que se generaron puzzles
            puzzle, solution = dataset[0]
            assert puzzle.shape == (9, 9)
            assert solution.shape == (9, 9)
    
    def test_different_grid_sizes(self):
        """Test diferentes tamaños de grid para laberintos"""
        for size in [5, 10, 15]:
            dataset = MazeDataset(num_samples=3, width=size, height=size)
            assert len(dataset) == 3
            
            maze, solution = dataset[0]
            assert maze.shape == (size, size)
            assert solution.shape == (size, size)


if __name__ == '__main__':
    pytest.main([__file__])
