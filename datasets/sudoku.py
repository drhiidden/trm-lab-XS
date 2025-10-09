"""
Dataset para tarea de Sudoku 9x9
Genera puzzles de Sudoku y sus soluciones para entrenar modelos de razonamiento estructurado.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import Tuple, List, Optional
import itertools


class SudokuGenerator:
    """
    Generador de puzzles de Sudoku
    """
    
    def __init__(self, grid_size: int = 9):
        self.grid_size = grid_size
        self.box_size = int(np.sqrt(grid_size))
        assert self.box_size * self.box_size == grid_size, "Grid size must be a perfect square"
    
    def is_valid(self, grid: np.ndarray, row: int, col: int, num: int) -> bool:
        """
        Verificar si un número es válido en una posición
        """
        # Verificar fila
        if num in grid[row, :]:
            return False
        
        # Verificar columna
        if num in grid[:, col]:
            return False
        
        # Verificar caja 3x3
        box_row = (row // self.box_size) * self.box_size
        box_col = (col // self.box_size) * self.box_size
        if num in grid[box_row:box_row + self.box_size, box_col:box_col + self.box_size]:
            return False
        
        return True
    
    def solve_sudoku(self, grid: np.ndarray) -> bool:
        """
        Resolver Sudoku usando backtracking
        """
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if grid[row, col] == 0:
                    for num in range(1, self.grid_size + 1):
                        if self.is_valid(grid, row, col, num):
                            grid[row, col] = num
                            if self.solve_sudoku(grid):
                                return True
                            grid[row, col] = 0
                    return False
        return True
    
    def generate_complete_sudoku(self) -> np.ndarray:
        """
        Generar un Sudoku completo válido
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Llenar diagonal principal (cajas 3x3)
        for i in range(0, self.grid_size, self.box_size):
            self.fill_box(grid, i, i)
        
        # Resolver el resto
        self.solve_sudoku(grid)
        return grid
    
    def fill_box(self, grid: np.ndarray, row: int, col: int):
        """
        Llenar una caja 3x3 con números únicos
        """
        nums = list(range(1, self.grid_size + 1))
        random.shuffle(nums)
        
        for i in range(self.box_size):
            for j in range(self.box_size):
                grid[row + i, col + j] = nums[i * self.box_size + j]
    
    def create_puzzle(self, complete_grid: np.ndarray, difficulty: str = 'medium') -> np.ndarray:
        """
        Crear un puzzle removiendo números del Sudoku completo
        """
        puzzle = complete_grid.copy()
        
        # Número de celdas a remover según dificultad
        if difficulty == 'easy':
            cells_to_remove = 40
        elif difficulty == 'medium':
            cells_to_remove = 50
        elif difficulty == 'hard':
            cells_to_remove = 60
        else:
            cells_to_remove = 50
        
        # Remover celdas aleatoriamente
        positions = list(itertools.product(range(self.grid_size), range(self.grid_size)))
        random.shuffle(positions)
        
        for i in range(min(cells_to_remove, len(positions))):
            row, col = positions[i]
            puzzle[row, col] = 0
        
        return puzzle
    
    def generate_puzzle_solution_pair(self, difficulty: str = 'medium') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generar un par puzzle-solución
        """
        solution = self.generate_complete_sudoku()
        puzzle = self.create_puzzle(solution, difficulty)
        return puzzle, solution


class SudokuDataset(Dataset):
    """
    Dataset de Sudoku para PyTorch
    """
    
    def __init__(self, num_samples: int = 10000, difficulty: str = 'medium', 
                 grid_size: int = 9, transform: Optional[callable] = None):
        self.num_samples = num_samples
        self.difficulty = difficulty
        self.grid_size = grid_size
        self.transform = transform
        
        self.generator = SudokuGenerator(grid_size)
        self.puzzles = []
        self.solutions = []
        
        # Generar datos
        self._generate_data()
    
    def _generate_data(self):
        """
        Generar puzzles y soluciones
        """
        print(f"Generando {self.num_samples} puzzles de Sudoku...")
        
        for i in range(self.num_samples):
            puzzle, solution = self.generator.generate_puzzle_solution_pair(self.difficulty)
            self.puzzles.append(puzzle)
            self.solutions.append(solution)
            
            if (i + 1) % 1000 == 0:
                print(f"Generados {i + 1}/{self.num_samples} puzzles")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        puzzle = torch.tensor(self.puzzles[idx], dtype=torch.long)
        solution = torch.tensor(self.solutions[idx], dtype=torch.long)
        
        if self.transform:
            puzzle = self.transform(puzzle)
            solution = self.transform(solution)
        
        return puzzle, solution


class SudokuDataModule:
    """
    Módulo de datos para Sudoku con train/val/test splits
    """
    
    def __init__(self, batch_size: int = 32, num_workers: int = 4, 
                 train_size: int = 8000, val_size: int = 1000, test_size: int = 1000,
                 difficulty: str = 'medium'):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.difficulty = difficulty
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """
        Configurar datasets
        """
        self.train_dataset = SudokuDataset(
            num_samples=self.train_size,
            difficulty=self.difficulty
        )
        
        self.val_dataset = SudokuDataset(
            num_samples=self.val_size,
            difficulty=self.difficulty
        )
        
        self.test_dataset = SudokuDataset(
            num_samples=self.test_size,
            difficulty=self.difficulty
        )
    
    def train_dataloader(self) -> DataLoader:
        """
        DataLoader para entrenamiento
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """
        DataLoader para validación
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """
        DataLoader para test
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


class SudokuMetrics:
    """
    Métricas específicas para evaluación de Sudoku
    """
    
    @staticmethod
    def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calcular precisión por celda
        """
        return (predictions == targets).float().mean().item()
    
    @staticmethod
    def grid_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calcular precisión por grid completo
        """
        correct_grids = (predictions == targets).all(dim=(1, 2)).float()
        return correct_grids.mean().item()
    
    @staticmethod
    def row_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calcular precisión por fila
        """
        correct_rows = (predictions == targets).all(dim=2).float()
        return correct_rows.mean().item()
    
    @staticmethod
    def column_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calcular precisión por columna
        """
        correct_cols = (predictions == targets).all(dim=1).float()
        return correct_cols.mean().item()
    
    @staticmethod
    def box_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calcular precisión por caja 3x3
        """
        batch_size = predictions.size(0)
        correct_boxes = 0
        total_boxes = 0
        
        for b in range(batch_size):
            for i in range(0, 9, 3):
                for j in range(0, 9, 3):
                    pred_box = predictions[b, i:i+3, j:j+3]
                    target_box = targets[b, i:i+3, j:j+3]
                    if torch.equal(pred_box, target_box):
                        correct_boxes += 1
                    total_boxes += 1
        
        return correct_boxes / total_boxes if total_boxes > 0 else 0.0


def create_sudoku_dataloaders(batch_size: int = 32, num_workers: int = 4,
                             train_size: int = 8000, val_size: int = 1000, 
                             test_size: int = 1000, difficulty: str = 'medium') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Función de conveniencia para crear dataloaders de Sudoku
    """
    data_module = SudokuDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        difficulty=difficulty
    )
    
    data_module.setup()
    
    return (
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        data_module.test_dataloader()
    )


# Ejemplo de uso
if __name__ == "__main__":
    # Crear dataset pequeño para prueba
    dataset = SudokuDataset(num_samples=100, difficulty='easy')
    
    # Mostrar ejemplo
    puzzle, solution = dataset[0]
    print("Puzzle:")
    print(puzzle.numpy())
    print("\nSolución:")
    print(solution.numpy())
    
    # Crear dataloaders
    train_loader, val_loader, test_loader = create_sudoku_dataloaders(
        batch_size=16,
        train_size=1000,
        val_size=100,
        test_size=100
    )
    
    # Probar dataloader
    for batch_idx, (puzzles, solutions) in enumerate(train_loader):
        print(f"Batch {batch_idx}: puzzles shape {puzzles.shape}, solutions shape {solutions.shape}")
        if batch_idx >= 2:  # Solo mostrar primeros 3 batches
            break
