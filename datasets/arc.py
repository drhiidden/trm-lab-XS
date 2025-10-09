"""
Dataset para ARC-AGI (Abstraction and Reasoning Corpus)
Implementación simplificada de tareas ARC para evaluación de razonamiento composicional.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import Tuple, List, Optional, Dict, Any
import json
import os


class ARCTaskGenerator:
    """
    Generador de tareas ARC simplificadas
    """
    
    def __init__(self, grid_size: int = 10, max_colors: int = 10):
        self.grid_size = grid_size
        self.max_colors = max_colors
        self.background_color = 0
        self.foreground_colors = list(range(1, max_colors + 1))
    
    def create_grid(self, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Crear grid vacío
        """
        if size is None:
            size = (self.grid_size, self.grid_size)
        return np.zeros(size, dtype=int)
    
    def add_shape(self, grid: np.ndarray, shape_type: str, color: int, 
                  position: Tuple[int, int], size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Añadir forma al grid
        """
        h, w = grid.shape
        x, y = position
        
        if shape_type == 'rectangle':
            if size is None:
                size = (3, 3)
            for i in range(min(size[0], h - y)):
                for j in range(min(size[1], w - x)):
                    if 0 <= y + i < h and 0 <= x + j < w:
                        grid[y + i, x + j] = color
        
        elif shape_type == 'line_horizontal':
            if size is None:
                size = (1, 5)
            for j in range(min(size[1], w - x)):
                if 0 <= y < h and 0 <= x + j < w:
                    grid[y, x + j] = color
        
        elif shape_type == 'line_vertical':
            if size is None:
                size = (5, 1)
            for i in range(min(size[0], h - y)):
                if 0 <= y + i < h and 0 <= x < w:
                    grid[y + i, x] = color
        
        elif shape_type == 'dot':
            if 0 <= y < h and 0 <= x < w:
                grid[y, x] = color
        
        return grid
    
    def create_copy_task(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crear tarea de copia: input -> output (mismo)
        """
        grid = self.create_grid()
        
        # Añadir algunas formas
        num_shapes = random.randint(2, 5)
        for _ in range(num_shapes):
            shape_type = random.choice(['rectangle', 'line_horizontal', 'line_vertical', 'dot'])
            color = random.choice(self.foreground_colors)
            position = (random.randint(0, self.grid_size - 3), random.randint(0, self.grid_size - 3))
            grid = self.add_shape(grid, shape_type, color, position)
        
        return grid, grid.copy()
    
    def create_color_change_task(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crear tarea de cambio de color: cambiar todos los colores
        """
        input_grid = self.create_grid()
        
        # Añadir formas con colores específicos
        colors_used = []
        num_shapes = random.randint(2, 4)
        for _ in range(num_shapes):
            shape_type = random.choice(['rectangle', 'line_horizontal', 'line_vertical'])
            color = random.choice(self.foreground_colors)
            colors_used.append(color)
            position = (random.randint(0, self.grid_size - 3), random.randint(0, self.grid_size - 3))
            input_grid = self.add_shape(input_grid, shape_type, color, position)
        
        # Crear output con colores cambiados
        output_grid = input_grid.copy()
        color_mapping = {old_color: random.choice(self.foreground_colors) 
                        for old_color in colors_used}
        
        for old_color, new_color in color_mapping.items():
            output_grid[output_grid == old_color] = new_color
        
        return input_grid, output_grid
    
    def create_mirror_task(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crear tarea de espejo: reflejar horizontalmente
        """
        input_grid = self.create_grid()
        
        # Añadir formas
        num_shapes = random.randint(2, 4)
        for _ in range(num_shapes):
            shape_type = random.choice(['rectangle', 'line_horizontal', 'line_vertical'])
            color = random.choice(self.foreground_colors)
            position = (random.randint(0, self.grid_size - 3), random.randint(0, self.grid_size - 3))
            input_grid = self.add_shape(input_grid, shape_type, color, position)
        
        # Crear output espejo
        output_grid = np.fliplr(input_grid)
        
        return input_grid, output_grid
    
    def create_rotation_task(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crear tarea de rotación: rotar 90 grados
        """
        input_grid = self.create_grid()
        
        # Añadir formas
        num_shapes = random.randint(2, 4)
        for _ in range(num_shapes):
            shape_type = random.choice(['rectangle', 'line_horizontal', 'line_vertical'])
            color = random.choice(self.foreground_colors)
            position = (random.randint(0, self.grid_size - 3), random.randint(0, self.grid_size - 3))
            input_grid = self.add_shape(input_grid, shape_type, color, position)
        
        # Crear output rotado
        output_grid = np.rot90(input_grid, k=1)
        
        return input_grid, output_grid
    
    def create_fill_task(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crear tarea de relleno: llenar área cerrada
        """
        input_grid = self.create_grid()
        
        # Crear contorno
        color = random.choice(self.foreground_colors)
        
        # Líneas horizontales
        for i in range(2, 6):
            for j in range(2, 8):
                input_grid[i, j] = color
        
        # Líneas verticales
        for i in range(2, 6):
            for j in range(2, 8):
                input_grid[i, j] = color
        
        # Crear contorno
        input_grid[2, 2:8] = color  # Top
        input_grid[5, 2:8] = color  # Bottom
        input_grid[2:6, 2] = color  # Left
        input_grid[2:6, 7] = color  # Right
        
        # Crear output con relleno
        output_grid = input_grid.copy()
        fill_color = random.choice([c for c in self.foreground_colors if c != color])
        output_grid[3:5, 3:7] = fill_color
        
        return input_grid, output_grid
    
    def generate_task(self, task_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generar tarea específica
        """
        if task_type == 'copy':
            return self.create_copy_task()
        elif task_type == 'color_change':
            return self.create_color_change_task()
        elif task_type == 'mirror':
            return self.create_mirror_task()
        elif task_type == 'rotation':
            return self.create_rotation_task()
        elif task_type == 'fill':
            return self.create_fill_task()
        else:
            raise ValueError(f"Tipo de tarea no soportado: {task_type}")


class ARCDataset(Dataset):
    """
    Dataset ARC para PyTorch
    """
    
    def __init__(self, num_samples: int = 10000, grid_size: int = 10, 
                 max_colors: int = 10, task_types: Optional[List[str]] = None,
                 transform: Optional[callable] = None):
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.max_colors = max_colors
        self.transform = transform
        
        if task_types is None:
            task_types = ['copy', 'color_change', 'mirror', 'rotation', 'fill']
        self.task_types = task_types
        
        self.generator = ARCTaskGenerator(grid_size, max_colors)
        self.inputs = []
        self.outputs = []
        self.task_labels = []
        
        # Generar datos
        self._generate_data()
    
    def _generate_data(self):
        """
        Generar tareas ARC
        """
        print(f"Generando {self.num_samples} tareas ARC...")
        
        for i in range(self.num_samples):
            task_type = random.choice(self.task_types)
            input_grid, output_grid = self.generator.generate_task(task_type)
            
            self.inputs.append(input_grid)
            self.outputs.append(output_grid)
            self.task_labels.append(task_type)
            
            if (i + 1) % 1000 == 0:
                print(f"Generados {i + 1}/{self.num_samples} ejemplos")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        input_grid = torch.tensor(self.inputs[idx], dtype=torch.long)
        output_grid = torch.tensor(self.outputs[idx], dtype=torch.long)
        task_label = self.task_labels[idx]
        
        if self.transform:
            input_grid = self.transform(input_grid)
            output_grid = self.transform(output_grid)
        
        return input_grid, output_grid, task_label


class ARCDataModule:
    """
    Módulo de datos para ARC
    """
    
    def __init__(self, batch_size: int = 32, num_workers: int = 4,
                 train_size: int = 8000, val_size: int = 1000, test_size: int = 1000,
                 grid_size: int = 10, max_colors: int = 10,
                 task_types: Optional[List[str]] = None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.grid_size = grid_size
        self.max_colors = max_colors
        self.task_types = task_types
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """
        Configurar datasets
        """
        self.train_dataset = ARCDataset(
            num_samples=self.train_size,
            grid_size=self.grid_size,
            max_colors=self.max_colors,
            task_types=self.task_types
        )
        
        self.val_dataset = ARCDataset(
            num_samples=self.val_size,
            grid_size=self.grid_size,
            max_colors=self.max_colors,
            task_types=self.task_types
        )
        
        self.test_dataset = ARCDataset(
            num_samples=self.test_size,
            grid_size=self.grid_size,
            max_colors=self.max_colors,
            task_types=self.task_types
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


class ARCMetrics:
    """
    Métricas específicas para ARC
    """
    
    @staticmethod
    def exact_match(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calcular coincidencia exacta
        """
        return (predictions == targets).all(dim=(1, 2)).float().mean().item()
    
    @staticmethod
    def pixel_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calcular precisión por píxel
        """
        return (predictions == targets).float().mean().item()
    
    @staticmethod
    def task_accuracy(predictions: torch.Tensor, targets: torch.Tensor, 
                     task_labels: List[str]) -> Dict[str, float]:
        """
        Calcular precisión por tipo de tarea
        """
        task_acc = {}
        unique_tasks = list(set(task_labels))
        
        for task in unique_tasks:
            task_mask = [i for i, label in enumerate(task_labels) if label == task]
            if task_mask:
                task_preds = predictions[task_mask]
                task_targets = targets[task_mask]
                task_acc[task] = ARCMetrics.exact_match(task_preds, task_targets)
        
        return task_acc


def visualize_arc_task(input_grid: np.ndarray, output_grid: np.ndarray, 
                      task_type: str = "ARC Task"):
    """
    Visualizar tarea ARC
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Input
    ax1.imshow(input_grid, cmap='tab10', vmin=0, vmax=10)
    ax1.set_title(f'Input - {task_type}')
    ax1.axis('off')
    
    # Output
    ax2.imshow(output_grid, cmap='tab10', vmin=0, vmax=10)
    ax2.set_title(f'Output - {task_type}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()


def create_arc_dataloaders(batch_size: int = 32, num_workers: int = 4,
                          train_size: int = 8000, val_size: int = 1000,
                          test_size: int = 1000, grid_size: int = 10,
                          max_colors: int = 10, task_types: Optional[List[str]] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Función de conveniencia para crear dataloaders de ARC
    """
    data_module = ARCDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        grid_size=grid_size,
        max_colors=max_colors,
        task_types=task_types
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
    dataset = ARCDataset(num_samples=100, grid_size=8, max_colors=5)
    
    # Mostrar ejemplo
    input_grid, output_grid, task_label = dataset[0]
    print(f"Tarea: {task_label}")
    print("Input:")
    print(input_grid.numpy())
    print("Output:")
    print(output_grid.numpy())
    
    # Crear dataloaders
    train_loader, val_loader, test_loader = create_arc_dataloaders(
        batch_size=16,
        train_size=1000,
        val_size=100,
        test_size=100,
        grid_size=8,
        max_colors=5
    )
    
    # Probar dataloader
    for batch_idx, (inputs, outputs, task_labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: inputs shape {inputs.shape}, outputs shape {outputs.shape}")
        print(f"Task labels: {task_labels[:5]}")
        if batch_idx >= 2:  # Solo mostrar primeros 3 batches
            break
    
    # Probar métricas
    predictions = torch.randint(0, 5, (32, 8, 8))
    targets = torch.randint(0, 5, (32, 8, 8))
    task_labels = ['copy'] * 16 + ['mirror'] * 16
    
    exact_match = ARCMetrics.exact_match(predictions, targets)
    pixel_acc = ARCMetrics.pixel_accuracy(predictions, targets)
    task_acc = ARCMetrics.task_accuracy(predictions, targets, task_labels)
    
    print(f"Exact Match: {exact_match:.4f}")
    print(f"Pixel Accuracy: {pixel_acc:.4f}")
    print(f"Task Accuracy: {task_acc}")
