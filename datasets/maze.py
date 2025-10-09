"""
Dataset para tarea de laberinto (Maze)
Genera laberintos 30x30 y sus soluciones para entrenar modelos de razonamiento espacial.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import Tuple, List, Optional, Set
from collections import deque
import matplotlib.pyplot as plt


class MazeGenerator:
    """
    Generador de laberintos usando algoritmo de backtracking
    """
    
    def __init__(self, width: int = 30, height: int = 30):
        self.width = width
        self.height = height
        self.wall = 1
        self.path = 0
        self.start = 2
        self.end = 3
    
    def is_valid(self, maze: np.ndarray, x: int, y: int) -> bool:
        """
        Verificar si una posición es válida
        """
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                maze[y, x] == self.wall)
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Obtener vecinos válidos
        """
        neighbors = []
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]  # Saltar una celda
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid(maze, nx, ny):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def generate_maze(self) -> np.ndarray:
        """
        Generar laberinto usando backtracking
        """
        # Inicializar con paredes
        maze = np.ones((self.height, self.width), dtype=int)
        
        # Empezar desde (1, 1) - debe ser impar
        start_x, start_y = 1, 1
        stack = [(start_x, start_y)]
        maze[start_y, start_x] = self.path
        
        while stack:
            current_x, current_y = stack[-1]
            neighbors = self.get_neighbors(current_x, current_y)
            
            if neighbors:
                # Elegir vecino aleatorio
                next_x, next_y = random.choice(neighbors)
                
                # Crear pasillo
                wall_x = current_x + (next_x - current_x) // 2
                wall_y = current_y + (next_y - current_y) // 2
                
                maze[wall_y, wall_x] = self.path
                maze[next_y, next_x] = self.path
                
                stack.append((next_x, next_y))
            else:
                stack.pop()
        
        return maze
    
    def find_path(self, maze: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Encontrar camino usando BFS
        """
        queue = deque([(start, [start])])
        visited = {start}
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            (x, y), path = queue.popleft()
            
            if (x, y) == end:
                return path
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < self.width and 
                    0 <= ny < self.height and 
                    (nx, ny) not in visited and 
                    maze[ny, nx] == self.path):
                    
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        
        return []  # No se encontró camino
    
    def add_start_end(self, maze: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Añadir puntos de inicio y fin
        """
        # Encontrar celdas de camino
        path_cells = np.where(maze == self.path)
        path_positions = list(zip(path_cells[1], path_cells[0]))  # (x, y)
        
        if len(path_positions) < 2:
            raise ValueError("No hay suficientes celdas de camino")
        
        # Elegir inicio y fin aleatoriamente
        start = random.choice(path_positions)
        end = random.choice([pos for pos in path_positions if pos != start])
        
        # Marcar inicio y fin
        maze[start[1], start[0]] = self.start
        maze[end[1], end[0]] = self.end
        
        return maze, start, end
    
    def create_maze_with_solution(self) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Crear laberinto con solución
        """
        maze = self.generate_maze()
        maze, start, end = self.add_start_end(maze)
        
        # Encontrar solución
        solution = self.find_path(maze, start, end)
        
        return maze, solution


class MazeDataset(Dataset):
    """
    Dataset de laberintos para PyTorch
    """
    
    def __init__(self, num_samples: int = 10000, width: int = 30, height: int = 30,
                 transform: Optional[callable] = None):
        self.num_samples = num_samples
        self.width = width
        self.height = height
        self.transform = transform
        
        self.generator = MazeGenerator(width, height)
        self.mazes = []
        self.solutions = []
        
        # Generar datos
        self._generate_data()
    
    def _generate_data(self):
        """
        Generar laberintos y soluciones
        """
        print(f"Generando {self.num_samples} laberintos...")
        
        for i in range(self.num_samples):
            try:
                maze, solution = self.generator.create_maze_with_solution()
                self.mazes.append(maze)
                self.solutions.append(solution)
                
                if (i + 1) % 1000 == 0:
                    print(f"Generados {i + 1}/{self.num_samples} laberintos")
            except ValueError:
                # Si no se puede crear laberinto válido, reintentar
                i -= 1
                continue
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        maze = torch.tensor(self.mazes[idx], dtype=torch.long)
        
        # Convertir solución a tensor de coordenadas
        solution = self.solutions[idx]
        solution_tensor = torch.zeros((self.height, self.width), dtype=torch.long)
        for x, y in solution:
            solution_tensor[y, x] = 1
        
        if self.transform:
            maze = self.transform(maze)
            solution_tensor = self.transform(solution_tensor)
        
        return maze, solution_tensor


class MazeDataModule:
    """
    Módulo de datos para laberintos con train/val/test splits
    """
    
    def __init__(self, batch_size: int = 32, num_workers: int = 4,
                 train_size: int = 8000, val_size: int = 1000, test_size: int = 1000,
                 width: int = 30, height: int = 30):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.width = width
        self.height = height
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """
        Configurar datasets
        """
        self.train_dataset = MazeDataset(
            num_samples=self.train_size,
            width=self.width,
            height=self.height
        )
        
        self.val_dataset = MazeDataset(
            num_samples=self.val_size,
            width=self.width,
            height=self.height
        )
        
        self.test_dataset = MazeDataset(
            num_samples=self.test_size,
            width=self.width,
            height=self.height
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


class MazeMetrics:
    """
    Métricas específicas para evaluación de laberintos
    """
    
    @staticmethod
    def path_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calcular precisión del camino predicho
        """
        # predictions: (batch_size, height, width) - camino predicho
        # targets: (batch_size, height, width) - camino real
        
        # Intersection over Union (IoU) para el camino
        intersection = (predictions * targets).sum(dim=(1, 2))
        union = (predictions + targets).clamp(max=1).sum(dim=(1, 2))
        
        iou = (intersection / (union + 1e-8)).mean().item()
        return iou
    
    @staticmethod
    def path_length_ratio(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calcular ratio de longitud del camino
        """
        pred_length = predictions.sum(dim=(1, 2)).float()
        target_length = targets.sum(dim=(1, 2)).float()
        
        ratio = (pred_length / (target_length + 1e-8)).mean().item()
        return ratio
    
    @staticmethod
    def connectivity_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calcular score de conectividad (si el camino conecta inicio y fin)
        """
        # Esta es una implementación simplificada
        # En la práctica, se necesitaría verificar si el camino predicho
        # realmente conecta el punto de inicio con el punto final
        
        # Por ahora, usamos la precisión como proxy
        return MazeMetrics.path_accuracy(predictions, targets)


def visualize_maze(maze: np.ndarray, solution: Optional[List[Tuple[int, int]]] = None, 
                   title: str = "Laberinto"):
    """
    Visualizar laberinto y solución
    """
    plt.figure(figsize=(10, 10))
    
    # Crear mapa de colores
    colors = ['white', 'black', 'green', 'red']  # path, wall, start, end
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    plt.imshow(maze, cmap=cmap, vmin=0, vmax=3)
    
    # Dibujar solución si se proporciona
    if solution:
        solution_x = [pos[0] for pos in solution]
        solution_y = [pos[1] for pos in solution]
        plt.plot(solution_x, solution_y, 'b-', linewidth=2, alpha=0.7)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def create_maze_dataloaders(batch_size: int = 32, num_workers: int = 4,
                           train_size: int = 8000, val_size: int = 1000,
                           test_size: int = 1000, width: int = 30, height: int = 30) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Función de conveniencia para crear dataloaders de laberintos
    """
    data_module = MazeDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        width=width,
        height=height
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
    dataset = MazeDataset(num_samples=10, width=15, height=15)
    
    # Mostrar ejemplo
    maze, solution = dataset[0]
    print("Laberinto:")
    print(maze.numpy())
    print(f"\nSolución tiene {len(dataset.solutions[0])} pasos")
    
    # Visualizar (opcional)
    # visualize_maze(maze.numpy(), dataset.solutions[0])
    
    # Crear dataloaders
    train_loader, val_loader, test_loader = create_maze_dataloaders(
        batch_size=8,
        train_size=100,
        val_size=20,
        test_size=20,
        width=15,
        height=15
    )
    
    # Probar dataloader
    for batch_idx, (mazes, solutions) in enumerate(train_loader):
        print(f"Batch {batch_idx}: mazes shape {mazes.shape}, solutions shape {solutions.shape}")
        if batch_idx >= 2:  # Solo mostrar primeros 3 batches
            break
