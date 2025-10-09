"""
Datasets module for TRM Lab
"""

from .sudoku import SudokuDataset, SudokuMetrics, create_sudoku_dataloaders
from .maze import MazeDataset, MazeMetrics, create_maze_dataloaders
from .mnist import MNISTDataset, MNISTMetrics, create_mnist_dataloaders
from .seqcopy import SequenceCopyDataset, SequenceCopyMetrics, create_seqcopy_dataloaders
from .cifar10 import CIFAR10Dataset, CIFAR10Metrics, create_cifar10_dataloaders
from .arc import ARCDataset, ARCMetrics, create_arc_dataloaders

__all__ = [
    # Sudoku
    'SudokuDataset', 'SudokuMetrics', 'create_sudoku_dataloaders',
    
    # Maze
    'MazeDataset', 'MazeMetrics', 'create_maze_dataloaders',
    
    # MNIST
    'MNISTDataset', 'MNISTMetrics', 'create_mnist_dataloaders',
    
    # Sequence Copy
    'SequenceCopyDataset', 'SequenceCopyMetrics', 'create_seqcopy_dataloaders',
    
    # CIFAR-10
    'CIFAR10Dataset', 'CIFAR10Metrics', 'create_cifar10_dataloaders',
    
    # ARC
    'ARCDataset', 'ARCMetrics', 'create_arc_dataloaders'
]
