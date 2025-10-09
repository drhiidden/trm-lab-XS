"""
Dataset para MNIST con soporte para TRM
Adapta MNIST para trabajar con la arquitectura TRM.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
import numpy as np


class MNISTDataset(Dataset):
    """
    Dataset MNIST adaptado para TRM
    """
    
    def __init__(self, root: str = './data', train: bool = True, 
                 transform: Optional[transforms.Compose] = None,
                 flatten: bool = False, normalize: bool = True):
        self.root = root
        self.train = train
        self.flatten = flatten
        self.normalize = normalize
        
        # Transformaciones por defecto
        if transform is None:
            transform_list = [transforms.ToTensor()]
            if normalize:
                transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = transform
        
        # Cargar dataset MNIST
        self.dataset = torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=self.transform
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[idx]
        
        if self.flatten:
            # Aplanar imagen para secuencia 1D
            image = image.view(-1)  # (784,)
            # Añadir dimensión de secuencia
            image = image.unsqueeze(0)  # (1, 784)
        else:
            # Mantener como imagen 2D
            image = image.squeeze(0)  # (28, 28)
        
        return image, torch.tensor(label, dtype=torch.long)


class MNISTSequenceDataset(Dataset):
    """
    Dataset MNIST como secuencia para TRM
    Convierte imágenes en secuencias de píxeles
    """
    
    def __init__(self, root: str = './data', train: bool = True, 
                 sequence_length: int = 784, patch_size: int = 1):
        self.root = root
        self.train = train
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        
        # Transformaciones
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Cargar dataset MNIST
        self.dataset = torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=self.transform
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[idx]
        
        # Convertir a secuencia
        if self.patch_size == 1:
            # Cada píxel es un token
            sequence = image.view(-1)  # (784,)
        else:
            # Agrupar píxeles en patches
            h, w = image.shape[1], image.shape[2]
            patches = []
            for i in range(0, h, self.patch_size):
                for j in range(0, w, self.patch_size):
                    patch = image[:, i:i+self.patch_size, j:j+self.patch_size]
                    patches.append(patch.mean())  # Promedio del patch
            sequence = torch.stack(patches)
        
        # Añadir dimensión de batch
        sequence = sequence.unsqueeze(0)  # (1, seq_len)
        
        return sequence, torch.tensor(label, dtype=torch.long)


class MNISTDataModule:
    """
    Módulo de datos para MNIST
    """
    
    def __init__(self, batch_size: int = 32, num_workers: int = 4,
                 root: str = './data', flatten: bool = False,
                 sequence_mode: bool = False, patch_size: int = 1):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root
        self.flatten = flatten
        self.sequence_mode = sequence_mode
        self.patch_size = patch_size
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """
        Configurar datasets
        """
        if self.sequence_mode:
            self.train_dataset = MNISTSequenceDataset(
                root=self.root,
                train=True,
                patch_size=self.patch_size
            )
            
            self.val_dataset = MNISTSequenceDataset(
                root=self.root,
                train=False,
                patch_size=self.patch_size
            )
            
            self.test_dataset = MNISTSequenceDataset(
                root=self.root,
                train=False,
                patch_size=self.patch_size
            )
        else:
            self.train_dataset = MNISTDataset(
                root=self.root,
                train=True,
                flatten=self.flatten
            )
            
            self.val_dataset = MNISTDataset(
                root=self.root,
                train=False,
                flatten=self.flatten
            )
            
            self.test_dataset = MNISTDataset(
                root=self.root,
                train=False,
                flatten=self.flatten
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


class MNISTMetrics:
    """
    Métricas específicas para MNIST
    """
    
    @staticmethod
    def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calcular precisión
        """
        pred_classes = predictions.argmax(dim=-1)
        return (pred_classes == targets).float().mean().item()
    
    @staticmethod
    def top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
        """
        Calcular precisión top-k
        """
        _, pred_indices = predictions.topk(k, dim=-1)
        correct = pred_indices.eq(targets.unsqueeze(-1).expand_as(pred_indices))
        return correct.any(dim=-1).float().mean().item()
    
    @staticmethod
    def per_class_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Calcular precisión por clase
        """
        pred_classes = predictions.argmax(dim=-1)
        per_class_acc = {}
        
        for class_id in range(10):
            class_mask = targets == class_id
            if class_mask.sum() > 0:
                class_acc = (pred_classes[class_mask] == targets[class_mask]).float().mean().item()
                per_class_acc[class_id] = class_acc
        
        return per_class_acc


def create_mnist_dataloaders(batch_size: int = 32, num_workers: int = 4,
                            root: str = './data', flatten: bool = False,
                            sequence_mode: bool = False, patch_size: int = 1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Función de conveniencia para crear dataloaders de MNIST
    """
    data_module = MNISTDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        root=root,
        flatten=flatten,
        sequence_mode=sequence_mode,
        patch_size=patch_size
    )
    
    data_module.setup()
    
    return (
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        data_module.test_dataloader()
    )


class MNISTAugmentation:
    """
    Transformaciones de aumento de datos para MNIST
    """
    
    @staticmethod
    def get_train_transforms():
        """
        Transformaciones para entrenamiento
        """
        return transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    @staticmethod
    def get_val_transforms():
        """
        Transformaciones para validación
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])


# Ejemplo de uso
if __name__ == "__main__":
    # Crear dataloaders
    train_loader, val_loader, test_loader = create_mnist_dataloaders(
        batch_size=32,
        sequence_mode=True,
        patch_size=2
    )
    
    # Probar dataloader
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
        print(f"Labels: {labels[:5]}")
        if batch_idx >= 2:  # Solo mostrar primeros 3 batches
            break
    
    # Probar métricas
    predictions = torch.randn(32, 10)  # Logits simulados
    targets = torch.randint(0, 10, (32,))
    
    acc = MNISTMetrics.accuracy(predictions, targets)
    top5_acc = MNISTMetrics.top_k_accuracy(predictions, targets, k=5)
    per_class_acc = MNISTMetrics.per_class_accuracy(predictions, targets)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    print(f"Per-class accuracy: {per_class_acc}")
