"""
Dataset para CIFAR-10 con soporte para TRM
Adapta CIFAR-10 para trabajar con la arquitectura TRM.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
import numpy as np


class CIFAR10Dataset(Dataset):
    """
    Dataset CIFAR-10 adaptado para TRM
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
                transform_list.append(transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                ))
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = transform
        
        # Cargar dataset CIFAR-10
        self.dataset = torchvision.datasets.CIFAR10(
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
            image = image.view(-1)  # (3072,)
            # Añadir dimensión de secuencia
            image = image.unsqueeze(0)  # (1, 3072)
        else:
            # Mantener como imagen 2D
            image = image  # (3, 32, 32)
        
        return image, torch.tensor(label, dtype=torch.long)


class CIFAR10SequenceDataset(Dataset):
    """
    Dataset CIFAR-10 como secuencia para TRM
    Convierte imágenes en secuencias de patches
    """
    
    def __init__(self, root: str = './data', train: bool = True, 
                 patch_size: int = 4, sequence_length: Optional[int] = None):
        self.root = root
        self.train = train
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        
        # Transformaciones
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
        
        # Cargar dataset CIFAR-10
        self.dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=self.transform
        )
        
        # Calcular longitud de secuencia
        if sequence_length is None:
            self.sequence_length = (32 // patch_size) ** 2
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[idx]
        
        # Convertir a secuencia de patches
        patches = self._image_to_patches(image, self.patch_size)
        
        # Añadir dimensión de batch
        patches = patches.unsqueeze(0)  # (1, seq_len, patch_dim)
        
        return patches, torch.tensor(label, dtype=torch.long)
    
    def _image_to_patches(self, image: torch.Tensor, patch_size: int) -> torch.Tensor:
        """
        Convertir imagen a secuencia de patches
        """
        # image: (3, 32, 32)
        c, h, w = image.shape
        
        # Calcular número de patches
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # Extraer patch
                patch = image[:, 
                             i * patch_size:(i + 1) * patch_size,
                             j * patch_size:(j + 1) * patch_size]
                # Aplanar patch
                patch_flat = patch.view(-1)  # (patch_size * patch_size * channels,)
                patches.append(patch_flat)
        
        return torch.stack(patches)  # (num_patches, patch_dim)


class CIFAR10DataModule:
    """
    Módulo de datos para CIFAR-10
    """
    
    def __init__(self, batch_size: int = 32, num_workers: int = 4,
                 root: str = './data', flatten: bool = False,
                 sequence_mode: bool = False, patch_size: int = 4):
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
            self.train_dataset = CIFAR10SequenceDataset(
                root=self.root,
                train=True,
                patch_size=self.patch_size
            )
            
            self.val_dataset = CIFAR10SequenceDataset(
                root=self.root,
                train=False,
                patch_size=self.patch_size
            )
            
            self.test_dataset = CIFAR10SequenceDataset(
                root=self.root,
                train=False,
                patch_size=self.patch_size
            )
        else:
            self.train_dataset = CIFAR10Dataset(
                root=self.root,
                train=True,
                flatten=self.flatten
            )
            
            self.val_dataset = CIFAR10Dataset(
                root=self.root,
                train=False,
                flatten=self.flatten
            )
            
            self.test_dataset = CIFAR10Dataset(
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


class CIFAR10Metrics:
    """
    Métricas específicas para CIFAR-10
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
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        for class_id in range(10):
            class_mask = targets == class_id
            if class_mask.sum() > 0:
                class_acc = (pred_classes[class_mask] == targets[class_mask]).float().mean().item()
                per_class_acc[class_names[class_id]] = class_acc
        
        return per_class_acc
    
    @staticmethod
    def confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        """
        Calcular matriz de confusión
        """
        pred_classes = predictions.argmax(dim=-1)
        
        # Crear matriz de confusión
        cm = np.zeros((10, 10), dtype=int)
        for pred, target in zip(pred_classes.cpu().numpy(), targets.cpu().numpy()):
            cm[target, pred] += 1
        
        return cm


class CIFAR10Augmentation:
    """
    Transformaciones de aumento de datos para CIFAR-10
    """
    
    @staticmethod
    def get_train_transforms():
        """
        Transformaciones para entrenamiento
        """
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
    
    @staticmethod
    def get_val_transforms():
        """
        Transformaciones para validación
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])


def create_cifar10_dataloaders(batch_size: int = 32, num_workers: int = 4,
                              root: str = './data', flatten: bool = False,
                              sequence_mode: bool = False, patch_size: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Función de conveniencia para crear dataloaders de CIFAR-10
    """
    data_module = CIFAR10DataModule(
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


# Ejemplo de uso
if __name__ == "__main__":
    # Crear dataloaders
    train_loader, val_loader, test_loader = create_cifar10_dataloaders(
        batch_size=32,
        sequence_mode=True,
        patch_size=4
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
    
    acc = CIFAR10Metrics.accuracy(predictions, targets)
    top5_acc = CIFAR10Metrics.top_k_accuracy(predictions, targets, k=5)
    per_class_acc = CIFAR10Metrics.per_class_accuracy(predictions, targets)
    cm = CIFAR10Metrics.confusion_matrix(predictions, targets)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    print(f"Per-class accuracy: {per_class_acc}")
    print(f"Confusion matrix shape: {cm.shape}")
