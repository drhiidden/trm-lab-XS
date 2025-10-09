"""
Dataset para tarea de copia de secuencias (Sequence Copying Task)
Genera secuencias aleatorias que el modelo debe copiar, evaluando razonamiento secuencial.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import Tuple, List, Optional
import string


class SequenceCopyGenerator:
    """
    Generador de tareas de copia de secuencias
    """
    
    def __init__(self, vocab_size: int = 10, max_length: int = 20, 
                 min_length: int = 5, delimiter_token: int = 0):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.min_length = min_length
        self.delimiter_token = delimiter_token
    
    def generate_sequence(self, length: Optional[int] = None) -> List[int]:
        """
        Generar secuencia aleatoria
        """
        if length is None:
            length = random.randint(self.min_length, self.max_length)
        
        # Generar secuencia de tokens (excluyendo delimiter)
        sequence = [random.randint(1, self.vocab_size - 1) for _ in range(length)]
        return sequence
    
    def create_copy_task(self, sequence: Optional[List[int]] = None) -> Tuple[List[int], List[int]]:
        """
        Crear tarea de copia: input = [sequence + delimiter], output = [sequence]
        """
        if sequence is None:
            sequence = self.generate_sequence()
        
        # Input: secuencia + delimiter
        input_seq = sequence + [self.delimiter_token]
        
        # Output: solo la secuencia original (sin delimiter)
        output_seq = sequence.copy()
        
        return input_seq, output_seq
    
    def create_reverse_copy_task(self, sequence: Optional[List[int]] = None) -> Tuple[List[int], List[int]]:
        """
        Crear tarea de copia reversa: input = [sequence + delimiter], output = [reversed_sequence]
        """
        if sequence is None:
            sequence = self.generate_sequence()
        
        # Input: secuencia + delimiter
        input_seq = sequence + [self.delimiter_token]
        
        # Output: secuencia reversa
        output_seq = sequence[::-1]
        
        return input_seq, output_seq
    
    def create_double_copy_task(self, sequence: Optional[List[int]] = None) -> Tuple[List[int], List[int]]:
        """
        Crear tarea de doble copia: input = [sequence + delimiter], output = [sequence + sequence]
        """
        if sequence is None:
            sequence = self.generate_sequence()
        
        # Input: secuencia + delimiter
        input_seq = sequence + [self.delimiter_token]
        
        # Output: secuencia duplicada
        output_seq = sequence + sequence
        
        return input_seq, output_seq


class SequenceCopyDataset(Dataset):
    """
    Dataset para tareas de copia de secuencias
    """
    
    def __init__(self, num_samples: int = 10000, vocab_size: int = 10, 
                 max_length: int = 20, min_length: int = 5,
                 task_type: str = 'copy', delimiter_token: int = 0,
                 transform: Optional[callable] = None):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.min_length = min_length
        self.task_type = task_type
        self.delimiter_token = delimiter_token
        self.transform = transform
        
        self.generator = SequenceCopyGenerator(
            vocab_size=vocab_size,
            max_length=max_length,
            min_length=min_length,
            delimiter_token=delimiter_token
        )
        
        self.inputs = []
        self.outputs = []
        
        # Generar datos
        self._generate_data()
    
    def _generate_data(self):
        """
        Generar secuencias de entrada y salida
        """
        print(f"Generando {self.num_samples} tareas de {self.task_type}...")
        
        for i in range(self.num_samples):
            if self.task_type == 'copy':
                input_seq, output_seq = self.generator.create_copy_task()
            elif self.task_type == 'reverse':
                input_seq, output_seq = self.generator.create_reverse_copy_task()
            elif self.task_type == 'double':
                input_seq, output_seq = self.generator.create_double_copy_task()
            else:
                raise ValueError(f"Tipo de tarea no soportado: {self.task_type}")
            
            self.inputs.append(input_seq)
            self.outputs.append(output_seq)
            
            if (i + 1) % 1000 == 0:
                print(f"Generados {i + 1}/{self.num_samples} ejemplos")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq = torch.tensor(self.inputs[idx], dtype=torch.long)
        output_seq = torch.tensor(self.outputs[idx], dtype=torch.long)
        
        if self.transform:
            input_seq = self.transform(input_seq)
            output_seq = self.transform(output_seq)
        
        return input_seq, output_seq


class SequenceCopyDataModule:
    """
    Módulo de datos para tareas de copia de secuencias
    """
    
    def __init__(self, batch_size: int = 32, num_workers: int = 4,
                 train_size: int = 8000, val_size: int = 1000, test_size: int = 1000,
                 vocab_size: int = 10, max_length: int = 20, min_length: int = 5,
                 task_type: str = 'copy'):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.min_length = min_length
        self.task_type = task_type
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """
        Configurar datasets
        """
        self.train_dataset = SequenceCopyDataset(
            num_samples=self.train_size,
            vocab_size=self.vocab_size,
            max_length=self.max_length,
            min_length=self.min_length,
            task_type=self.task_type
        )
        
        self.val_dataset = SequenceCopyDataset(
            num_samples=self.val_size,
            vocab_size=self.vocab_size,
            max_length=self.max_length,
            min_length=self.min_length,
            task_type=self.task_type
        )
        
        self.test_dataset = SequenceCopyDataset(
            num_samples=self.test_size,
            vocab_size=self.vocab_size,
            max_length=self.max_length,
            min_length=self.min_length,
            task_type=self.task_type
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


class SequenceCopyMetrics:
    """
    Métricas específicas para tareas de copia de secuencias
    """
    
    @staticmethod
    def sequence_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calcular precisión de secuencia completa
        """
        # predictions: (batch_size, seq_len, vocab_size)
        # targets: (batch_size, seq_len)
        
        pred_tokens = predictions.argmax(dim=-1)
        correct_sequences = (pred_tokens == targets).all(dim=1).float()
        return correct_sequences.mean().item()
    
    @staticmethod
    def token_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calcular precisión por token
        """
        pred_tokens = predictions.argmax(dim=-1)
        return (pred_tokens == targets).float().mean().item()
    
    @staticmethod
    def length_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calcular precisión de longitud de secuencia
        """
        # Asumir que las predicciones tienen la misma longitud que los targets
        # En la práctica, esto podría requerir padding o truncamiento
        pred_lengths = (predictions.argmax(dim=-1) != 0).sum(dim=1)
        target_lengths = (targets != 0).sum(dim=1)
        
        return (pred_lengths == target_lengths).float().mean().item()
    
    @staticmethod
    def edit_distance(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calcular distancia de edición promedio
        """
        pred_tokens = predictions.argmax(dim=-1)
        
        total_distance = 0
        batch_size = predictions.size(0)
        
        for i in range(batch_size):
            pred_seq = pred_tokens[i].cpu().numpy()
            target_seq = targets[i].cpu().numpy()
            
            # Calcular distancia de Levenshtein
            distance = SequenceCopyMetrics._levenshtein_distance(pred_seq, target_seq)
            total_distance += distance
        
        return total_distance / batch_size
    
    @staticmethod
    def _levenshtein_distance(s1: np.ndarray, s2: np.ndarray) -> int:
        """
        Calcular distancia de Levenshtein entre dos secuencias
        """
        if len(s1) < len(s2):
            return SequenceCopyMetrics._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


def create_seqcopy_dataloaders(batch_size: int = 32, num_workers: int = 4,
                              train_size: int = 8000, val_size: int = 1000,
                              test_size: int = 1000, vocab_size: int = 10,
                              max_length: int = 20, min_length: int = 5,
                              task_type: str = 'copy') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Función de conveniencia para crear dataloaders de copia de secuencias
    """
    data_module = SequenceCopyDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        vocab_size=vocab_size,
        max_length=max_length,
        min_length=min_length,
        task_type=task_type
    )
    
    data_module.setup()
    
    return (
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        data_module.test_dataloader()
    )


def visualize_sequence_task(input_seq: List[int], output_seq: List[int], 
                           task_type: str = 'copy', vocab_size: int = 10):
    """
    Visualizar tarea de secuencia
    """
    print(f"Tarea: {task_type}")
    print(f"Input:  {input_seq}")
    print(f"Output: {output_seq}")
    
    # Mostrar como caracteres si el vocabulario es pequeño
    if vocab_size <= 26:
        input_chars = [chr(ord('A') + token - 1) if token > 0 else '|' for token in input_seq]
        output_chars = [chr(ord('A') + token - 1) if token > 0 else '|' for token in output_seq]
        print(f"Input:  {''.join(input_chars)}")
        print(f"Output: {''.join(output_chars)}")


# Ejemplo de uso
if __name__ == "__main__":
    # Crear dataset pequeño para prueba
    dataset = SequenceCopyDataset(num_samples=100, task_type='copy', vocab_size=10)
    
    # Mostrar ejemplo
    input_seq, output_seq = dataset[0]
    print("Ejemplo de tarea de copia:")
    print(f"Input:  {input_seq.tolist()}")
    print(f"Output: {output_seq.tolist()}")
    
    # Crear dataloaders
    train_loader, val_loader, test_loader = create_seqcopy_dataloaders(
        batch_size=16,
        train_size=1000,
        val_size=100,
        test_size=100,
        task_type='copy'
    )
    
    # Probar dataloader
    for batch_idx, (inputs, outputs) in enumerate(train_loader):
        print(f"Batch {batch_idx}: inputs shape {inputs.shape}, outputs shape {outputs.shape}")
        if batch_idx >= 2:  # Solo mostrar primeros 3 batches
            break
    
    # Probar métricas
    predictions = torch.randn(32, 20, 10)  # Logits simulados
    targets = torch.randint(0, 10, (32, 20))
    
    seq_acc = SequenceCopyMetrics.sequence_accuracy(predictions, targets)
    token_acc = SequenceCopyMetrics.token_accuracy(predictions, targets)
    edit_dist = SequenceCopyMetrics.edit_distance(predictions, targets)
    
    print(f"Sequence Accuracy: {seq_acc:.4f}")
    print(f"Token Accuracy: {token_acc:.4f}")
    print(f"Edit Distance: {edit_dist:.2f}")
