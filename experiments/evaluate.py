"""
Script para evaluación de modelos TRM
Evalúa modelos entrenados en diferentes tareas y genera reportes detallados.
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from core.registry import create_model
from core.utils import get_device
from datasets.sudoku import create_sudoku_dataloaders, SudokuMetrics
from datasets.maze import create_maze_dataloaders, MazeMetrics
from datasets.mnist import create_mnist_dataloaders, MNISTMetrics
from datasets.seqcopy import create_seqcopy_dataloaders, SequenceCopyMetrics
from datasets.cifar10 import create_cifar10_dataloaders, CIFAR10Metrics
from datasets.arc import create_arc_dataloaders, ARCMetrics


class Evaluator:
    """
    Evaluador para modelos TRM
    """
    
    def __init__(self, config: Dict[str, Any], checkpoint_path: str):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = get_device()
        
        # Configurar logging
        self._setup_logging()
        
        # Inicializar componentes
        self.model = None
        self.test_loader = None
        self.metrics = None
        
        # Crear directorio de resultados
        self.results_dir = Path(config.get('results_dir', 'results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configurar logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self):
        """Cargar modelo desde checkpoint"""
        # Crear modelo
        model_config = self.config['model_config'].copy()
        model_name = self.config['model_name']
        
        self.model = create_model(model_name, model_config)
        self.model = self.model.to(self.device)
        
        # Cargar pesos
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.logger.info(f"Modelo cargado desde {self.checkpoint_path}")
        self.logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    def _create_dataloader(self):
        """Crear dataloader de test"""
        dataset_config = self.config['dataset']
        dataset_name = dataset_config['name']
        
        if dataset_name == 'sudoku':
            _, _, self.test_loader = create_sudoku_dataloaders(
                **dataset_config['config']
            )
            self.metrics = SudokuMetrics()
        elif dataset_name == 'maze':
            _, _, self.test_loader = create_maze_dataloaders(
                **dataset_config['config']
            )
            self.metrics = MazeMetrics()
        elif dataset_name == 'mnist':
            _, _, self.test_loader = create_mnist_dataloaders(
                **dataset_config['config']
            )
            self.metrics = MNISTMetrics()
        elif dataset_name == 'seqcopy':
            _, _, self.test_loader = create_seqcopy_dataloaders(
                **dataset_config['config']
            )
            self.metrics = SequenceCopyMetrics()
        elif dataset_name == 'cifar10':
            _, _, self.test_loader = create_cifar10_dataloaders(
                **dataset_config['config']
            )
            self.metrics = CIFAR10Metrics()
        elif dataset_name == 'arc':
            _, _, self.test_loader = create_arc_dataloaders(
                **dataset_config['config']
            )
            self.metrics = ARCMetrics()
        else:
            raise ValueError(f"Dataset no soportado: {dataset_name}")
    
    def _compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, 
                        metric_names: list) -> Dict[str, float]:
        """Calcular métricas"""
        results = {}
        
        for metric_name in metric_names:
            if hasattr(self.metrics, metric_name):
                metric_func = getattr(self.metrics, metric_name)
                results[metric_name] = metric_func(predictions, targets)
            else:
                self.logger.warning(f"Métrica no encontrada: {metric_name}")
        
        return results
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluar modelo"""
        self.logger.info("Iniciando evaluación...")
        
        # Cargar modelo y dataloader
        self._load_model()
        self._create_dataloader()
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_inputs = []
        inference_times = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Evaluating'):
                if len(batch) == 2:
                    inputs, targets = batch
                    task_labels = None
                else:
                    inputs, targets, task_labels = batch
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Medir tiempo de inferencia
                start_time = time.time()
                
                # Forward pass
                if hasattr(self.model, 'forward'):
                    outputs = self.model(inputs)
                    if isinstance(outputs, dict):
                        predictions = outputs.get('logits', outputs.get('output', outputs.get('predictions')))
                    else:
                        predictions = outputs
                else:
                    predictions = self.model(inputs)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                all_inputs.append(inputs.cpu())
        
        # Combinar resultados
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_inputs = torch.cat(all_inputs, dim=0)
        
        # Calcular métricas
        metric_names = self.config.get('metrics', ['accuracy'])
        metrics = self._compute_metrics(all_predictions, all_targets, metric_names)
        
        # Estadísticas de inferencia
        inference_stats = {
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'total_inference_time': np.sum(inference_times),
            'samples_per_second': len(inference_times) / np.sum(inference_times)
        }
        
        # Resultados completos
        results = {
            'metrics': metrics,
            'inference_stats': inference_stats,
            'num_samples': len(all_targets),
            'model_config': self.config['model_config'],
            'dataset_config': self.config['dataset']
        }
        
        return results, all_predictions, all_targets, all_inputs
    
    def save_results(self, results: Dict[str, Any], predictions: torch.Tensor, 
                    targets: torch.Tensor, inputs: torch.Tensor):
        """Guardar resultados"""
        # Guardar métricas
        results_path = self.results_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Guardar predicciones
        predictions_path = self.results_dir / 'predictions.pt'
        torch.save({
            'predictions': predictions,
            'targets': targets,
            'inputs': inputs
        }, predictions_path)
        
        self.logger.info(f"Resultados guardados en {self.results_dir}")
    
    def generate_report(self, results: Dict[str, Any]):
        """Generar reporte detallado"""
        report_path = self.results_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=== REPORTE DE EVALUACIÓN ===\n\n")
            
            # Información del modelo
            f.write("INFORMACIÓN DEL MODELO:\n")
            f.write(f"Nombre: {self.config['model_name']}\n")
            f.write(f"Configuración: {json.dumps(self.config['model_config'], indent=2)}\n\n")
            
            # Información del dataset
            f.write("INFORMACIÓN DEL DATASET:\n")
            f.write(f"Nombre: {self.config['dataset']['name']}\n")
            f.write(f"Configuración: {json.dumps(self.config['dataset']['config'], indent=2)}\n\n")
            
            # Métricas
            f.write("MÉTRICAS:\n")
            for metric_name, value in results['metrics'].items():
                f.write(f"{metric_name}: {value:.4f}\n")
            f.write("\n")
            
            # Estadísticas de inferencia
            f.write("ESTADÍSTICAS DE INFERENCIA:\n")
            for stat_name, value in results['inference_stats'].items():
                f.write(f"{stat_name}: {value:.4f}\n")
            f.write("\n")
            
            # Resumen
            f.write("RESUMEN:\n")
            f.write(f"Número de muestras evaluadas: {results['num_samples']}\n")
            f.write(f"Tiempo total de inferencia: {results['inference_stats']['total_inference_time']:.2f} segundos\n")
            f.write(f"Muestras por segundo: {results['inference_stats']['samples_per_second']:.2f}\n")
        
        self.logger.info(f"Reporte generado en {report_path}")
    
    def plot_results(self, predictions: torch.Tensor, targets: torch.Tensor, 
                    inputs: torch.Tensor, results: Dict[str, Any]):
        """Generar gráficos de resultados"""
        dataset_name = self.config['dataset']['name']
        
        if dataset_name == 'mnist' or dataset_name == 'cifar10':
            self._plot_classification_results(predictions, targets, inputs, results)
        elif dataset_name == 'sudoku':
            self._plot_sudoku_results(predictions, targets, inputs, results)
        elif dataset_name == 'maze':
            self._plot_maze_results(predictions, targets, inputs, results)
        elif dataset_name == 'seqcopy':
            self._plot_sequence_results(predictions, targets, inputs, results)
        elif dataset_name == 'arc':
            self._plot_arc_results(predictions, targets, inputs, results)
    
    def _plot_classification_results(self, predictions: torch.Tensor, targets: torch.Tensor, 
                                   inputs: torch.Tensor, results: Dict[str, Any]):
        """Gráficos para tareas de clasificación"""
        # Matriz de confusión
        if hasattr(self.metrics, 'confusion_matrix'):
            cm = self.metrics.confusion_matrix(predictions, targets)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Matriz de Confusión')
            plt.xlabel('Predicción')
            plt.ylabel('Verdadero')
            plt.savefig(self.results_dir / 'confusion_matrix.png')
            plt.close()
        
        # Distribución de predicciones
        pred_classes = predictions.argmax(dim=-1)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(targets.numpy(), bins=10, alpha=0.7, label='Verdadero')
        plt.hist(pred_classes.numpy(), bins=10, alpha=0.7, label='Predicción')
        plt.xlabel('Clase')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Clases')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(predictions.max(dim=-1)[0].numpy(), bins=50, alpha=0.7)
        plt.xlabel('Confianza de Predicción')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Confianza')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'classification_distributions.png')
        plt.close()
    
    def _plot_sudoku_results(self, predictions: torch.Tensor, targets: torch.Tensor, 
                           inputs: torch.Tensor, results: Dict[str, Any]):
        """Gráficos para Sudoku"""
        # Ejemplos de predicciones
        num_examples = min(9, len(predictions))
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for i in range(num_examples):
            pred_grid = predictions[i].argmax(dim=-1).numpy()
            target_grid = targets[i].numpy()
            input_grid = inputs[i].numpy()
            
            # Mostrar input, predicción y target
            ax = axes[i]
            ax.imshow(input_grid, cmap='viridis', alpha=0.7)
            ax.set_title(f'Ejemplo {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'sudoku_examples.png')
        plt.close()
    
    def _plot_maze_results(self, predictions: torch.Tensor, targets: torch.Tensor, 
                         inputs: torch.Tensor, results: Dict[str, Any]):
        """Gráficos para laberintos"""
        # Ejemplos de predicciones
        num_examples = min(6, len(predictions))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(num_examples):
            pred_path = (predictions[i] > 0.5).numpy()
            target_path = targets[i].numpy()
            maze = inputs[i].numpy()
            
            # Combinar laberinto con camino predicho
            combined = maze.copy()
            combined[pred_path] = 2  # Camino predicho
            combined[target_path] = 3  # Camino real
            
            ax = axes[i]
            ax.imshow(combined, cmap='tab10')
            ax.set_title(f'Laberinto {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'maze_examples.png')
        plt.close()
    
    def _plot_sequence_results(self, predictions: torch.Tensor, targets: torch.Tensor, 
                             inputs: torch.Tensor, results: Dict[str, Any]):
        """Gráficos para tareas de secuencias"""
        # Longitud de secuencias
        pred_lengths = (predictions.argmax(dim=-1) != 0).sum(dim=1)
        target_lengths = (targets != 0).sum(dim=1)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(target_lengths.numpy(), bins=20, alpha=0.7, label='Verdadero')
        plt.hist(pred_lengths.numpy(), bins=20, alpha=0.7, label='Predicción')
        plt.xlabel('Longitud de Secuencia')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Longitudes')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(target_lengths.numpy(), pred_lengths.numpy(), alpha=0.5)
        plt.plot([0, target_lengths.max()], [0, target_lengths.max()], 'r--')
        plt.xlabel('Longitud Verdadera')
        plt.ylabel('Longitud Predicha')
        plt.title('Longitud Predicha vs Verdadera')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'sequence_analysis.png')
        plt.close()
    
    def _plot_arc_results(self, predictions: torch.Tensor, targets: torch.Tensor, 
                        inputs: torch.Tensor, results: Dict[str, Any]):
        """Gráficos para ARC"""
        # Ejemplos de tareas ARC
        num_examples = min(6, len(predictions))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(num_examples):
            pred_grid = predictions[i].argmax(dim=-1).numpy()
            target_grid = targets[i].numpy()
            input_grid = inputs[i].numpy()
            
            # Mostrar input, predicción y target
            ax = axes[i]
            ax.imshow(input_grid, cmap='tab10')
            ax.set_title(f'Tarea ARC {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'arc_examples.png')
        plt.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """Cargar configuración desde archivo YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Evaluar modelo TRM')
    parser.add_argument('--config', type=str, required=True, help='Ruta al archivo de configuración')
    parser.add_argument('--checkpoint', type=str, required=True, help='Ruta al checkpoint')
    parser.add_argument('--results_dir', type=str, help='Directorio para guardar resultados')
    parser.add_argument('--device', type=str, help='Dispositivo a usar (cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Sobrescribir configuración si se especifica
    if args.results_dir:
        config['results_dir'] = args.results_dir
    if args.device:
        config['device'] = args.device
    
    # Crear evaluador
    evaluator = Evaluator(config, args.checkpoint)
    
    # Evaluar
    results, predictions, targets, inputs = evaluator.evaluate()
    
    # Guardar resultados
    evaluator.save_results(results, predictions, targets, inputs)
    
    # Generar reporte
    evaluator.generate_report(results)
    
    # Generar gráficos
    evaluator.plot_results(predictions, targets, inputs, results)
    
    # Mostrar resumen
    print("\n=== RESUMEN DE EVALUACIÓN ===")
    for metric_name, value in results['metrics'].items():
        print(f"{metric_name}: {value:.4f}")
    print(f"\nTiempo de inferencia promedio: {results['inference_stats']['mean_inference_time']:.4f} segundos")
    print(f"Muestras por segundo: {results['inference_stats']['samples_per_second']:.2f}")


if __name__ == '__main__':
    main()
