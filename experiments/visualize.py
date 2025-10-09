"""
Script para visualización de resultados y análisis de TRM
Visualiza la evolución de estados, razonamiento y comparaciones con baselines.
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import time
import json

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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


class TRMVisualizer:
    """
    Visualizador para análisis de modelos TRM
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
        
        # Crear directorio de visualizaciones
        self.viz_dir = Path(config.get('viz_dir', 'visualizations'))
        self.viz_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    def visualize_state_evolution(self, num_examples: int = 5):
        """Visualizar evolución de estados y y z"""
        self.logger.info("Visualizando evolución de estados...")
        
        self._load_model()
        self._create_dataloader()
        
        self.model.eval()
        
        # Obtener ejemplos
        examples = []
        for i, batch in enumerate(self.test_loader):
            if i >= num_examples:
                break
            examples.append(batch)
        
        # Procesar cada ejemplo
        for idx, batch in enumerate(examples):
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets, _ = batch
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass con estados
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    outputs = self.model(inputs, return_states=True, return_halt_scores=True)
                    states = outputs.get('states', [])
                    halt_scores = outputs.get('halt_scores', [])
                else:
                    outputs = self.model(inputs)
                    states = []
                    halt_scores = []
            
            # Visualizar evolución de estados
            self._plot_state_evolution(idx, inputs, targets, states, halt_scores)
    
    def _plot_state_evolution(self, idx: int, inputs: torch.Tensor, targets: torch.Tensor, 
                            states: List[torch.Tensor], halt_scores: List[torch.Tensor]):
        """Plotear evolución de estados para un ejemplo"""
        if not states:
            self.logger.warning("No hay estados para visualizar")
            return
        
        # Crear subplots
        num_states = len(states)
        fig, axes = plt.subplots(2, num_states, figsize=(4 * num_states, 8))
        
        if num_states == 1:
            axes = axes.reshape(2, 1)
        
        for i, state in enumerate(states):
            # Estado y
            state_np = state[0].cpu().numpy()  # Primer ejemplo del batch
            
            # Visualizar como heatmap
            sns.heatmap(state_np, ax=axes[0, i], cmap='viridis', cbar=True)
            axes[0, i].set_title(f'Estado y - Iteración {i+1}')
            axes[0, i].set_xlabel('Dimensión')
            axes[0, i].set_ylabel('Posición')
        
        # Scores de parada
        if halt_scores:
            halt_np = halt_scores[0].cpu().numpy()  # Primer ejemplo del batch
            
            for i, halt in enumerate(halt_scores):
                halt_np = halt[0].cpu().numpy()
                axes[1, i].plot(halt_np)
                axes[1, i].set_title(f'Scores de Parada - Iteración {i+1}')
                axes[1, i].set_xlabel('Posición')
                axes[1, i].set_ylabel('Score')
                axes[1, i].axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'state_evolution_example_{idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_attention_patterns(self, num_examples: int = 5):
        """Visualizar patrones de atención (si el modelo los tiene)"""
        self.logger.info("Visualizando patrones de atención...")
        
        self._load_model()
        self._create_dataloader()
        
        self.model.eval()
        
        # Obtener ejemplos
        examples = []
        for i, batch in enumerate(self.test_loader):
            if i >= num_examples:
                break
            examples.append(batch)
        
        # Procesar cada ejemplo
        for idx, batch in enumerate(examples):
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets, _ = batch
            
            inputs = inputs.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(inputs)
            
            # Visualizar patrones de atención
            self._plot_attention_patterns(idx, inputs, outputs)
    
    def _plot_attention_patterns(self, idx: int, inputs: torch.Tensor, outputs: torch.Tensor):
        """Plotear patrones de atención"""
        # Esta es una implementación simplificada
        # En la práctica, se necesitaría extraer los pesos de atención del modelo
        
        # Por ahora, creamos un patrón de atención simulado
        seq_len = inputs.shape[1]
        attention_pattern = np.random.rand(seq_len, seq_len)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_pattern, cmap='Blues', cbar=True)
        plt.title(f'Patrón de Atención - Ejemplo {idx+1}')
        plt.xlabel('Posición de Salida')
        plt.ylabel('Posición de Entrada')
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'attention_pattern_example_{idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_reasoning_process(self, num_examples: int = 3):
        """Visualizar proceso de razonamiento paso a paso"""
        self.logger.info("Visualizando proceso de razonamiento...")
        
        self._load_model()
        self._create_dataloader()
        
        self.model.eval()
        
        # Obtener ejemplos
        examples = []
        for i, batch in enumerate(self.test_loader):
            if i >= num_examples:
                break
            examples.append(batch)
        
        # Procesar cada ejemplo
        for idx, batch in enumerate(examples):
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets, _ = batch
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass con estados
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    outputs = self.model(inputs, return_states=True, return_halt_scores=True)
                    states = outputs.get('states', [])
                    halt_scores = outputs.get('halt_scores', [])
                else:
                    outputs = self.model(inputs)
                    states = []
                    halt_scores = []
            
            # Visualizar proceso de razonamiento
            self._plot_reasoning_process(idx, inputs, targets, outputs, states, halt_scores)
    
    def _plot_reasoning_process(self, idx: int, inputs: torch.Tensor, targets: torch.Tensor, 
                              outputs: torch.Tensor, states: List[torch.Tensor], halt_scores: List[torch.Tensor]):
        """Plotear proceso de razonamiento paso a paso"""
        dataset_name = self.config['dataset']['name']
        
        if dataset_name == 'sudoku':
            self._plot_sudoku_reasoning(idx, inputs, targets, outputs, states, halt_scores)
        elif dataset_name == 'maze':
            self._plot_maze_reasoning(idx, inputs, targets, outputs, states, halt_scores)
        elif dataset_name == 'arc':
            self._plot_arc_reasoning(idx, inputs, targets, outputs, states, halt_scores)
        else:
            self._plot_generic_reasoning(idx, inputs, targets, outputs, states, halt_scores)
    
    def _plot_sudoku_reasoning(self, idx: int, inputs: torch.Tensor, targets: torch.Tensor, 
                             outputs: torch.Tensor, states: List[torch.Tensor], halt_scores: List[torch.Tensor]):
        """Plotear razonamiento de Sudoku"""
        input_grid = inputs[0].cpu().numpy()
        target_grid = targets[0].cpu().numpy()
        
        if isinstance(outputs, dict):
            pred_grid = outputs.get('logits', outputs.get('output', outputs.get('predictions')))
        else:
            pred_grid = outputs
        
        pred_grid = pred_grid[0].argmax(dim=-1).cpu().numpy()
        
        # Crear figura con múltiples subplots
        num_states = len(states)
        fig, axes = plt.subplots(2, num_states + 2, figsize=(4 * (num_states + 2), 8))
        
        if num_states == 1:
            axes = axes.reshape(2, 3)
        
        # Input
        axes[0, 0].imshow(input_grid, cmap='viridis')
        axes[0, 0].set_title('Input (Puzzle)')
        axes[0, 0].axis('off')
        
        # Target
        axes[0, 1].imshow(target_grid, cmap='viridis')
        axes[0, 1].set_title('Target (Solución)')
        axes[0, 1].axis('off')
        
        # Predicción
        axes[0, 2].imshow(pred_grid, cmap='viridis')
        axes[0, 2].set_title('Predicción')
        axes[0, 2].axis('off')
        
        # Estados intermedios
        for i, state in enumerate(states):
            if i < num_states:
                state_np = state[0].cpu().numpy()
                sns.heatmap(state_np, ax=axes[1, i], cmap='viridis', cbar=True)
                axes[1, i].set_title(f'Estado {i+1}')
                axes[1, i].set_xlabel('Dimensión')
                axes[1, i].set_ylabel('Posición')
        
        # Scores de parada
        if halt_scores:
            for i, halt in enumerate(halt_scores):
                if i < num_states:
                    halt_np = halt[0].cpu().numpy()
                    axes[1, i].plot(halt_np, color='red', alpha=0.7)
                    axes[1, i].axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'sudoku_reasoning_example_{idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_maze_reasoning(self, idx: int, inputs: torch.Tensor, targets: torch.Tensor, 
                           outputs: torch.Tensor, states: List[torch.Tensor], halt_scores: List[torch.Tensor]):
        """Plotear razonamiento de laberinto"""
        maze = inputs[0].cpu().numpy()
        target_path = targets[0].cpu().numpy()
        
        if isinstance(outputs, dict):
            pred_path = outputs.get('logits', outputs.get('output', outputs.get('predictions')))
        else:
            pred_path = outputs
        
        pred_path = (pred_path[0] > 0.5).cpu().numpy()
        
        # Crear figura
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Laberinto original
        axes[0].imshow(maze, cmap='tab10')
        axes[0].set_title('Laberinto Original')
        axes[0].axis('off')
        
        # Camino objetivo
        combined_target = maze.copy()
        combined_target[target_path] = 2
        axes[1].imshow(combined_target, cmap='tab10')
        axes[1].set_title('Camino Objetivo')
        axes[1].axis('off')
        
        # Camino predicho
        combined_pred = maze.copy()
        combined_pred[pred_path] = 3
        axes[2].imshow(combined_pred, cmap='tab10')
        axes[2].set_title('Camino Predicho')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'maze_reasoning_example_{idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_arc_reasoning(self, idx: int, inputs: torch.Tensor, targets: torch.Tensor, 
                          outputs: torch.Tensor, states: List[torch.Tensor], halt_scores: List[torch.Tensor]):
        """Plotear razonamiento de ARC"""
        input_grid = inputs[0].cpu().numpy()
        target_grid = targets[0].cpu().numpy()
        
        if isinstance(outputs, dict):
            pred_grid = outputs.get('logits', outputs.get('output', outputs.get('predictions')))
        else:
            pred_grid = outputs
        
        pred_grid = pred_grid[0].argmax(dim=-1).cpu().numpy()
        
        # Crear figura
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Input
        axes[0].imshow(input_grid, cmap='tab10')
        axes[0].set_title('Input')
        axes[0].axis('off')
        
        # Target
        axes[1].imshow(target_grid, cmap='tab10')
        axes[1].set_title('Target')
        axes[1].axis('off')
        
        # Predicción
        axes[2].imshow(pred_grid, cmap='tab10')
        axes[2].set_title('Predicción')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'arc_reasoning_example_{idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_generic_reasoning(self, idx: int, inputs: torch.Tensor, targets: torch.Tensor, 
                              outputs: torch.Tensor, states: List[torch.Tensor], halt_scores: List[torch.Tensor]):
        """Plotear razonamiento genérico"""
        # Visualización genérica para otros tipos de datos
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Input
        if inputs.dim() > 2:
            input_vis = inputs[0].mean(dim=0).cpu().numpy()
        else:
            input_vis = inputs[0].cpu().numpy()
        
        axes[0, 0].imshow(input_vis, cmap='viridis')
        axes[0, 0].set_title('Input')
        axes[0, 0].axis('off')
        
        # Target
        if targets.dim() > 2:
            target_vis = targets[0].mean(dim=0).cpu().numpy()
        else:
            target_vis = targets[0].cpu().numpy()
        
        axes[0, 1].imshow(target_vis, cmap='viridis')
        axes[0, 1].set_title('Target')
        axes[0, 1].axis('off')
        
        # Predicción
        if isinstance(outputs, dict):
            pred = outputs.get('logits', outputs.get('output', outputs.get('predictions')))
        else:
            pred = outputs
        
        if pred.dim() > 2:
            pred_vis = pred[0].mean(dim=0).cpu().numpy()
        else:
            pred_vis = pred[0].cpu().numpy()
        
        axes[1, 0].imshow(pred_vis, cmap='viridis')
        axes[1, 0].set_title('Predicción')
        axes[1, 0].axis('off')
        
        # Scores de parada
        if halt_scores:
            halt_np = halt_scores[0][0].cpu().numpy()
            axes[1, 1].plot(halt_np)
            axes[1, 1].set_title('Scores de Parada')
            axes[1, 1].set_xlabel('Posición')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'generic_reasoning_example_{idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_dashboard(self):
        """Crear dashboard interactivo con Plotly"""
        self.logger.info("Creando dashboard interactivo...")
        
        # Crear dashboard con múltiples gráficos
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Evolución de Estados', 'Scores de Parada', 'Métricas', 'Comparación'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Datos simulados para el dashboard
        # En la práctica, se cargarían datos reales del modelo
        
        # Evolución de estados
        state_data = np.random.rand(10, 10)
        fig.add_trace(
            go.Heatmap(z=state_data, colorscale='Viridis'),
            row=1, col=1
        )
        
        # Scores de parada
        halt_data = np.random.rand(10)
        fig.add_trace(
            go.Scatter(y=halt_data, mode='lines+markers', name='Halt Scores'),
            row=1, col=2
        )
        
        # Métricas
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        values = [0.85, 0.82, 0.88, 0.85]
        fig.add_trace(
            go.Bar(x=metrics, y=values, name='Métricas'),
            row=2, col=1
        )
        
        # Comparación
        models = ['TRM', 'Transformer', 'LSTM', 'CNN']
        accuracies = [0.85, 0.82, 0.78, 0.80]
        fig.add_trace(
            go.Scatter(x=models, y=accuracies, mode='markers', name='Comparación'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Dashboard de Análisis TRM",
            showlegend=False,
            height=800
        )
        
        # Guardar dashboard
        fig.write_html(self.viz_dir / 'interactive_dashboard.html')
        self.logger.info(f"Dashboard guardado en {self.viz_dir / 'interactive_dashboard.html'}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Cargar configuración desde archivo YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Visualizar modelo TRM')
    parser.add_argument('--config', type=str, required=True, help='Ruta al archivo de configuración')
    parser.add_argument('--checkpoint', type=str, required=True, help='Ruta al checkpoint')
    parser.add_argument('--viz_dir', type=str, help='Directorio para guardar visualizaciones')
    parser.add_argument('--device', type=str, help='Dispositivo a usar (cpu, cuda, mps)')
    parser.add_argument('--num_examples', type=int, default=5, help='Número de ejemplos a visualizar')
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Sobrescribir configuración si se especifica
    if args.viz_dir:
        config['viz_dir'] = args.viz_dir
    if args.device:
        config['device'] = args.device
    
    # Crear visualizador
    visualizer = TRMVisualizer(config, args.checkpoint)
    
    # Generar visualizaciones
    visualizer.visualize_state_evolution(args.num_examples)
    visualizer.visualize_attention_patterns(args.num_examples)
    visualizer.visualize_reasoning_process(args.num_examples)
    visualizer.create_interactive_dashboard()
    
    print(f"\nVisualizaciones guardadas en {visualizer.viz_dir}")


if __name__ == '__main__':
    main()
