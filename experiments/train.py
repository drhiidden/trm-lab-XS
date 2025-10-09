"""
Script unificado para entrenamiento de modelos TRM
Soporta diferentes tareas y configuraciones mediante archivos YAML.
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from core.registry import create_model
from core.utils import get_device, count_parameters, initialize_weights
from datasets.sudoku import create_sudoku_dataloaders, SudokuMetrics
from datasets.maze import create_maze_dataloaders, MazeMetrics
from datasets.mnist import create_mnist_dataloaders, MNISTMetrics
from datasets.seqcopy import create_seqcopy_dataloaders, SequenceCopyMetrics
from datasets.cifar10 import create_cifar10_dataloaders, CIFAR10Metrics
from datasets.arc import create_arc_dataloaders, ARCMetrics


class Trainer:
    """
    Entrenador unificado para modelos TRM
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device()
        
        # Configurar logging
        self._setup_logging()
        
        # Configurar semilla
        self._setup_seed()
        
        # Inicializar componentes
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.metrics = None
        
        # Configurar wandb
        if config.get('logging', {}).get('use_wandb', False):
            self._setup_wandb()
        
        # Crear directorio de checkpoints
        self.checkpoint_dir = Path(config['checkpoints']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar configuración
        self._save_config()
    
    def _setup_logging(self):
        """Configurar logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.checkpoint_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_seed(self):
        """Configurar semilla para reproducibilidad"""
        seed = self.config.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _setup_wandb(self):
        """Configurar Weights & Biases"""
        wandb.init(
            project=self.config['logging']['project'],
            name=self.config['logging']['experiment_name'],
            config=self.config
        )
    
    def _save_config(self):
        """Guardar configuración"""
        config_path = self.checkpoint_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def _create_model(self):
        """Crear modelo"""
        model_config = self.config['model_config'].copy()
        model_name = self.config['model_name']
        
        self.model = create_model(model_name, model_config)
        self.model = self.model.to(self.device)
        
        # Inicializar pesos si es necesario
        if hasattr(self.model, 'apply'):
            self.model.apply(lambda m: initialize_weights(m, 'xavier_uniform'))
        
        self.logger.info(f"Modelo creado: {count_parameters(self.model)} parámetros")
    
    def _create_optimizer(self):
        """Crear optimizador"""
        opt_config = self.config['optimizer']
        
        if opt_config['name'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0),
                betas=opt_config.get('betas', [0.9, 0.999]),
                eps=opt_config.get('eps', 1e-8)
            )
        elif opt_config['name'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0),
                betas=opt_config.get('betas', [0.9, 0.95]),
                eps=opt_config.get('eps', 1e-8)
            )
        else:
            raise ValueError(f"Optimizador no soportado: {opt_config['name']}")
    
    def _create_scheduler(self):
        """Crear scheduler"""
        sched_config = self.config.get('scheduler', {})
        
        if sched_config.get('name') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.get('max_steps', 100000),
                eta_min=1e-6
            )
        elif sched_config.get('name') == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 20),
                gamma=sched_config.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
    
    def _create_criterion(self):
        """Crear función de pérdida"""
        loss_config = self.config['loss']
        
        if loss_config['name'] == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=loss_config.get('label_smoothing', 0)
            )
        elif loss_config['name'] == 'bce_with_logits':
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(loss_config.get('pos_weight', 1.0))
            )
        elif loss_config['name'] == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Función de pérdida no soportada: {loss_config['name']}")
    
    def _create_dataloaders(self):
        """Crear dataloaders"""
        dataset_config = self.config['dataset']
        dataset_name = dataset_config['name']
        
        if dataset_name == 'sudoku':
            self.train_loader, self.val_loader, self.test_loader = create_sudoku_dataloaders(
                **dataset_config['config']
            )
            self.metrics = SudokuMetrics()
        elif dataset_name == 'maze':
            self.train_loader, self.val_loader, self.test_loader = create_maze_dataloaders(
                **dataset_config['config']
            )
            self.metrics = MazeMetrics()
        elif dataset_name == 'mnist':
            self.train_loader, self.val_loader, self.test_loader = create_mnist_dataloaders(
                **dataset_config['config']
            )
            self.metrics = MNISTMetrics()
        elif dataset_name == 'seqcopy':
            self.train_loader, self.val_loader, self.test_loader = create_seqcopy_dataloaders(
                **dataset_config['config']
            )
            self.metrics = SequenceCopyMetrics()
        elif dataset_name == 'cifar10':
            self.train_loader, self.val_loader, self.test_loader = create_cifar10_dataloaders(
                **dataset_config['config']
            )
            self.metrics = CIFAR10Metrics()
        elif dataset_name == 'arc':
            self.train_loader, self.val_loader, self.test_loader = create_arc_dataloaders(
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
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Entrenar una época"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 2:
                inputs, targets = batch
                task_labels = None
            else:
                inputs, targets, task_labels = batch
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'forward'):
                outputs = self.model(inputs)
                if isinstance(outputs, dict):
                    predictions = outputs.get('logits', outputs.get('output', outputs.get('predictions')))
                else:
                    predictions = outputs
            else:
                predictions = self.model(inputs)
            
            # Calcular pérdida
            if self.criterion is not None:
                loss = self.criterion(predictions, targets)
            else:
                loss = torch.tensor(0.0, device=self.device)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Actualizar scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Acumular métricas
            total_loss += loss.item()
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())
            
            # Logging
            if batch_idx % self.config['logging']['log_interval'] == 0:
                pbar.set_postfix({'loss': loss.item()})
                
                if self.config.get('logging', {}).get('use_wandb', False):
                    wandb.log({
                        'train_loss': loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'batch': batch_idx
                    })
        
        # Calcular métricas de la época
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        epoch_metrics = self._compute_metrics(
            all_predictions, all_targets, 
            self.config.get('metrics', ['accuracy'])
        )
        epoch_metrics['train_loss'] = total_loss / len(self.train_loader)
        
        return epoch_metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validar modelo"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                if len(batch) == 2:
                    inputs, targets = batch
                    task_labels = None
                else:
                    inputs, targets, task_labels = batch
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward'):
                    outputs = self.model(inputs)
                    if isinstance(outputs, dict):
                        predictions = outputs.get('logits', outputs.get('output', outputs.get('predictions')))
                    else:
                        predictions = outputs
                else:
                    predictions = self.model(inputs)
                
                # Calcular pérdida
                if self.criterion is not None:
                    loss = self.criterion(predictions, targets)
                    total_loss += loss.item()
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Calcular métricas
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        val_metrics = self._compute_metrics(
            all_predictions, all_targets, 
            self.config.get('metrics', ['accuracy'])
        )
        val_metrics['val_loss'] = total_loss / len(self.val_loader)
        
        return val_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Guardar checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        # Guardar checkpoint regular
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Guardar mejor modelo
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
        
        # Guardar último modelo
        last_path = self.checkpoint_dir / 'last_model.pt'
        torch.save(checkpoint, last_path)
    
    def train(self):
        """Entrenar modelo"""
        self.logger.info("Iniciando entrenamiento...")
        
        # Crear componentes
        self._create_model()
        self._create_optimizer()
        self._create_scheduler()
        self._create_criterion()
        self._create_dataloaders()
        
        # Variables de entrenamiento
        best_metric = float('-inf') if self.config['checkpoints']['mode'] == 'max' else float('inf')
        monitor_metric = self.config['checkpoints']['monitor_metric']
        
        start_time = time.time()
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Entrenar
            train_metrics = self.train_epoch(epoch)
            
            # Validar
            val_metrics = self.validate(epoch)
            
            # Combinar métricas
            all_metrics = {**train_metrics, **val_metrics}
            
            # Logging
            self.logger.info(f"Epoch {epoch}: {all_metrics}")
            
            if self.config.get('logging', {}).get('use_wandb', False):
                wandb.log(all_metrics, step=epoch)
            
            # Guardar checkpoint
            current_metric = all_metrics.get(monitor_metric, 0)
            is_best = (current_metric > best_metric) if self.config['checkpoints']['mode'] == 'max' else (current_metric < best_metric)
            
            if is_best:
                best_metric = current_metric
            
            if epoch % self.config['logging']['save_interval'] == 0 or is_best:
                self.save_checkpoint(epoch, all_metrics, is_best)
        
        # Guardar métricas finales
        total_time = time.time() - start_time
        final_metrics = {
            'total_time': total_time,
            'best_metric': best_metric,
            'final_epoch': epoch
        }
        
        with open(self.checkpoint_dir / 'final_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        self.logger.info(f"Entrenamiento completado en {total_time:.2f} segundos")
        self.logger.info(f"Mejor métrica ({monitor_metric}): {best_metric:.4f}")
        
        if self.config.get('logging', {}).get('use_wandb', False):
            wandb.finish()


def load_config(config_path: str) -> Dict[str, Any]:
    """Cargar configuración desde archivo YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Entrenar modelo TRM')
    parser.add_argument('--config', type=str, required=True, help='Ruta al archivo de configuración')
    parser.add_argument('--resume', type=str, help='Ruta al checkpoint para reanudar entrenamiento')
    parser.add_argument('--device', type=str, help='Dispositivo a usar (cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Sobrescribir dispositivo si se especifica
    if args.device:
        config['device'] = args.device
    
    # Crear entrenador
    trainer = Trainer(config)
    
    # Entrenar
    trainer.train()


if __name__ == '__main__':
    main()
