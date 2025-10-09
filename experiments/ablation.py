"""
Módulo de ablaciones automáticas para TRM
Permite variar hiperparámetros y generar reportes comparativos.
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
import itertools
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.train import Trainer
from experiments.evaluate import Evaluator


class AblationStudy:
    """
    Estudio de ablaciones para modelos TRM
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_config = config['base_config']
        self.sweep_params = config.get('sweep_params', {})
        self.experiments = config.get('experiments', {})
        
        # Configurar logging
        self._setup_logging()
        
        # Crear directorio de resultados
        self.results_dir = Path(config.get('results_dir', 'results/ablation'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Almacenar resultados
        self.results = []
        self.completed_experiments = 0
        self.total_experiments = 0
    
    def _setup_logging(self):
        """Configurar logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _generate_config_combinations(self, experiment_name: str) -> List[Dict[str, Any]]:
        """Generar combinaciones de configuración para un experimento"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experimento no encontrado: {experiment_name}")
        
        exp_config = self.experiments[experiment_name]
        base_config = self.base_config.copy()
        
        # Aplicar parámetros fijos
        if 'fixed_params' in exp_config:
            base_config['model_config'].update(exp_config['fixed_params'])
        
        # Generar combinaciones de parámetros variables
        sweep_params = exp_config.get('sweep_params', {})
        param_names = list(sweep_params.keys())
        param_values = list(sweep_params.values())
        
        combinations = []
        for combination in itertools.product(*param_values):
            config = base_config.copy()
            config['model_config'] = base_config['model_config'].copy()
            
            # Aplicar combinación
            for param_name, param_value in zip(param_names, combination):
                config['model_config'][param_name] = param_value
            
            # Añadir metadatos
            config['experiment_name'] = experiment_name
            config['combination_id'] = len(combinations)
            config['params'] = dict(zip(param_names, combination))
            
            combinations.append(config)
        
        return combinations
    
    def _run_single_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar un solo experimento"""
        try:
            # Crear directorio para este experimento
            exp_dir = self.results_dir / f"{config['experiment_name']}_{config['combination_id']}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Actualizar configuración
            config['checkpoints']['save_dir'] = str(exp_dir)
            config['logging']['experiment_name'] = f"{config['experiment_name']}_{config['combination_id']}"
            
            # Entrenar modelo
            trainer = Trainer(config)
            trainer.train()
            
            # Evaluar modelo
            checkpoint_path = exp_dir / 'best_model.pt'
            if checkpoint_path.exists():
                evaluator = Evaluator(config, str(checkpoint_path))
                results, _, _, _ = evaluator.evaluate()
                
                # Combinar resultados
                experiment_results = {
                    'experiment_name': config['experiment_name'],
                    'combination_id': config['combination_id'],
                    'params': config['params'],
                    'metrics': results['metrics'],
                    'inference_stats': results['inference_stats'],
                    'config': config
                }
                
                return experiment_results
            else:
                self.logger.error(f"No se encontró checkpoint para {config['experiment_name']}_{config['combination_id']}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error en experimento {config['experiment_name']}_{config['combination_id']}: {str(e)}")
            return None
    
    def run_experiment(self, experiment_name: str, max_workers: int = 4) -> List[Dict[str, Any]]:
        """Ejecutar un experimento de ablación"""
        self.logger.info(f"Ejecutando experimento: {experiment_name}")
        
        # Generar combinaciones
        combinations = self._generate_config_combinations(experiment_name)
        self.total_experiments = len(combinations)
        
        self.logger.info(f"Total de combinaciones: {self.total_experiments}")
        
        # Ejecutar experimentos
        results = []
        
        if max_workers > 1:
            # Ejecución paralela
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_config = {
                    executor.submit(self._run_single_experiment, config): config 
                    for config in combinations
                }
                
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                        self.completed_experiments += 1
                        self.logger.info(f"Completado {self.completed_experiments}/{self.total_experiments}")
                    except Exception as e:
                        self.logger.error(f"Error en experimento {config['experiment_name']}_{config['combination_id']}: {str(e)}")
        else:
            # Ejecución secuencial
            for config in combinations:
                result = self._run_single_experiment(config)
                if result is not None:
                    results.append(result)
                self.completed_experiments += 1
                self.logger.info(f"Completado {self.completed_experiments}/{self.total_experiments}")
        
        # Guardar resultados
        self._save_experiment_results(experiment_name, results)
        
        return results
    
    def run_all_experiments(self, max_workers: int = 4) -> Dict[str, List[Dict[str, Any]]]:
        """Ejecutar todos los experimentos"""
        all_results = {}
        
        for experiment_name in self.experiments.keys():
            self.logger.info(f"Ejecutando experimento: {experiment_name}")
            results = self.run_experiment(experiment_name, max_workers)
            all_results[experiment_name] = results
        
        return all_results
    
    def _save_experiment_results(self, experiment_name: str, results: List[Dict[str, Any]]):
        """Guardar resultados de un experimento"""
        # Guardar resultados completos
        results_path = self.results_dir / f"{experiment_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Crear DataFrame para análisis
        df_data = []
        for result in results:
            row = {
                'experiment_name': result['experiment_name'],
                'combination_id': result['combination_id']
            }
            
            # Añadir parámetros
            row.update(result['params'])
            
            # Añadir métricas
            row.update(result['metrics'])
            
            # Añadir estadísticas de inferencia
            row.update(result['inference_stats'])
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Guardar CSV
        csv_path = self.results_dir / f"{experiment_name}_results.csv"
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Resultados guardados en {results_path} y {csv_path}")
    
    def generate_report(self, experiment_name: str) -> Dict[str, Any]:
        """Generar reporte de ablación"""
        results_path = self.results_dir / f"{experiment_name}_results.json"
        
        if not results_path.exists():
            raise FileNotFoundError(f"No se encontraron resultados para {experiment_name}")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        if not results:
            return {}
        
        # Crear DataFrame
        df_data = []
        for result in results:
            row = {
                'experiment_name': result['experiment_name'],
                'combination_id': result['combination_id']
            }
            row.update(result['params'])
            row.update(result['metrics'])
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Análisis estadístico
        report = {
            'experiment_name': experiment_name,
            'total_combinations': len(results),
            'successful_runs': len(results),
            'failed_runs': 0,  # Se puede calcular si se guarda información de fallos
            'best_combination': {},
            'worst_combination': {},
            'parameter_analysis': {},
            'correlation_analysis': {}
        }
        
        # Mejor y peor combinación
        if 'val_grid_accuracy' in df.columns:
            best_idx = df['val_grid_accuracy'].idxmax()
            worst_idx = df['val_grid_accuracy'].idxmin()
            
            report['best_combination'] = df.iloc[best_idx].to_dict()
            report['worst_combination'] = df.iloc[worst_idx].to_dict()
        
        # Análisis por parámetro
        param_columns = [col for col in df.columns if col not in ['experiment_name', 'combination_id'] and not col.startswith('val_') and not col.startswith('train_')]
        
        for param in param_columns:
            if param in df.columns:
                param_analysis = {
                    'mean': df[param].mean(),
                    'std': df[param].std(),
                    'min': df[param].min(),
                    'max': df[param].max(),
                    'unique_values': df[param].unique().tolist()
                }
                report['parameter_analysis'][param] = param_analysis
        
        # Análisis de correlación
        metric_columns = [col for col in df.columns if col.startswith('val_') or col.startswith('train_')]
        
        if metric_columns:
            correlation_matrix = df[param_columns + metric_columns].corr()
            report['correlation_analysis'] = correlation_matrix.to_dict()
        
        # Guardar reporte
        report_path = self.results_dir / f"{experiment_name}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def generate_comparison_plots(self, experiment_name: str):
        """Generar gráficos de comparación"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        results_path = self.results_dir / f"{experiment_name}_results.json"
        
        if not results_path.exists():
            return
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        if not results:
            return
        
        # Crear DataFrame
        df_data = []
        for result in results:
            row = {
                'experiment_name': result['experiment_name'],
                'combination_id': result['combination_id']
            }
            row.update(result['params'])
            row.update(result['metrics'])
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Crear gráficos
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Gráfico 1: Distribución de métricas
        metric_columns = [col for col in df.columns if col.startswith('val_')]
        if metric_columns:
            df[metric_columns].boxplot(ax=axes[0, 0])
            axes[0, 0].set_title('Distribución de Métricas')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Gráfico 2: Correlación entre parámetros y métricas
        param_columns = [col for col in df.columns if col not in ['experiment_name', 'combination_id'] and not col.startswith('val_') and not col.startswith('train_')]
        
        if param_columns and metric_columns:
            correlation_matrix = df[param_columns + metric_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
            axes[0, 1].set_title('Matriz de Correlación')
        
        # Gráfico 3: Mejores combinaciones
        if 'val_grid_accuracy' in df.columns:
            top_10 = df.nlargest(10, 'val_grid_accuracy')
            axes[1, 0].bar(range(len(top_10)), top_10['val_grid_accuracy'])
            axes[1, 0].set_title('Top 10 Combinaciones')
            axes[1, 0].set_xlabel('Combinación')
            axes[1, 0].set_ylabel('Accuracy')
        
        # Gráfico 4: Análisis de parámetros específicos
        if 'T' in df.columns and 'val_grid_accuracy' in df.columns:
            df.groupby('T')['val_grid_accuracy'].mean().plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Accuracy por T')
            axes[1, 1].set_xlabel('T')
            axes[1, 1].set_ylabel('Accuracy Promedio')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f"{experiment_name}_comparison_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generar reporte resumen de todos los experimentos"""
        summary = {
            'total_experiments': len(self.experiments),
            'experiment_results': {},
            'overall_best': {},
            'recommendations': []
        }
        
        for experiment_name in self.experiments.keys():
            try:
                report = self.generate_report(experiment_name)
                summary['experiment_results'][experiment_name] = report
                
                # Actualizar mejor resultado general
                if 'best_combination' in report and report['best_combination']:
                    if not summary['overall_best'] or report['best_combination'].get('val_grid_accuracy', 0) > summary['overall_best'].get('val_grid_accuracy', 0):
                        summary['overall_best'] = report['best_combination']
                        summary['overall_best']['experiment_name'] = experiment_name
                
            except Exception as e:
                self.logger.error(f"Error generando reporte para {experiment_name}: {str(e)}")
        
        # Generar recomendaciones
        summary['recommendations'] = self._generate_recommendations(summary)
        
        # Guardar reporte resumen
        summary_path = self.results_dir / 'summary_report.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generar recomendaciones basadas en los resultados"""
        recommendations = []
        
        # Analizar mejores parámetros
        if 'overall_best' in summary and summary['overall_best']:
            best = summary['overall_best']
            
            if 'T' in best:
                recommendations.append(f"Usar T={best['T']} para mejor rendimiento")
            
            if 'n_inner' in best:
                recommendations.append(f"Usar n_inner={best['n_inner']} para mejor rendimiento")
            
            if 'use_attention' in best:
                attention_type = "atención" if best['use_attention'] else "MLP-Mixer"
                recommendations.append(f"Usar {attention_type} para mejor rendimiento")
            
            if 'ema_decay' in best:
                recommendations.append(f"Usar EMA decay={best['ema_decay']} para mejor rendimiento")
        
        # Analizar correlaciones
        for exp_name, exp_results in summary.get('experiment_results', {}).items():
            if 'correlation_analysis' in exp_results:
                # Buscar correlaciones fuertes
                correlations = exp_results['correlation_analysis']
                # Implementar lógica de análisis de correlaciones
        
        return recommendations


def load_config(config_path: str) -> Dict[str, Any]:
    """Cargar configuración desde archivo YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Ejecutar estudio de ablaciones TRM')
    parser.add_argument('--config', type=str, required=True, help='Ruta al archivo de configuración')
    parser.add_argument('--experiment', type=str, help='Nombre del experimento específico a ejecutar')
    parser.add_argument('--max_workers', type=int, default=4, help='Número máximo de workers paralelos')
    parser.add_argument('--results_dir', type=str, help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Sobrescribir configuración si se especifica
    if args.results_dir:
        config['results_dir'] = args.results_dir
    
    # Crear estudio de ablaciones
    ablation_study = AblationStudy(config)
    
    # Ejecutar experimentos
    if args.experiment:
        # Ejecutar experimento específico
        results = ablation_study.run_experiment(args.experiment, args.max_workers)
        
        # Generar reporte
        report = ablation_study.generate_report(args.experiment)
        
        # Generar gráficos
        ablation_study.generate_comparison_plots(args.experiment)
        
        print(f"\n=== RESULTADOS DEL EXPERIMENTO {args.experiment} ===")
        print(f"Total de combinaciones: {len(results)}")
        if report and 'best_combination' in report:
            print(f"Mejor combinación: {report['best_combination']}")
    else:
        # Ejecutar todos los experimentos
        all_results = ablation_study.run_all_experiments(args.max_workers)
        
        # Generar reporte resumen
        summary = ablation_study.generate_summary_report()
        
        print(f"\n=== RESUMEN DE ABLACIONES ===")
        print(f"Total de experimentos: {summary['total_experiments']}")
        if summary['overall_best']:
            print(f"Mejor resultado general: {summary['overall_best']}")
        print(f"Recomendaciones: {summary['recommendations']}")


if __name__ == '__main__':
    main()
