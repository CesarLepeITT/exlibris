import os
from time import time

import json

import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.base import clone

from .dataset import Dataset

class Stats:
    def __init__(self, experiment_name, n_runs=30, models=None, datasets=None, normalize = False, scaler_type = "minmax"):
        if models is None:
            models = {}
        if datasets is None:
            datasets = {}
        self.experiment_name = experiment_name
        self.n_runs = n_runs
        self.models = models
        self.datasets = datasets
        self.set_models(models=models)
        self.set_datasets(datasets=datasets, normalize=normalize, scaler_type=scaler_type)

        self.stats_path = os.path.join(os.getcwd(), f'stats_{self.experiment_name}')

        self.figures_path = os.path.join(self.stats_path, f'fig_{self.experiment_name}')
        
        self.training_status = {}
        self.status_file = os.path.join(self.stats_path, "training_status.json")
        
        os.makedirs(self.stats_path, exist_ok=True)
        
        self._load_training_status()
        
    def rename_model(self, old_name, new_name):
        # Verifica si el modelo existe en la lista de modelos
        if old_name not in self.models:
            raise ValueError(f"El modelo '{old_name}' no existe en los modelos cargados.")
        
        new_models = {}
        for key, value in self.models.items():
            if key == old_name:
                new_models[new_name] = value  
            else:
                new_models[key] = value

        self.models = new_models        
        
        stats_path = os.path.join(os.getcwd(), f'stats_{self.experiment_name}')
        
        for dataset_name in self.datasets.keys():
            dataset_stats_path = os.path.join(stats_path, f'stats_{dataset_name}')
            model_path_old = os.path.join(dataset_stats_path, old_name)
            model_path_new = os.path.join(dataset_stats_path, new_name)
            
            if os.path.exists(model_path_old):
                os.rename(model_path_old, model_path_new)
                
                for file_name in os.listdir(model_path_new):
                    old_file_path = os.path.join(model_path_new, file_name)
                    new_file_path = os.path.join(model_path_new, file_name.replace(old_name, new_name))
                    if old_file_path != new_file_path:
                        shutil.move(old_file_path, new_file_path)
            else:
                raise Exception(f'No se encontró el modelo {old_name} en el conjunto de datos {dataset_name}.')
        
        if os.path.exists(self.status_file):
            with open(self.status_file, 'r') as f:
                self.training_status = json.load(f)
            
            for dataset_name in self.training_status.keys():
                if old_name in self.training_status[dataset_name]:
                    self.training_status[dataset_name][new_name] = self.training_status[dataset_name].pop(old_name)
            
            self._save_training_status()
        
        print(f"El modelo '{old_name}' ha sido renombrado a '{new_name}' y todos sus archivos han sido actualizados.")
        
    def _load_training_status(self):
        if os.path.exists(self.status_file):
            with open(self.status_file, 'r') as f:
                self.training_status = json.load(f)

            for dataset_name in self.datasets.keys():
                for model_name in self.models.keys():
                    if model_name not in self.training_status[dataset_name]:
                        self.training_status[dataset_name][model_name] = 0
                
            self._save_training_status()
        else:
            self.training_status = {dataset: {model_name: 0 for model_name in self.models.keys()} 
                                    for dataset in self.datasets.keys()}
            
    def _save_training_status(self):
        with open(self.status_file, 'w') as f:
            json.dump(self.training_status, f, indent=4)        
        
        
    def get_experiment_name(self):
        return self.experiment_name
    
    def get_n_runs(self):
        return self.n_runs
    
    def get_models(self):
        return self.models
    
    def get_datasets(self):
        return self.datasets
    
    def set_datasets(self, datasets=None, normalize = False, scaler_type = "minmax"):
        if datasets is None:
            datasets = {}

        if not datasets:
            dataset_path = Dataset.path_datasets()        
            files = os.listdir(dataset_path)
            for file in files:
                if file.endswith('.csv'):
                    #dataset_file_path = os.path.join(dataset_path, file)
                    dataset_name = os.path.splitext(file)[0]
                    aux = Dataset(dataset_name)
                    datasets[dataset_name] = aux.load_dataset(split_target = False, normalize=normalize, scaler_type=scaler_type)

        if not isinstance(datasets, dict):
            raise TypeError(f'Datasets must be a dictionary. {type(datasets)} was provided.')

        for dataset in datasets.values():
            if not isinstance(dataset, pd.DataFrame):
                raise TypeError(f'Dataset must be a DataFrame. {type(dataset)} was provided.')
        
        self.datasets = datasets                    
    
    def set_models(self, models=None):
        if models is None:
            models = {}

        if not models:
            raise Exception('No models were provided.')
        
        self.models = models
    
    def _write_csv(self, df, path, mode='a', header=False):
        if not os.path.exists(path):
            df.to_csv(path, index=False)    
        else:
            df.to_csv(path, mode=mode, header=header, index=False)
    
    def evaluate(self):
        stats_path = os.path.join(os.getcwd(), f'stats_{self.experiment_name}')
        os.makedirs(stats_path, exist_ok=True)
        
        for dataset_name, dataset in self.datasets.items():
            path_stats = os.path.join(stats_path, f'stats_{dataset_name}')
            os.makedirs(path_stats, exist_ok=True)
            
            X, y = Dataset.split_target(dataset)

            for run in range(self.n_runs):
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=run)
                
                for model_name, model in self.models.items():
                    if self.training_status[dataset_name][model_name] >= self.n_runs:
                        print(f"Model '{model_name}' has already completed {self.n_runs} runs on the '{dataset_name}' dataset.")
                        continue
                    
                    est = clone(model)
                    path_model = os.path.join(path_stats, f'{model_name}')
                    os.makedirs(path_model, exist_ok=True)

                    path_predictions_test = os.path.join(path_model, f'predictions_test_{model_name}.csv')
                    path_y_true = os.path.join(path_model, f'y_true_{model_name}.csv')
                    path_metrics = os.path.join(path_model, f'metrics_{model_name}.csv')
                    
                    start_total = time()
                    start_fit = time()
                    est.fit(X_train, y_train)
                    fit_time = time() - start_fit

                    start_predict = time()
                    y_pred = est.predict(X_test)
                    predict_time = time() - start_predict

                    runtime = time() - start_total
                    
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    auc_roc = roc_auc_score(y_test, y_pred)
                    accuracy = accuracy_score(y_test, y_pred)
                    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
                    
                    df_metrics = pd.DataFrame({
                        'Runtime':[runtime],
                        'Fit Time' :[fit_time],
                        'Prediction Time' :[predict_time],
                        'Precision': [precision],
                        'Recall': [recall],
                        'F1 Score': [f1],
                        'AUC ROC': [auc_roc],
                        'Accuracy': [accuracy],
                        'Balanced Accuracy': [balanced_accuracy]
                    })
                    
                    if getattr(est, '__class__', None) is not None:
                        if self.is_gsgp(est):
                            df_metrics['name_run1'] = [est.name_run1]

                    self._write_csv(df_metrics, path_metrics)
                    
                    self._save_predictions(path_y_true, y_test, run)
                    self._save_predictions(path_predictions_test, y_pred, run)
                
                    self.training_status[dataset_name][model_name] = run + 1
                    self._save_training_status()
    @staticmethod
    def is_gsgp(model):
        return model.__class__.__name__ in ["gsgpcudaregressor", "GsgpCudaClassifier"]
    
    def _save_predictions(self, path, values, run):
        values = pd.Series(np.ravel(values)).reset_index(drop=True)
        df_test = pd.DataFrame({f"y_{run + 1}": values})
        
        if os.path.exists(path):
            aux = pd.read_csv(path)
            df_test_csv = pd.concat([aux, df_test], axis=1)
        else:
            df_test_csv = df_test
        df_test_csv.to_csv(path, index=False)
    
    def _read_metrics(self, error_selection, dataset_name, only_gsgp=False):
        models = self._filter_models(only_gsgp)
        save_path = os.path.join(os.getcwd(), f'stats_{self.experiment_name}')
        
        if not os.path.exists(save_path):
            raise Exception(f'No experiments found. Path does not exist: {save_path}')
        
        path_stats = os.path.join(save_path, f'stats_{dataset_name}')
        if not os.path.exists(path_stats):
            raise Exception(f'No statistics found for dataset: {dataset_name}')
        
        df_data = pd.DataFrame()
        
        for model_name, model in models.items():
            model_path = os.path.join(path_stats, f'{model_name}')
            metrics_path = os.path.join(model_path, f'metrics_{model_name}.csv')
            predictions_test_path = os.path.join(model_path, f'predictions_test_{model_name}.csv')
            y_true_path = os.path.join(model_path, f'y_true_{model_name}.csv')
            
            if not os.path.exists(model_path):
                raise Exception(f'No experiments have been conducted. The path could not be found. {model_path}')          
            if not os.path.exists(predictions_test_path):
                raise Exception(f'No experiments have been conducted. The path could not be found. {predictions_test_path}')         
            if not os.path.exists(y_true_path):
                raise Exception(f'No experiments have been conducted. The path could not be found. {y_true_path}')         
            if not os.path.exists(metrics_path):
                raise Exception(f'No experiments have been conducted. The path could not be found. {metrics_path}')    
                                 
            df_metrics = pd.read_csv(metrics_path)
            df_data[f'{model_name}_{error_selection}'] = df_metrics[f'{error_selection}']
        
        return df_data

    def _filter_models(self, only_gsgp):
        if only_gsgp:
            return {model_name: model for model_name, model in self.models.items() if self.is_gsgp(model)}
        return self.models

    def get_violin(self, error_selection, ncols=2, nrows=5, figsize=(12, 20)):
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
        axes = axes.flatten()  

        datasets = list(self.datasets.keys())
        
        num_colors = len(self.models)
        colors = cm.tab20(np.linspace(0, 1, num_colors))  

        for i, ax in enumerate(axes):
            df_data = self._read_metrics(error_selection, datasets[i])

            ax.boxplot([df_data[col] for col in df_data.columns], widths=0.2,
                    showfliers=False, showcaps=False, showmeans=False,
                    medianprops=dict(color="red"))

            violin = ax.violinplot([df_data[col] for col in df_data.columns])

            for pc, color in zip(violin['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_edgecolor('gray')
                pc.set_alpha(1)

            for partname in ('cbars', 'cmins', 'cmaxes'):
                violin[partname].set_edgecolor('gray')

            font_size = 17
            ax.set_ylabel(error_selection, fontweight='bold', fontsize=font_size)

            ax.tick_params(axis='y', labelsize=font_size, labelrotation=0, which='both')
            for tick in ax.get_yticklabels():
                tick.set_fontweight('bold')
            ax.set_xticks(np.arange(1, len(list(self.models.keys())) + 1))
            ax.set_xticklabels(list(self.models.keys()), fontsize=font_size, fontweight='bold', rotation=45, ha="center")
            
            ax.tick_params(axis='y', labelsize=font_size)
            ax.set_title(f'({chr(97 + i)}) {datasets[i]}', fontweight='bold', fontsize=font_size)

        if not os.path.exists(self.figures_path):
            os.makedirs(self.figures_path)
        
        violin_path = os.path.join(self.figures_path, f'violin_{error_selection}.pdf')

        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=model)
                        for model, color in zip(list(self.models.keys()), colors)]

        fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=20)

        plt.tight_layout()

        plt.savefig(violin_path, bbox_inches='tight')

        plt.show()
        
    def get_convergencia(self, error_selection='Accuracy'):
        fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(12, 20))
        axes = axes.flatten()
        
        datasets = list(self.datasets.keys())
        colors = plt.cm.tab20b(np.linspace(0, 1, len(self.models)))

        for i, ax in enumerate(axes):
            for idx, (model_name, model) in enumerate(self.models.items()):
                if model.__class__.__name__ in ["gsgpcudaregressor", "GsgpCudaClassifier"]:
                    df_error = self._read_metrics(error_selection, datasets[i], only_gsgp=True)
                    df_name_run1 = self._read_metrics('name_run1', datasets[i], only_gsgp=True)
                    best_gsgp_name = str(int(df_name_run1.loc[df_error[f'{model_name}_{error_selection}'].idxmax(), f'{model_name}_name_run1']))
                    
                    path = os.path.join(os.getcwd(), best_gsgp_name, f'{best_gsgp_name}_fitnestrain.csv')
                    data_gsgp = pd.read_csv(path, header=None, index_col=0)
                    data_traces = list(data_gsgp[1])
                    x = range(0, len(data_traces))
                    
                    ax.plot(x, data_traces, color=colors[idx], label=model_name)

            ax.set_ylabel('Fitness', fontweight='bold', fontsize=17)
            ax.set_title(f'({chr(97 + i)}) {datasets[i]}', fontweight='bold', fontsize=17)
            ax.set_xlabel('Generations', fontweight='bold', fontsize=17)
            ax.tick_params(axis='y', labelsize=17)

        legend_elements = [
            Line2D([0], [0], color=colors[idx], lw=2, label=model_name) 
            for idx, model_name in enumerate(self.models.keys())
            if self.is_gsgp(self.models[model_name])
        ]
        
        fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=20)
        plt.tight_layout()
        
        path = os.path.join(os.getcwd(), self.figures_path, f'convergence_{error_selection}.pdf')
        os.makedirs(self.figures_path, exist_ok=True)
        plt.savefig(path)
        plt.show()