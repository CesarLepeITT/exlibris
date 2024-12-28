import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.base import clone
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from time import time
from .dataset import Dataset
from .stats import Stats

class StatsGSGP(Stats):
    def __init__(self, experiment_name, n_runs=30, models=None, datasets=None):
        super().__init__(experiment_name, n_runs, models or {}, datasets or {})
    
    def evaluate(self):
        path_save = os.path.join(os.getcwd(), f'stats_{self.experiment_name}')
        os.makedirs(path_save, exist_ok=True)
        
        for dataset_name, dataset in self.datasets.items():
            path_stats = os.path.join(path_save, f'stats_{dataset_name}')
            os.makedirs(path_stats, exist_ok=True)
            
            X, y = Dataset.split_target(dataset)
            df_test = {}

            for run in range(self.n_runs):
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=run)
                
                for model_name, model in self.models.items():
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

                    total_time = time() - start_total
                    
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    auc_roc = roc_auc_score(y_test, y_pred)
                    accuracy = accuracy_score(y_test, y_pred)
                    balanced = balanced_accuracy_score(y_test, y_pred)
                    
                    df_metrics = pd.DataFrame({
                        'Total Time':[total_time],
                        'Fit Time' :[fit_time],
                        'Prediction Time' :[predict_time],
                        'Precision': [precision],
                        'Recall': [recall],
                        'F1 Score': [f1],
                        'AUC ROC': [auc_roc],
                        'Accuracy': [accuracy],
                        'Balanced Accuracy': [balanced]
                    })
                    
                    if getattr(est, '__class__', None) is not None:
                        if est.__class__.__name__ in ["gsgpcudaregressor", "GsgpCudaClassifier"]:
                            df_metrics['name_run1'] = [est.name_run1]

                    self._write_csv(df_metrics, path_metrics)
                    
                    self._save_predictions(path_y_true, y_test, run)
                    self._save_predictions(path_predictions_test, y_pred, run)

    def _save_predictions(self, path, values, run):
        values = values.squeeze().reset_index(drop=True)
        df_test = pd.DataFrame({f"y_{run + 1}": values})
        
        if os.path.exists(path):
            aux = pd.read_csv(path)
            df_test_csv = pd.concat([aux, df_test], axis=1)
        else:
            df_test_csv = df_test
        df_test_csv.to_csv(path, index=False)

    def _read_metrics(self, error_selection, dataset_name, only_gsgp=False):
        models = self._filter_models(only_gsgp)
        path_save = os.path.join(os.getcwd(), f'stats_{self.experiment_name}')
        
        if not os.path.exists(path_save):
            raise Exception(f'No experiments found. Path does not exist: {path_save}')
        
        path_stats = os.path.join(path_save, f'stats_{dataset_name}')
        
        if not os.path.exists(path_stats):
            raise Exception(f'No statistics found for dataset: {dataset_name}')
        
        df_data = pd.DataFrame()
        
        for model_name, model in models.items():
            path_model = os.path.join(path_stats, f'{model_name}')
            path_metrics = os.path.join(path_model, f'metrics_{model_name}.csv')
            
            if not os.path.exists(path_metrics):
                raise Exception(f'Metrics file not found for model: {model_name}')
            
            df_metrics = pd.read_csv(path_metrics)
            df_data[f'{model_name}_{error_selection}'] = df_metrics[f'{error_selection}']
        
        return df_data
    
    def _filter_models(self, only_gsgp):
        if only_gsgp:
            return {model_name: model for model_name, model in self.models.items()
                    if model.__class__.__name__ in ["gsgpcudaregressor", "GsgpCudaClassifier"]}
        return self.models

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
            if self.models[model_name].__class__.__name__ in ["gsgpcudaregressor", "GsgpCudaClassifier"]
        ]
        
        fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=20)
        plt.tight_layout()
        
        path = os.path.join(os.getcwd(), self.figures_path, f'convergence_{error_selection}.pdf')
        os.makedirs(self.figures_path, exist_ok=True)
        plt.savefig(path)
        plt.show()
