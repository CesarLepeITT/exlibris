import os
import pandas as pd
from .dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.base import clone
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from time import time
import matplotlib.cm as cm

class Stats:
    def __init__(self, experiment_name, n_runs=30, models=None, datasets=None):
        if models is None:
            models = {}
        if datasets is None:
            datasets = {}

        self.experiment_name = experiment_name
        self.n_runs = n_runs
        self.models = models
        self.datasets = datasets
        self.set_models(models=models)
        self.set_datasets(datasets=datasets)
        
        self.figures_path = os.path.join(os.getcwd(), f'stats_{self.experiment_name}', f'fig_{self.experiment_name}')
        
    def get_experiment_name(self):
        return self.experiment_name
    
    def get_n_runs(self):
        return self.n_runs
    
    def get_models(self):
        return self.models
    
    def get_datasets(self):
        return self.datasets
    
    def set_datasets(self, datasets=None):
        if datasets is None:
            datasets = {}

        if not datasets:
            dataset_path = Dataset.path_datasets()        
            files = os.listdir(dataset_path)
            for file in files:
                if file.endswith('.csv'):
                    dataset_file_path = os.path.join(dataset_path, file)
                    dataset_name = os.path.splitext(file)[0]
                    datasets[dataset_name] = pd.read_csv(dataset_file_path)

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
        save_path = os.path.join(os.getcwd(), f'stats_{self.experiment_name}')
        if not os.path.exists(save_path):
            os.makedirs(save_path) 
        
        for dataset_name, dataset in self.datasets.items():
            stats_path = os.path.join(save_path, f'stats_{dataset_name}')    
            if not os.path.exists(stats_path):
                os.makedirs(stats_path)
            
            X, y = Dataset.split_target(dataset)
            y.squeeze()
            df_test = {}

            for run in range(self.n_runs):
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7) 
                for model_name, model in self.models.items():
                    estimator = clone(model)
                    model_path = os.path.join(stats_path, f'{model_name}')
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)           
                    
                    predictions_test_path = os.path.join(model_path, f'predictions_test_{model_name}.csv')
                    y_true_path = os.path.join(model_path, f'y_true_{model_name}.csv')
                    metrics_path = os.path.join(model_path, f'metrics_{model_name}.csv')
                    
                    start_total = time()
                    start_fit = time()
                    estimator.fit(X_train, y_train)
                    fit_time = time() - start_fit

                    start_predict = time()
                    y_pred = estimator.predict(X_test)
                    predict_time = time() - start_predict

                    total_time = time() - start_total

                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    auc_roc = roc_auc_score(y_test, y_pred)
                    accuracy = accuracy_score(y_test, y_pred)
                    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
                    
                    metrics_df = pd.DataFrame()
                    metrics_df['Total Time'] = [total_time]
                    metrics_df['Fit Time'] = [fit_time]
                    metrics_df['Prediction Time'] = [predict_time]
                    metrics_df['Precision'] = [precision]
                    metrics_df['Recall'] = [recall]
                    metrics_df['F1 Score'] = [f1]
                    metrics_df['AUC ROC'] = [auc_roc]
                    metrics_df['Accuracy'] = [accuracy]
                    metrics_df['Balanced Accuracy'] = [balanced_accuracy]
                    
                    self._write_csv(metrics_df, metrics_path)
                    
                    df_test = pd.DataFrame({
                        f"y_true_{run + 1}": y_test
                    })
                    
                    if os.path.exists(y_true_path):
                        aux = pd.read_csv(y_true_path)
                        df_test_csv = pd.concat([aux, df_test], axis=1)
                    else:
                        df_test_csv = df_test
                    df_test_csv.to_csv(y_true_path, index=False)
                    
                    df_test = pd.DataFrame({
                        f"y_pred_{run + 1}": y_pred
                    })
                    
                    if os.path.exists(predictions_test_path):
                        aux = pd.read_csv(predictions_test_path)
                        df_test_csv = pd.concat([aux, df_test], axis=1)
                    else:
                        df_test_csv = df_test
                    df_test_csv.to_csv(predictions_test_path, index=False)
    
    def _read_metrics(self, error_selection, dataset_name):
        save_path = os.path.join(os.getcwd(), f'stats_{self.experiment_name}')
        if not os.path.exists(save_path):
            raise Exception(f'No experiments have been conducted. The path could not be found. {save_path}')
        
        stats_path = os.path.join(save_path, f'stats_{dataset_name}')              
        if not os.path.exists(stats_path):
            raise Exception(f'No experiments have been conducted. The path could not be found. {stats_path}')
        
        df_data = pd.DataFrame()
        
        for model_name, model in self.models.items():
            model_path = os.path.join(stats_path, f'{model_name}')
            predictions_test_path = os.path.join(model_path, f'predictions_test_{model_name}.csv')
            y_true_path = os.path.join(model_path, f'y_true_{model_name}.csv')
            metrics_path = os.path.join(model_path, f'metrics_{model_name}.csv')
            
            if not os.path.exists(model_path):
                raise Exception(f'No experiments have been conducted. The path could not be found. {model_path}')          
            if not os.path.exists(predictions_test_path):
                raise Exception(f'No experiments have been conducted. The path could not be found. {predictions_test_path}')         
            if not os.path.exists(y_true_path):
                raise Exception(f'No experiments have been conducted. The path could not be found. {y_true_path}')         
            if not os.path.exists(metrics_path):
                raise Exception(f'No experiments have been conducted. The path could not be found. {metrics_path}')    
                                                                    
            metrics_df = pd.read_csv(metrics_path)
            df_data[f'{model_name}_{error_selection}'] = metrics_df[f'{error_selection}']
        
        return df_data

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