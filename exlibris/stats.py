#from . import initialize, load_datasets
import os
import pandas as pd

from .dataset import Dataset
#from dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score
from sklearn.base import clone

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

import numpy as np

from time import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.cm as cm


class Stats():
    def __init__(self, experiment_name, n_runs = 30, models = {}, datasets = {}):
        self.experiment_name = experiment_name
        self.n_runs = n_runs
        self.models = {}
        self.datasets = {}
        self.set_models(models=models)
        self.set_datasets(datasets=datasets)
        
        self.path_figures = os.path.join(os.getcwd(), f'stats_{self.experiment_name}' ,f'fig_{self.experiment_name}')
        
        
    def get_experiment_name(self):
        return self.experiment_name
    
    def get_n_runs(self):
        return self.n_runs
    
    def get_models(self):
        return self.models
    
    def get_datasets(self):
        return self.datasets
    
    
    def set_datasets(self, datasets = {}):
        if datasets == {}:
            path_datasets = Dataset.path_datasets()        
            files = os.listdir(path_datasets)
            for file in files:
                if file.endswith('.csv'):
                    path_dataset = os.path.join(path_datasets, file)
                    name_dataset = os.path.splitext(file)[0]
                    datasets[name_dataset] = pd.read_csv(path_dataset)
        
        if not(isinstance(datasets, dict)):
            raise TypeError(f'Datasets must be a diccionary. {type(datasets)} was provided.')
        
        for value in datasets.values():
            if not isinstance(value, pd.DataFrame):
                raise TypeError(f'Dataset must be a DataFrame. {type(value)} was provided.')
        
        self.datasets = datasets                    
    
    def set_models(self, models=None):
        if models is None:
            models = {}  # Inicializa un diccionario vacío si no se proporciona nada

        #if not isinstance(models, dict):
        #    raise TypeError(f'Models must be a dictionary. {type(models)} was provided.')

        if not models:
            raise Exception('No models were provided.')

        self.models = models
    
    def _write_csv(self, df, path, mode = 'a', header = False):
        if not os.path.exists(path):
            df.to_csv(path, index = False)    
        else:
            df.to_csv(path, mode= mode, header=header, index=False)
        
    
    def evaluate(self):
        path_save = os.path.join(os.getcwd(), f'stats_{self.experiment_name}')#  os.getcwd() + f'/stats_{self.experiment_name}'
        if not os.path.exists(path_save):
            os.makedirs(path_save) 
        
        for dataset_name, dataset in self.datasets.items():
            path_stats = os.path.join(path_save, f'stats_{(dataset_name)}')  # f'{path_save}/stats_{(dataset_name)}'            
            if not os.path.exists(path_stats):
                os.makedirs(path_stats)
            
            X, y = Dataset.split_target(dataset)
            y.squeeze()
            df_test = {}
            
            for run in range(self.n_runs):
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7) #random_state=)
                for model_name, model in self.models.items():
                    est = clone(model)
                    # Carpeta
                    path_model = os.path.join(path_stats, f'{model_name}') #path_stats + f'/{model_name}'
                    if not os.path.exists(path_model):
                        os.makedirs(path_model)           
                    
                    path_predictions_test = os.path.join(path_model , f'predictions_test_{model_name}.csv')#path_model + f'/predictions_{model_name}.csv'
                    path_y_true = os.path.join(path_model , f'y_true_{model_name}.csv')#path_model + f'/predictions_{model_name}.csv'

                    path_metrics = os.path.join(path_model, f'metrics_{model_name}.csv')#path_model + f'/metrics_{model_name}.csv'      
                    
                    start_total = time()

                    start_fit = time()
                    est.fit(X_train, y_train)
                    runtime_fit = time() - start_fit

                    start_predict = time()
                    y_pred = est.predict(X_test)
                    runtime_predict = time() - start_predict

                    runtime = time() - start_total

                    
                    # Calcular estadísticas
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    auc_roc = roc_auc_score(y_test, y_pred)
                    accuracy = accuracy_score(y_test, y_pred)
                    balanced = balanced_accuracy_score(y_test, y_pred)
                    
                    df_metrics = pd.DataFrame()
                    df_metrics['Runtime'] = [runtime]
                    df_metrics['Runtime_fit'] = [runtime_fit]
                    df_metrics['Runtime_predict'] = [runtime_predict]
                    df_metrics['Precision'] = [precision]
                    df_metrics['Recall'] = [recall]
                    df_metrics['F1 Score'] = [f1]
                    df_metrics['auc_roc'] = [auc_roc]
                    df_metrics['Accuracy'] = [accuracy]
                    df_metrics['Balanced Accuracy'] = [balanced]
                    
                    
                    self._write_csv(df_metrics, path_metrics)
                    
                    # Guardamos y_true
                    
                    #y_test = y_test.squeeze()
                    #y_test = y_test.reset_index(drop=True)
    
                    df_test = pd.DataFrame({
                        f"y_true_{run+1}": y_test#.ravel()
                    })
                    
                    # Guardar o actualizar el archivo CSV
                    if os.path.exists(path_y_true):
                        aux = pd.read_csv(path_y_true)
                        df_test_csv = pd.concat([aux, df_test], axis=1)
                    else:
                        df_test_csv = df_test
                    df_test_csv.to_csv(path_y_true, index=False)
                    
                    # Guardamos predicciones 
                    #y_pred = y_pred.squeeze()
                    #y_pred = y_pred.reset_index(drop=True)
    
                    df_test = pd.DataFrame({
                        f"y_pred_{run+1}": y_pred
                    })
                    
                    # Guardar o actualizar el archivo CSV
                    if os.path.exists(path_predictions_test):
                        aux = pd.read_csv(path_predictions_test)
                        df_test_csv = pd.concat([aux, df_test], axis=1)
                    else:
                        df_test_csv = df_test
                    df_test_csv.to_csv(path_predictions_test, index=False)
    
    def _read_metrics(self, error_selection, dataset_name):
        # Va a leer los modelos y los problemas y va    
        path_save = os.path.join(os.getcwd(), f'stats_{self.experiment_name}')#  os.getcwd() + f'/stats_{self.experiment_name}'
        if not os.path.exists(path_save):
            raise Exception(f'No experiments have been conducted. The path could not be found. {path_save}')
        
        #for dataset_name, dataset in self.datasets.items():
        path_stats = os.path.join(path_save, f'stats_{(dataset_name)}')  # f'{path_save}/stats_{(dataset_name)}'            
        if not os.path.exists(path_stats):
            raise Exception(f'No experiments have been conducted. The path could not be found. {path_stats}')
        
        df_data = pd.DataFrame()
        
        for model_name, model in self.models.items():
            path_model = os.path.join(path_stats, f'{model_name}') #path_stats + f'/{model_name}'
            path_predictions_test = os.path.join(path_model , f'predictions_test_{model_name}.csv')#path_model + f'/predictions_{model_name}.csv'
            path_y_true = os.path.join(path_model , f'y_true_{model_name}.csv')#path_model + f'/predictions_{model_name}.csv'
            path_metrics = os.path.join(path_model, f'metrics_{model_name}.csv')#path_model + f'/metrics_{model_name}.csv'      
            
            if not os.path.exists(path_model):
                raise Exception(f'No experiments have been conducted. The path could not be found. {path_model}')          
            if not os.path.exists(path_predictions_test):
                raise Exception(f'No experiments have been conducted. The path could not be found. {path_predictions_test}')         
            if not os.path.exists(path_y_true):
                raise Exception(f'No experiments have been conducted. The path could not be found. {path_y_true}')         
            if not os.path.exists(path_metrics):
                raise Exception(f'No experiments have been conducted. The path could not be found. {path_metrics}')    
                                                                    
            df_metrics = pd.read_csv(path_metrics)
            df_data[f'{model_name}_{error_selection}'] = df_metrics[f'{error_selection}']
        
        return df_data

    def get_violin(self, error_selection, ncols=2, nrows=5, figsize=(12, 20)):

        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
        axes = axes.flatten()  # Aplanar la matriz de ejes para fácil acceso por índice.

        datasets = list(self.datasets.keys())
        
        # Generar la paleta de colores automáticamente usando 'tab20' de matplotlib
        num_colors = len(self.models)
        colors = cm.tab20(np.linspace(0, 1, num_colors))  # Usar 'tab20' como mapa de colores

        # Generar un gráfico violin para cada uno de los problemas.
        for i, ax in enumerate(axes):
            # Crear un diccionario con los resultados de error para cada modelo
            df_data = self._read_metrics(error_selection, datasets[i])

            # Crear el gráfico boxplot
            ax.boxplot([df_data[col] for col in df_data.columns], widths=0.2,
                    showfliers=False, showcaps=False, showmeans=False,
                    medianprops=dict(color="red"))

            # Crear el gráfico violin
            violin = ax.violinplot([df_data[col] for col in df_data.columns])

            # Aplicar los colores especificados a cada "cuerpo" del gráfico violin
            for pc, color in zip(violin['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_edgecolor('gray')
                pc.set_alpha(1)

            # Establecer el color de las partes del gráfico violin (barras de contorno)
            for partname in ('cbars', 'cmins', 'cmaxes'):
                violin[partname].set_edgecolor('gray')

            tamanoletra = 17
            # Configurar etiquetas y ticks del gráfico
            ax.set_ylabel(error_selection, fontweight='bold', fontsize=tamanoletra)

            ax.tick_params(axis='y', labelsize=tamanoletra, labelrotation=0, which='both')
            for tick in ax.get_yticklabels():
                tick.set_fontweight('bold')
            ax.set_xticks(np.arange(1, len(list(self.models.keys())) + 1))
            ax.set_xticklabels(list(self.models.keys()), fontsize=tamanoletra, fontweight='bold', rotation=45, ha="center")
            
            ax.tick_params(axis='y', labelsize=tamanoletra)
            ax.set_title(f'({chr(97 + i)}) {datasets[i]}', fontweight='bold', fontsize=tamanoletra)

        # Definir la ruta donde se guardará el archivo PDF con todos los gráficos generados.
        if not os.path.exists(self.path_figures):
            os.makedirs(self.path_figures)
        
        path_violin = os.path.join(self.path_figures, f'violin_{error_selection}.pdf')

        # Crear una lista de objetos Line2D para la leyenda
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=model)
                        for model, color in zip(list(self.models.keys()), colors)]

        # Agregar la leyenda fuera del gráfico (esto es para todos los subgráficos).
        fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=20)

        # Ajustar el diseño del gráfico para evitar solapamientos.
        plt.tight_layout()

        # Guardar el gráfico como un archivo PDF.
        plt.savefig(path_violin, bbox_inches='tight')

        # Mostrar el gráfico generado.
        plt.show()
 

if __name__ == '__main__':
    pass