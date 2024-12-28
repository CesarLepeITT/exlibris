#from . import initialize, load_datasets
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score
from sklearn.base import clone

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

import numpy as np

from time import time

from .dataset import Dataset
from .stats import Stats

class stats_gsgp(Stats):
    def __init__(self, experiment_name, n_runs = 30, models = {}, datasets = {}):
        super().__init__(experiment_name, n_runs, models, datasets)
        
        self
        
    def evaluate(self):
        path_save = os.path.join(os.getcwd(), f'stats_{self.experiment_name}')#  os.getcwd() + f'/stats_{self.experiment_name}'
        if not os.path.exists(path_save):
            os.makedirs(path_save) 
        
        for dataset_name, dataset in self.datasets.items():
            path_stats = os.path.join(path_save, f'stats_{(dataset_name)}')  # f'{path_save}/stats_{(dataset_name)}'            
            if not os.path.exists(path_stats):
                os.makedirs(path_stats)
            
            X, y = Dataset.split_target(dataset)
            df_test = {}
            
            for run in range(self.n_runs):
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=run) #random_state=)
                for model_name, model in self.models.items():
                    est = clone(model)#model
                    # Carpeta
                    path_model = os.path.join(path_stats, f'{model_name}') #path_stats + f'/{model_name}'
                    if not os.path.exists(path_model):
                        os.makedirs(path_model)           
                    
                    path_predictions_test = os.path.join(path_model , f'predictions_test_{model_name}.csv')#path_model + f'/predictions_{model_name}.csv'
                    path_y_true = os.path.join(path_model , f'y_true_{model_name}.csv')#path_model + f'/predictions_{model_name}.csv'

                    path_metrics = os.path.join(path_model, f'metrics_{model_name}.csv')#path_model + f'/metrics_{model_name}.csv'      
                    
                    start = time()
                    est.fit(X_train, y_train)
                    y_pred = est.predict(X_test)
                    runtime = time() - start
                    
                    # Calcular estadísticas
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    auc_roc = roc_auc_score(y_test, y_pred)
                    accuracy = accuracy_score(y_test, y_pred)
                    balanced = balanced_accuracy_score(y_test, y_pred)
                    
                    df_metrics = pd.DataFrame()
                    df_metrics['Runtime'] = [runtime]
                    df_metrics['Precision'] = [precision]
                    df_metrics['Recall'] = [recall]
                    df_metrics['F1 Score'] = [f1]
                    df_metrics['auc_roc'] = [auc_roc]
                    df_metrics['Accuracy'] = [accuracy]
                    df_metrics['Balanced Accuracy'] = [balanced]
                    
                    # Suponiendo que las clases están definidas en el módulo 'gsgp'
                    # Verificar si el objeto 'est' es de tipo 'gsgpcudaregressor' o 'GsgpCudaClassifier'
                    if getattr(est, '__class__', None) is not None:
                        if est.__class__.__name__ == "gsgpcudaregressor" or est.__class__.__name__ == "GsgpCudaClassifier":
                            df_metrics['name_run1'] = [est.name_run1]

                    
                    self._write_csv(df_metrics, path_metrics)
                    
                    # Guardamos y_true
                    
                    y_test = y_test.squeeze()
                    y_test = y_test.reset_index(drop=True)
    
                    df_test = pd.DataFrame({
                        f"y_true_{run+1}": y_test
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
  
        
    def _read_metrics(self, error_selection, dataset_name, only_gsgp = False):
        if only_gsgp:
            models = {model_name: model for model_name, model in self.models.items()
                    if model.__class__.__name__ == "gsgpcudaregressor" or model.__class__.__name__ == "GsgpCudaClassifier"}
        else:
            models = self.models
        # Va a leer los modelos y los problemas y va    
        path_save = os.path.join(os.getcwd(), f'stats_{self.experiment_name}')#  os.getcwd() + f'/stats_{self.experiment_name}'
        if not os.path.exists(path_save):
            raise Exception(f'No experiments have been conducted. The path could not be found. {path_save}')
        
        #for dataset_name, dataset in self.datasets.items():
        path_stats = os.path.join(path_save, f'stats_{(dataset_name)}')  # f'{path_save}/stats_{(dataset_name)}'            
        if not os.path.exists(path_stats):
            raise Exception(f'No experiments have been conducted. The path could not be found. {path_stats}')
        
        df_data = pd.DataFrame()
        
        for model_name, model in models.items():
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

    def get_convergencia(self, error_selection='Accuracy'):
        fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(12, 20))  
        axes = axes.flatten()  # Aplanar la matriz de ejes para fácil acceso por índice.
        
        datasets = list(self.datasets.keys())
        
        # Generar un mapa de colores utilizando 'tab20b' de matplotlib
        colors = plt.cm.tab20b(np.linspace(0, 1, len(self.models)))

        for i, ax in enumerate(axes):       
            # Cargar y graficar los datos de convergencia para cada variante de GSGP.
            for idx, (model_name, model) in enumerate(self.models.items()):
                if getattr(model, '__class__', None) is not None:
                    if model.__class__.__name__ in ["gsgpcudaregressor", "GsgpCudaClassifier"]:
                        print(model_name)
                        
                        df_error = self._read_metrics(error_selection, datasets[i], only_gsgp=True)  
                        df_name_run1 = self._read_metrics('name_run1', datasets[i], only_gsgp=True)
                        best_gsgp_name = str(int(df_name_run1.loc[df_error[f'{model_name}_{error_selection}'].idxmax(), f'{model_name}_name_run1']))
                        
                        # Definir la ruta al archivo CSV con los datos de convergencia (fitness de entrenamiento).
                        path = os.path.join(os.getcwd(), best_gsgp_name, f'{best_gsgp_name}_fitnestrain.csv')
                        
                        # Leer los datos de convergencia del archivo CSV.
                        data_gsgp = pd.read_csv(path, header=None, index_col=0)
                        
                        # Almacenar la lista de valores de fitness de la variante de GSGP.
                        data_traces = list(data_gsgp[1])
                        
                        # Crear el eje X (iteraciones) y el eje Y (valores de fitness).
                        x = range(0, len(data_traces))
                        
                        # Asignar un color específico del mapa de colores
                        ax.plot(x, data_traces, color=colors[idx], label=model_name)

            tamanoletra = 17
            # Configurar etiquetas y ticks del gráfico
            ax.set_ylabel('Fitness', fontweight='bold', fontsize=tamanoletra)
            ax.tick_params(axis='y', labelsize=tamanoletra, labelrotation=0, which='both')
            for tick in ax.get_yticklabels():
                tick.set_fontweight('bold')   
            
            # Establecer el título para cada subgráfico con la letra y el nombre del problema.
            ax.set_title(f'({chr(97 + i)}) {datasets[i]}', fontweight='bold', fontsize=tamanoletra)
            ax.set_xlabel('Generations', fontweight='bold', fontsize=tamanoletra)

        # Crear la leyenda
        legend_elements = [
            Line2D([0], [0], color=colors[idx], lw=2, label=model_name) 
            for idx, model_name in enumerate(self.models.keys())
            if self.models[model_name].__class__.__name__ in ["gsgpcudaregressor", "GsgpCudaClassifier"]
        ]

        # Agregar la leyenda fuera del gráfico (esto es para todos los subgráficos).
        fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=20)
        
        # Ajustar el diseño del gráfico para evitar solapamientos.
        plt.tight_layout()
        
        # Guardar el gráfico como un archivo PDF.
        path = os.path.join(os.getcwd(), self.path_figures, f'convergence_{error_selection}.pdf')
        if not os.path.exists(self.path_figures):
            os.makedirs(self.path_figures)
        plt.savefig(path)
        
        # Mostrar el gráfico generado.
        plt.show()

