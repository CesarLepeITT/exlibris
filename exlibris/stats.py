#from . import initialize, load_datasets
import os
import pandas as pd

from .dataset import Dataset
#from dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.base import clone

from time import time



class Stats():
    def __init__(self, experiment_name, n_runs = 30, models = {}, datasets = {}):
        self.experiment_name = experiment_name
        self.n_runs = n_runs
        self.models = {}
        self.datasets = {}
        self.set_models(models=models)
        self.set_datasets(datasets=datasets)
        
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
            print('A\n'*2,type(y))
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
                    
                    df_metrics = pd.DataFrame()
                    df_metrics['runtime'] = [runtime]
                    df_metrics['precision'] = [precision]
                    df_metrics['recall'] = [recall]
                    df_metrics['f1'] = [f1]
                    df_metrics['auc_roc'] = [auc_roc]
                    df_metrics['accuracy'] = [accuracy]
                    
                    self._write_csv(df_metrics, path_metrics)
                    
                    # Guardamos y_true
                    
                    #y_test = y_test.squeeze()
                    #y_test = y_test.reset_index(drop=True)
    
                    df_test = pd.DataFrame({
                        f"y_true_{run+1}": y_test.ravel()
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
    
if __name__ == '__main__':
    pass
    #est = RandomForestClassifier()
    #est2 = LogisticRegression(max_iter=200)
    #stats = Stats('prueba', 2, {"RF": est, "logi": est2})
    #stats.evaluate()




