import os
import os.path
import pandas as pd

class Dataset():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.set_dataset_path()
    
    def get_dataset_name(self):
        return self.dataset_name
        
    def get_dataset_path(self):
        return self.dataset_path
    
    def set_dataset_path(self):
        lib_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(lib_path, 'datasets', self.dataset_name + '.csv')#os.path.join(os.getcwd(), 'datasets', self.dataset_name + '.csv')
        if os.path.exists(path):
            self.dataset_path = path
        else:
            raise OSError('File not found: ' + path)
    
    def path_datasets():
        lib_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(lib_path, 'datasets')
        #path = os.path.join(os.getcwd(), 'datasets')
        if os.path.exists(path):
            return path
        else:
            raise OSError('File not found: ' + path)
    
    
    def load_dataset(self, split_target = False):
        df = pd.read_csv(self.dataset_path)
        
        if split_target:
            return self.split_target(df)
        
        return df
        
    def split_target(dataset):
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(f'Dataset must be a DataFrame. {type(dataset)} was provided.')
        
        X = dataset.drop(columns=dataset.columns[-1])
        y = dataset.iloc[:, -1:]
        
        return X, y
            
if __name__ == '__main__':
    dataset = print(Dataset('1_fertility').load_dataset(split_target=True))

