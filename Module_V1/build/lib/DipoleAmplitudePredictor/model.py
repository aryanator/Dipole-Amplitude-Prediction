import pickle
#from sklearn.preprocessing import StandardScaler
import gzip
import numpy as np
import os

class RandomForestModel:
    def __init__(self, model_path=None, scaler_path=None):
        # Get the base directory of the module
        base_dir = os.path.dirname(os.path.abspath(__file__))
        module_dir = os.path.abspath(os.path.join(base_dir, '..'))

        # Search for model and scaler files if paths are not provided
        self.model_path = model_path if model_path is not None else self.find_file(module_dir, 'rf_model_33.pkl')
        self.scaler_path = scaler_path if scaler_path is not None else self.find_file(module_dir, 'scaler_33.pkl')
        
        # If not found in module_dir, search the entire system
        if not self.model_path:
            print("Model not found in module directory. Searching the entire system...")
            self.model_path = self.find_file('/', 'rf_model_33.pkl')
        else:
            print("Model found at: ", self.model_path)
        if not self.scaler_path:
            print("Scaler not found in module directory. Searching the entire system...")
            self.scaler_path = self.find_file('/', 'scaler_33.pkl')
        else:
            print("Scaler found at: ", self.scaler_path)

        # Raise an error if the files are still not found
        if not self.model_path:
            raise FileNotFoundError("Model file rf_model_33.pkl not found.")
        else:
            print("Model found at: ", self.model_path)
        if not self.scaler_path:
            raise FileNotFoundError("Scaler file scaler_33.pkl not found.")
        else:
            print("Scaler found at: ", self.scaler_path)

        # Load the model and scaler
        self.model = self.load_pickle(self.model_path)
        self.scaler = self.load_pickle(self.scaler_path)

    def find_file(self, directory, filename):
        # Walk through the directory to find the file
        for root, dirs, files in os.walk(directory):
            if filename in files:
                return os.path.join(root, filename)
        return None


    def load_pickle(self, path):
        with gzip.open(path, 'rb') as file:
            return pickle.load(file)

    def predict(self, X):
        X = np.array(X)
        X.reshape(1, -1)
        X_standardized = self.scaler.transform(X)
        predictions = self.model.predict(X_standardized)
        return predictions
    
    r_values = [
    1.01000000e-06, 1.21200000e-06, 1.45440000e-06, 1.74528000e-06,
    2.09433600e-06, 2.51320320e-06, 3.01584384e-06, 3.61901261e-06,
    4.34281513e-06, 5.21137816e-06, 6.25365379e-06, 7.50438454e-06,
    9.00526145e-06, 1.08063137e-05, 1.29675765e-05, 1.55610918e-05,
    1.86733102e-05, 2.24079722e-05, 2.68895666e-05, 3.22674799e-05,
    3.87209759e-05, 4.64651711e-05, 5.57582053e-05, 6.69098464e-05,
    8.02918157e-05, 9.63501788e-05, 1.15620215e-04, 1.38744257e-04,
    1.66493109e-04, 1.99791731e-04, 2.39750077e-04, 2.87700092e-04,
    3.45240111e-04, 4.14288133e-04, 4.97145760e-04, 5.96574911e-04,
    7.15889894e-04, 8.59067873e-04, 1.03088145e-03, 1.23705774e-03,
    1.48446928e-03, 1.78136314e-03, 2.13763577e-03, 2.56516292e-03,
    3.07819551e-03, 3.69383461e-03, 4.43260153e-03, 5.31912184e-03,
    6.38294620e-03, 7.65953544e-03, 9.19144253e-03, 1.10297310e-02,
    1.32356772e-02, 1.58828127e-02, 1.90593752e-02, 2.28712503e-02,
    2.74455003e-02, 3.29346004e-02, 3.95215205e-02, 4.74258246e-02,
    5.69109895e-02, 6.82931874e-02, 8.19518249e-02, 9.83421898e-02,
    1.18010628e-01, 1.41612753e-01, 1.69935304e-01, 2.03922365e-01,
    2.44706838e-01, 2.93648205e-01, 3.52377846e-01, 4.22853416e-01,
    5.07424099e-01, 6.08908919e-01, 7.30690703e-01, 8.76828843e-01,
    1.05219461e+00, 1.26263353e+00, 1.51516024e+00, 1.81819229e+00,
    2.18183075e+00, 2.61819690e+00, 3.14183628e+00, 3.77020353e+00,
    4.52424424e+00, 5.42909308e+00, 6.51491170e+00, 7.81789404e+00,
    9.38147285e+00, 1.12577674e+01, 1.35093209e+01, 1.62111851e+01,
    1.94534221e+01, 2.33441065e+01, 2.80129278e+01, 3.36155134e+01,
    4.03386161e+01, 4.84063393e+01, 5.80876071e+01, 6.97051286e+00,
    8.36461543e+00
]
    def Rgrid(self):
        return self.r_values
    
    
