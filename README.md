---

# Dipole Amplitude Predictor
description: |
  The latest version can be found at: **[DipoleAmplitudeModule](https://github.com/aryanator/DipoleAmplitudeModule)**, which has eased installation!



## Overview

The Dipole Amplitude Predictor module provides a RandomForest-based model for predicting dipole amplitudes. This guide will help you install and use the module effectively.

## Prerequisites

- **Python** (3.x)
- **numpy** (1.25.2)
- **scikit-learn** (1.2.2)

## Installation

1. **Download and Extract:**

   Download the zip file containing the module and extract it. This will be your main directory.

2. **Install the Module:**

   Open a terminal or command prompt and navigate to the main directory where the module is located. Run the following command to install the module:

   ```sh
   cd /path/to/folder/
   pip install -e .
   ```

   For example:

   ```sh
   cd C:/Users/aryan/Downloads/Module_V0
   pip install -e .
   ```

## Usage

After installation, you can use the module as follows:

1. **Create a Python Script:**

   Create a Python script in the main directory and import the module as shown below.

2. **Example Usage:**

   ```python
   from DipoleAmplitudePredictor import RandomForestModel
   import numpy as np

   # Paths to your saved model and scaler
   model_path = r"C:\Users\aryan\Downloads\module2\models\rf_model_33.pkl"
   scaler_path = r"C:\Users\aryan\Downloads\module2\models\scaler_33.pkl"

   # Initialize the model
   rf_model = RandomForestModel()  # Scans the OS for models
   rf_model_paths = RandomForestModel(model_path, scaler_path)

   # Example input array
   X_new = [6.80213521e-05, 7.60105631e-05, 8.49278899e-05, 9.48961447e-05, 1.06028156e-04, 1.18472109e-04, 1.32375933e-04, 1.47911787e-04, 1.65270689e-04, 1.84666744e-04, 2.06338769e-04, 2.30553961e-04, 2.57610535e-04, 2.87841893e-04, 3.21620436e-04, 3.59362142e-04, 4.01532030e-04, 4.48649085e-04, 5.01293915e-04, 5.60114007e-04, 6.25834181e-04, 6.99262232e-04, 7.81302824e-04, 8.72963753e-04, 9.75373848e-04, 1.08979043e-03, 1.21762157e-03, 1.36043585e-03, 1.51998909e-03, 1.69823794e-03, 1.89737105e-03, 2.11982869e-03, 2.36833796e-03, 2.64594136e-03, 2.95603541e-03, 3.30241181e-03, 3.68929871e-03, 4.12141876e-03, 4.60403227e-03, 5.14301691e-03, 5.74491243e-03, 6.41702695e-03, 7.16748365e-03, 8.00535812e-03, 8.94072927e-03, 9.98485026e-03, 1.11502092e-02, 1.24507328e-02, 1.39018663e-02, 1.55208053e-02, 1.73266076e-02, 1.93404444e-02, 2.15857595e-02, 2.40885269e-02, 2.68774740e-02, 2.99843248e-02, 3.34441065e-02, 3.72953477e-02, 4.15804846e-02, 4.63459745e-02, 5.16428064e-02, 5.75264818e-02, 6.40576123e-02, 7.13017086e-02, 7.93298126e-02, 8.82180267e-02, 9.80480664e-02, 1.08906458e-01, 1.20884816e-01, 1.34078623e-01, 1.48586950e-01, 1.64510703e-01, 1.81951410e-01, 2.01008806e-01, 2.21778229e-01, 2.44347425e-01, 2.68792050e-01, 2.95171693e-01, 3.23523061e-01, 3.53855443e-01, 3.86141501e-01, 4.20312850e-01, 4.56249093e-01, 4.93774961e-01, 5.32649387e-01, 5.72566639e-01, 6.13148821e-01, 6.53954286e-01, 6.94478267e-01, 7.34171849e-01, 7.72455242e-01, 8.08748270e-01, 8.42497531e-01, 8.73213445e-01, 9.00505412e-01, 9.24111984e-01, 9.43929546e-01, 9.60016623e-01, 9.72598746e-01, 9.82034747e-01, 9.88802034e-01]
   c2_value = 2.5
   x_bj_target = 1e-3
   X_new.append(c2_value)
   X_new.append(x_bj_target)
   X_new = np.array(X_new).reshape(1, -1)

   # Make predictions
   predictions = rf_model.predict(X_new)
   print(predictions)
   ```

   **Note:** Ensure that `X_new` has a shape of `(no. of samples, 103)`, where the first 101 features correspond to the R_grid. Append the C2 value and the x_bj value for the prediction.

## File Structure

Ensure the following structure in your main directory to compile the module:

```
Module_V0/
|-- DipoleAmplitudePredictor/
|   |-- __init__.py
|   |-- model.py
|-- models/
|   |-- rf_model_33.pkl
|   |-- scaler_33.pkl
|-- README.md
|-- setup.py
```

## Contributing

Feel free to submit issues and pull requests. Ensure your contributions follow the module’s coding style and add tests for new features.

## License

# Models:

V1: rf_1 
Trained on a sample of data. Just for the purpose of setting things up

V2: rf_2
Pruned model, trained on whole data. Fairly accurate.

V3: rf_3
Finetuned large model, trained on whole data. Most accurate.



---
