# my_ml_package/setup.py

from setuptools import setup, find_packages

setup(
    name='DipoleAmplitudePredictor',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
    ],
    description='A package for making predictions for Dipole Amplitude using a pre-trained Random Forest model',
    author='Aryan Patil',
    author_email='aryansanjay.patil@stonybrook.edu',
    url='https://github.com/aryanator/DipoleAmplitudePredictor', 
)
