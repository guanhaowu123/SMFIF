from setuptools import setup, find_packages

setup(
    name='smfif',
    version='0.1.0',
    description='SMFIF stacking model package',
    author='Liu shu Wu guan hao et al'
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'xgboost',
        'lightgbm',
        'joblib'
    ],
    python_requires='>=3.7',
)