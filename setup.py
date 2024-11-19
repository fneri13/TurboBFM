from setuptools import setup, find_packages

setup(
    name='TurboBFM',
    version='1.0.0',
    author='Francesco Neri, TU Delft',
    license='MIT',
    description='CFD solver with BFM modeling for turbomachinery simulations',
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy'
    ],
    packages=find_packages(),
)
